import argparse
import os

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from hrrr_utils import DynamicBBoxDataset, custom_collate_fn, load_velocity_dynamic, nll_loss
from hrrr_model_function import ClimateEncoderFreeUncertain, count_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train ClimODE on HRRR data.")
    parser.add_argument("--input-path", default="hrrr_data_train/", help="Input folder path")
    parser.add_argument("--output-folder", default="hrrr_training/", help="Output folder path")
    parser.add_argument("--vel-name", default="hrrr_train_dynamic", help="Velocity file name")
    parser.add_argument("--vel-type", default="file", choices=["file", "folder"], help="Velocity storage type")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--model-name', type=str, default="best_model_hrrr.pth", help='Model file name')
    parser.add_argument('--cache-name', type=str, default=None, help='Metadata cache name in storage')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument("--pretrained-model-path", type=str, default=None, help="Path to a pretrained model to load")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="Train/val/test split ratios", type=lambda s: [float(r) for r in s.split(",")])
    args = parser.parse_args()
    return args


args = parse_arguments()

# Setup distributed training
if args.distributed:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_rank = 0
    world_size = 1
    rank = 0

data_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith('_dataset.nc')]
data_files = sorted(data_files)

const_path = "constants/constants_hrrr.nc"

num_train_files = int(len(data_files) * args.split[0])
num_val_files = int(len(data_files) * args.split[1])
num_test_files = len(data_files) - num_train_files - num_val_files

train_files = data_files[:num_train_files]
val_files = data_files[num_train_files:num_train_files + num_val_files]
test_files = data_files[num_train_files + num_val_files:]

train_files = train_files[:args.limit] if args.limit else train_files
val_files = val_files[:args.limit] if args.limit else val_files
test_files = test_files[:args.limit] if args.limit else test_files
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")
print(f"Number of test files: {len(test_files)}")


if args.pretrained_model_path is None:
    data_vars = ["z", "t", "t2m", "u10", "v10"]
else:
    print(f"Will be fine-tuning: {args.pretrained_model_path}")

    # data_vars = ["z", "t", "t2m", "u10", "v10"]
    data_vars =  ["t2m", "u10", "v10", "z", "t"]

    data_min = {
        "t2m": 232.2499,
        "u10": -28.6147,
        "v10": -29.0914,
        "z": 5101.502,
        "t": 240.2032
    }
    data_max = {
        "t2m": 319.6834, 
        "u10": 33.7216, 
        "v10": 28.5466, 
        "z": 5946.5474, 
        "t": 306.761
    }

if args.cache_name is not None:
    cache_name = args.cache_name
else:
    cache_name = f"metadata_cache_{args.limit if args.limit else 'full'}"

train_dataset = DynamicBBoxDataset(
    train_files, 
    const_path, 
    data_vars, 
    metadata_cache_path=os.path.join(args.output_folder, f"{cache_name}_train.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_train_vel{'.npz' if args.vel_type == 'file' else ''}"),
)
val_dataset = DynamicBBoxDataset(
    val_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"{cache_name}_val.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_val_vel{'.npz' if args.vel_type == 'file' else ''}")
)
test_dataset = DynamicBBoxDataset(
    test_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"{cache_name}_test.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_test_vel{'.npz' if args.vel_type == 'file' else ''}")
)

if args.pretrained_model_path is not None:
    train_dataset.data_min = data_min
    train_dataset.data_max = data_max
    train_dataset.data_min_tensor = torch.tensor([train_dataset.data_min[var] for var in train_dataset.data_vars])
    train_dataset.data_max_tensor = torch.tensor([train_dataset.data_max[var] for var in train_dataset.data_vars])

# Use training dataset stats for normalization
val_dataset.data_min = train_dataset.data_min
val_dataset.data_max = train_dataset.data_max
val_dataset.data_min_tensor = torch.tensor([val_dataset.data_min[var] for var in val_dataset.data_vars])
val_dataset.data_max_tensor = torch.tensor([val_dataset.data_max[var] for var in val_dataset.data_vars])
test_dataset.data_min = train_dataset.data_min
test_dataset.data_max = train_dataset.data_max
test_dataset.data_min_tensor = torch.tensor([test_dataset.data_min[var] for var in test_dataset.data_vars])
test_dataset.data_max_tensor = torch.tensor([test_dataset.data_max[var] for var in test_dataset.data_vars])

# Create distributed samplers if using distributed training
if args.distributed:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
else:
    train_sampler = None
    val_sampler = None
    test_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    pin_memory=True,
    collate_fn=custom_collate_fn,
    drop_last=True,
    num_workers=2 if not args.distributed else 4
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=val_sampler,
    pin_memory=True,
    collate_fn=custom_collate_fn,
    drop_last=True,
    num_workers=2 if not args.distributed else 4
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=test_sampler,
    pin_memory=True,
    collate_fn=custom_collate_fn,
    drop_last=True,
    num_workers=2 if not args.distributed else 4
)


num_channels = len(data_vars)
const_channels = train_dataset.const_cache[0]["const_info"].shape[1]
method = "euler"

print(f"Number of input channels: {num_channels}")
print(f"Number of constant channels: {const_channels}")

model = ClimateEncoderFreeUncertain(
    num_channels=num_channels,
    const_channels=const_channels,
    method=method,
    out_types=num_channels,
    use_att=True,
    use_err=True,
    use_pos=False,
).to(device)
# Load pretrained model if specified
if args.pretrained_model_path is not None:
    if rank == 0:
        print(f"Loading pretrained model from {args.pretrained_model_path}")
    pretrained_dict = torch.load(args.pretrained_model_path, map_location=device)
    model.load_state_dict(pretrained_dict)

# Wrap model with DDP if distributed
if args.distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    model_without_ddp = model.module
else:
    model_without_ddp = model

if rank == 0:  # Only print on main process
    print(model_without_ddp)
    param = count_parameters(model_without_ddp)
    print(f"Number of trainable parameters: {param}")



optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

if rank == 0:
    print(f"\nTraining on {world_size} GPU(s)" if args.distributed else "Training on single GPU")
    print(f"Effective batch size: {args.batch_size * world_size}\n")

val_best_loss = float('inf')
train_best_loss = float('inf')
best_epoch = float('inf')
epochs = args.epochs




for epoch in range(args.epochs):
    # Set epoch for distributed sampler
    if args.distributed:
        train_sampler.set_epoch(epoch)
    
    total_train_loss = 0
    total_val_loss = 0
    total_test_loss = 0
    
    if epoch == 0:
        var_coeff = 0.001
    else:
        var_coeff = 2*scheduler.get_last_lr()[0]
    
    model.train()
    for entry, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad(set_to_none=True)
        data = batch["data"] # 8 x 3 x 5 x 320 x 320
        data = batch["data"][:, 1, :, :, :].to(device) # use t-1 as input, since we want to predict t. shape: 8 x 5 x 320 x 320
        past_sample = batch["velocity"].to(device)
        past_sample = past_sample.view(data.shape[0], 2 * num_channels, data.shape[-2], data.shape[-1])
        # past sample shape: 8 x 10 x 320 x 320
        const_channels_info = batch["const_info"].to(device)
        lat_map = batch["lat_map"].to(device)
        lon_map = batch["lon_map"].to(device)
        time_vals = batch["time_values"].float().to(device)
        time_groups = batch["time_groups"].to(device)
        # print("--- shape info ---")
        # print("data shape:", data.shape)
        # print("past_sample shape:", past_sample.shape)
        # print("const_channels_info shape:", const_channels_info.shape)
        # print("lat_map shape:", lat_map.shape)
        # print("lon_map shape:", lon_map.shape)
        # print("time_vals shape:", time_vals.shape)
        # print("time values:", time_vals)
        # print("time_groups shape:", time_groups.shape)
        # print("time_groups:", time_groups)
        # print("------------------")


        model_without_ddp.update_param(
            [
                past_sample,
                const_channels_info,
                lat_map,
                lon_map
            ]
        )
        mean, std = model(time_vals, time_groups, data)
        
        if rank == 0:
            print(f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB peak")

        target = batch["data"][:, 2, :, :, :].to(device)  # use t as target
        loss = nll_loss(mean, std, target, var_coeff)
        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if rank == 0:
            print("Loss for batch is ",loss.item())
        total_train_loss = total_train_loss + loss.detach().item()
        del data, target, past_sample, const_channels_info, lat_map, lon_map
        del mean, std, loss
        torch.cuda.empty_cache()
        # if torch.isnan(loss) : 
        #     print("Quitting due to Nan loss")
        #     quit()
        # total_train_loss = total_train_loss + loss.item()

    lr_val = scheduler.get_last_lr()[0]
    scheduler.step()
    
    # Synchronize losses across GPUs
    if args.distributed:
        train_loss_tensor = torch.tensor([total_train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        total_train_loss = train_loss_tensor.item() / world_size
    
    avg_train_loss = total_train_loss / len(train_loader)

    if avg_train_loss < train_best_loss:
        train_best_loss = avg_train_loss
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs} - Training Complete")
        print(f"  Current Train Loss: {avg_train_loss:.6f}")
        print(f"  Best Train Loss:    {train_best_loss:.6f}")
        print(f"  Learning Rate:      {lr_val:.6f}")
        print(f"{'='*70}\n")

    model.eval()
    with torch.no_grad():
        for entry, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            data = batch["data"] # 8 x 3 x 5 x 320 x 320 for batch size of 8
            data = batch["data"][:, 1, :, :, :].to(device) # use t-1 as input, since we want to predict t. shape: 8 x 5 x 320 x 320
            past_sample = batch["velocity"].to(device)
            past_sample = past_sample.view(data.shape[0], 2 * num_channels, data.shape[-2], data.shape[-1])
            # past sample shape: 8 x 10 x 320 x 320
            const_channels_info = batch["const_info"].to(device)
            lat_map = batch["lat_map"].to(device)
            lon_map = batch["lon_map"].to(device)
            time_vals = batch["time_values"].float().to(device)
            time_groups = batch["time_groups"].to(device)

            model_without_ddp.update_param(
                [
                    past_sample,
                    const_channels_info,
                    lat_map,
                    lon_map
                ]
            )
            mean, std = model(time_vals, time_groups, data)
            target = batch["data"][:, 2, :, :, :].to(device)  # use t as target
            loss = nll_loss(mean, std, target, var_coeff)
            total_val_loss = total_val_loss + loss.item()
            if rank == 0:
                print("Val Loss for batch is ",loss.item())
    
    # Synchronize validation losses across GPUs
    if args.distributed:
        val_loss_tensor = torch.tensor([total_val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        total_val_loss = val_loss_tensor.item() / world_size

    avg_val_loss = total_val_loss / len(val_loader)
    
    model_saved = ""
    if avg_val_loss < val_best_loss:
        val_best_loss = avg_val_loss
        if rank == 0:  # Only save on rank 0
            save_dict = model_without_ddp.state_dict() if args.distributed else model.state_dict()
            torch.save(save_dict, os.path.join(args.output_folder, args.model_name))
            model_saved = "✓ Model saved!"
    
    if rank == 0:
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs} - Validation Complete")
        print(f"  Current Val Loss: {avg_val_loss:.6f}")
        print(f"  Best Val Loss:    {val_best_loss:.6f} {model_saved}")
        print(f"{'='*70}\n")

total_test_loss = 0.0

model.eval()
with torch.no_grad():
    for entry, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        data = batch["data"] # 8 x 3 x 5 x 320 x 320 for batch size of 8
        data = batch["data"][:, 1, :, :, :].to(device) # use t-1 as input, since we want to predict t. shape: 8 x 5 x 320 x 320

        past_sample = batch["velocity"].to(device)
        past_sample = past_sample.view(data.shape[0], 2 * num_channels, data.shape[-2], data.shape[-1])
        # past sample shape: 8 x 10 x 320 x 320
        const_channels_info = batch["const_info"].to(device)
        lat_map = batch["lat_map"].to(device)
        lon_map = batch["lon_map"].to(device)
        time_vals = batch["time_values"].float().to(device)
        time_groups = batch["time_groups"].to(device)
        model_without_ddp.update_param(
            [
                past_sample,
                const_channels_info,
                lat_map,
                lon_map
            ]
        )
        mean, std = model(time_vals, time_groups, data)
        target = batch["data"][:, 2, :, :, :].to(device)  # use t as target
        loss = nll_loss(mean, std, target, var_coeff)
        total_test_loss = total_test_loss + loss.item()
        if rank == 0:
            print("Test Loss for batch is ",loss.item())

# Synchronize test losses across GPUs
if args.distributed:
    test_loss_tensor = torch.tensor([total_test_loss], device=device)
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    total_test_loss = test_loss_tensor.item() / world_size

avg_test_loss = total_test_loss / len(test_loader)
if rank == 0:
    print(f"\n{'='*70}")
    print(f"Testing Complete")
    print(f"  Final Test Loss: {avg_test_loss:.6f}")
    print(f"{'='*70}\n")

# Cleanup distributed training
if args.distributed:
    dist.destroy_process_group()
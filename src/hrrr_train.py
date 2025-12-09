import argparse
import os

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from hrrr_utils import DynamicBBoxDataset, custom_collate_fn, load_velocity_dynamic, nll_loss
from hrrr_model_function import ClimateEncoderFreeUncertain, count_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train ClimODE on HRRR data.")
    parser.add_argument("--input-path", default="hrrr_data_train/", help="Input folder path")
    parser.add_argument("--output-folder", default="hrrr_training/", help="Output folder path")
    parser.add_argument("--vel-name", default="hrrr_train_dynamic", help="Velocity file name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--model-name', type=str, default="best_model_hrrr.pth", help='Model file name')
    args = parser.parse_args()
    return args


args = parse_arguments()

data_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith('.nc')]
data_files = sorted(data_files)

const_path = "constants/constants_hrrr.nc"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 80% train, 10% val, 10% test split
num_train_files = int(len(data_files) * 0.8)
num_val_files = int(len(data_files) * 0.1)
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

data_vars = ["z", "t", "t2m", "u10", "v10"]


train_dataset = DynamicBBoxDataset(
    train_files, 
    const_path, 
    data_vars, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_train.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_train_vel.npz")
)
val_dataset = DynamicBBoxDataset(
    val_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_val.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_val_vel.npz")
)
test_dataset = DynamicBBoxDataset(
    test_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_test.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_test_vel.npz")
)

# Use training dataset stats for normalization
val_dataset.data_min = train_dataset.data_min
val_dataset.data_max = train_dataset.data_max
val_dataset.data_min_tensor = torch.tensor([val_dataset.data_min[var] for var in val_dataset.data_vars])
val_dataset.data_max_tensor = torch.tensor([val_dataset.data_max[var] for var in val_dataset.data_vars])
test_dataset.data_min = train_dataset.data_min
test_dataset.data_max = train_dataset.data_max
test_dataset.data_min_tensor = torch.tensor([test_dataset.data_min[var] for var in test_dataset.data_vars])
test_dataset.data_max_tensor = torch.tensor([test_dataset.data_max[var] for var in test_dataset.data_vars])

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn,
    drop_last=True
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

print(model)

param = count_parameters(model)
print(f"Number of trainable parameters: {param}")



optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

val_best_loss = float('inf')
train_best_loss = float('inf')
best_epoch = float('inf')
epochs = args.epochs




for epoch in range(args.epochs):
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


        model.update_param(
            [
                past_sample,
                const_channels_info,
                lat_map,
                lon_map
            ]
        )
        mean, std = model(time_vals, time_groups, data)
        
        print(f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB peak")

        target = batch["data"][:, 2, :, :, :].to(device)  # use t as target
        loss = nll_loss(mean, std, target, var_coeff)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
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
    avg_train_loss = total_train_loss / len(train_loader)

    if avg_train_loss < train_best_loss:
        train_best_loss = avg_train_loss
    
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

            model.update_param(
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
            print("Val Loss for batch is ",loss.item())

    avg_val_loss = total_val_loss / len(val_loader)
    
    model_saved = ""
    if avg_val_loss < val_best_loss:
        val_best_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.output_folder, args.model_name))
        model_saved = "Model saved!"
    
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
        model.update_param(
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
        print("Test Loss for batch is ",loss.item())

avg_test_loss = total_test_loss / len(test_loader)
print(f"\n{'='*70}")
print(f"Testing Complete")
print(f"  Final Test Loss: {avg_test_loss:.6f}")
print(f"{'='*70}\n")
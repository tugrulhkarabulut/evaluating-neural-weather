import argparse
import os

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

from hrrr_utils import DynamicBBoxDataset, custom_collate_fn
from hrrr_model_function import ClimateEncoderFreeUncertain, count_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train ClimODE on HRRR data.")
    parser.add_argument("--input-path", default="hrrr_data_train/", help="Input folder path")
    parser.add_argument("--output-folder", default="hrrr_training/", help="Output folder path")
    parser.add_argument("--vel-name", default="hrrr_train_dynamic", help="Velocity file name")
    parser.add_argument("--vel-type", default="file", choices=["file", "folder"], help="Velocity storage type")
    parser.add_argument('--cache-name', type=str, default="metadata_cache_full", help="Cache name prefix for metadata files")
    parser.add_argument('--metrics-name', type=str, default="metrics", help="Output file name prefix for evaluation metrics")
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size.')
    parser.add_argument('--model-path', type=str, default="hrrr_training/best_model_hrrr.pth", help='Path to the model file')
    parser.add_argument("--split", default="0.8,0.1,0.1", help="Train/val/test split ratios", type=lambda s: [float(r) for r in s.split(",")])
    parser.add_argument('--precomputed-stats', action='store_true', help='Whether to use precomputed dataset statistics.')
    args = parser.parse_args()
    return args


args = parse_arguments()

data_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith('_dataset.nc')]
data_files = sorted(data_files)

const_path = "constants/constants_hrrr.nc"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 80% train, 10% val, 10% test split
num_train_files = int(len(data_files) * args.split[0])
num_val_files = int(len(data_files) * args.split[1])
num_test_files = len(data_files) - num_train_files - num_val_files

train_files = data_files[:num_train_files]
val_files = data_files[num_train_files:num_train_files + num_val_files]
test_files = data_files[num_train_files + num_val_files:]

# train_files = train_files[:100]
# val_files = val_files[:100]
# test_files = test_files[:100]
print(f"Number of training files: {len(train_files)}")
print(f"Number of validation files: {len(val_files)}")
print(f"Number of test files: {len(test_files)}")

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

train_dataset = DynamicBBoxDataset(
    train_files, 
    const_path, 
    data_vars, 
    metadata_cache_path=os.path.join(args.output_folder, f"{args.cache_name}_train.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_train_vel{'.npz' if args.vel_type == 'file' else ''}"),
    compute_stats=not args.precomputed_stats,
)
val_dataset = DynamicBBoxDataset(
    val_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"{args.cache_name}_val.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_val_vel{'.npz' if args.vel_type == 'file' else ''}")
)
test_dataset = DynamicBBoxDataset(
    test_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"{args.cache_name}_test.pkl"),
    velocity_path=os.path.join(args.output_folder, f"{args.vel_name}_test_vel{'.npz' if args.vel_type == 'file' else ''}"),
    include_masks=True,
)


if args.precomputed_stats:
    train_dataset.data_min = data_min
    train_dataset.data_max = data_max
    train_dataset.data_min_tensor = torch.tensor([train_dataset.data_min[var] for var in train_dataset.data_vars])
    train_dataset.data_max_tensor = torch.tensor([train_dataset.data_max[var] for var in train_dataset.data_vars])


    print("\nData statistics per variable:")
    for var in train_dataset.data_vars:
        print(f"  {var}: min={train_dataset.data_min[var]:.4f}, max={train_dataset.data_max[var]:.4f}")

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

loaders = {
    # 'train': train_loader,
    # 'val': val_loader,
    'test': test_loader
}


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
model.load_state_dict(torch.load(os.path.join(args.model_path), map_location=device))

print(model)

param = count_parameters(model)
print(f"Number of trainable parameters: {param}")


def evaluation_rmsd_mm(targets, pred_means, lats):
    rmse_list = []
    for target, pred, lat in tqdm(zip(targets, pred_means, lats), total=len(targets), desc="Calculating RMSD"):
        error = pred - target # (4, 5, 320, 320)
        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat = weights_lat / weights_lat.mean()
        rmse = np.sqrt(((error)**2 * weights_lat).mean(axis=(-1, -2)).mean(axis=0))
        rmse_list.append(rmse)
    rmse_array = np.concatenate(rmse_list, axis=0)
    return rmse_array.mean(axis=0)


def evaluation_acc_mm(targets, pred_means, lats):
    acc_list = []
    
    for target, pred, lat in tqdm(zip(targets, pred_means, lats), total=len(targets), desc="Calculating ACC"):
        target_anomaly = target - target.mean(axis=0, keepdims=True)
        pred_anomaly = pred - pred.mean(axis=0, keepdims=True)
        
        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat = weights_lat / weights_lat.mean()
        weights_lat_broadcast = weights_lat[None, None, :, None]
        
        numerator = (weights_lat_broadcast * pred_anomaly * target_anomaly).sum(axis=(2, 3))
        
        pred_var = (weights_lat_broadcast * pred_anomaly**2).sum(axis=(2, 3))
        target_var = (weights_lat_broadcast * target_anomaly**2).sum(axis=(2, 3))
        denominator = np.sqrt(pred_var * target_var)
        
        acc = numerator / (denominator + 1e-10)
        
        acc_list.append(acc.mean(axis=0))
    acc_array = np.stack(acc_list, axis=0)
    return acc_array.mean(axis=0)


class Evaluator:
    def __init__(self):
        self.rmse_list = []
        self.acc_list = []

    def update_rmse(self, target, pred, lat, mask=None):
        error = pred - target # (4, 5, 320, 320)
        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat = weights_lat / weights_lat.mean()
        
        if mask is not None:
            # mask shape: (batch, height, width)
            # Expand mask to match error shape: (batch, channels, height, width)
            mask_expanded = mask[:, None, :, :]
            weights_lat_masked = weights_lat * mask_expanded
            # Compute weighted sum and divide by sum of weights
            squared_error_weighted = (error ** 2) * weights_lat_masked
            sum_weights = weights_lat_masked.sum(axis=(-2, -1), keepdims=True)
            # Avoid division by zero
            sum_weights = np.where(sum_weights > 0, sum_weights, 1.0)
            rmse = np.sqrt((squared_error_weighted.sum(axis=(-2, -1)) / sum_weights.squeeze((-2, -1))).mean(axis=0))
        else:
            rmse = np.sqrt(((error)**2 * weights_lat).mean(axis=(-2, -1)).mean(axis=0))
        
        self.rmse_list.append(rmse)

    def update_acc(self, target, pred, lat, mask=None):
        if mask is not None:
            # mask shape: (batch, height, width)
            mask_expanded = mask[:, None, :, :]
            # Compute anomalies only over masked points
            target_masked = np.where(mask_expanded, target, np.nan)
            pred_masked = np.where(mask_expanded, pred, np.nan)
            # Compute mean over spatial dimensions (2, 3) only, per batch and channel
            target_anomaly = target - np.nanmean(target_masked, axis=(2, 3), keepdims=True)
            pred_anomaly = pred - np.nanmean(pred_masked, axis=(2, 3), keepdims=True)
        else:
            target_anomaly = target - target.mean(axis=0, keepdims=True)
            pred_anomaly = pred - pred.mean(axis=0, keepdims=True)

        weights_lat = np.cos(np.deg2rad(lat))
        weights_lat = weights_lat / weights_lat.mean()
        
        if mask is not None:
            mask_expanded = mask[:, None, :, :]
            weights_lat_masked = weights_lat * mask_expanded
            numerator = (weights_lat_masked * pred_anomaly * target_anomaly).sum(axis=(-2, -1))
            pred_var = (weights_lat_masked * pred_anomaly**2).sum(axis=(-2, -1))
            target_var = (weights_lat_masked * target_anomaly**2).sum(axis=(-2, -1))
        else:
            numerator = (weights_lat * pred_anomaly * target_anomaly).sum(axis=(-2, -1))
            pred_var = (weights_lat * pred_anomaly**2).sum(axis=(-2, -1))
            target_var = (weights_lat * target_anomaly**2).sum(axis=(-2, -1))
        
        denominator = np.sqrt(pred_var * target_var)
        acc = numerator / (denominator + 1e-10)
        acc = acc.mean(axis=0)
        self.acc_list.append(acc)

    def update(self, target, pred, lat, mask=None):
        self.update_rmse(target, pred, lat, mask)
        self.update_acc(target, pred, lat, mask)

    def compute_metrics(self):
        print(len(self.rmse_list))
        rmse_array = np.concatenate(self.rmse_list, axis=0)
        acc_array = np.concatenate(self.acc_list, axis=0)
        self.rmse_array = rmse_array
        self.acc_array = acc_array

        return {
            "rmse": rmse_array.mean(axis=0),
            "acc": acc_array.mean(axis=0)
        }

model.eval()
with torch.no_grad():
    for loader_name, loader in loaders.items():
        print(f"\n{'='*30} Starting evaluation on {loader_name} set {'='*30}\n")

        evaluator = Evaluator()

        for entry, batch in tqdm(enumerate(loader), total=len(loader)):
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
            if entry % 100 == 0:
                print(f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB peak")
            mean = mean.cpu()
            target = batch["data"][:, 2, :, :, :].cpu()  # use t as target
            mean = train_dataset.denormalize(mean)
            target = train_dataset.denormalize(target)
            mean = mean.numpy()
            target = target.numpy()

            std = std.cpu()
            std = std * (train_dataset.data_max_tensor.view(1, -1, 1, 1) - train_dataset.data_min_tensor.view(1, -1, 1, 1))
            std = std.numpy()

            lat = batch["lat_map"].cpu().numpy()
            lon = batch["lon_map"].cpu().numpy()

            mask = batch["mask"].cpu().numpy() if batch["mask"] is not None else None

            evaluator.update(target, mean, lat, mask)

        metrics = evaluator.compute_metrics()

        output_file = os.path.join(args.output_folder, f"{args.metrics_name}_{loader_name}.npz")
        np.savez(
            output_file,
            rmse_array=evaluator.rmse_array,
            acc_array=evaluator.acc_array,
            rmse_mean=metrics["rmse"],
            acc_mean=metrics["acc"],
            data_vars=data_vars
        )
        print(f"Saved metrics to {output_file}")
        
        # Print results in a pretty table
        var_names =  ["t2m", "u10", "v10", "z", "t"]
        print(f"\n{'='*60}")
        print(f"{'Evaluation Results for ' + loader_name + ' set':^60}")
        print(f"{'='*60}")
        print(f"{'Variable':<15} {'RMSE':>15} {'ACC':>15}")
        print(f"{'-'*60}")
        for var_idx, var_name in enumerate(var_names):
            rmse_val = metrics["rmse"][var_idx]
            acc_val = metrics["acc"][var_idx]
            print(f"{var_name:<15} {rmse_val:>15.4f} {acc_val:>15.4f}")
        print(f"{'='*60}\n")
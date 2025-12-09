import argparse
import os

from torch.utils.data import DataLoader
import torch

from hrrr_utils import DynamicBBoxDataset, custom_collate_fn, fit_velocity_dynamic
from hrrr_model_function import OptimVelocity


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fit velocity model for HRRR data.")
    parser.add_argument("--input-path", default="hrrr_data_train/", help="Input folder path")
    parser.add_argument("--output-folder", default="hrrr_training/", help="Output folder path")
    parser.add_argument("--output-name", default="hrrr_dynamic", help="Output name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()
    return args


args = parse_arguments()

data_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith('.nc')]
data_files = sorted(data_files)
if args.limit is not None:
    data_files = data_files[:args.limit]

const_path = "constants/constants_hrrr.nc"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_files = int(len(data_files) * 0.8)
num_val_files = int(len(data_files) * 0.1)
num_test_files = len(data_files) - num_train_files - num_val_files

train_files = data_files[:num_train_files]
val_files = data_files[num_train_files:num_train_files + num_val_files]
test_files = data_files[num_train_files + num_val_files:]

data_vars = ["z", "t", "t2m", "u10", "v10"]


train_dataset = DynamicBBoxDataset(
    train_files, 
    const_path, 
    data_vars, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_train.pkl")
)
val_dataset = DynamicBBoxDataset(
    val_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_val.pkl")
)
test_dataset = DynamicBBoxDataset(
    test_files, 
    const_path, 
    data_vars, 
    compute_stats=False, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_{args.limit if args.limit else 'full'}_test.pkl")
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
    batch_size=8,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn
)

loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}


for key in loaders:
    print(f"Fitting velocity model for {key} dataset...")
    fit_velocity_dynamic(
        data_loader=loaders[key], 
        device=device, 
        vel_model=OptimVelocity, 
        H=320, 
        W=320,
        output_folder=args.output_folder,
        output_name=f"{args.output_name}_{key}",
    )
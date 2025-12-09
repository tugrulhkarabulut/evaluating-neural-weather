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

data_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith('_dataset.nc')]
data_files = sorted(data_files)
if args.limit is not None:
    data_files = data_files[:args.limit]

const_path = "constants/constants_hrrr.nc"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Data statistics per variable:
#   t2m: min=0.0000, max=326.1181
#   u10: min=-38.8419, max=34.8059
#   v10: min=-37.4751, max=33.8374
#   z: min=0.0000, max=6014.8604
#   t: min=0.0000, max=311.1464


dataset = DynamicBBoxDataset(
    data_files, 
    const_path, 
    data_vars, 
    metadata_cache_path=os.path.join(args.output_folder, f"metadata_cache_hr_extreme_{args.limit if args.limit else 'full'}_train.pkl")
)




# Use training dataset stats for normalization
dataset.data_min = data_min
dataset.data_max = data_max
dataset.data_min_tensor = torch.tensor([dataset.data_min[var] for var in dataset.data_vars])
dataset.data_max_tensor = torch.tensor([dataset.data_max[var] for var in dataset.data_vars])

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    pin_memory=False,
    collate_fn=custom_collate_fn
)

loaders = {
    'test': loader,
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
        save_individual=True,
    )
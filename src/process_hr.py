import os
from utils import get_train_test_data_without_scales_batched, add_constant_info
import torch
from torch.utils.data import DataLoader
import argparse
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
import numpy as np
import tarfile
import xarray as xr
import xesmf as xe
from tqdm import tqdm
import warnings 
import re
import pandas as pd
import concurrent.futures
import shutil


# ClimODE variables geopotential_500, temperature_850, t2m, 10u, 10m,
# INDICES = [1, 2, 3, 11, 53]
INDICES = [11, 53, 1, 2, 3]
DATA_VARS = ["z", "t", "t2m", "u10", "v10"]

WIDTH = 320
HEIGHT = 320

LATITUDE_START = 21.1
LATITUDE_END = 52.6
LONGITUDE_START = 225.9
LONGITUDE_END = 299.1

HR_GRID_WIDTH = 1799
HR_GRID_HEIGHT = 1059




def compute_latlon_bounding_box(bounding_box):
    width_start, height_start, width_end, height_end = bounding_box
    x, y = width_start, height_start
    x_max, y_max = width_end, height_end
    left, lower = x, y
    height, width = 1059, 1799
    if y+320>height and x+320>width:
        right, upper = x_max, y_max     # right lower left upper are area need to crop
        lat = coords[upper-320:upper, 0, 0]
        lon = coords[0, right-320:right, 1]
    elif y+320>height and x+320<=width:
        right, upper = min(x+320, x_max), min(y+320,y_max) 
        lat = coords[upper-320:upper, 0, 0]
        lon = coords[0, left:left+320, 1]
    elif x+320>width:
        right, upper = min(x+320, x_max), min(y+320,y_max)
        lat = coords[lower:lower+320, 0, 0]
        lon = coords[0, right-320:right, 1]
    else:
        right, upper = min(x + 320, width), min(y + 320, height)
        lat = coords[lower:upper, 0, 0]
        lon = coords[0, left:right, 1]

    return lat, lon
	#print(f"[DEBUG] Bounding box lats: {lat0}, {lat1} (should be in [-90, 90])")
	#return (lon0, lat0, lon1, lat1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process high-resolution climate data")
    parser.add_argument("--data-dir", type=str, default="./hr-extreme", help="Path to the data directory")
    parser.add_argument("--output-dir", type=str, default="./hr-processed", help="Path to the output directory")
    parser.add_argument("--output-resolution", type=float, default=5.625, help="Output resolution in degrees")
    parser.add_argument("--output-width", type=int, default=32, help="Output width in degrees")
    parser.add_argument("--output-height", type=int, default=64, help="Output height in degrees")
    parser.add_argument("--num-files", type=int, help="Number of data files to process for testing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for parallel processing")
    parser.add_argument('--downloadRaw', type=bool, default=False)
    return parser.parse_args()

def process_data_file(file, output_dir):
    print(f"\nDownloading data file: {file}")
    file = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=local_dir,
        force_download=True
    )
    # Create a unique extraction directory for this tar file
    tar_basename = os.path.splitext(os.path.basename(file))[0]
    extract_dir = os.path.join(local_dir, tar_basename)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(file, "r") as tar:
        tar.extractall(path=extract_dir)

    for npz_file in os.listdir(extract_dir):
        if not npz_file.endswith('.npz'):
            continue

        match = re.match(r"(\d{10})", npz_file)
        if match:
            date_str = match.group(1)  # example: "2020070100"
            base_date = pd.to_datetime(date_str, format="%Y%m%d%H")
        else:
            print(f"Filename {npz_file} does not match expected datetime pattern.")
            continue

        # getting bounding box
        npz_path = npz_file
        numbers = npz_path.split('_')[-4:]
        numbers[-1] = numbers[-1].replace('.npz', '')
        bounding_box = list(map(int, numbers))
        print("Bounding box in pixel coordinates:", bounding_box)
        lat_coords, lon_coords = compute_latlon_bounding_box(bounding_box)

        try:
            data = np.load(os.path.join(extract_dir, npz_file), allow_pickle=True)
            inputs = data['inputs'][:, :, INDICES, :, :][0]  # shape: (2, 5, 320, 320)
            targets = data['targets'][:, :, INDICES, :, :][0]  # shape: (1, 5, 320, 320)
            inputs = np.concatenate([inputs, targets], axis=0)  # shape: (3, 5, 320, 320)
            masks = data['masks'][0]  # (320, 320)
        except KeyError as e:
            os.remove(os.path.join(extract_dir, npz_file))
            print(f"KeyError: {e} in file {npz_file}. Skipping this file.")
            continue
        except Exception as e:
            os.remove(os.path.join(extract_dir, npz_file))
            print(f"Error: {e} while processing file {npz_file}. Skipping this file.")
            continue

        input_dims = ["time", "channel", "lat", "lon"]
        mask_dims = ["lat", "lon"]

        n_time = inputs.shape[0]
        time_coords = [base_date + pd.Timedelta(hours=i) for i in range(n_time)]

        data = {'inputs': inputs, 'masks': masks}
        data_dims = {'inputs': input_dims, 'masks': mask_dims}

        # Split channels into separate variables in a Dataset
        data_vars = {
            var: (["time", "lat", "lon"], inputs[:, i, :, :])
            for i, var in enumerate(DATA_VARS)
        }
        #lat_coords = np.linspace(bounding_box[1], bounding_box[3], WIDTH)
        #lon_coords = np.linspace(bounding_box[0], bounding_box[2], HEIGHT)
        coords = {
            'time': time_coords,
            'lat': lat_coords,
            'lon': lon_coords
        }
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
        mask_xr = xr.DataArray(
            masks,
            coords={'lat': lat_coords, 'lon': lon_coords},
            dims=mask_dims,
            name='mask'
        )
        dataset_path=os.path.join(output_dir, f'processed_{npz_file.replace(".npz", "")}_dataset.nc')
        dataset.to_netcdf(dataset_path)
        mask_path=os.path.join(output_dir, f'processed_{npz_file.replace(".npz", "")}_mask.nc')
        mask_xr.to_netcdf(mask_path)

        os.remove(os.path.join(extract_dir, npz_file))
    os.remove(file)
    shutil.rmtree(extract_dir)


if __name__ == "__main__":
    args = parse_arguments()

    # List all files in the dataset
    all_files = list_repo_files("NianRan1/HR-Extreme", repo_type="dataset")
    print(f"Total files in dataset: {len(all_files)}")

    # Files to download
    metadata_files = [
        'index_file/data_all202007_info_new.csv', 
        'index_file/extreme_data_info1.csv', 
        'index_file/extreme_data_info2.csv', 
        'make_dataset_by_index_file.py', 
        'make_nwp_predictions.py', 
        'output/fuxi.png', 
        'output/fuxi2020070222.png', 
        'output/hrheim.png', 
        'output/hrheim2020070222.png', 
        'output/nwp.png', 
        'output/nwp2020081520.png', 
        'output/pangu.png', 
        'output/pangu2020081520.png', 
        'statistics/means_stds_hrrr.npz', 
        'test_nwp.py'
    ]
    
    # Download the files
    repo_id = "NianRan1/HR-Extreme"
    local_dir = args.data_dir
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)


    coords = np.load('latlon_grid_hrrr.npy')


    for filename in metadata_files:
        filepath = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir
        )

    
    data_files = [f for f in all_files if f.startswith("202007_202012/")]
    if args.num_files:
        data_files = data_files[:args.num_files]  # Limit to first N files for testing

    #coords = np.load('latlon_grid_hrrr.npy')

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_data_file, file, args.output_dir): file
            for file in data_files
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as exc:
                print(f"Error processing file: {futures[future]}: {exc}")

    # for file in tqdm(data_files, desc="Processing data files"):
    #     process_data_file(file, args.output_dir)


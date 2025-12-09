import os
import re

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

# DATA_VARS = ["z", "t", "t2m", "u10", "v10"]
DATA_VARS = ["t2m", "u10", "v10", "z", "t"]

WIDTH = 320
HEIGHT = 320

coords = np.load('latlon_grid_hrrr.npy')

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

if __name__ == "__main__":
        
    input_path = "hrrr_data/"
    output_path = "hrrr_data_train/"
    os.makedirs(output_path, exist_ok=True)

    for npz_file in tqdm(os.listdir(input_path)):
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
            data = np.load(os.path.join(input_path, npz_file), allow_pickle=True)
            inputs = data['inputs'][0]  # shape: (2, 5, 320, 320)
            targets = data['targets'][0]  # shape: (1, 5, 320, 320)
            inputs = np.concatenate([inputs, targets], axis=0)  # shape: (3, 5, 320, 320)
        except Exception as e:
            os.remove(os.path.join(input_path, npz_file))
            print(f"Error: {e} while processing file {npz_file}. Skipping this file.")
            continue


        n_time = inputs.shape[0]
        time_coords = [base_date + pd.Timedelta(hours=i) for i in range(n_time)]

        data_vars = {
            var: (["time", "lat", "lon"], inputs[:, i, :, :])
            for i, var in enumerate(DATA_VARS)
        }
        #lat_coords = np.linspace(bounding_box[1], bounding_box[3], WIDTH)
        #lon_coords = np.linspace(bounding_box[0], bounding_box[2], HEIGHT)
        coords_xr = {
            'time': time_coords,
            'lat': lat_coords,
            'lon': lon_coords
        }
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords_xr
        )
        dataset_path=os.path.join(output_path, f'processed_{npz_file.replace(".npz", "")}_dataset.nc')
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        dataset.to_netcdf(dataset_path)
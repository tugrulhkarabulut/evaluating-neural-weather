import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta
import json
from tqdm import tqdm
import pandas as pd
import warnings
import argparse
from herbie import FastHerbie
from concurrent.futures import ThreadPoolExecutor,as_completed

#### this file is the old version stored in kmsw0eastau

warnings.filterwarnings("ignore")
def crop_images(image, x_min, x_max, y_min, y_max):
    cropped_images = []
    height, width = 1059,1799
    for y in range(y_min, y_max, 320):
        for x in range(x_min, x_max, 320):
            left, upper = x,y
            if y+320>height and x+320>width:
                right, lower = x_max,y_max     # right lower left upper are area need to crop
                crop = image[...,lower-320:lower,right-320:right]
            elif y+320>height and x+320<=width:
                right, lower = min(x+320, x_max), min(y+320,y_max) 
                crop = image[...,lower-320:lower,left:left+320]
            elif x+320>width:
                right, lower = min(x+320, x_max), min(y+320,y_max)
                crop = image[...,upper:upper+320,right-320:right]
            else:
                right, lower = min(x + 320, width), min(y + 320, height)
                crop = image[...,upper:lower, left:right]
            # Determine if padding is needed (at the edges of the original image)
            pad_height = 320 - crop.shape[-2]
            pad_width = 320 - crop.shape[-1]
            # Create a mask for the valid area within the specified range
            # Apply padding if necessary
            if pad_height > 0 or pad_width > 0:
                print('should not have pad')
                print(crop.shape)
                assert False
            cropped_images.append(crop)
    return cropped_images
        
def get_cropped_images(files,names,x_min, x_max, y_min, y_max):
    images = np.concatenate( [files[i.strftime("%Y%m%d%H")] for i in names], axis=0)
    #print(images.shape)
    cropped_images = crop_images(images,x_min, x_max, y_min, y_max)
    #print(len(cropped_images),cropped_images[0].shape,len(masks),masks[0].shape)
    return cropped_images

def get_date_range(dates,pres=2,after=2):
    results = []
    for d in dates:
        results.append(pd.date_range(start=d-timedelta(hours=pres),end=d+timedelta(hours=after),freq='H'))
    return results

single_vars = ["2t", "10u", "10v"]
# atmos_vars = ["hgtn", "u", "v", "t"]
atmos_vars = ["hgtn_500", "t_850"]

var_name = single_vars + atmos_vars
var_name = np.array(var_name)
herbie_maps = {'hgtn': 'HGT',
    'msl': 'MSLMA:mean sea level',
    '2t': 'TMP:2 m above',
    '10u': 'UGRD:10 m above',
    '10v': 'VGRD:10 m above',
    't': 'TMP'}
new_var_names = []
for var in var_name:
    if var in ['msl','2t','10u','10v']:
        new_var_names.append(herbie_maps[var])
    else:
        var,level_ = var.split('_')
        searchString = f"{herbie_maps[var]}:{level_} mb"
        new_var_names.append(searchString)
            
def load_variable(H_pre,searchString):
    return H_pre.xarray(searchString).to_array().values[0]

def multi_core(H_pre,var_name,max_workers=16):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_variable = {executor.submit(load_variable, H_pre, var): var for var in var_name}
        for future in as_completed(future_to_variable):
            variable = future_to_variable[future]
            try:
                data = future.result()
                results[variable] = data
            except Exception as exc:
                print(f'{variable} generated an exception: {exc}')
    return results


def load_files_herbie(filenames,new_var_names=new_var_names):
    #### multi core version
    filenames = [f.strftime("%Y%m%d%H")+'.npy' for f in filenames]
    files = {}
    #filenames = ['2019051108.npy', '2019051109.npy', '2019051110.npy']
    if len(filenames)==0:
        return files
    for f in filenames:
        date_str = f.split('.')[0]
        files[date_str] = np.zeros([1,5,1059,1799])
        date = pd.to_datetime(date_str, format='%Y%m%d%H')
        H_pre = FastHerbie([date], model="hrrr", fxx=[0],product='prs',max_threads=50)
        pre_array = multi_core(H_pre,new_var_names)
        for i,var in enumerate(new_var_names):
            files[date_str][0,i] = pre_array[var]
    return files

'''
def load_files_herbie(filenames,new_var_names=new_var_names):
    #### single-core version
    filenames = [f.strftime("%Y%m%d%H")+'.npy' for f in filenames]
    files = {}
    #filenames = ['2019051108.npy', '2019051109.npy', '2019051110.npy']
    if len(filenames)==0:
        return files
    for f in filenames:
        date_str = f.split('.')[0]
        files[date_str] = np.zeros([1,69,1059,1799])
        date = pd.to_datetime(date_str, format='%Y%m%d%H')
        H_pre = FastHerbie([date], model="hrrr", fxx=[0],product='prs',max_threads=50)
        #pre_array = multi_core(H_pre,new_var_names)
        for i,var in enumerate(new_var_names):
            files[date_str][0,i] = H_pre.xarray(var).to_array().values[0]
    return files
'''

def need_pad(x_min, x_max, y_min, y_max):
    has_pad = False
    height, width = 1059,1799
    image = np.zeros([1,1059,1799])
    for y in range(y_min, y_max, 320):
        for x in range(x_min, x_max, 320):
            left, upper = x,y
            right, lower = min(x + 320, width), min(y + 320, height)
            crop = image[...,upper:lower, left:right]
            pad_height = 320 - crop.shape[-2]
            pad_width = 320 - crop.shape[-1]
            if pad_height > 0 or pad_width > 0:
                has_pad = True
    return has_pad
# /#pde
# /#blob/kmsw0eastau/data


def generate_data_points(begin_datetime, end_datetime, num_points=100):
    height, width = 1059,1799
    date_range = pd.date_range(start=begin_datetime,end=end_datetime, freq='h')

    data_points = []
    for _ in range(num_points):
        type = "Nice_Weather"
        begin_time = date_range[np.random.randint(0, len(date_range))]
        end_time = begin_time + timedelta(hours=np.random.randint(4, 24))
        min_x = np.random.randint(0, width - 320)
        min_y = np.random.randint(0, height - 320)
        max_x = min_x + 320
        max_y = min_y + 320
        bounding_box = f"{min_x}_{min_y}_{max_x}_{max_y}"


        data_points.append((type, begin_time, end_time, bounding_box))

    df_index = pd.DataFrame(data_points, columns=['type', 'begin_time', 'end_time', 'bounding_box'])
    return df_index

parser = argparse.ArgumentParser(description='Process an integer input.')

parser.add_argument('begin_datetime', type=str, help='The begin datetime in format YYYYMMDD')
parser.add_argument('end_datetime', type=str, help='The end datetime in format YYYYMMDD')
# Parse the arguments

args = parser.parse_args()
print(f'start {args.begin_datetime} {args.end_datetime}')
# df = pd.read_csv('/kmsw/extreme_dataset/data_all202007_info_new.csv',parse_dates=['begin_time','end_time'])
df = generate_data_points(args.begin_datetime, args.end_datetime, num_points=500)
df = df[(df.begin_time>=args.begin_datetime)&(df.begin_time<args.end_datetime)]
#print('len df',len(df))
save_dir = "./hrrr_data"
exist_files = os.listdir(save_dir)
pres = 2 # input images number
after = 0 # output images number
for idx, row in tqdm(df.iterrows(),total=len(df)):
    event_span = pd.date_range(start=row['begin_time'],end=row['end_time'], freq='h')
    datetimes_all = get_date_range(event_span,pres,after) # a list of 5 time stamps
    
    gap = 24 # each time process 24 hours at most
    for i in range(0,len(datetimes_all),gap):
        datetimes = datetimes_all[i:i+gap]
        x_min,y_min, x_max,y_max = [int(i) for i in row['bounding_box'].split('_')]
        unique_datetimes = []
        for i in datetimes:
            for j in i:
                if j not in unique_datetimes:
                    unique_datetimes.append(j)
        print('len of unique datetimes:',len(unique_datetimes))
        
        need_save = False
        for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
            save_name =  '_'.join([timestamp[pres].strftime("%Y%m%d%H"),row['type'],row['bounding_box']]) + '.npz'
            if save_name not in exist_files:
                need_save = True
                break

        if need_save:
            print('start',x_min,x_max,y_min,y_max)
            try:
                imgs = load_files_herbie(unique_datetimes,new_var_names=new_var_names)
                #imgs = load_images(unique_datetimes)
                # imgs = dict {'2020070116':np.array(1,69,1059,1799)}
                for timestamp in datetimes: # timestamp ['2020-07-01 16:00:00', '2020-07-01 17:00:00','2020-07-01 18:00:00', '2020-07-01 19:00:00','2020-07-01 20:00:00']
                    save_name =  '_'.join([timestamp[pres].strftime("%Y%m%d%H"),row['type'],row['bounding_box']]) + '.npz'
                    
                    cropped_images = get_cropped_images(imgs,timestamp,x_min, x_max, y_min, y_max)
                    cropped_images = np.stack(cropped_images)
                    # (4,5,69,320,320) numberOfImagesToCoverArea, timestamp, channels, height, width
                    print('save name:',os.path.join(save_dir,save_name))
                    
                    np.savez(os.path.join(save_dir,save_name),inputs= cropped_images[:,:pres],targets=cropped_images[:,pres:])
            except Exception as e:
                print("Error processing ",row['begin_time'],row['end_time'],row['bounding_box'])
                print(str(e))
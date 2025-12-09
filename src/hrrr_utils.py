import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
from tqdm import tqdm
import pickle
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)


def add_constant_info_dynamic(constants, lat_coords, lon_coords):
    const_interp = constants.interp(lat=lat_coords, lon=lon_coords, method='linear')
    
    const_channels = ['orography', 'lsm']
    const_list = []
    
    for channel in const_channels:
        if channel in const_interp:
            const_list.append(torch.from_numpy(const_interp[channel].values).float())
    
    const_channels_info = torch.stack(const_list, dim=0).unsqueeze(0)  # [1, C, H, W]
    
    H, W = len(lat_coords), len(lon_coords)
    lat_map = torch.zeros(1, 1, H, W)
    lon_map = torch.zeros(1, 1, H, W)
    
    for i in range(H):
        lat_map[0, 0, i, :] = float(lat_coords[i])
    for j in range(W):
        lon_map[0, 0, :, j] = float(lon_coords[j])

    return const_channels_info, lat_map, lon_map


def timestamps_to_hour_indices_vectorized(timestamps, reference_year=2020):
    timestamps = np.array(timestamps, dtype='datetime64[ns]')
    year_start = np.datetime64(f'{reference_year}-01-01T00:00:00', 'ns')
    hours_since_start = (timestamps - year_start) / np.timedelta64(1, 'h')
    return hours_since_start.astype(int)


class DynamicBBoxDataset(Dataset):
    def __init__(self, nc_file_paths, const_path, data_vars, precompute_constants=True, compute_stats=True, metadata_cache_path=None, velocity_path=None, include_masks=False):
        self.nc_files = nc_file_paths
        self.const_path = const_path
        self.data_vars = data_vars
        self.metadata_cache_path = metadata_cache_path
        self.compute_stats = compute_stats
        self.precompute_constants = precompute_constants
        self.velocity_path = velocity_path
        self.include_masks = include_masks

        self.data_min = {var: float('inf') for var in data_vars}
        self.data_max = {var: float('-inf') for var in data_vars}


        self.constants = xr.open_dataset(self.const_path)

        if self.velocity_path is not None:
            if os.path.isdir(self.velocity_path):
                vel_files = [f for f in os.listdir(self.velocity_path) if f.startswith('vel_') and f.endswith('.npy')]
                self.velocity_keys = sorted(vel_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            else:
                with np.load(self.velocity_path) as vel_file:
                    self.velocity_keys = sorted(vel_file.keys(), key=lambda x: int(x.split('_')[1]))
            self.velocities = None 
        else:
            self.velocity_keys = None

        print("Loading dataset metadata...")
        if self.metadata_cache_path and os.path.exists(self.metadata_cache_path):
            self._load_metadata_from_cache(self.metadata_cache_path)
        else:
            self.metadata = []
            for nc_file in tqdm(nc_file_paths, desc="Reading netCDF metadata"):
                ds = xr.open_dataset(nc_file)
                self.metadata.append({
                    'file': nc_file,
                    'lat_coords': ds.lat.values,
                    'lon_coords': ds.lon.values,
                    'time_coords': ds.time.values,
                    'time_values': timestamps_to_hour_indices_vectorized(ds.time.values),
                    'shape': (len(ds.lat), len(ds.lon))
                })
                
                if compute_stats:
                    for var in self.data_vars:
                        if var in ds:
                            var_data = ds[var].values
                            self.data_min[var] = min(self.data_min[var], float(np.nanmin(var_data)))
                            self.data_max[var] = max(self.data_max[var], float(np.nanmax(var_data)))
                
                ds.close()


            self._save_metadata_to_cache(self.metadata_cache_path) if self.metadata_cache_path else None
            
        self.data_min_tensor = torch.tensor([self.data_min[var] for var in self.data_vars])
        self.data_max_tensor = torch.tensor([self.data_max[var] for var in self.data_vars])

        if compute_stats:
            print("\nData statistics per variable:")
            for var in self.data_vars:
                print(f"  {var}: min={self.data_min[var]:.4f}, max={self.data_max[var]:.4f}")
        
        self.const_cache = {}
        self.data_cache = {}
        if precompute_constants:
            print("Precomputing constant information for all bounding boxes...")
            self._precompute_constants()
    
    def add_vel(self, velocity_path):
        if velocity_path is not None:
            self.velocity_path = velocity_path
            if os.path.isdir(self.velocity_path):
                vel_files = [f for f in os.listdir(self.velocity_path) if f.startswith('vel_') and f.endswith('.npy')]
                self.velocity_keys = sorted(vel_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            else:
                with np.load(self.velocity_path) as vel_file:
                    self.velocity_keys = sorted(vel_file.keys(), key=lambda x: int(x.split('_')[1]))
            self.velocities = None
    def _save_metadata_to_cache(self, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'metadata': self.metadata,
        }
        if self.compute_stats:
            cache_data['data_min'] = self.data_min
            cache_data['data_max'] = self.data_max
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_metadata_from_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        for key in cache_data:
            setattr(self, key, cache_data[key])


    def _precompute_constants(self):
        unique_bboxes = {}
        
        for idx, meta in enumerate(tqdm(self.metadata, desc="Analyzing bboxes")):
            lat_min, lat_max = float(meta['lat_coords'].min()), float(meta['lat_coords'].max())
            lon_min, lon_max = float(meta['lon_coords'].min()), float(meta['lon_coords'].max())
            shape = meta['shape']
            bbox_key = (lat_min, lat_max, lon_min, lon_max, shape[0], shape[1])
            
            if bbox_key not in unique_bboxes:
                unique_bboxes[bbox_key] = {
                    'lat_coords': meta['lat_coords'],
                    'lon_coords': meta['lon_coords'],
                    'shape': shape,
                    'indices': []
                }
            unique_bboxes[bbox_key]['indices'].append(idx)
        
        print(f"Found {len(unique_bboxes)} unique bounding boxes")
        
        for bbox_key, bbox_info in tqdm(unique_bboxes.items(), desc="Precomputing constants"):
            lat_coords = bbox_info['lat_coords']
            lon_coords = bbox_info['lon_coords']
            
            const_info, lat_map, lon_map = add_constant_info_dynamic(
                self.constants, lat_coords, lon_coords
            )
            
            for idx in bbox_info['indices']:
                self.const_cache[idx] = {
                    'const_info': const_info,
                    'lat_map': lat_map,
                    'lon_map': lon_map,
                    'lat_coords': lat_coords,
                    'lon_coords': lon_coords,
                    'shape': bbox_info['shape']
                }
    
    def normalize(self, data_tensor):
        normalized = data_tensor.clone()
        normalized = (data_tensor - self.data_min_tensor.view(1, -1, 1, 1)) / \
                     (self.data_max_tensor.view(1, -1, 1, 1) - self.data_min_tensor.view(1, -1, 1, 1))
        
        return normalized
    
    def denormalize(self, normalized_tensor):
        denormalized = normalized_tensor.clone()
        denormalized = normalized_tensor * (self.data_max_tensor.view(1, -1, 1, 1) - self.data_min_tensor.view(1, -1, 1, 1)) + \
                       self.data_min_tensor.view(1, -1, 1, 1)
        
        return denormalized
    
    def __len__(self):
        return len(self.nc_files)
    
    def __getitem__(self, idx):
        if idx in self.data_cache:
            ds = self.data_cache[idx]
        else:
            ds = xr.open_dataset(self.nc_files[idx])
            ds_mask = xr.open_dataset(self.nc_files[idx].replace('_dataset.nc','_mask.nc')) if self.include_masks else None
            data_tensor = ds[self.data_vars].to_array().to_numpy()
            data_tensor = np.transpose(data_tensor, (1, 0, 2, 3))
            data_tensor = torch.from_numpy(data_tensor).float()
            mask = torch.from_numpy(ds_mask.to_array().to_numpy().squeeze()).float() if ds_mask is not None else None

            ds.close()
            ds_mask.close() if ds_mask is not None else None

        sample = {
            'data': self.normalize(data_tensor[:3]),
            'time_coords': self.metadata[idx]['time_coords'][:3],
            'time_values': self.metadata[idx]['time_values'][:3],
            'idx': idx,
            'file': self.nc_files[idx],
            'mask': mask,
        }
        
        # Add precomputed constants
        if idx in self.const_cache:
            sample.update(self.const_cache[idx])
        else:
            # Fallback: compute on-the-fly (shouldn't happen if precompute=True)
            meta = self.metadata[idx]
            const_info, lat_map, lon_map = add_constant_info_dynamic(
                self.const_path, meta['lat_coords'], meta['lon_coords']
            )
            sample.update({
                'const_info': const_info,
                'lat_map': lat_map,
                'lon_map': lon_map,
                'lat_coords': meta['lat_coords'],
                'lon_coords': meta['lon_coords'],
                'shape': meta['shape']
            })

        if self.velocity_path is not None:
            if os.path.isdir(self.velocity_path):
                vel_file = self.velocity_keys[idx]
                file_path = os.path.join(self.velocity_path, vel_file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Velocity file {file_path} not found for idx {idx}")
                vel_data = np.load(file_path)
            else:
                # Aggregated NPZ case
                with np.load(self.velocity_path) as vel_file:
                    vel_key = self.velocity_keys[idx]
                    vel_data = vel_file[vel_key]
            sample['velocity'] = torch.from_numpy(vel_data).float()
            
        return sample
    


def custom_collate_fn(batch):
    collated = {}
    
    collated['data'] = torch.stack([item['data'] for item in batch])
    collated['const_info'] = torch.stack([item['const_info'] for item in batch])
    collated['time_values'] = torch.stack([torch.from_numpy(item['time_values']).long() for item in batch])
    time_sums = collated['time_values'].sum(dim=1)
    _, inverse_indices = torch.unique(time_sums, return_inverse=True)
    collated['time_groups'] = inverse_indices
    
    collated['time_coords'] = [item['time_coords'] for item in batch]
    
    collated['idx'] = [item['idx'] for item in batch]
    
    if 'lat_map' in batch[0]:
        collated['lat_map'] = torch.stack([item['lat_map'] for item in batch])
        collated['lon_map'] = torch.stack([item['lon_map'] for item in batch])
        collated['lat_coords'] = [item['lat_coords'] for item in batch]
        collated['lon_coords'] = [item['lon_coords'] for item in batch]
        collated['shape'] = [item['shape'] for item in batch]

    if 'velocity' in batch[0]:
        collated['velocity'] = torch.stack([item['velocity'] for item in batch])

    collated['file'] = [item['file'] for item in batch]
    if 'mask' in batch[0] and batch[0]['mask'] is not None:
        collated['mask'] = torch.stack([item['mask'] for item in batch])
    else:
        collated['mask'] = None

    return collated

def get_delta_u(u_vel,t_steps):
    t = t_steps.flatten().float()
    input_u_vel = u_vel.view(u_vel.shape[0],u_vel.shape[1],-1)
    coeffs = natural_cubic_spline_coeffs(t, input_u_vel)
    spline = NaturalCubicSpline(coeffs)
    point = t[-1]
    out = spline.derivative(point).view(-1,u_vel.shape[2],u_vel.shape[3],u_vel.shape[4])
    
    return out


def optimize_vel_dynamic(data,delta_u,vel_model,kernel,H,W,steps=200):
    model = vel_model(data.shape[0],H,W).to(data.device)
    optimizer = optim.Adam(model.parameters(),lr=2)
    best_loss = float('inf')
    loss_step = []
    for step in range(steps):
        optimizer.zero_grad()
        out,v_x,v_y = model(data)
        grad_vx = torch.gradient(v_x, dim=[-2, -1]) # local smoothness
        grad_vy = torch.gradient(v_y, dim=[-2, -1])
        smoothness_loss = (grad_vx[0]**2 + grad_vx[1]**2 + grad_vy[0]**2 + grad_vy[1]**2).mean()


        vel_loss = nn.MSELoss()(delta_u,out) + 1e-7 * smoothness_loss
        loss_step.append(vel_loss.item())
        if vel_loss.item() < best_loss:
            best_loss = vel_loss.item()
            final_vx = v_x
            final_vy = v_y
            final_out = out
        vel_loss.backward()
        optimizer.step()

    return final_vx, final_vy, loss_step, final_out

def fit_velocity_dynamic(data_loader, device, vel_model, H, W, output_folder, output_name, save_individual=False):
    
    npz_file_path = os.path.join(output_folder, output_name + "_vel.npz")
    if (not save_individual) and os.path.exists(npz_file_path):
        os.remove(npz_file_path)

    velocity_dict = {} if not save_individual else None
    sample_idx = 0

    if save_individual:
        individual_dir = os.path.join(output_folder, output_name)
        os.makedirs(individual_dir, exist_ok=True)
    
    for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc="Computing velocities")):
        sample = batch["data"].to(device)
        data = sample[:, 1, :, :, :]
        past_sample = sample[:, [0,1], :, :, :]
        delta_u = get_delta_u(past_sample,torch.tensor([0,1]).to(device))

        v_x,v_y,loss_terms,out = optimize_vel_dynamic(data,delta_u,vel_model,None,H,W)
        final_v = torch.cat([v_x,v_y],dim=1)  # [batch_size, 2*channels, H, W]
        
        final_v_cpu = final_v.detach().cpu().numpy()
        for i in range(final_v_cpu.shape[0]):
            if save_individual:
                np.save(os.path.join(individual_dir, f'vel_{sample_idx}.npy'), final_v_cpu[i:i+1])
            else:
                velocity_dict[f'vel_{sample_idx}'] = final_v_cpu[i:i+1]  # Keep [1, channels, H, W] shape
            sample_idx += 1
        
        del v_x, v_y, final_v, final_v_cpu, data, past_sample, delta_u, out, sample
        torch.cuda.empty_cache()
    
    if save_individual:
        print(f"Saved individual velocities under {output_folder}/{output_name}/")
    else:
        np.savez_compressed(npz_file_path, **velocity_dict)
        print(f"Saved aggregated velocities to {output_name}_vel.npz")

def load_velocity_dynamic(vel_path):
    if vel_path.endswith('.npz'):
        with np.load(vel_path) as data:
            keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
            velocities = [data[k] for k in keys]
        return torch.from_numpy(np.concatenate(velocities, axis=0))
    else:
        velocities = np.load(vel_path, allow_pickle=True)
        return torch.from_numpy(velocities)

def nll_loss(mean,std,truth,var_coeff):
    normal_lkl = torch.distributions.normal.Normal(mean, 1e-3 + std)
    lkl = -normal_lkl.log_prob(truth)
    loss_val = lkl.mean() + var_coeff*(std**2).mean()
    return loss_val

def evaluation_rmsd_mm(Pred,Truth,lat,lon,max_vals,min_vals,H,W,levels):
    RMSD_final = []
    RMSD_lat_lon = []
    true_lat_lon = []
    pred_lat_lon = []
    for idx,lev in enumerate(levels):
        true_idx = idx
        das_pred = []
        das_true = []
        pred_spectral = Pred[idx].detach().cpu().numpy()
        true_spectral = Truth[true_idx,:,:].detach().cpu().numpy()

        pred = pred_spectral*(max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_pred.append(xr.DataArray(pred.reshape(1,H,W),dims=['time','lat','lon'],coords={'time':[0],'lat':lat,'lon':lon},name=lev))
        Pred_xr = xr.merge(das_pred)
        
        true = true_spectral*(max_vals[idx] - min_vals[idx]) + min_vals[idx]

        das_true.append(xr.DataArray(true.reshape(1,H,W),dims=['time','lat','lon'],coords={'time':[0],'lat':lat,'lon':lon},name=lev))
        True_xr = xr.merge(das_true)
        error = Pred_xr - True_xr
        weights_lat = np.cos(np.deg2rad(error.lat))
        weights_lat /= weights_lat.mean()
        rmse = np.sqrt(((error)**2 * weights_lat).mean(dim=['lat','lon'])).mean(dim=['time'])
        lat_lon_rmse = np.sqrt((error)**2)
        RMSD_lat_lon.append(lat_lon_rmse[lev].values)
        RMSD_final.append(rmse[lev].values.tolist())

    return RMSD_final
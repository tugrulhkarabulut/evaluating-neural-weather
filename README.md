# Evaluating ClimODE on HRRR and HR-Extreme

This repository contains code for training and evaluating ClimODE on High-Resolution Rapid Refresh (HRRR) and HR-Extreme datasets.

HRRR dataset generation script is adapted from: [https://github.com/HuskyNian/HR-Extreme](https://github.com/HuskyNian/HR-Extreme)

Model training and evaluation scripts are adapted from: [https://github.com/Aalto-QuML/ClimODE](https://github.com/Aalto-QuML/ClimODE).

All the source code is under src folder. Refer to the doc folder for project report and slides. 

## Setup

Either build a Docker image using the Dockerfile or set up a virtual environment and install the packages that are mentioned in it.

## Data Preparation

### Generating HRRR Dataset

Generate HRRR training dataset:

Example:
```bash
python make_hrrr_dataset.py 20200101 20200630
```

This script downloads HRRR data from NOAA for the specified date range, extracts 320×320 spatial patches, and saves them as NPZ files in the `hrrr_data/` directory.

### Processing HRRR Dataset

Convert raw HRRR data into training-ready NetCDF:

```bash
python process_hrrr.py
```

This script reads NPZ files from `hrrr_data/` and extracts 320x320 patches with lat/lon coordinates, and saves the processed files to `hrrr_data_train/`.

### Processing HR-Extreme Dataset

Process HR-Extreme data from Hugging Face or local archives:

```bash
python process_hr.py --num-workers 8 --output-dir hr-processed
```

## Initial Velocity Estimation

Before training, estimate the initial velocity field for the ODE solver:

### For HRRR Data

```bash
python hrrr_fit_velocity.py \
    --input-path hrrr_data_train/ \
    --output-folder hrrr_training/ \
    --output-name hrrr_train_dynamic \
```

### For HR-Extreme Data

```bash
python hr_extreme_fit_velocity.py \
    --input-path hr-processed/ \
    --output-folder hrrr_training/ \
    --output-name hr_extreme_dynamic
```

## Training

### Single-GPU Training

Train ClimODE on HRRR data:

```bash
python hrrr_train.py \
    --input-path hrrr_data_train/ \
    --output-folder hrrr_training/ \
    --vel-name hrrr_train_dynamic \
    --vel-type folder \
    --batch-size 4 \
    --lr 0.0005 \
    --epochs 30 \
    --model-name best_model_hrrr.pth
```

### Distributed Training


```bash
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    hrrr_train_dist.py \
    --input-path hrrr_data_train/ \
    --output-folder hrrr_training/ \
    --vel-name hrrr_train_dynamic \
    --vel-type folder \
    --batch-size 4 \
    --epochs 30
```

### Fine-Tuning on HR-Extreme

Fine-tune on HR-Extreme data:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --use_env \
    hrrr_train_dist.py \
    --distributed \
    --world-size 3 \
    --input-path hr-processed/ \
    --output-folder hrrr_training/ \
    --vel-name hr_extreme_dynamic_ft \
    --vel-type folder \
    --model-name best_model_hrrr_dist_ft.pth \
    --pretrained-model-path hrrr_training/best_model_hrrr_dist.pth \
    --batch-size 2 \
    --lr 0.000001 \
    --epochs 10 \
    --cache-name metadata_cache_hr_extreme \
    --split 0.2,0.1,0.7
```

## Evaluation

Evaluate trained models on test data:

### HRRR Evaluation

```bash
python hrrr_eval.py \
    --input-path hrrr_data_train/ \
    --output-folder hrrr_training/ \
    --vel-name hrrr_train_dynamic \
    --vel-type folder \
    --batch-size 4 \
    --model-path hrrr_training/best_model_hrrr.pth \
    --metrics-name metrics_hrrr \
    --split 0.8,0.1,0.1
```

### HR-Extreme Evaluation

```bash
python hrrr_eval.py \
    --input-path hr-processed/ \
    --output-folder hrrr_training/ \
    --vel-name hr_extreme_dynamic_ft \
    --vel-type folder \
    --cache-name metadata_cache_hr_extreme \
    --metrics-name metrics_ft \
    --batch-size 4 \
    --model-path hrrr_training/best_model_hrrr_dist_ft_v3.pth \
    --precomputed-stats \
    --split 0.2,0.1,0.7
```
# PSG-MAE

This repository contains the official source code for **PSG-MAE**, as described in the paper: **"PSG-MAE: Robust Multitask Sleep Event Monitoring using Multichannel PSG Reconstruction and Inter-channel Contrastive Learning"**. PSG-MAE is a masked autoencoder framework for multichannel polysomnography (PSG) signals, supporting robust pretraining and transfer to multiple downstream tasks such as sleep staging and sleep apnea detection.

## Pretraining
```bash
python pre-train.py --epochs 200 --batch-size 64
```

## Downstream Tasks

### Sleep Staging
```bash
python train_staging.py \
  --epochs 100 \
  --batch_size 256 \
  --h5_path /data3/wyf/SHHS_npy_staging_h5 \
  --model_weight encoder_weight/best_encoder.pth \
  --val_interval 10 \
  --augment \
  --save_dir /home/wyf/PSG-MAE
```

### Sleep Apnea Detection
```bash
python train_apnea.py \
  --epochs 100 \
  --batch_size 256 \
  --h5_path /data3/wyf/SHHS_npy_apnea_h5 \
  --model_weight encoder_weight/best_encoder.pth \
  --val_interval 10 \
  --augment \
  --save_dir /home/wyf/PSG-MAE
```

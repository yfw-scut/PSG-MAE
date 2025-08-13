## Pretraining
```bash
python pre-train.py --epochs 200 --batch-size 64
```

## Downstream Tasks

### Sleep Staging
```bash
python train_staging.py \
  --h5_path /data3/wyf/SHHS_npy_staging_h5 \
  --model_weight encoder_weight/best_encoder.pth \
  --epochs 100 \
  --val_interval 10 \
  --augment \
  --save_dir /home/wyf/PSG-MAE
```

### Sleep Apnea Detection
```bash
python train_apnea.py \
  --h5_path /data3/wyf/SHHS_npy_apnea_h5 \
  --model_weight encoder_weight/best_encoder.pth \
  --epochs 100 \
  --val_interval 10 \
  --augment \
  --save_dir /home/wyf/PSG-MAE
```

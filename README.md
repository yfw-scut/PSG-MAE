# PSG-MAE

This repository contains the official source code for **PSG-MAE**, as described in the paper: **“PSG-MAE: Robust Multitask Sleep Event Monitoring using Multichannel PSG Reconstruction and Inter-channel Contrastive Learning.”**
PSG-MAE is a masked autoencoder framework for multichannel polysomnography (PSG) signals, enabling robust pretraining and transfer to downstream tasks such as sleep staging and sleep apnea detection.

---

## Environment

> Tested with **Python 3.8**. You can install via Conda or pip.

**requirements**
```txt
numpy==1.24.4
scipy==1.10.1
pandas==2.0.3
joblib==1.3.2
h5py==3.11.0
torch==1.12.0+cu113
torchvision==0.13.0+cu113
scikit-learn==1.3.2
mne==1.6.0
PyWavelets==1.4.1
matplotlib==3.7.4
seaborn==0.13.2
pillow==10.1.0
tqdm==4.66.1
umap-learn==0.5.7
pynndescent==0.5.13
```
---

## Data & Preprocessing

**Model input shape** per epoch: **(5, 3000)** = **5 channels × 30 s × 100 Hz**.
For faster training, preprocessed samples are packed into **HDF5 (.h5)** files.

### Notebooks (in repository root)
- **SHHS**
  - `SHHS_preprocessing_staging.ipynb` (sleep staging/pretraining)
  - `SHHS_preprocessing_apnea.ipynb` (apnea)
- **MESA**
  - `MESA_staging.ipynb` (sleep staging/pretraining)
  - `MESA_apnea.ipynb` (apnea)
- **PSG audio**
  - `psg_audio_preprocessing.ipynb`(pretraining)
- **HDF5 packing**
  - `h5.ipynb` — converts per-epoch NumPy files into compressed `.h5` datasets for efficient training

### What preprocessing produces
1. Run the dataset notebooks above: the **output directory** will contain multiple **subfolders**.
   Each subfolder corresponds to one raw PSG **.edf** file and contains all **30 s epochs** saved as `.npy`.
2. Run **`h5.ipynb`** to pack those `.npy` epochs into **HDF5** datasets used by the training scripts.
---

## Pretraining the PSG Encoder (PSG-MAE)
To pretrain the PSG encoder in PSG-MAE, first preprocess the data intended for pretraining and pack it using **`h5.ipynb`**. Then run:
```bash
python pre-train.py --epochs 200 --batch-size 64
```

> The encoder weights are saved to `encoder_weight/` (e.g., `encoder_weight/best_encoder.pth`) and can be loaded by downstream tasks.

---

## Downstream Task Training and Evaluation
To train and evaluate the downstream tasks, preprocess the data for the corresponding task, pack it using **`h5.ipynb`**, and then run:

### Sleep Staging
```bash
python train_staging.py   --epochs 100   --batch_size 256   --h5_path /data3/wyf/SHHS_npy_staging_h5   --model_weight encoder_weight/best_encoder.pth   --val_interval 5   --augment   --save_dir /home/wyf/PSG-MAE
```

### Sleep Apnea Detection
```bash
python train_apnea.py   --epochs 100   --batch_size 256   --h5_path /data3/wyf/SHHS_npy_apnea_h5   --model_weight encoder_weight/best_encoder.pth   --val_interval 5   --augment   --save_dir /home/wyf/PSG-MAE
```
---

## Downstream Task Fine-tuning
To fine-tune the downstream tasks, preprocess the MESA data used for downstream training, pack it using **`h5.ipynb`**, place the best model weights under the `best_model/` folder, and then run:

### Sleep Staging
```bash
python fine_tuning_staging.py   --epochs 100   --batch_size 256   --h5_path /data3/wyf/MESA_npy_staging_h5   --model_weight best_model/best_staging_model.pth   --val_interval 2   --augment   --save_dir /home/wyf/PSG-MAE
```

### Sleep Apnea Detection
```bash
python fine_tuning_apnea.py   --epochs 10   --batch_size 256   --h5_path /data3/wyf/MESA_npy_apnea_h5   --model_weight best_model/best_apnea_model.pth   --val_interval 2   --augment   --save_dir /home/wyf/PSG-MAE
```
---

## Citation
If you find this repository useful, please cite:

```bibtex
@article{psgmae2025,
  title   = {PSG-MAE: Robust Multitask Sleep Event Monitoring using Multichannel PSG Reconstruction and Inter-channel Contrastive Learning},
  author  = {Yifei Wang, Qi Liu, Fuli Min, and Honghao Wang},
  year    = {2025}
}
```



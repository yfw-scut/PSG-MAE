import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class DownstreamSleepDataset(Dataset):
    def __init__(self, h5_directory, subject_list, transform=None):
        self.h5_directory = h5_directory
        self.subject_list = subject_list
        self.transform = transform

        self.samples = [] 
        for subject in subject_list:
            h5_path = os.path.join(h5_directory, f"{subject}.h5")
            if not os.path.exists(h5_path):
                print(f"[Warning] {h5_path} not found. Skipped.")
                continue

            with h5py.File(h5_path, 'r') as h5f:
                for key in h5f.keys():
                    label = self._extract_label_from_key(key)
                    if label is not None:
                        self.samples.append((subject, key, label))
                    else:
                        print(f"[Skip] Invalid label in key: {key} of {subject}")

        self.file_cache = {} 

        print(f"[Dataset Init] Loading subjects: {len(subject_list)}")
        print(f"[Dataset Init] Total valid samples loaded: {len(self.samples)}")

    def _extract_label_from_key(self, key):
        try:
            label_char = key[-1]
            if label_char.isdigit():
                return int(label_char)
        except:
            pass
        return None

    def __getitem__(self, idx):
        subject, key, label = self.samples[idx]
        h5_path = os.path.join(self.h5_directory, f"{subject}.h5")

        if subject not in self.file_cache:
            self.file_cache[subject] = h5py.File(h5_path, 'r')

        h5f = self.file_cache[subject]
        data = h5f[key][()] 
        tensor = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label, subject

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        for h5f in self.file_cache.values():
            try:
                h5f.close()
            except:
                pass


# ------------ Augmentations ------------ #

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01, prob=0.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if random.random() < self.prob:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor


class RandomChannelDrop:
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, tensor):
        if random.random() < self.drop_prob:
            num_channels = tensor.size(0)
            drop_ch = random.randint(0, num_channels - 1)
            tensor[drop_ch] = 0
        return tensor


class RandomTimeStretch:
    def __init__(self, stretch_range=(0.9, 1.1), prob=0.3):
        self.stretch_range = stretch_range
        self.prob = prob

    def __call__(self, tensor):
        if random.random() < self.prob:
            factor = random.uniform(*self.stretch_range)
            time_len = tensor.size(1)
            new_len = int(time_len * factor)

            stretched = F.interpolate(
                tensor.unsqueeze(0), size=new_len, mode='linear', align_corners=False
            ).squeeze(0)

            if new_len > time_len:
                return stretched[:, :time_len]
            else:
                pad = time_len - new_len
                return F.pad(stretched, (0, pad))
        return tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor):
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

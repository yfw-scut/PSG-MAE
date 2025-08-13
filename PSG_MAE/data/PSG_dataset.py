import os
import torch
import numpy as np
import h5py
from torch.utils import data

class PSG_dataset(data.Dataset):
    def __init__(self, h5_directory=None, subject_list=[], transform=None, wt_mask=False):

        self.h5_directory = h5_directory
        self.subjects = subject_list
        self.wt_mask = wt_mask

        if not os.path.isdir(self.h5_directory):
            raise ValueError(f"HDF5 directory not found: {self.h5_directory}")

        self.samples = self.make_dataset()

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found! Check your subject list and directory.")

        print(f"Dataset initialized with {len(self)} samples")

    def make_dataset(self):
        samples = []
        for subject in self.subjects:
            file_path = os.path.join(self.h5_directory, f"{subject}.h5")
            if not os.path.isfile(file_path):
                print(f"[Warning] File not found for subject: {subject}")
                continue
            with h5py.File(file_path, 'r') as h5_file:
                for dataset_name in h5_file.keys():
                    samples.append((file_path, subject, dataset_name))
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def generate_masks(num_channels, seq_length, num_patches=10, num_masked_channels=2):
        patch_length = seq_length // num_patches
        mask1 = torch.ones((num_channels, seq_length))

        for i in range(num_patches):
            start_idx = i * patch_length
            end_idx = start_idx + patch_length
            masked_channels = torch.randperm(num_channels)[:num_masked_channels]
            for channel in masked_channels:
                mask1[channel, start_idx:end_idx] = 0

        mask2 = 1 - mask1
        return mask1, mask2

    def __getitem__(self, index):
        file_path, subject, dataset_name = self.samples[index]

        with h5py.File(file_path, 'r') as h5_file:
            psg_data = h5_file[dataset_name][()]

        psg_tensor = torch.from_numpy(psg_data.astype(np.float32))

        if self.wt_mask:
            mask1, mask2 = self.generate_masks(5, 3000)
            return psg_tensor, mask1, mask2
        else:
            return psg_tensor

import csv
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def get_filelists(listpath):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)
    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    csv_file.close()
    return train_filelist, val_filelist, test_filelist


class SEN12MSCR_Dataset(Dataset):
    def __init__(self, filelist, inputs_dir, targets_dir):
        self.filelist = filelist
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir

        self.clip_min = 0
        self.clip_max = 10000
        self.scale = 10000

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fileID = self.filelist[index][-1]
        input_path = os.path.join(self.inputs_dir, fileID)
        target_path = os.path.join(self.targets_dir, fileID)

        input_image = self.get_image(input_path).astype('float32')
        target_image = self.get_image(target_path).astype('float32')

        input_image = self.get_normalized_data(input_image)
        target_image = self.get_normalized_data(target_image)

        return {'input': torch.from_numpy(input_image),
                'target': torch.from_numpy(target_image)}

    def get_image(self, path):
        with rasterio.open(path, 'r') as src:
            image = src.read()
            image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
        return image

    def get_normalized_data(self, data_image):
        data_image = np.clip(data_image, self.clip_min, self.clip_max)
        data_image = data_image / self.scale
        return data_image

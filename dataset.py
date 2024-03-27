import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

import csv


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
    def __init__(self, filelist, inputs_dir, targets_dir, sar_dir=None, inputs_dir2=None):
        self.filelist = filelist
        self.inputs_dir = inputs_dir
        self.inputs_dir2 = inputs_dir2
        self.sar_dir = sar_dir
        self.targets_dir = targets_dir

        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                         [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fileID = self.filelist[index][-1]

        input_path = os.path.join(self.inputs_dir, fileID)
        target_path = os.path.join(self.targets_dir, fileID)

        input_image = self.get_image(input_path).astype('float32')
        target_image = self.get_image(target_path).astype('float32')

        input_image = self.get_normalized_data(input_image, data_type=2)
        target_image = self.get_normalized_data(target_image, data_type=3)

        result = {'input': torch.from_numpy(input_image),
                  'target': torch.from_numpy(target_image)}

        if self.sar_dir is not None:
            sar_path = os.path.join(self.sar_dir, fileID)
            sar_image = self.get_image(sar_path).astype('float32')
            sar_image = self.get_normalized_data(sar_image, data_type=1)
            result['sar'] = torch.from_numpy(sar_image)
        if self.inputs_dir2 is not None:
            input_path2 = os.path.join(self.inputs_dir2, fileID)
            input_image2 = self.get_image(input_path2).astype('float32')
            input_image2 = self.get_normalized_data(input_image2, data_type=4)
            result['input2'] = torch.from_numpy(input_image2)
        return result

    def get_image(self, path):
        with rasterio.open(path, 'r', driver='GTiff') as src:
            image = src.read()
            image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
        return image

    # def get_normalized_data(self, data_image):
    #     data_image = np.clip(data_image, self.clip_min, self.clip_max)
    #     data_image = data_image / self.scale
    #     return data_image
    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
            data_image /= self.scale
        elif data_type == 4:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
        return data_image

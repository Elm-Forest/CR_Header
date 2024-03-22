import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt

from model import SpectralCloudRemover

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_image(path):
    with rasterio.open(path, 'r') as src:
        image = src.read()
        image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
    return image


def get_normalized_data(data_image):
    clip_min = 0
    clip_max = 10000
    scale = 10000
    data_image = np.clip(data_image, clip_min, clip_max)
    data_image = data_image / scale
    return data_image


def build_data(input_path, target_path):
    input_img = get_image(input_path).astype('float32')
    target_img = get_image(target_path).astype('float32')
    input_img = get_normalized_data(input_img)
    target_img = get_normalized_data(target_img)
    return {'input': torch.from_numpy(input_img),
            'target': torch.from_numpy(target_img)}


def load_model(path):
    meta_learner = SpectralCloudRemover(input_channels=13, output_channels=3).to(device)
    checkpoint = torch.load(path)
    meta_learner.load_state_dict(checkpoint, strict=True)
    return meta_learner


def get_rgb_preview(r, g, b, sar_composite=False):
    if not sar_composite:
        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)
        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))
        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)
        return rgb.astype(np.uint8)
    else:
        # generate SAR composite
        HH = r
        HV = g
        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6
        rgb = np.dstack((np.zeros_like(HH), HH, HV))
        return rgb.astype(np.uint8)


if __name__ == '__main__':
    input_image = 'K:\dataset\ensemble\dsen2\ROIs1158_spring_6_p742.tif'
    target_image = 'K:\dataset\selected_data_folder\s2_cloudFree\ROIs1158_spring_6_p742.tif'
    meta_path = '../weights/meta_eca.pt'
    images = build_data(input_image, target_image)
    inputs = images["input"].to(device)
    targets = images["target"]
    meta_learner = load_model(meta_path)
    print(inputs.unsqueeze(dim=0).shape)
    outputs = meta_learner(inputs.unsqueeze(dim=0))
    outputs_rgb = outputs.cpu().detach().numpy()
    targets_R_channel = targets[3, :, :]
    targets_G_channel = targets[2, :, :]
    targets_B_channel = targets[1, :, :]
    output_R_channel = outputs_rgb[0, 2, :, :]
    output_G_channel = outputs_rgb[0, 1, :, :]
    output_B_channel = outputs_rgb[0, 0, :, :]
    targets_rgb = get_rgb_preview(targets_R_channel, targets_G_channel, targets_B_channel)
    output_rgb = get_rgb_preview(output_R_channel, output_G_channel, output_B_channel)
    plt.figure(figsize=(6, 6))
    plt.imshow(targets_rgb)
    plt.title('gt')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(output_rgb)
    plt.title('pred')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()

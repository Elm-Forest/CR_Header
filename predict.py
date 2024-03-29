import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt

from ssim_tools import ssim
from uent_model import UNet_new

device = torch.device("cpu")


def get_image(path):
    with rasterio.open(path, 'r') as src:
        image = src.read()
        image = np.nan_to_num(image, nan=np.nanmean(image))  # fill NaN with the mean
    return image


def get_normalized_data(data_image, data_type=2):
    clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    clip_max = [[0, 0],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

    max_val = 1
    scale = 10000
    # SAR
    if data_type == 1:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                          clip_max[data_type - 1][channel])
            data_image[channel] -= clip_min[data_type - 1][channel]
            data_image[channel] = max_val * (data_image[channel] / (
                    clip_max[data_type - 1][channel] - clip_min[data_type - 1][channel]))
    # OPT
    elif data_type == 2 or data_type == 3:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel],
                                          clip_max[data_type - 1][channel])
        data_image /= scale

    return data_image


def build_data(input_path, target_path, cloudy_path, sar_path, input_path2):
    input_img = get_image(input_path).astype('float32')
    input_img2 = get_image(input_path2).astype('float32')
    target_img = get_image(target_path).astype('float32')
    cloudy_img = get_image(cloudy_path).astype('float32')
    sar_img = get_image(sar_path).astype('float32')
    input_img = get_normalized_data(input_img, 2)
    input_img2 = get_normalized_data(input_img2, 2)
    target_img = get_normalized_data(target_img, 3)
    cloudy_img = get_normalized_data(cloudy_img, 2)
    sar_img = get_normalized_data(sar_img, 1)
    return {'input': torch.from_numpy(input_img),
            'input2': torch.from_numpy(input_img2),
            'target': torch.from_numpy(target_img),
            'cloudy': torch.from_numpy(cloudy_img),
            'sar': torch.from_numpy(sar_img)}


def load_model(path):
    meta_learner = UNet_new(2, 26, 3, bilinear=True).to(device)
    checkpoint = torch.load(path)
    try:
        meta_learner.load_state_dict(checkpoint, strict=True)
    except:
        # 创建一个新的状态字典，其中去掉了 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # 加载新的状态字典到模型
        meta_learner.load_state_dict(new_state_dict)
    return meta_learner


def get_rgb_preview(r, g, b, brighten_limit=None, sar_composite=False):
    if brighten_limit is not None:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)

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
    name = 'ROIs1158_spring_8_p403.tif'  # 113p169
    input_image = f'K:/dataset/ensemble/dsen2/{name}'
    input_image2 = f'K:/dataset/ensemble/clf/{name}'
    cloudy_image = f'K:\dataset\selected_data_folder\s2_cloudy\\{name}'
    target_image = f'K:\dataset\selected_data_folder\s2_cloudFree\\{name}'
    sar_image = f'K:\dataset\selected_data_folder\s1\\{name}'
    meta_path = 'checkpoint/checkpoint_7.pth'
    images = build_data(input_image, target_image, cloudy_image, sar_image, input_image2)
    inputs = images["input"]
    inputs2 = images["input2"] * 10000
    avg = (inputs + inputs2) / 2
    targets = images["target"]
    print(inputs.max())
    cloudy = images['cloudy']
    sar = images['sar']
    meta_learner = load_model(meta_path)
    print(inputs.unsqueeze(dim=0).shape)
    concatenated = torch.cat((inputs.unsqueeze(dim=0), inputs2.unsqueeze(dim=0)), dim=1)
    outputs = meta_learner(sar.unsqueeze(dim=0), concatenated) * 10000
    outputs_rgb = outputs.cpu().detach()
    outputs_rgb = get_normalized_data(outputs_rgb.squeeze(dim=0).numpy(), 2)
    print(ssim(inputs2[1:4, :, :].unsqueeze(0), targets[1:4, :, :].unsqueeze(0)))
    print(ssim(inputs[1:4, :, :].unsqueeze(0), targets[1:4, :, :].unsqueeze(0)))
    print(ssim(torch.from_numpy(outputs_rgb).unsqueeze(0), targets[1:4, :, :].unsqueeze(0)))
    inputs_R_channel = inputs[3, :, :]
    inputs_G_channel = inputs[2, :, :]
    inputs_B_channel = inputs[1, :, :]
    inputs_R_channel2 = inputs2[3, :, :]
    inputs_G_channel2 = inputs2[2, :, :]
    inputs_B_channel2 = inputs2[1, :, :]
    targets_R_channel = targets[3, :, :]
    targets_G_channel = targets[2, :, :]
    targets_B_channel = targets[1, :, :]
    cloudy_R_channel = cloudy[3, :, :]
    cloudy_G_channel = cloudy[2, :, :]
    cloudy_B_channel = cloudy[1, :, :]
    output_R_channel = outputs_rgb[2, :, :]
    output_G_channel = outputs_rgb[1, :, :]
    output_B_channel = outputs_rgb[0, :, :]
    avg_R_channel = avg[3, :, :]
    avg_G_channel = avg[2, :, :]
    avg_B_channel = avg[1, :, :]
    # print(nn.functional.l1_loss(outputs_rgb, targets[1:4, :, :]))
    inputs_rgb = get_rgb_preview(inputs_R_channel, inputs_G_channel, inputs_B_channel, brighten_limit=2000)
    inputs_rgb2 = get_rgb_preview(inputs_R_channel2, inputs_G_channel2, inputs_B_channel2, brighten_limit=2000)
    targets_rgb = get_rgb_preview(targets_R_channel, targets_G_channel, targets_B_channel, brighten_limit=2000)
    cloudy_rgb = get_rgb_preview(cloudy_R_channel, cloudy_G_channel, cloudy_B_channel, brighten_limit=2000)
    output_rgb = get_rgb_preview(output_R_channel, output_G_channel, output_B_channel, brighten_limit=2000)
    avg_rgb = get_rgb_preview(avg_R_channel, avg_G_channel, avg_B_channel, brighten_limit=2000)
    plt.imsave('im.png', output_rgb)
    plt.figure(figsize=(6, 6))
    plt.imshow(inputs_rgb)
    plt.title('input')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(inputs_rgb2)
    plt.title('input2')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(targets_rgb)
    plt.title('gt')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(cloudy_rgb)
    plt.title('cloudy')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(output_rgb)
    plt.title('pred')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_rgb)
    plt.title('avg')
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()

from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(outputs, targets):
    ssim_scores = []
    for b in range(outputs.shape[0]):
        output = outputs[b]
        target = targets[b]

        # 确定合适的win_size，这里假设为5，但您可以根据实际情况调整
        win_size = min(5, min(output.shape[:-1]), min(target.shape[:-1]))  # 确保win_size不大于图像的任一维度
        if win_size % 2 == 0:  # 确保win_size是奇数
            win_size -= 1

        # 计算SSIM，确保图像尺寸大于win_size，并指定channel_axis
        if output.shape[0] >= win_size and output.shape[1] >= win_size:
            score = ssim(output, target, win_size=win_size, multichannel=True, channel_axis=-1)
            ssim_scores.append(score)
        else:
            ssim_scores.append(np.nan)  # 对于尺寸不足的图像，可以赋值为np.nan或其他适当的值

    # 返回SSIM分数，跳过尺寸不足的图像
    return [score for score in ssim_scores if not np.isnan(score)]


def getRGBImg(r, g, b, img_size=256):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)

        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out


def Get2Img(img_fake, img_truth, img_size=256):
    output_img = np.zeros((img_size, 2 * img_size, 3), dtype=np.uint8)
    img_fake = uint16to8((torch.squeeze(img_fake).cpu().detach().numpy() * 10000).astype("uint16")).transpose(1, 2, 0)
    img_truth = uint16to8((torch.squeeze(img_truth).cpu().detach().numpy() * 10000).astype("uint16")).transpose(1, 2, 0)

    img_fake_RGB = getRGBImg(img_fake[:, :, 2], img_fake[:, :, 1], img_fake[:, :, 0], img_size)
    img_truth_RGB = getRGBImg(img_truth[:, :, 2], img_truth[:, :, 1], img_truth[:, :, 0], img_size)

    output_img[:, 0 * img_size:1 * img_size, :] = img_fake_RGB
    output_img[:, 1 * img_size:2 * img_size, :] = img_truth_RGB
    return output_img

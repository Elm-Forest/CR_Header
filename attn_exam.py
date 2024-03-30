import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

from feature_detectors import get_cloud_cloudshadow_mask
from predict import get_image, get_normalized_data, get_rgb_preview

name = 'ROIs1158_spring_44_p528.tif'  # ROIs1158_spring_17_p324
input = f'K:\dataset\ensemble\dsen2\\{name}'
input2 = f'K:\dataset\ensemble\clf\\{name}'
cloudfree = f'K:\dataset\selected_data_folder\s2_cloudFree\\{name}'
cloudy_path = f'K:\dataset\selected_data_folder\s2_cloudy\\{name}'
input = get_image(input).astype('float32')  # np 13*256*256
input2 = get_image(input2).astype('float32')  # np 13*256*256
cloudfree = get_image(cloudfree).astype('float32')  # np 13*256*256
cloudy_ori = get_image(cloudy_path).astype('float32')  # np 13*256*256
cloud_mask = get_cloud_cloudshadow_mask(input, 0.5)
# cloud_mask[cloud_mask != 0] = 1
x = get_normalized_data(input, 2)  # np 13*256*256
y = get_normalized_data(input2, 4)  # np 13*256*256
t = get_normalized_data(cloudfree, 2)  # np 13*256*256
z = cloudy_ori.copy()  # np 13*256*256

inputs_R_channel = x[3, :, :]
inputs_G_channel = x[2, :, :]
inputs_B_channel = x[1, :, :]
inputs2_R = y[3]
inputs2_G = y[2]
inputs2_B = y[1]
targets_R_channel = t[3, :, :]
targets_G_channel = t[2, :, :]
targets_B_channel = t[1, :, :]
cloudy_ori_R_channel = z[3]
cloudy_ori_G_channel = z[2]
cloudy_ori_B_channel = z[1]
x = get_rgb_preview(inputs_R_channel, inputs_G_channel, inputs_B_channel, brighten_limit=2000)  # np 256* 256 *3
y = get_rgb_preview(inputs2_R, inputs2_G, inputs2_B, brighten_limit=2000)
t = get_rgb_preview(targets_R_channel, targets_G_channel, targets_B_channel, brighten_limit=2000)  # np 256* 256 *3
z = get_rgb_preview(cloudy_ori_R_channel, cloudy_ori_G_channel, cloudy_ori_B_channel,
                    brighten_limit=2000)  # np 256* 256 *3
M = np.mean(np.abs(t - x), axis=2)
# 假设x和t是我们的两幅图像，这里直接使用它们的灰度版本来简化处理
x_gray = cv2.cvtColor(x.astype('uint8'), cv2.COLOR_RGB2GRAY)
t_gray = cv2.cvtColor(t.astype('uint8'), cv2.COLOR_RGB2GRAY)

# 局部直方图均衡化
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
x_enhanced = clahe.apply(x_gray)
t_enhanced = clahe.apply(t_gray)
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.util import img_as_float

# 转换为浮点数图像
x_float = img_as_float(x_enhanced)
t_float = img_as_float(t_enhanced)

# 局部二值模式（LBP）纹理特征提取
radius = 3
n_points = 8 * radius
x_lbp = local_binary_pattern(x_float, n_points, radius, method='uniform')
t_lbp = local_binary_pattern(t_float, n_points, radius, method='uniform')

# Gabor滤波器处理
freq = 0.6
x_gabor, _ = gabor(x_float, frequency=freq)
t_gabor, _ = gabor(t_float, frequency=freq)

# 计算纹理特征的差异
texture_diff = np.abs(x_lbp - t_lbp) + np.abs(x_gabor - t_gabor)

# 计算增强的绝对差异作为基础的M
base_diff = np.abs(x_enhanced - t_enhanced)

# 综合多种特征差异
M_combined = base_diff + texture_diff

# 标准化M以便于可视化
M_normalized = cv2.normalize(M_combined, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(M_normalized)
plt.title('M Matrix')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()
# 计算 SSIM，需要确保图像是灰度的，所以我们先转换 RGB 图像到灰度
x_gray = np.mean(x, axis=2).astype(np.float32)
y_gray = np.mean(y, axis=2).astype(np.float32)
t_gray = np.mean(t, axis=2).astype(np.float32)

# 计算 SSIM 和差异图（这里的差异图可以直接作为 M）
_, M_ssim_x = ssim(t_gray, x_gray, full=True, data_range=255.0, win_size=7)
# M_ssim_x = (M_ssim_x - M_ssim_x.min()) / (M_ssim_x.max() - M_ssim_x.min())
M_ssim_x = 1 - M_ssim_x
_, M_ssim_y = ssim(t_gray, y_gray, full=True, data_range=255.0, win_size=7)
# M_ssim_y = (M_ssim_y - M_ssim_y.min()) / (M_ssim_y.max() - M_ssim_y.min())
M_ssim_y = 1 - M_ssim_y
# 真实图像
plt.imshow(t.astype(np.uint8))
plt.title('True Image (t)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()

# 有云的图像
plt.imshow(x.astype(np.uint8))
plt.title('Input Image (x)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()

# 有云的图像
plt.imshow(z)
plt.title('Cloudy Image (z)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()

# M ssim矩阵
plt.imshow(M_ssim_x)
plt.title('M SSIM (x)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()

plt.imshow(M_ssim_y)
plt.title('M SSIM (y)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()

M_ssim = (M_ssim_x + M_ssim_y) * 0.5
M_ssim = (M_ssim - M_ssim.min()) / (M_ssim.max() - M_ssim.min())
plt.imshow(M_ssim)
plt.title('M SSIM (sum)')
plt.axis('off')  # 关闭坐标轴标号和刻度
plt.show()
# # M 矩阵
# axs[4].imshow(cloud_mask)
# axs[4].set_title('M Matrix')
# axs[4].axis('off')

# M 矩阵
# plt.imshow(M_ssim_use)
# plt.title('M SSIM USE')
# plt.axis('off')  # 关闭坐标轴标号和刻度
# plt.show()

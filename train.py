import os

import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
from torchvision import transforms

from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from dataset import SEN12MSCR_Dataset, get_filelists
from model2 import CRHeader_L
from ssim_tools import ssim

transform = transforms.Compose([
    transforms.ToTensor(),
])

csv_filepath = 'E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv'
inputs_dir = 'K:/dataset/ensemble/dsen2'
inputs_val_dir = 'K:/dataset/ensemble/dsen2'
targets_dir = 'K:/dataset/selected_data_folder/s2_cloudFree'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_filelist, val_filelist, _ = get_filelists(csv_filepath)
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_val_dir, targets_dir)

batch_size = 3

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

meta_learner = CRHeader_L(input_channels=13, output_channels=3).to(device)

optimizer = optim.Adam(meta_learner.parameters(), lr=1e-4)
# criterion = torch.nn.L1Loss()
criterion = MS_SSIM_L1_LOSS()
num_epochs = 15

print('start training')

# 准备在日志打印时使用的变量
log_step = 10  # 每处理log_step个batch打印一次日志

meta_learner.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0

    for i, images in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = images["input"].to(device)
        targets = images["target"].to(device)
        outputs = meta_learner(inputs)

        # print(outputs.shape)

        # 计算损失
        outputs_rgb = outputs[:, :, :, :]
        targets_rgb = targets[:, 1:4, :, :]
        loss = criterion(outputs_rgb, targets_rgb)
        loss.backward()
        optimizer.step()

        # 更新运行损失
        running_loss += loss.item()

        # 将Tensor转换为NumPy数组以计算SSIM和PSNR
        outputs_np = outputs_rgb.cpu().detach().numpy()
        targets_np = targets_rgb.cpu().detach().numpy()

        # 计算并更新SSIM和PSNR
        batch_ssim = ssim(outputs_rgb, targets_rgb)
        batch_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
        running_ssim += batch_ssim
        running_psnr += batch_psnr

        # 打印日志
        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss: {running_loss / log_step:.4f}, "
                  f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}")
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    print('start val')
    # 验证
    meta_learner.eval()  # 设置模型为评估模式
    with torch.no_grad():
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        running_val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        for i, images in enumerate(val_dataloader):
            inputs = images["input"].to(device)
            targets = images["target"].to(device)
            outputs = meta_learner(inputs)
            outputs_rgb = outputs
            targets_rgb = targets[:, 1:4, :, :]
            # 计算验证集上的损失
            loss = criterion(outputs_rgb, targets_rgb)
            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss
            # 计算SSIM和PSNR
            outputs_np = outputs_rgb.cpu().numpy()
            targets_np = targets_rgb.cpu().numpy()
            val_ssim = ssim(outputs_rgb, targets_rgb)
            val_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
            total_psnr += val_psnr
            total_ssim += val_ssim
            running_psnr += val_psnr
            running_ssim += val_ssim

            # 打印日志
            if (i + 1) % log_step == 0:
                print(f"VAL: Step [{i + 1}/{len(val_dataloader)}], "
                      f"Loss: {running_loss / log_step:.4f}, "
                      f"SSIM: {running_ssim / len(val_dataloader):.4f}, "
                      f"PSNR: {running_psnr / log_step:.4f}")
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0
        # 打印验证结果
        print(f"Validation Results - Epoch: {epoch + 1}, Loss: {val_loss / len(val_dataloader):.4f}, "
              f"SSIM: {total_ssim / len(val_dataloader):.4f}, "
              f"PSNR: {total_psnr / len(val_dataloader):.4f}")
    meta_learner.train()  # 重新设置模型为训练
    if epoch % 1 == 0:
        torch.save(meta_learner.state_dict(), os.path.join('./checkpoint', f'checkpoint_{epoch}.pth'))
torch.save(meta_learner.state_dict(), './weights/meta_eca.pth')

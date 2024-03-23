import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SEN12MSCR_Dataset, get_filelists
from model import CRHeader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=7, help='batch size used for training')
parser.add_argument('--inputs_dir', type=str, default='K:/dataset/ensemble/dsen2')
parser.add_argument('--inputs_val_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudy')
parser.add_argument('--targets_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudFree')
parser.add_argument('--data_list_filepath', type=str,
                    default='E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=2, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=1, help='epoch to start lr decay')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='./weights/meta_eca.pth',
                    help='directory used to store trained networks')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint")
parser.add_argument('--frozen', type=bool, default=False)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--input_channels', type=int, default=13)
parser.add_argument('--use_rgb', type=bool, default=True)
opts = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
])

csv_filepath = opts.data_list_filepath
inputs_dir = opts.inputs_dir
inputs_val_dir = opts.inputs_val_dir
targets_dir = opts.targets_dir
output_channels = 3
if opts.use_rgb:
    output_channels = 3
else:
    output_channels = 13
train_filelist, val_filelist, _ = get_filelists(csv_filepath)
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_val_dir, targets_dir)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
meta_learner = CRHeader(input_channels=opts.input_channels, output_channels=output_channels).to(device)

if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    meta_learner = nn.DataParallel(meta_learner)

optimizer = optim.Adam(meta_learner.parameters(), lr=opts.lr)
criterion = torch.nn.L1Loss()

num_epochs = opts.epoch
log_step = opts.log_freq

print('start training')

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
        if opts.use_rgb:
            targets = targets[:, 1:4, :, :]
        else:
            targets = targets
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs_np = outputs.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        # 计算并更新SSIM和PSNR
        # batch_ssim = np.mean(
        #     [calculate_ssim(outputs_np, targets_np)(outputs_np[b], targets_np[b]) for b in range(outputs_np.shape[0])])
        batch_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
        # running_ssim += batch_ssim
        running_psnr += batch_psnr

        # 打印日志
        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss: {running_loss / log_step:.4f}, "
                  # f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}")
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    print('start val')
    # 验证
    meta_learner.eval()
    with torch.no_grad():
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        running_val_loss = 0.0
        total_psnr = 0.0
        for i, images in enumerate(val_dataloader):
            inputs = images["input"].to(device)
            targets = images["target"].to(device)
            outputs = meta_learner(inputs)
            if opts.use_rgb:
                targets = targets[:, 1:4, :, :]
            else:
                targets = targets
            # 计算验证集上的损失
            loss = criterion(outputs, targets)
            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss

            # 计算SSIM和PSNR
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            # val_ssim += np.mean(
            #     [calculate_ssim(outputs_np, targets_np)(outputs_np[b], targets_np[b]) for b in range(outputs_np.shape[0])])
            val_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
            # running_ssim += batch_ssim
            total_psnr += val_psnr
            running_psnr += val_psnr

            # 打印日志
            if (i + 1) % log_step == 0:
                print(f"VAL: , Step [{i + 1}/{len(val_dataloader)}], "
                      f"Loss: {running_loss / log_step:.4f}, "
                      f"PSNR: {running_psnr / log_step:.4f}")
                running_loss = 0.0
                running_psnr = 0.0
        # 打印验证结果
        print(f"Validation Results - Epoch: {epoch + 1}, Loss: {val_loss / len(val_dataloader):.4f}, "
              # f"SSIM: {val_ssim / len(val_dataloader):.4f}, "
              f"PSNR: {total_psnr / len(val_dataloader):.4f}")
    meta_learner.train()
    if epoch % opts.save_freq == 0:
        torch.save(meta_learner.state_dict(), os.path.join(opts.checkpoint, f'_{epoch}'))

torch.save(meta_learner.state_dict(), opts.save_model_dir)

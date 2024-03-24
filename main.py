import argparse
import os
import warnings

import numpy as np
import torch
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from dataset import SEN12MSCR_Dataset, get_filelists
from ssim_tools import ssim
from tools import weights_init
from uent_model import UNet

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=7, help='batch size used for training')
parser.add_argument('--inputs_dir', type=str, default='K:/dataset/ensemble/dsen2')
parser.add_argument('--inputs_val_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudy')
parser.add_argument('--targets_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudFree')
parser.add_argument('--sar_dir', type=str, default='K:/dataset/selected_data_folder/s1')
parser.add_argument('--data_list_filepath', type=str,
                    default='E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=2, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=1, help='epoch to start lr decay')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='./weights/meta_cbam.pth',
                    help='directory used to store trained networks')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint")
parser.add_argument('--frozen', type=bool, default=False)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--input_channels', type=int, default=13)
parser.add_argument('--use_sar', type=bool, default=False)
parser.add_argument('--use_rgb', type=bool, default=True)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--weights_path', type=str, default='weights/unet_carvana_scale0.5_epoch2.pth')
parser.add_argument('--weight_decay', type=float, default=0.0001)
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
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_val_dir, targets_dir, sar_dir=opts.sar_dir)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opts.sar_dir is not None:
    meta_learner = UNet(in_channels=opts.input_channels + 2, out_channels=output_channels).to(device)
else:
    meta_learner = UNet(in_channels=opts.input_channels, out_channels=output_channels).to(device)

# meta_learner.apply(weights_init)
if opts.load_weights and opts.weights_path is not None:
    weights = torch.load(opts.weights_path)
    meta_learner.down1.load_state_dict(weights, strict=False)
    meta_learner.down2.load_state_dict(weights, strict=False)
    meta_learner.down3.load_state_dict(weights, strict=False)
    meta_learner.down4.load_state_dict(weights, strict=False)
    meta_learner.up1.load_state_dict(weights, strict=False)
    meta_learner.up2.load_state_dict(weights, strict=False)
    meta_learner.up3.load_state_dict(weights, strict=False)
    meta_learner.up4.load_state_dict(weights, strict=False)
if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    meta_learner = nn.DataParallel(meta_learner)

optimizer = optim.Adam(meta_learner.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
criterion = MS_SSIM_L1_LOSS().to(device)

num_epochs = opts.epoch
log_step = opts.log_freq


def lr_lambda(ep):
    initial_lr = 1e-4
    final_lr = 1e-5
    lr_decay = final_lr / initial_lr
    return 1 - (1 - lr_decay) * (ep / (num_epochs - 1))


scheduler = LambdaLR(optimizer, lr_lambda)

print('Start Training!')

meta_learner.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    running_ssim = 0.0
    original_ssim = 0.0
    running_psnr = 0.0

    for i, images in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = images["input"].to(device)
        targets = images["target"].to(device)
        if opts.sar_dir is not None:
            sars = images["sar"].to(device)
            concatenated = torch.cat((inputs, sars), dim=1)
            outputs = meta_learner(concatenated)
        else:
            outputs = meta_learner(inputs)
        if opts.use_rgb:
            targets_rgb = targets[:, 1:4, :, :]
        else:
            targets_rgb = targets
        loss = criterion(outputs, targets_rgb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs_np = outputs.cpu().detach().numpy()
        targets_np = targets_rgb.cpu().detach().numpy()

        batch_ssim = ssim(outputs, targets_rgb)
        if opts.use_rgb:
            ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
        else:
            ori_ssim = ssim(inputs, targets_rgb)

        batch_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
        running_ssim += batch_ssim
        running_psnr += batch_psnr
        original_ssim += ori_ssim

        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss: {running_loss / log_step:.4f}, "
                  f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}, "
                  f"ORI_SSIM: {original_ssim / log_step:.4f}")
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            original_ssim = 0.0
    scheduler.step()

    print('start val')
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    original_ssim = 0.0
    meta_learner.eval()
    with torch.no_grad():
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        running_val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_ori_ssim = 0.0
        for i, images in enumerate(val_dataloader):
            inputs = images["input"].to(device)
            targets = images["target"].to(device)
            if opts.sar_dir is not None:
                sars = images["sar"].to(device)
                concatenated = torch.cat((inputs, sars), dim=1)
                outputs = meta_learner(concatenated)
            else:
                outputs = meta_learner(inputs)
            if opts.use_rgb:
                targets_rgb = targets[:, 1:4, :, :]
            else:
                targets_rgb = targets

            loss = criterion(outputs, targets_rgb)

            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss

            outputs_np = outputs.cpu().numpy()
            targets_np = targets_rgb.cpu().numpy()

            val_ssim = ssim(outputs, targets_rgb)
            if opts.use_rgb:
                ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
            else:
                ori_ssim = ssim(inputs, targets_rgb)
            val_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])

            total_psnr += val_psnr
            total_ssim += val_ssim
            total_ori_ssim += ori_ssim
            running_psnr += val_psnr
            running_ssim += val_ssim
            original_ssim += ori_ssim
            if (i + 1) % log_step == 0:
                print(f"VAL: Step [{i + 1}/{len(val_dataloader)}], "
                      f"Loss: {running_loss / log_step:.4f}, "
                      f"SSIM: {running_ssim / log_step:.4f}, "
                      f"PSNR: {running_psnr / log_step:.4f}, "
                      f"ORI_SSIM: {original_ssim / log_step:.4f}")
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0

        print(f"Validation Results - Epoch: {epoch + 1}, Loss: {val_loss / len(val_dataloader):.4f}, "
              f"SSIM: {total_ssim / len(val_dataloader):.4f}, "
              f"PSNR: {total_psnr / len(val_dataloader):.4f}, "
              f"ORI_SSIM: {total_ori_ssim / len(val_dataloader):.4f}")

    meta_learner.train()

    if epoch % opts.save_freq == 0:
        torch.save(meta_learner.state_dict(), os.path.join(opts.checkpoint, f'checkpoint_{epoch}.pth'))

torch.save(meta_learner.state_dict(), opts.save_model_dir)

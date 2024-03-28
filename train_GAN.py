import argparse
import os
import warnings

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from DIS import Discriminator
from dataset import SEN12MSCR_Dataset, get_filelists
from ssim_tools import ssim
from uent_model import UNet_new

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='batch size used for training')
parser.add_argument('--inputs_dir', type=str, default='K:/dataset/ensemble/dsen2')
parser.add_argument('--inputs_dir2', type=str, default='K:/dataset/ensemble/clf')
parser.add_argument('--targets_dir', type=str, default='K:/dataset/selected_data_folder/s2_cloudFree')
parser.add_argument('--sar_dir', type=str, default='K:/dataset/selected_data_folder/s1')
parser.add_argument('--data_list_filepath', type=str,
                    default='E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_dis', type=float, default=5e-5, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=2, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=1, help='epoch to start lr decay')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--dis_backward_delay', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='./weights',
                    help='directory used to store trained networks')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint")
parser.add_argument('--frozen', type=bool, default=False)
parser.add_argument('--speed', type=bool, default=False)
parser.add_argument('--input_channels', type=int, default=13)
parser.add_argument('--use_sar', type=bool, default=True)
parser.add_argument('--use_rgb', type=bool, default=True)
parser.add_argument('--use_input2', type=bool, default=True)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--weights_path', type=str, default='weights/unet_carvana_scale0.5_epoch2.pth')
parser.add_argument('--weight_decay', type=float, default=0.0001)
opts = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
])

csv_filepath = opts.data_list_filepath
inputs_dir = opts.inputs_dir
inputs_dir2 = opts.inputs_dir2
targets_dir = opts.targets_dir
output_channels = 3
if opts.use_rgb:
    output_channels = 3
else:
    output_channels = 13
train_filelist, val_filelist, _ = get_filelists(csv_filepath)
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir,
                                  inputs_dir2=inputs_dir2, crop_size=opts.crop_size)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir, inputs_dir2=inputs_dir2,
                                crop_size=opts.crop_size)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = UNet_new(2, 26, 3, bilinear=True).to(device)
discriminator = Discriminator(in_ch=3, out_ch=3 + 3 + 2, gpu_ids=0).to(device)

criterion_L1 = nn.SmoothL1Loss().to(device)
criterionSoftplus = nn.Softplus()
criterion_GAN = nn.MSELoss()
num_epochs = opts.epoch
log_step = opts.log_freq
if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opts.lr_gen, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opts.lr_dis, betas=(0.5, 0.999))
lambda_L1 = 100


def lr_lambda_gen(ep):
    initial_lr = 1e-4
    final_lr = 1e-5
    lr_decay = final_lr / initial_lr
    return 1 - (1 - lr_decay) * (ep / (num_epochs - 1))


def lr_lambda_dis(ep):
    initial_lr = 1e-5
    final_lr = 6e-6
    lr_decay = final_lr / initial_lr
    return 1 - (1 - lr_decay) * (ep / (num_epochs - 1))


scheduler_G = LambdaLR(optimizer_G, lr_lambda_gen)
scheduler_D = LambdaLR(optimizer_D, lr_lambda_dis)
print('Start Training!')

for epoch in range(num_epochs):
    running_loss = 0.0
    running_ssim = 0.0
    original_ssim = 0.0
    original_ssim2 = 0.0
    running_psnr = 0.0
    running_loss_dis = 0.0
    running_loss_L1 = 0.0
    running_loss_GAN = 0.0
    delay_steps = 0
    for i, images in enumerate(train_dataloader):

        # 准备真实图像和标签
        real_images = images["target"].to(device)[:, 1:4, :, :]  # 假设去雾后的真实RGB图像
        sars = images["sar"].to(device)
        inputs = images["input"].to(device)
        inputs2 = images["input2"].to(device)

        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        concatenated = torch.cat((inputs, inputs2), dim=1)  # 假设这是另一种形式的输入
        fake_images = generator(sars, concatenated)
        if delay_steps % opts.dis_backward_delay == 0:
            # 训练判别器D
            optimizer_D.zero_grad()

            # 真实图像通过D
            real_ab = torch.cat((real_images, inputs[:, 1:4, :, :], inputs2[:, 1:4, :, :], sars), 1)  # 真实图像对
            pred_real = discriminator(real_ab)
            loss_D_real = (torch.sum(criterionSoftplus(-pred_real))
                           / pred_real.size(0) / pred_real.size(2) / pred_real.size(3))

            # 生成假图像并通过D
            fake_ab = torch.cat((fake_images.detach(), inputs[:, 1:4, :, :], inputs2[:, 1:4, :, :], sars), 1)  # 使用生成的图像
            pred_fake = discriminator(fake_ab)
            loss_D_fake = (torch.sum(criterionSoftplus(pred_fake))
                           / pred_fake.size(0) / pred_fake.size(2) / pred_fake.size(3))

            # 更新D
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            running_loss_dis += loss_D.item()
            loss_D.backward()
            optimizer_D.step()
            delay_steps = 0
        delay_steps += 1
        # 训练生成器G
        optimizer_G.zero_grad()
        fake_ab = torch.cat((fake_images, inputs[:, 1:4, :, :], inputs2[:, 1:4, :, :], sars), 1)  # 使用生成的图像
        pred_fake = discriminator(fake_ab)
        # 更新G，使D将生成的图像分类为真
        loss_G_GAN = (torch.sum(criterionSoftplus(-pred_fake))
                      / pred_fake.size(0) / pred_fake.size(2) / pred_fake.size(3))

        # L1损失，确保像素级相似度
        loss_G_L1 = criterion_L1(fake_images, real_images)
        # 总损失为GAN损失和L1损失的组合
        loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        running_loss += loss_G.item()
        running_loss_L1 += loss_G_L1.item()
        running_loss_GAN += loss_G_GAN.item()
        targets_rgb = real_images
        outputs_np = fake_images.cpu().detach().numpy()
        targets_np = targets_rgb.cpu().detach().numpy()

        batch_ssim = ssim(fake_images, targets_rgb)
        ori_ssim = 0.0
        ori_ssim2 = 0.0
        if opts.use_rgb:
            ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
            ori_ssim2 = ssim(inputs2[:, 1:4, :, :], targets_rgb)
        else:
            ori_ssim = ssim(inputs, targets_rgb)
            ori_ssim2 = ssim(inputs2, targets_rgb)

        batch_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])
        running_ssim += batch_ssim
        running_psnr += batch_psnr
        original_ssim += ori_ssim
        original_ssim2 += ori_ssim2

        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss_gen: {running_loss / log_step:.4f}, "
                  f"Loss_dis: {running_loss_dis / (log_step / opts.dis_backward_delay):.4f}, "
                  f"Loss_L1: {running_loss_L1 / log_step:.4f} * {lambda_L1}, "
                  f"Loss_GAN: {running_loss_GAN / log_step:.4f}, "
                  f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}, "
                  f"ORI_SSIM: {original_ssim / log_step:.4f}, "
                  f"ORI_SSIM2: {original_ssim2 / log_step:.4f}")
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            original_ssim = 0.0
            original_ssim2 = 0.0
            running_loss_dis = 0.0
            running_loss_GAN = 0.0
            running_loss_L1 = 0.0
    scheduler_G.step()
    scheduler_D.step()
    print('start val')
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    original_ssim = 0.0
    original_ssim2 = 0.0
    generator.eval()
    with torch.no_grad():
        val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
        running_val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_ori_ssim = 0.0
        total_ori_ssim2 = 0.0
        for i, images in enumerate(val_dataloader):
            inputs = images["input"].to(device)
            targets = images["target"].to(device)
            inputs2 = torch.zeros(inputs.shape)
            if opts.use_sar is not None and opts.use_input2 is None:
                sars = images["sar"].to(device)
                outputs = generator(sars, inputs)
            elif opts.use_sar is not None and opts.use_input2 is not None:
                sars = images["sar"].to(device)
                inputs2 = images["input2"].to(device)
                concatenated = torch.cat((inputs, inputs2), dim=1)
                outputs = generator(sars, concatenated)
            else:
                outputs = generator(inputs)
            if opts.use_rgb:
                targets_rgb = targets[:, 1:4, :, :]
            else:
                targets_rgb = targets

            loss = criterion_L1(outputs, targets_rgb)

            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss

            outputs_np = outputs.cpu().numpy()
            targets_np = targets_rgb.cpu().numpy()

            val_ssim = ssim(outputs, targets_rgb)
            ori_ssim = 0.0
            ori_ssim2 = 0.0
            if opts.use_rgb:
                ori_ssim = ssim(inputs[:, 1:4, :, :], targets_rgb)
                ori_ssim2 = ssim(inputs2[:, 1:4, :, :], targets_rgb)
            else:
                ori_ssim = ssim(inputs, targets_rgb)
                ori_ssim2 = ssim(inputs2, targets_rgb)
            val_psnr = np.mean([psnr(targets_np[b], outputs_np[b]) for b in range(outputs_np.shape[0])])

            total_psnr += val_psnr
            total_ssim += val_ssim
            total_ori_ssim += ori_ssim
            total_ori_ssim2 += ori_ssim2
            running_psnr += val_psnr
            running_ssim += val_ssim
            original_ssim += ori_ssim
            original_ssim2 += ori_ssim2
            if (i + 1) % log_step == 0:
                print(f"VAL: Step [{i + 1}/{len(val_dataloader)}], "
                      f"Loss: {running_loss / log_step:.4f}, "
                      f"SSIM: {running_ssim / log_step:.4f}, "
                      f"PSNR: {running_psnr / log_step:.4f}, "
                      f"ORI_SSIM: {original_ssim / log_step:.4f}, "
                      f"ORI_SSIM2: {original_ssim2 / log_step:.4f}")
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0
                original_ssim = 0.0
                original_ssim2 = 0.0
        print(f"Validation Results - Epoch: {epoch + 1}, Loss: {val_loss / len(val_dataloader):.4f}, "
              f"SSIM: {total_ssim / len(val_dataloader):.4f}, "
              f"PSNR: {total_psnr / len(val_dataloader):.4f}, "
              f"ORI_SSIM: {total_ori_ssim / len(val_dataloader):.4f}, "
              f"ORI_SSIM2: {total_ori_ssim2 / len(val_dataloader):.4f}")

    generator.train()

    if epoch % opts.save_freq == 0:
        torch.save(generator.state_dict(), os.path.join(opts.checkpoint, f'checkpoint_gen_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(opts.checkpoint, f'checkpoint_dis_{epoch}.pth'))

torch.save(generator.state_dict(), os.path.join(opts.save_model_dir, f'gen_{num_epochs}.pth'))
torch.save(discriminator.state_dict(), os.path.join(opts.save_model_dir, f'dis_{num_epochs}.pth'))

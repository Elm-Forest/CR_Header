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

from Charbonnier_Loss import EdgeLoss, L1_Charbonnier_loss
from MemoryNet import MemoryNet
from dataset import SEN12MSCR_Dataset, get_filelists
from ssim_tools import ssim
from uent_model import UNet_new
from unet_m import NestedUNet

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size used for training')
parser.add_argument('--inputs_dir', type=str, default='K:/dataset/ensemble/dsen2')
parser.add_argument('--inputs_dir2', type=str, default='K:/dataset/ensemble/clf')
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
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
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
                                  inputs_dir2=inputs_dir2)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_dir, targets_dir, sar_dir=opts.sar_dir, inputs_dir2=inputs_dir2)

train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    if opts.local_rank != -1:
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
if opts.use_sar and opts.use_input2 is False:
    print('create unet_new inc=13')
    meta_learner = UNet_new(2, 13, 3).to(device)
elif opts.use_sar and opts.use_input2:
    print('create unet_new inc=26')
    #  meta_learner = UNet_new(2, 26, 3, bilinear=True).to(device)
    meta_learner = MemoryNet(in_c=26 + 2).to(device)
else:
    meta_learner = NestedUNet(in_channels=opts.input_channels, out_channels=output_channels).to(device)

# meta_learner.apply(weights_init)
if opts.load_weights and opts.weights_path is not None:
    print("Loading weights!")
    weights = torch.load(opts.weights_path)
    try:
        meta_learner.load_state_dict(weights['state_dict'], strict=False)
    except:
        pass
train_sampler = None
if len(opts.gpu_ids) > 1:
    print("Parallel training!")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        meta_learner = nn.parallel.DistributedDataParallel(meta_learner, device_ids=[opts.local_rank],
                                                           output_device=opts.local_rank)
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=opts.batch_size,
                                      num_workers=opts.num_workers, pin_memory=True, shuffle=False)

    meta_learner = nn.DataParallel(meta_learner)

optimizer = optim.Adam(meta_learner.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
criterion_char = L1_Charbonnier_loss().to(device)
criterion_edge = EdgeLoss().to(device)
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
    original_ssim2 = 0.0
    running_psnr = 0.0
    if len(opts.gpu_ids) > 1:
        train_sampler.set_epoch(epoch)
    for i, images in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs = images["input"].to(device)
        targets = images["target"].to(device)
        inputs2 = torch.zeros(inputs.shape)
        if opts.use_sar is not None and opts.use_input2 is None:
            sars = images["sar"].to(device)
            outputs = meta_learner(sars, inputs)
        elif opts.use_sar is not None and opts.use_input2 is not None:
            sars = images["sar"].to(device)
            inputs2 = images["input2"].to(device)
            concatenated = torch.cat((inputs, inputs2, sars), dim=1)
            outputs = meta_learner(concatenated)
        else:
            outputs = meta_learner(inputs)
        if opts.use_rgb:
            targets_rgb = targets[:, 1:4, :, :]
        else:
            targets_rgb = targets
        loss_char = torch.sum(
            torch.stack([criterion_char(outputs[j], targets[:, 1:4, :, :]) for j in range(len(outputs))]))
        loss_edge = torch.sum(
            torch.stack([criterion_edge(outputs[j], targets[:, 1:4, :, :]) for j in range(len(outputs))]))
        loss = loss_char + (0.05 * loss_edge)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs = outputs[0]
        outputs_np = outputs.cpu().detach().numpy()
        targets_np = targets_rgb.cpu().detach().numpy()

        batch_ssim = ssim(outputs, targets_rgb)
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
                  f"Loss: {running_loss / log_step:.4f}, "
                  f"SSIM: {running_ssim / log_step:.4f}, "
                  f"PSNR: {running_psnr / log_step:.4f}, "
                  f"ORI_SSIM: {original_ssim / log_step:.4f}, "
                  f"ORI_SSIM2: {original_ssim2 / log_step:.4f}")
            running_loss = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            original_ssim = 0.0
            original_ssim2 = 0.0
    scheduler.step()

    print('start val')
    running_loss = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    original_ssim = 0.0
    original_ssim2 = 0.0
    meta_learner.eval()
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
                outputs = meta_learner(sars, inputs)
            elif opts.use_sar is not None and opts.use_input2 is not None:
                sars = images["sar"].to(device)
                inputs2 = images["input2"].to(device)
                concatenated = torch.cat((inputs, inputs2, sars), dim=1)
                outputs = meta_learner(concatenated)
            else:
                outputs = meta_learner(inputs)
            if opts.use_rgb:
                targets_rgb = targets[:, 1:4, :, :]
            else:
                targets_rgb = targets

            loss_char = torch.sum(
                torch.stack([criterion_char(outputs[j], targets[:, 1:4, :, :]) for j in range(len(outputs))]))
            loss_edge = torch.sum(
                torch.stack([criterion_edge(outputs[j], targets[:, 1:4, :, :]) for j in range(len(outputs))]))
            loss = loss_char + (0.05 * loss_edge)
            running_val_loss = loss.item()
            running_loss += running_val_loss
            val_loss += running_val_loss
            outputs = outputs[0]
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

    meta_learner.train()

    if epoch % opts.save_freq == 0:
        torch.save(meta_learner.state_dict(), os.path.join(opts.checkpoint, f'checkpoint_{epoch}.pth'))

torch.save(meta_learner.state_dict(), opts.save_model_dir)

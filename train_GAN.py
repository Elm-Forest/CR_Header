import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from dataset import SEN12MSCR_Dataset, get_filelists
from DIS import Discriminator
from model2 import CRHeader_L

csv_filepath = 'E:/Development Program/Pycharm Program/ECANet/csv/datasetfilelist.csv'
inputs_dir = 'K:/dataset/ensemble/dsen2'
inputs_val_dir = 'K:/dataset/ensemble/dsen2'
targets_dir = 'K:/dataset/selected_data_folder/s2_cloudFree'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_filelist, val_filelist, _ = get_filelists(csv_filepath)
train_dataset = SEN12MSCR_Dataset(train_filelist, inputs_dir, targets_dir)
val_dataset = SEN12MSCR_Dataset(val_filelist, inputs_val_dir, targets_dir)

batch_size = 2
lr_gen = 1e-4
lr_dis = 1e-4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# real_a = torch.FloatTensor(batch_size, 13, 256, 256).to(device)
# real_b = torch.FloatTensor(batch_size, 3, 256, 256).to(device)
# M = torch.FloatTensor(batch_size, 256, 256).to(device)
# real_a = Variable(real_a)
# real_b = Variable(real_b)

gen = CRHeader_L(input_channels=13, output_channels=3).to(device)
dis = Discriminator(in_ch=13, out_ch=3, gpu_ids=0)

optimizer_G = optim.Adam(gen.parameters(), lr=lr_gen, weight_decay=0.00001)
optimizer_D = optim.Adam(dis.parameters(), lr=lr_dis, weight_decay=0.00001)
# criterion = torch.nn.L1Loss()
criterion = MS_SSIM_L1_LOSS().to(device)
criterionSoftplus = nn.Softplus().to(device)
num_epochs = 15

print('start training')

# 准备在日志打印时使用的变量
log_step = 10  # 每处理log_step个batch打印一次日志

gen.train()
for epoch in range(num_epochs):
    running_loss_G = 0.0
    running_loss_D = 0.0
    for i, data in enumerate(train_dataloader):
        real_images = data["input"].to(device)
        targets = data["target"].to(device)

        # 真实图像标签为1，生成图像标签为0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 训练鉴别器
        optimizer_D.zero_grad()

        # 使用真实图像训练鉴别器
        real_outputs = dis(real_images)
        d_loss_real = criterionSoftplus(real_outputs, real_labels)

        # 使用生成图像训练鉴别器
        fake_images = gen(real_images)
        fake_outputs = dis(fake_images.detach())
        d_loss_fake = criterionSoftplus(fake_outputs, fake_labels)

        # 合并鉴别器的损失并反向传播
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        # 生成器的目标是让鉴别器误将生成图像判断为真实图像
        fake_outputs = dis(fake_images)
        g_loss = criterion(fake_images, targets[:, 1:4, :, :]) + criterionSoftplus(fake_outputs, real_labels)  # 合并对抗损失和内容损失

        g_loss.backward()
        optimizer_G.step()

        running_loss_G += g_loss.item()
        running_loss_D += d_loss.item()

        if (i + 1) % log_step == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], "
                  f"Loss_G: {running_loss_G / log_step:.4f}, "
                  f"Loss_D: {running_loss_D / log_step:.4f}")
            running_loss_G = 0.0
            running_loss_D = 0.0

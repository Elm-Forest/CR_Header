import torch
from torch import nn

from uent_model import AttnCGAN_CR


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入维度: [bs, in_channels, h, w]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 维度: [bs, 64, h/2, w/2]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 维度: [bs, 128, h/4, w/4]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 维度: [bs, 256, h/8, w/8]
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 维度: [bs, 512, h/8, w/8]
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            # 最终维度: [bs, 1, h/8, w/8]
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    torch.save(Discriminator(3).state_dict(), './model_test/d.pth')
    torch.save(AttnCGAN_CR(2, 13, 3, bilinear=True).state_dict(), './model_test/g.pth')

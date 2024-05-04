import torch
from torch import nn
from torch.nn import init


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDB_Lite(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB)
    来源：ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)
    作用：通过稠密残差块强化特征提取，有效提升图像的细节和清晰度。
    """

    def __init__(self, channels=64, growth_channel=32):
        super(RRDB_Lite, self).__init__()
        # 第一个DenseBlock
        self.conv1 = nn.Conv2d(channels, growth_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channel, growth_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channel, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        for m in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        concat1 = torch.cat([x, out], 1)
        out = self.relu(self.conv2(concat1))
        concat2 = torch.cat([concat1, out], 1)
        out = self.conv3(concat2)
        # 残差缩放
        out *= 0.2
        return x + out
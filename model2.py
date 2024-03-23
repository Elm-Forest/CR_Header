from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from cbam import CBAM


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ColorCorrectionLayer(nn.Module):
    def __init__(self, channels):
        super(ColorCorrectionLayer, self).__init__()
        self.adjust = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.adjust(x)
        return x


class CMAM_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(CMAM_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class CRHeader_L(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CRHeader_L, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(input_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.res_block1 = Bottleneck(64, 64)
        self.res_block2 = Bottleneck(64, 64)
        self.res_block3 = Bottleneck(64, 64)
        self.res_block4 = Bottleneck(64, 64)
        self.cbam_block1 = CMAM_Block(inplanes=64, planes=64)
        self.cbam_block2 = CMAM_Block(inplanes=64, planes=64)
        self.conv_out = nn.Sequential(
            conv3x3(64, output_channels),
            nn.ReLU(True)
        )
        self.color_correction = ColorCorrectionLayer(output_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = F.relu(self.res_block1(x) + x)
        x = F.relu(self.res_block2(x) + x)
        x = self.cbam_block1(x)
        x = F.relu(self.res_block3(x) + x)
        x = F.relu(self.res_block4(x) + x)
        x = self.cbam_block2(x)
        x = self.conv_out(x)
        x = self.color_correction(x)
        return x


if __name__ == '__main__':
    net = CRHeader_L(13, 3)

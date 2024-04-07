""" Full assembly of the parts to form the complete network """
from collections import OrderedDict

from SPANet import conv3x3
from cbam import CBAM
from unet_parts import *


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class Encoder(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(Encoder, self).__init__()
        self.n_channels = in_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['bc1'] = nn.BatchNorm2d(out_channels)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['bc2'] = nn.BatchNorm2d(out_channels)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        m['cbam'] = CBAM(out_channels)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class AttnCGAN_CR(nn.Module):
    def __init__(self, in_channels_sar=2, in_channels_s2=13, out_channels=3, ensemble_num=2, bilinear=True,
                 feature_c=32):
        super(AttnCGAN_CR, self).__init__()
        self.bilinear = bilinear
        self.encoder_sar = Encoder(in_channels_sar, bilinear)  # official Unet Encoder
        self.encoder_s2 = Encoder(in_channels_s2 * ensemble_num, bilinear)
        factor = 2 if bilinear else 1
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128 // factor, bilinear))
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.cbam5 = CBAM(2048 // factor)
        # self.res_input = nn.Sequential(
        #     conv3x3(in_channels_s2, 128 // factor),
        #     nn.BatchNorm2d(128 // factor),
        #     nn.ReLU(True),
        # )
        self.out_s2 = (OutConv(128 // factor, in_channels_s2))
        self.relu = nn.ReLU(inplace=True)
        self.conv_in_reg = nn.Sequential(
            conv3x3(128 // factor, feature_c),
            nn.BatchNorm2d(feature_c, affine=True),
            nn.ReLU(True),
        )
        self.res_block1 = Bottleneck(feature_c, feature_c)
        self.res_block2 = Bottleneck(feature_c, feature_c)
        self.res_block3 = Bottleneck(feature_c, feature_c)
        self.res_block4 = Bottleneck(feature_c, feature_c)
        self.res_block5 = Bottleneck(feature_c, feature_c)
        self.outc = (OutConv(feature_c, out_channels))

    def forward(self, x11, x12, x2):
        x1 = torch.cat((x11, x12), dim=1)
        # x_mean = (x11 + x12) / 2
        x11, x12, x13, x14, x15 = self.encoder_s2(x1)
        x21, x22, x23, x24, x25 = self.encoder_sar(x2)
        x1 = torch.cat((x11, x21), dim=1)
        x1 = F.relu(self.cbam1(x1) + x1)
        x2 = torch.cat((x12, x22), dim=1)
        x2 = F.relu(self.cbam2(x2) + x2)
        x3 = torch.cat((x13, x23), dim=1)
        x3 = F.relu(self.cbam3(x3) + x3)
        x4 = torch.cat((x14, x24), dim=1)
        x4 = F.relu(self.cbam4(x4) + x4)
        x5 = torch.cat((x15, x25), dim=1)
        x5 = F.relu(self.cbam5(x5) + x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        s2 = self.out_s2(x)
        # x = self.relu(self.res_input(x_mean) + x)
        out = self.conv_in_reg(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)
        out = F.relu(self.res_block4(out) + out)
        out = F.relu(self.res_block5(out) + out)
        out = self.outc(out)
        out = F.tanh(out)
        return out, s2


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    s21 = torch.zeros((1, 13, 256, 256))
    s22 = torch.zeros((1, 13, 256, 256))
    s1 = torch.zeros((1, 2, 256, 256))
    unet = AttnCGAN_CR(2, 13, 3, ensemble_num=2, bilinear=False)
    output = unet(s21, s22, s1)
    torch.save(unet.state_dict(), f'checkpoint_test.pth')
    print(output[0].shape)

""" Full assembly of the parts to form the complete network """

from cbam import CBAM
from unet_parts import *


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


class UNet_new(nn.Module):
    def __init__(self, in_channels_sar, in_channels_s2, out_channels, bilinear=True):
        super(UNet_new, self).__init__()
        self.bilinear = bilinear
        self.encoder_sar = Encoder(in_channels_sar, bilinear)  # official Unet Encoder
        self.encoder_s2 = Encoder(in_channels_s2, bilinear)
        factor = 2 if bilinear else 1
        self.up1 = (Up(2048, 1024 // factor, bilinear))
        self.up2 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up4 = (Up(256, 128 // factor, bilinear))
        self.cbam_block = CBAM_Block(128 // factor, 128 // factor)
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.cbam5 = CBAM(2048 // factor)
        self.outc = (OutConv(128 // factor, out_channels))

    def forward(self, x1, x2):
        x11, x12, x13, x14, x15 = self.encoder_sar(x1)
        x21, x22, x23, x24, x25 = self.encoder_s2(x2)
        x1 = torch.cat((x11, x21), dim=1)
        x1 = self.cbam1(x1)
        x2 = torch.cat((x12, x22), dim=1)
        x2 = self.cbam2(x2)
        x3 = torch.cat((x13, x23), dim=1)
        x3 = self.cbam3(x3)
        x4 = torch.cat((x14, x24), dim=1)
        x4 = self.cbam4(x4)
        x5 = torch.cat((x15, x25), dim=1)
        x5 = self.cbam5(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.cbam_block(x)
        x = self.outc(x)
        return x


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
    s2 = torch.zeros((1, 13, 256, 256))
    s1 = torch.zeros((1, 2, 256, 256))
    unet = UNet_new(2, 13, 3, bilinear=False)
    output = unet(s1, s2)
    torch.save(unet.state_dict(), f'checkpoint_test.pth')
    print(output.shape)

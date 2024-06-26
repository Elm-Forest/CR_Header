from collections import OrderedDict

import torch
import torch.nn as nn

import memory
from SPANet import conv3x3
from rrdb import RRDB


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, in_c=3):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, in_c, kernel_size, bias=bias)
        self.conv3 = conv(in_c, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class MemoryNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(MemoryNet, self).__init__()
        # self.memory = memory.MemModule()
        self.memory = memory.MemModule(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_c)
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail1 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail2 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail3 = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img, x_rgb):
        H = x3_img.size(2)
        W = x3_img.size(3)
        ##通过memory模块使得变为三个分支
        x1, x2, x3 = self.memory(x3_img)
        # x3bot_img  = x3_img
        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        # Two Patches for Stage 2
        x2top_img = x2[:, :, 0:int(H / 2), :]
        x2bot_img = x2[:, :, int(H / 2):H, :]

        x3top_img = x3[:, :, 0:int(H / 2), :]
        x3bot_img = x3[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x3top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x3top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x3bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x3bot_img[:, :, :, int(W / 2):W]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        stage1_img = self.tail1(stage1_img)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x1)
        stage2_img = self.tail2(stage2_img)
        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x1)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        # x3_cat = self.relu(self.tail3(x3_cat) + x3_img)
        stage3_img = self.tail3(x3_cat)
        return [stage3_img + x_rgb, stage2_img, stage1_img]


class MemoryNet_C(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(MemoryNet_C, self).__init__()
        # self.memory = memory.MemModule()
        self.memory = memory.MemModule(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_c)
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail1 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail2 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail3 = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, s2_img, s1_img):
        x3_img = torch.cat((s2_img, s1_img), dim=1)
        H = x3_img.size(2)
        W = x3_img.size(3)
        ##通过memory模块使得变为三个分支
        x1, x2, x3 = self.memory(x3_img)
        # x3bot_img  = x3_img
        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        # Two Patches for Stage 2
        x2top_img = x2[:, :, 0:int(H / 2), :]
        x2bot_img = x2[:, :, int(H / 2):H, :]

        x3top_img = x3[:, :, 0:int(H / 2), :]
        x3bot_img = x3[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x3top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x3top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x3bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x3bot_img[:, :, :, int(W / 2):W]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        stage1_img = self.tail1(stage1_img)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x1)
        stage2_img = self.tail2(stage2_img)
        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x1)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        # x3_cat = self.relu(self.tail3(x3_cat) + x3_img)
        stage3_img = self.tail3(x3_cat)
        return [stage3_img, stage2_img, stage1_img]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if use_norm:
            m['bc1'] = nn.BatchNorm2d(out_channels)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        if use_norm:
            m['bc2'] = nn.BatchNorm2d(out_channels)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(True)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.group1(x)
        out = self.relu(out + x)
        return out


class MemoryNet2(nn.Module):
    def __init__(self, in_c=3, in_s1=2, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8,
                 kernel_size=3,
                 reduction=4, bias=False):
        super(MemoryNet2, self).__init__()
        self.memory = memory.MemModule(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_c)
        self.memory_sar = memory.MemModule(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_s1)
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_sar_1 = nn.Sequential(conv(in_s1, n_feat, kernel_size, bias=bias),
                                                CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_sar_2 = nn.Sequential(conv(in_s1, n_feat, kernel_size, bias=bias),
                                                CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat_sar_3 = nn.Sequential(conv(in_s1, n_feat, kernel_size, bias=bias),
                                                CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.concat_xy = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.tail1 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail2 = conv(in_c, out_c, kernel_size, bias=bias)
        self.tail3 = conv(n_feat + scale_orsnetfeats, in_c, kernel_size=3, bias=True)
        self.relu = nn.ReLU(True)
        self.res_in = nn.Sequential(
            conv3x3(in_c, n_feat),
            nn.ReLU(True),
        )
        self.rrdb_blocks = nn.Sequential(*[RRDB(n_feat) for _ in range(4)])
        # self.cbam1 = CBAM(n_feat, no_spatial=True)
        # self.cbam2 = CBAM(n_feat, no_spatial=True)
        # self.res_blocks1 = nn.Sequential(
        #     Bottleneck(n_feat, n_feat),
        #     Bottleneck(n_feat, n_feat),
        # )
        # self.res_blocks2 = nn.Sequential(
        #     Bottleneck(n_feat, n_feat),
        #     Bottleneck(n_feat, n_feat),
        # )
        # self.res_blocks3 = nn.Sequential(
        #     Bottleneck(n_feat, n_feat),
        #     Bottleneck(n_feat, n_feat),
        # )
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, s2_img, s1_img):
        H = s2_img.size(2)
        W = s2_img.size(3)
        x1, x2, x3 = self.memory(s2_img)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        # Two Patches for Stage 2
        x2top_img = x2[:, :, 0:int(H / 2), :]
        x2bot_img = x2[:, :, int(H / 2):H, :]
        x3top_img = x3[:, :, 0:int(H / 2), :]
        x3bot_img = x3[:, :, int(H / 2):H, :]
        # Four Patches for Stage 1
        x1ltop_img = x3top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x3top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x3bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x3bot_img[:, :, :, int(W / 2):W]

        H = s1_img.size(2)
        W = s1_img.size(3)
        y1, y2, y3 = self.memory_sar(s1_img)
        y2top_img = y2[:, :, 0:int(H / 2), :]
        y2bot_img = y2[:, :, int(H / 2):H, :]
        y3top_img = y3[:, :, 0:int(H / 2), :]
        y3bot_img = y3[:, :, int(H / 2):H, :]

        y1ltop_img = y3top_img[:, :, :, 0:int(W / 2)]
        y1rtop_img = y3top_img[:, :, :, int(W / 2):W]
        y1lbot_img = y3bot_img[:, :, :, 0:int(W / 2)]
        y1rbot_img = y3bot_img[:, :, :, int(W / 2):W]
        # -------------- Stage 1---------------------
        # Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        # Add SAR
        x1ltop += self.shallow_feat_sar_1(y1ltop_img)
        x1rtop += self.shallow_feat_sar_1(y1rtop_img)
        x1lbot += self.shallow_feat_sar_1(y1lbot_img)
        x1rbot += self.shallow_feat_sar_1(y1rbot_img)

        # Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        # Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        # Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        # Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        # Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        stage1_img = self.tail1(stage1_img)
        # -------------- Stage 2---------------------
        # Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)
        x2top += self.shallow_feat_sar_2(y2top_img)
        x2bot += self.shallow_feat_sar_2(y2bot_img)

        # Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        # Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        # Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        # Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        # Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x1)
        stage2_img = self.tail2(stage2_img)
        # -------------- Stage 3---------------------
        # Compute Shallow Features
        x3 = self.shallow_feat3(x1)
        y3 = self.shallow_feat_sar_3(y1)
        x3 = self.concat_xy(torch.cat((x3, y3), dim=1))
        # Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        x3_cat = self.relu(self.tail3(x3_cat) + s2_img)
        outs = self.res_in(x3_cat)
        outs = self.rrdb_blocks(outs)
        # out = nn.functional.relu(self.cbam1(out) + out)
        # out = self.res_blocks1(out)
        # out = nn.functional.relu(self.cbam2(out) + out)
        # out = self.res_blocks2(out)
        # out = self.res_blocks3(out)
        stage3_img = self.tail(outs)
        return [stage3_img, stage2_img, stage1_img]


if __name__ == '__main__':
    cloudy = torch.zeros((1, 13, 256, 256))
    s21 = torch.zeros((1, 13, 256, 256))
    s22 = torch.zeros((1, 13, 256, 256))
    s1 = torch.zeros((1, 2, 256, 256))
    # meta_learner = MemoryNet2(in_c=39, in_s1=2, n_feat=64, scale_unetfeats=32)
    meta_learner = MemoryNet_C(in_c=13 + 2, n_feat=64, scale_unetfeats=32)
    # for name, param in meta_learner.named_parameters():
    #     # 默认情况下冻结所有参数
    #     param.requires_grad = False
    #
    #     # 检查参数名前缀，如果匹配特定模块，则解冻
    #     if name.startswith('res_in') or name.startswith('cbam1') or \
    #             name.startswith('cbam2') or name.startswith('res_blocks1') or \
    #             name.startswith('res_blocks2') or name.startswith('res_blocks3') or \
    #             name.startswith('tail'):
    #         param.requires_grad = True
    outp = meta_learner(cloudy, s1)
    torch.save(meta_learner.state_dict(), './model_test/mem.pth')
    # outp = meta_learner(torch.cat((cloudy, s21, s22), dim=1), s1)

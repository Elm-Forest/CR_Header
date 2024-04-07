from collections import OrderedDict

import torch
import torch.nn as nn

from layers import CBR
from tools import weights_init
import torch.nn.functional as F

class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch

        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c11 = CBR(128, 128, bn=True, sample='stay', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c22 = CBR(256, 256, bn=True, sample='stay', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c33 = CBR(512, 512, bn=True, sample='stay', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.in_ch]
        x_1 = x[:, self.in_ch:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = F.relu(self.c11(h)+h)
        h = self.c2(h)
        h = F.relu(self.c22(h) + h)
        h = self.c3(h)
        h = F.relu(self.c33(h) + h)
        h = self.c4(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        return self.dis(x)


if __name__ == '__main__':
    torch.save(Discriminator(3, 3, 0).state_dict(), './model_test/d2.pth')
    # real_labels_smooth = torch.full((4, 1), 0.9)
    # fake_labels_smooth = torch.full((4, 1), 0.1)
    # fake_labels_smooth = torch.full((4, 1), 0.1)

import torch.nn as nn

from cbam import CBAM


class CRHeader(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CRHeader, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        # self.eca = DynamicECAModule(128)
        self.cbma = CBAM(gate_channels=128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbma(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class SAR_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SAR_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        # self.eca = DynamicECAModule(128)
        self.cbma = CBAM(gate_channels=128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbma(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

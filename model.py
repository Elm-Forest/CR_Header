import torch.nn as nn

from cbma import CBAM


class DynamicECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super(DynamicECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False,
                              groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        y = y.unsqueeze(-1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1)
        return x * y


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

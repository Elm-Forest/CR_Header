import torch.nn as nn


class DynamicECAModule(nn.Module):
    def __init__(self, channels, k_size=3):
        super(DynamicECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使conv层能够处理动态数量的通道
        self.conv = nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False,
                              groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # 在全局平均池化之后移除末尾两个维度，形状变为(batch_size, channels)
        y = y.squeeze(-1).squeeze(-1)
        # Conv1d期望的输入形状为(batch_size, channels, length)，增加一个假的长度维度
        y = y.unsqueeze(-1)
        y = self.conv(y)
        y = self.sigmoid(y)
        # 此时y的形状为(batch_size, channels, 1)，我们需要在末尾增加一个维度以便与x的形状(batch_size, channels, height, width)匹配
        y = y.unsqueeze(-1)
        # 不再使用expand_as，而是直接使用广播机制进行乘法操作
        return x * y


class SpectralCloudRemover(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpectralCloudRemover, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        # 应用改进的DynamicECA
        self.eca = DynamicECAModule(128)
        # 添加额外的卷积层以增强模型的特征提取能力
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.eca(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

# class ECAModule(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super(ECAModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)
#
#
# class CNNWithECA(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(CNNWithECA, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.eca = ECAModule(64)
#         self.conv2 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.eca(x)
#         x = self.conv2(x)
#         return x

# import torch
# # 假设基模型和元学习器的定义
# # 加载基模型A, B, C，并冻结它们的权重
# model_A = ...  # 加载模型A
# model_B = ...  # 加载模型B
# model_C = ...  # 加载模型C
# for param in model_A.parameters():
#     param.requires_grad = False
# for param in model_B.parameters():
#     param.requires_grad = False
# for param in model_C.parameters():
#     param.requires_grad = False
#
# # 假设输入图像的通道数为13，与基模型输出维度相同
# meta_learner = CNNWithECA(input_channels=13 * 3, output_channels=13)  # 假设有3个基模型
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(meta_learner.parameters(), lr=0.001)
# num_epochs = ...
# dataloader = ...
# targets = ...
# # 训练过程
# for epoch in range(num_epochs):
#     for inputs in dataloader:  # 假设dataloader已经定义
#         # 假设inputs的形状为[batch_size, 13, H, W]
#
#         # 获取基模型的输出，并冻结它们
#         with torch.no_grad():
#             output_A = model_A(inputs)
#             output_B = model_B(inputs)
#             output_C = model_C(inputs)
#
#         # Concatenate基模型的输出
#         concatenated_outputs = torch.cat((output_A, output_B, output_C), dim=1)
#
#         # 通过元学习器处理concatenated_outputs
#         outputs = meta_learner(concatenated_outputs)
#
#         # 计算损失
#         loss = criterion(outputs, targets)  # 假设targets已经定义
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

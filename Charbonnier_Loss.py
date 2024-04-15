import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = L1_Charbonnier_loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg19_model = vgg19(pretrained=True).features[:36].eval()  # 使用VGG的前36层
        for param in vgg19_model.parameters():
            param.requires_grad = False  # 冻结VGG参数，避免更新它们
        self.vgg19_model = vgg19_model
        self.loss = nn.MSELoss()  # 使用MSE计算特征之间的差异

    def forward(self, generated_image, target_image):
        generated_features = self.vgg19_model(generated_image)
        target_features = self.vgg19_model(target_image)
        return self.loss(generated_features, target_features)


class AdjustedPerceptualLoss(nn.Module):
    def __init__(self):
        super(AdjustedPerceptualLoss, self).__init__()
        self.vgg19_model = vgg19(pretrained=True).features.eval()
        self.selected_layers = [2, 7, 12, 21]  # 选择VGG的前几层
        for param in self.vgg19_model.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, generated_image, target_image):
        loss = 0.0
        x = generated_image
        y = target_image
        for i, layer in enumerate(self.vgg19_model):
            x = layer(x)
            y = layer(y)
            if i in self.selected_layers:
                loss += self.loss(x, y)
        return loss


# 使用调整后的感知损失
perceptual_loss_module = AdjustedPerceptualLoss()

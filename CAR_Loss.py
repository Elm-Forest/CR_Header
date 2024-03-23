import torch
import torch.nn as nn


class CARLoss(nn.Module):
    def __init__(self):
        super(CARLoss, self).__init__()

    def forward(self, y_true, y_pred):
        """Computes the Cloud-Adaptive Regularized Loss (CARL) for RGB images"""
        # Assuming the last channel is the mask for cloud/cloudshadow
        cloud_cloudshadow_mask = y_true[:, -1:, :, :]
        clearmask = torch.ones_like(y_true[:, -1:, :, :]) - cloud_cloudshadow_mask

        # Only the first 3 channels are used for RGB images
        predicted = y_pred[:, 0:3, :, :]
        input_cloudy = y_pred[:, -4:-1, :, :]
        target = y_true[:, 0:3, :, :]

        # Compute Cloud-Shadow-Clear Masked Absolute Error (CSCMAE)
        # The loss computation remains the same, just the channel indices have changed
        cscmae = torch.mean(clearmask * torch.abs(predicted - input_cloudy) +
                            cloud_cloudshadow_mask * torch.abs(predicted - target)) + 1.0 * torch.mean(
            torch.abs(predicted - target))

        return cscmae

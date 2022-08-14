import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Implementation of Facal Loss"""

    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weighted_cs = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.cs = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predicted, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        pt = 1 / torch.exp(self.cs(predicted, target))
        # shape: [batch_size]
        entropy_loss = self.weighted_cs(predicted, target)
        # shape: [batch_size]
        focal_loss = ((1 - pt) ** self.gamma) * entropy_loss
        # shape: [batch_size]
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()

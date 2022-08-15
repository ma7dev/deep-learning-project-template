# import torch
# import torch.nn as nn


# class FocalLoss(nn.Module):
#     """Implementation of Facal Loss"""

#     def __init__(
#         self: nn.Module,
#         weight: float = None,
#         gamma: int = 2,
#         reduction: str = "mean",
#     ) -> None:
#         super(FocalLoss).__init__(weight, gamma, reduction)
#         self.weighted_cs = nn.CrossEntropyLoss(weight=weight, reduction="none")
#         self.cs = nn.CrossEntropyLoss(reduction="none")
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(
#         self: nn.Module, predicted: torch.Tensor, target: torch.Tensor
#     ):
#         """
#         predicted: [batch_size, n_classes]
#         target: [batch_size]
#         """
#         pt = 1 / torch.exp(self.cs(predicted, target))
#         # shape: [batch_size]
#         entropy_loss = self.weighted_cs(predicted, target)
#         # shape: [batch_size]
#         focal_loss = ((1 - pt) ** self.gamma) * entropy_loss
#         # shape: [batch_size]
#         if self.reduction == "none":
#             return focal_loss
#         elif self.reduction == "mean":
#             return focal_loss.mean()
#         else:
#             return focal_loss.sum()

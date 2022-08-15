import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(
        self: nn.Module,
        in_channels: int = 28 * 28,
        hidden_size: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__(in_channels, hidden_size, num_classes)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

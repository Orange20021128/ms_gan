# incomplete/recover.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelRecover(nn.Module):
    """
    使用 3x3 邻域（8 个体素）预测中心体素
    """
    def __init__(self, n_class=4):
        super().__init__()
        self.n_class = n_class

        self.net = nn.Sequential(
            nn.Linear(8 * n_class, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        # x: (B, 8, n_class)
        x = x.view(x.size(0), -1)
        return self.net(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


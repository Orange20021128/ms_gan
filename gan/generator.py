# gan/generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    输入: 9 根钻孔
    输出: 16 根预测钻孔
    """
    def __init__(self, nz, n_class=4):
        super().__init__()
        self.nz = nz
        self.n_class = n_class

        self.net = nn.Sequential(
            nn.Conv1d(9 * n_class, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 16 * n_class, 1)
        )

    def forward(self, x):
        # x: (B, 9, nz, n_class)
        b = x.size(0)
        x = x.view(b, 9 * self.n_class, self.nz)
        out = self.net(x)
        out = out.view(b, 16, self.n_class, self.nz)
        return out

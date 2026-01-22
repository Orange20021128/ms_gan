# gan/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nz, n_class=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(16 * n_class, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 16, nz, n_class)
        b = x.size(0)
        x = x.view(b, 16 * x.size(3), x.size(2))
        return self.net(x)

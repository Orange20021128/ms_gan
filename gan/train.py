# gan/train.py
import torch
from torch import nn, optim
from generator import Generator
from discriminator import Discriminator

def train_gan(train_loader, nz, device):
    G = Generator(nz).to(device)
    D = Discriminator(nz).to(device)

    opt_g = optim.Adam(G.parameters(), lr=1e-4)
    opt_d = optim.Adam(D.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    for epoch in range(100):
        for x_in, x_gt in train_loader:
            x_in, x_gt = x_in.to(device), x_gt.to(device)

            # 训练 D
            fake = G(x_in).detach()
            real_pred = D(x_gt)
            fake_pred = D(fake)

            loss_d = loss_fn(real_pred, torch.ones_like(real_pred)) + \
                     loss_fn(fake_pred, torch.zeros_like(fake_pred))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # 训练 G
            fake = G(x_in)
            loss_g = loss_fn(D(fake), torch.ones_like(real_pred))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        print(f"Epoch {epoch}, D={loss_d.item():.3f}, G={loss_g.item():.3f}")

    return G

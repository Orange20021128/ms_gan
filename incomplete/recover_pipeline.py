# incomplete/recover_pipeline.py
import numpy as np
import torch
import torch.nn.functional as F
from .utils import extract_neighbors


def recover_volume(
    volume,
    recover_model,
    device="cpu",
    n_class=4,
    max_iter=2
):
    """
    volume: (Nx, Ny, Nz)，含 -1
    max_iter: 多轮扫描，增强稳定性
    """
    vol = volume.copy()
    nx, ny, nz = vol.shape
    recover_model.eval().to(device)

    for _ in range(max_iter):
        for z in range(nz):
            for x in range(1, nx - 1):
                for y in range(1, ny - 1):

                    if vol[x, y, z] >= 0:
                        continue

                    neigh = extract_neighbors(vol, x, y, z)

                    # 邻域必须已知
                    if np.any(neigh < 0):
                        continue

                    # one-hot
                    neigh = torch.from_numpy(neigh).long().to(device)
                    neigh = F.one_hot(neigh, num_classes=n_class).float()
                    neigh = neigh.unsqueeze(0)

                    with torch.no_grad():
                        pred = recover_model.predict(neigh)

                    vol[x, y, z] = pred.item()

    return vol

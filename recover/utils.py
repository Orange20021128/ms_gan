# incomplete/utils.py
import numpy as np
import torch
import torch.nn.functional as F


def extract_neighbors(vol, x, y, z):
    """
    返回中心体素的 8 邻域（不含自己）
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbors.append(vol[x + dx, y + dy, z])
    return np.array(neighbors)

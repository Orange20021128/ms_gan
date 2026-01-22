# multiscale/template.py
import numpy as np

def extract_5x5(volume, x, y):
    """
    返回:
    input_boreholes: 9 × nz
    target_boreholes: 16 × nz
    """
    block = volume[x-2:x+3, y-2:y+3, :]
    inputs = []
    targets = []

    for i in range(5):
        for j in range(5):
            if i % 2 == 0 and j % 2 == 0:
                inputs.append(block[i, j])
            else:
                targets.append(block[i, j])

    return np.array(inputs), np.array(targets)

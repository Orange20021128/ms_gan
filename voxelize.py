# voxelize.py
import numpy as np

def build_training_image(surfaces, nx, ny, nz):
    """
    surfaces: dict, 每个地层界面的 2D array
    return: (nx, ny, nz) int
    """
    volume = np.zeros((nx, ny, nz), dtype=np.int32)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                depth = z  # 已换算为 index
                if depth < surfaces["marine"][x, y]:
                    volume[x, y, z] = 0
                elif depth < surfaces["alluvium"][x, y]:
                    volume[x, y, z] = 1
                elif depth < surfaces["weathered"][x, y]:
                    volume[x, y, z] = 2
                else:
                    volume[x, y, z] = 3
    return volume

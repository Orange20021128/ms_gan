# ms_gan/gan/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from ms_gan.multiscale.template import extract_5x5
from ms_gan.voxelize import build_training_image

class GANDataset(Dataset):
    def __init__(self, training_image_path, n_class=4, transform=None):
        self.training_image = np.load(training_image_path)
        self.n_class = n_class
        self.transform = transform
        self.blocks = self._get_all_blocks()

    def _get_all_blocks(self):
        """提取训练图像中所有有效的5x5块"""
        nx, ny, _ = self.training_image.shape
        blocks = []
        # 遍历所有不越界的5x5块
        for x in range(2, nx-2):
            for y in range(2, ny-2):
                inputs, targets = extract_5x5(self.training_image, x, y)
                blocks.append((inputs, targets))
        return blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        inputs, targets = self.blocks[idx]
        
        # 转换为one-hot编码
        inputs = torch.nn.functional.one_hot(
            torch.from_numpy(inputs).long(), num_classes=self.n_class
        ).float()
        targets = torch.nn.functional.one_hot(
            torch.from_numpy(targets).long(), num_classes=self.n_class
        ).float()

        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)
        
        return inputs, targets

# 数据加载器构建函数
def build_dataloader(training_image_path, batch_size=32, n_class=4):
    dataset = GANDataset(training_image_path, n_class=n_class)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader
# ms_gan/multiscale/ms_gan.py
import numpy as np
import torch
from .template import extract_5x5

class MSGAN:
    def __init__(self, generators, scales=(4, 2, 1), device="cpu"):
        self.generators = generators  # list of trained Generator
        self.scales = scales          # 多尺度下采样因子
        self.device = device

    def _get_valid_blocks(self, volume, scale):
        """获取当前尺度下有效的5x5块坐标（避免越界）"""
        nx, ny, _ = volume.shape
        step = self.scales[scale]
        valid_x = np.arange(2, nx-2, step)
        valid_y = np.arange(2, ny-2, step)
        return [(x, y) for x in valid_x for y in valid_y]

    def _get_inputs(self, volume, x, y):
        """提取5x5块的输入（9个采样点）并转换为模型输入格式"""
        inputs, _ = extract_5x5(volume, x, y)
        # 转换为one-hot + 适配模型输入 (B=1, 9, nz, n_class)
        nz = inputs.shape[1]
        n_class = len(np.unique(volume[volume != -1]))
        inputs_onehot = torch.nn.functional.one_hot(
            torch.from_numpy(inputs).long(), num_classes=n_class
        ).float().unsqueeze(0)  # (1, 9, nz, n_class)
        return inputs_onehot.to(self.device)

    def _fill_predictions(self, volume, pred, x, y):
        """将模型预测的16个点填充回体素"""
        # pred: (1, 16, n_class, nz) → 转类别索引
        pred = torch.argmax(pred, dim=2).squeeze(0).cpu().numpy()  # (16, nz)
        
        # 5x5块的非采样点位置（16个）
        idx = 0
        for i in range(5):
            for j in range(5):
                if i % 2 == 0 and j % 2 == 0:
                    continue  # 跳过输入采样点
                volume[x-2+i, y-2+j, :] = pred[idx]
                idx += 1
        return volume

    def generate(self, borehole_volume):
        """多尺度生成完整体素"""
        vol = borehole_volume.copy()
        nx, ny, nz = vol.shape
        n_class = len(np.unique(vol[vol != -1]))

        for scale, G in enumerate(self.generators):
            G.eval()
            valid_blocks = self._get_valid_blocks(vol, scale)
            print(f"[MS-GAN] Scale {scale}, processing {len(valid_blocks)} blocks")
            
            with torch.no_grad():
                for x, y in valid_blocks:
                    inp = self._get_inputs(vol, x, y)
                    pred = G(inp)  # 生成16个点
                    vol = self._fill_predictions(vol, pred, x, y)
        return vol
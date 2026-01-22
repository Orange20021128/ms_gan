# ms_gan/data/generate_data.py
import numpy as np
import yaml
from pathlib import Path

# 确保目录存在
Path("ms_gan/data").mkdir(parents=True, exist_ok=True)
Path("ms_gan/models").mkdir(parents=True, exist_ok=True)
Path("ms_gan/result").mkdir(parents=True, exist_ok=True)

# ===================== 1. 配置文件生成 =====================
config = {
    "grid": {
        "nx": 100,  # x方向体素数
        "ny": 100,  # y方向体素数
        "nz": 50,   # z方向体素数（深度）
        "spacing": [10, 10, 5]  # 体素间距 (m)
    },
    "scales": [4, 2, 1],  # MS-GAN 多尺度
    "n_class": 4,         # 地质类别数（海洋/冲积/风化/基岩）
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "lr": 1e-4
    },
    "recover": {
        "max_iter": 2
    }
}

# 保存配置文件
with open("ms_gan/data/config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config, f, indent=4, sort_keys=False)

# ===================== 2. 训练图像（Training Image）生成 =====================
nx, ny, nz = config["grid"]["nx"], config["grid"]["ny"], config["grid"]["nz"]
n_class = config["n_class"]

# 模拟4类地质界面（随机场 + 分层趋势）
np.random.seed(42)
surfaces = {
    "marine": np.random.normal(10, 2, size=(nx, ny)).clip(5, 15),  # 海洋层顶界面
    "alluvium": np.random.normal(25, 3, size=(nx, ny)).clip(20, 30),  # 冲积层顶界面
    "weathered": np.random.normal(40, 4, size=(nx, ny)).clip(35, 45)  # 风化层顶界面
}

# 从界面构建3D训练图像（复用voxelize.py逻辑）
from ms_gan.voxelize import build_training_image
training_image = build_training_image(surfaces, nx, ny, nz)

# 保存训练图像
np.save("ms_gan/data/training_image.npy", training_image)

# ===================== 3. 稀疏钻孔体生成（含-1不完整值） =====================
# 1. 先生成完整钻孔坐标（稀疏采样，比如每10个点取1个）
sample_step = 10
x_coords = np.arange(0, nx, sample_step)
y_coords = np.arange(0, ny, sample_step)
coords = np.array([(x, y) for x in x_coords for y in y_coords])

# 2. 生成钻孔体（初始为全-1，仅采样点填充真实值）
borehole_volume = np.full((nx, ny, nz), -1, dtype=np.int32)
for x, y in coords:
    # 从训练图像中提取该钻孔的真实值
    borehole_volume[x, y, :] = training_image[x, y, :]
    
    # 模拟钻孔缺失：随机mask掉30%的深度点
    mask = np.random.choice([True, False], size=nz, p=[0.3, 0.7])
    borehole_volume[x, y, mask] = -1

# 保存钻孔体和坐标
np.save("ms_gan/data/boreholes.npy", borehole_volume)
np.save("ms_gan/data/coords.npy", coords)

# ===================== 输出数据信息 =====================
print("✅ 数据生成完成：")
print(f"  - 训练图像: {training_image.shape} (nx×ny×nz)，类别数: {np.unique(training_image)}")
print(f"  - 钻孔体: {borehole_volume.shape}，缺失值占比: {np.mean(borehole_volume == -1):.2%}")
print(f"  - 配置文件: ms_gan/data/config.yaml")
print(f"  - 钻孔坐标数: {len(coords)} 个")
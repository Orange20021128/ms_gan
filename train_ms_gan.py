# ms_gan/train_ms_gan.py
import yaml
import torch
from gan.train import train_gan
from gan.dataset import build_dataloader
from gan.generator import Generator

# 加载配置
with open("ms_gan/data/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nx, ny, nz = config["grid"]["nx"], config["grid"]["ny"], config["grid"]["nz"]
n_class = config["n_class"]

# 逐尺度训练Generator
for scale_idx, scale in enumerate(config["scales"]):
    print(f"\n===== 训练尺度 {scale_idx} (下采样因子: {scale}) =====")
    dataloader = build_dataloader(
        training_image_path="ms_gan/data/training_image.npy",
        batch_size=config["training"]["batch_size"],
        n_class=n_class
    )
    # 训练GAN
    G = train_gan(
        train_loader=dataloader,
        nz=nz,
        device=device
    )
    # 保存模型
    torch.save(G, f"ms_gan/models/G_scale{scale_idx}.pt")
print("✅ 所有尺度Generator训练完成！")
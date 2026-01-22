# ms_gan/run.py
import numpy as np
import torch
import yaml
from pathlib import Path
from pipeline import MSGANPipeline
from multiscale.ms_gan import MSGAN
from incomplete.recover import VoxelRecover

# 加载配置
with open("ms_gan/data/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

# 1️⃣ 载入初始钻孔
borehole_volume = np.load("ms_gan/data/boreholes.npy")
nx, ny, nz = config["grid"]["nx"], config["grid"]["ny"], config["grid"]["nz"]
n_class = config["n_class"]

# 2️⃣ 加载/初始化多尺度 Generator（如果无预训练模型，初始化空模型示例）
generators = []
for scale in range(len(config["scales"])):
    model_path = f"ms_gan/models/G_scale{scale}.pt"
    if Path(model_path).exists():
        G = torch.load(model_path, map_location=device)
    else:
        # 初始化空Generator（实际需先训练）
        from ms_gan.gan.generator import Generator
        G = Generator(nz=nz, n_class=n_class).to(device)
        torch.save(G, model_path)  # 保存空模型占位
    generators.append(G)

ms_gan = MSGAN(
    generators=generators,
    scales=config["scales"],
    device=device
)

# 3️⃣ Recover 模型
recover_model = VoxelRecover(n_class=n_class).to(device)
recover_model_path = "ms_gan/models/recover.pt"
if Path(recover_model_path).exists():
    recover_model.load_state_dict(torch.load(recover_model_path, map_location=device))
else:
    # 保存初始模型占位
    torch.save(recover_model.state_dict(), recover_model_path)

# 4️⃣ Pipeline
pipeline = MSGANPipeline(
    ms_gan=ms_gan,
    recover_model=recover_model,
    device=device,
    n_class=n_class
)

# 5️⃣ 运行
print("🚀 启动MS-GAN + 体素恢复流程...")
final_volume = pipeline.run(borehole_volume)

# 6️⃣ 保存结果
np.save("ms_gan/result/final_model.npy", final_volume)
print("✅ 流程完成，结果保存至: ms_gan/result/final_model.npy")

# 可选：不确定性分析（多轮生成）
print("📊 开始不确定性分析...")
realizations = []
for i in range(5):  # 生成5个实现
    np.random.seed(i)
    torch.manual_seed(i)
    realization = pipeline.run(borehole_volume)
    realizations.append(realization)
    np.save(f"ms_gan/result/realization_{i+1:02d}.npy", realization)

# 计算离散度
from ms_gan.uncertainty import dispersion
realizations = np.array(realizations)
mpp, disp = dispersion(realizations)
np.save("ms_gan/result/dispersion.npy", disp)
print("✅ 不确定性分析完成，离散度保存至: ms_gan/result/dispersion.npy")
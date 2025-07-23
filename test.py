import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
from train import Generator, mask_image

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_path = "model/generator_UNET.pth"
test_image_path = "test_data/img.png"
output_image_path = "test_results/generated_image.png"
uncertainty_image_path = "test_results/uncertainty_image.png"
mean_residual_path = "test_results/residual_image.png"
num_samples = 50
mask_size = 64
image_size = 256

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载模型
generator = Generator().to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.train()   # 重点！推论阶段也必须train模式以启用Dropout

# 加载测试图像
image = Image.open(test_image_path).convert("RGB")
image = transform(image).to(device)

# 添加遮挡
masked_image = mask_image(image.clone(), mask_size=mask_size)

# MC Dropout 采样生成
outputs = []
with torch.no_grad():
    for _ in range(num_samples):
        gen_img = generator(masked_image.unsqueeze(0))
        outputs.append(gen_img.cpu().numpy())
outputs = np.stack(outputs, axis=0)   # (num_samples, B, C, H, W)
mean_img = outputs.mean(axis=0)[0]    # (C, H, W)
std_img = outputs.std(axis=0)[0]      # (C, H, W)

# 反归一化
mean_img_tensor = torch.from_numpy(mean_img)
std_img_tensor = torch.from_numpy(std_img)
masked_image = (masked_image * 0.5) + 0.5
original_image = (image * 0.5) + 0.5
mean_img_tensor = (mean_img_tensor * 0.5) + 0.5
std_img_tensor = std_img_tensor  # std本身就无须归一化

# 计算残差
residual = torch.abs(original_image - mean_img_tensor)
residual_mean = residual.mean().item()

# 保存补全结果
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
save_image(masked_image, "test_results/masked_image.png")
save_image(mean_img_tensor, output_image_path)
save_image(residual, mean_residual_path)
print(f"平均残差值: {residual_mean:.6f}")

# 保存uncertainty热度图（取各像素std的均值，转成灰度热度）
import matplotlib.pyplot as plt
std_gray = std_img_tensor.mean(dim=0).numpy()  # (H, W)
plt.imshow(std_gray, cmap='hot')
plt.axis('off')
plt.colorbar()
plt.title("Uncertainty (STD)")
plt.savefig(uncertainty_image_path, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"信心热度图已保存至 {uncertainty_image_path}")

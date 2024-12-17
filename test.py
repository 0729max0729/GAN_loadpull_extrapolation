import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from train import Generator, mask_image
# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_path = "model/generator_UNET.pth"
test_image_path = "test_data/img.png"  # 测试图像路径
output_image_path = "test_results/generated_image.png"  # 输出图像路径

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 图像归一化到 [-1, 1]
])


# 加载模型
generator = Generator().to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()




# 加载测试图像
image = Image.open(test_image_path).convert("RGB")
image = transform(image).to(device)

# 添加遮挡
masked_image = mask_image(image.clone(), mask_size=64)

# 使用生成器修复图像
with torch.no_grad():
    generated_image = generator(masked_image.unsqueeze(0))

# 去归一化
generated_image = (generated_image * 0.5) + 0.5  # [-1, 1] -> [0, 1]
masked_image = (masked_image * 0.5) + 0.5  # [-1, 1] -> [0, 1]
original_image = (image * 0.5) + 0.5  # [-1, 1] -> [0, 1]

# 计算残差
residual = torch.abs(original_image - generated_image)  # 绝对误差 (L1 残差)
residual_mean = residual.mean().item()  # 平均残差值

# 保存结果
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
save_image(masked_image, "test_results/,masked_image.png")
save_image(generated_image, output_image_path)
save_image(residual, "test_results/residual_image.png")  # 保存残差图
print(f"生成的图像已保存至 {output_image_path}")
print(f"残差图已保存至 test_results/residual_image.png")
print(f"平均残差值: {residual_mean:.6f}")


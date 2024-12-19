import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os

from Transformer_model import UNetTransformer
from diffusion_transformer_train import forward_diffusion, beta_schedule, alphas, posterior_variance, transform, \
    image_size, model_path
import matplotlib.pyplot as plt
from diffusion_transformer_train import backward_diffusion, generate_random_square_mask, sqrt_alphas_cumprod, \
    sqrt_one_minus_alphas_cumprod, diffusion_model

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 255 # 扩散过程的总时间步长



def visualize_xt_distribution(xt, step, save_path=None):
    """
    可视化 xt 的分布
    :param xt: 当前时间步的 xt
    :param step: 当前时间步编号
    :param save_path: 保存分布图的路径（可选）
    """
    xt_flat = xt.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(xt_flat, bins=100, color='blue', label=f'Step {step}')
    plt.xlabel('xt Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of xt at Step {step}')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_step_{step}.png")




# 初始化模型并加载权重

diffusion_model.load_state_dict(torch.load(model_path))

# 加载输入图像并生成遮罩
input_image_path = "test_results/generated_image.png"  # 替换为您的输入图像路径
input_image = Image.open(input_image_path).convert("RGB")

input_tensor = transform(input_image).unsqueeze(0).to(device)


diffusion_model.eval()
imgs = input_tensor.to(device)

# 随机生成随机方形遮罩
mask = generate_random_square_mask(image_size=imgs.size())

# 随机时间步长 t


# 正向扩散（仅遮罩部分加噪声）
xt, noise = forward_diffusion(imgs, T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
#xt = torch.randn_like(imgs)
with torch.no_grad():
    for t in reversed(range(T)):
        # 计算当前时间步的 alpha 和 sqrt(1 - alpha)
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_oneminus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # 预测噪声
        predicted_noise = diffusion_model(xt * (1 - mask) + imgs * mask, torch.tensor([t]).to(device),torch.tensor([0]).to(device))
        # 反向扩散
        xt = backward_diffusion(xt * (1 - mask), t, predicted_noise * (1 - mask), alphas,sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,beta_schedule,posterior_variance)
        # 保存当前步的图像
        save_image(xt*(1-mask)+imgs*mask, f"test_results/{t}xt.png", normalize=False)
        # 打印当前步的值范围和 alpha 信息
        print(f"Step {t}: xt min {xt.min().item()}, max {xt.max().item()}, std {xt.std().item()}")
        print(f"Step {t}: predicted_noise min {predicted_noise.min().item()}, max {predicted_noise.max().item()}")

    save_image(imgs*mask, f"test_results/0.png", normalize=True)
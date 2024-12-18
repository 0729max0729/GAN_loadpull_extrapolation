import torch
from torchvision.utils import save_image
from stable_diffusion_train import Autoencoder, LatentDiffusionModel, forward_diffusion, beta_schedule

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 1000  # 扩散过程的总时间步长
num_samples = 8  # 要生成的样本数量

# 加载模型
autoencoder = Autoencoder().to(device)
latent_diffusion_model = LatentDiffusionModel().to(device)

autoencoder.load_state_dict(torch.load("model/autoencoder.pth"))
latent_diffusion_model.load_state_dict(torch.load("model/latent_diffusion.pth"))

autoencoder.eval()
latent_diffusion_model.eval()


# 定义扩散采样过程
@torch.no_grad()
def reverse_diffusion(latent_shape, latent_diffusion_model, T):
    x = torch.randn(latent_shape).to(device)  # 从高斯噪声开始
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t] * x.size(0)).to(device)
        predicted_noise = latent_diffusion_model(x, t_tensor.float())

        sqrt_alpha_t = torch.sqrt(1. - beta_schedule[t]).view(1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(beta_schedule[t]).view(1, 1, 1, 1)

        # 更新公式: x_t -> x_t-1
        x = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        if t > 0:
            noise = torch.randn_like(x)
            x += sqrt_one_minus_alpha_t * noise
    return x


# 生成测试样本
@torch.no_grad()
def generate_samples(autoencoder, latent_diffusion_model, num_samples, T):
    # 从噪声生成潜在向量
    latent_shape = (num_samples, 512, 16, 16)  # 假设 autoencoder 的 latent 形状为 (512, 16, 16)
    generated_latent = reverse_diffusion(latent_shape, latent_diffusion_model, T)

    # 使用 Autoencoder 的解码器将潜在向量还原为图像
    generated_images = autoencoder.decoder(generated_latent)
    # 将图像的值从 [-1, 1] 转换为 [0, 1]
    generated_images = (generated_images + 1) / 2
    return generated_images


# 生成并保存测试样本
samples = generate_samples(autoencoder, latent_diffusion_model, num_samples, T)
save_image(samples, "generated_samples.png", nrow=4)
print("生成的样本已保存为 generated_samples.png")

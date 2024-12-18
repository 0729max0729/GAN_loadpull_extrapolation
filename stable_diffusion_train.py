import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
lr = 1e-4
epochs = 10001
T = 1000  # 扩散过程的总时间步长

# 定义 Beta 调度表
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    alphas = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    alphas = alphas / alphas[0]  # 归一化到 [0, 1]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.9999)  # 确保 beta 值合理

beta_schedule = torch.tensor(cosine_beta_schedule(T)).to(device)

# 定义 Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# 初始化 Autoencoder
autoencoder = Autoencoder().to(device)
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=lr)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 加载数据集
dataset = CustomDataset(root_dir="ADS_smith_chart_plots", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class DoubleConv(nn.Module):
    """
    两层卷积模块：Conv2D -> ReLU -> Conv2D -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super(UNet, self).__init__()
        self.time_dim = time_dim

        # 编码器部分
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # 中间部分
        self.middle = DoubleConv(512, 512)

        # 解码器部分
        self.dec4 = DoubleConv(512 + time_dim + 512, 256)
        self.dec3 = DoubleConv(256 + 256, 128)
        self.dec2 = DoubleConv(128 + 128, 64)
        self.dec1 = DoubleConv(64 + 64, 64)

        # 上采样
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # 最终输出
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # 中间部分
        middle = self.middle(e4)

        # 将时间嵌入扩展到与特征图相同的形状
        time_embedding = time_embedding[:, :, None, None].repeat(1, 1, middle.shape[2], middle.shape[3])

        # 解码器
        d4 = self.dec4(torch.cat([middle, e4, time_embedding], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)

# Diffusion 模型
class LatentDiffusionModel(nn.Module):
    def __init__(self):
        super(LatentDiffusionModel, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.unet = UNet(in_channels=512,out_channels=512)

    def forward(self, x, t):
        t_embedding = self.time_embedding(t.view(-1, 1))
        return self.unet(x, t_embedding)

diffusion_model = LatentDiffusionModel().to(device)
optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=lr)

alphas = 1. - beta_schedule
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

@torch.no_grad()
def forward_diffusion(latent, t):
    noise = torch.randn_like(latent)
    sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_t * latent + sqrt_one_minus_alpha_t * noise
    return xt, noise

if __name__=="__main__":
    # 训练 Autoencoder 和扩散模型
    for epoch in range(epochs):
        autoencoder.train()
        diffusion_model.train()
        for imgs in dataloader:
            imgs = imgs.to(device)

            # Autoencoder 训练
            optimizer_ae.zero_grad()
            latent, reconstructed = autoencoder(imgs)
            loss_ae = F.mse_loss(reconstructed, imgs)
            loss_ae.backward()
            optimizer_ae.step()

            # Diffusion 模型训练
            optimizer_diffusion.zero_grad()
            t = torch.randint(0, T, (imgs.size(0),)).to(device)
            xt, noise = forward_diffusion(latent, t)
            predicted_noise = diffusion_model(xt, t.float())
            loss_diffusion = F.mse_loss(predicted_noise, noise)
            loss_diffusion.backward()
            optimizer_diffusion.step()

        print(f"Epoch {epoch}: AE Loss = {loss_ae.item()}, Diffusion Loss = {loss_diffusion.item()}")

        if epoch % 100 == 0:
            torch.save(autoencoder.state_dict(), "model/autoencoder.pth")
            torch.save(diffusion_model.state_dict(), "model/latent_diffusion.pth")
            print("模型已保存。")

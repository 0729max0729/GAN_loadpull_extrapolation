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
from UNet import UNet


# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
lr = 1e-3
epochs = 10001
T = 1000  # 扩散过程的总时间步长

# 定义 Beta 调度表
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    alphas = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    alphas = alphas / alphas[0]  # 归一化到 [0, 1]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.9999)  # 确保 beta 值合理

def linear_beta_schedule(T, start=1e-4, end=2e-2):
    return torch.linspace(start, end, T)

beta_schedule = torch.tensor(cosine_beta_schedule(T)).to(device)

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

# Diffusion 模型
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()

        # 时间嵌入层
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256),  # 时间步映射到高维
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        # 编码器部分
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        self.enc6 = self.conv_block(1024, 2048)
        self.enc7 = self.conv_block(2048, 4096)  # 增加深度

        # 解码器部分
        self.dec7 = self.upconv_block(4096 + 256, 2048)  # 拼接时间嵌入
        self.dec6 = self.upconv_block(2048 + 2048, 1024)
        self.dec5 = self.upconv_block(1024 + 1024, 512)
        self.dec4 = self.upconv_block(512 + 512, 256)
        self.dec3 = self.upconv_block(256 + 256, 128)
        self.dec2 = self.upconv_block(128 + 128, 64)
        self.dec1 = self.upconv_block(64 + 64, 64)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1,stride=2):
        """卷积块：Conv2d -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels, kernel_size=3, padding=1,stride=2):
        """反卷积块：ConvTranspose2d -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, img, t):
        # 时间嵌入
        t_embedding = self.time_embedding(t.view(-1, 1))  # 将时间步映射为 (batch_size, 256)
        t_embedding = t_embedding.view(-1, 256, 1, 1)  # 调整为 (batch_size, 256, 1, 1)

        # 编码器
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # 上采样时间嵌入以匹配 e7 的形状
        t_embedding = nn.functional.interpolate(t_embedding, size=(e7.shape[2], e7.shape[3]), mode='nearest')
        # 解码器
        d7 = self.dec7(torch.cat([e7, t_embedding], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # 输出
        return self.output_conv(d1)


# 初始化模型
diffusion_model = DiffusionModel().to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr, weight_decay=1e-4)

def forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_oneminus_alphas_cumprod):
    """
    正向扩散过程：从原始数据 x_0 添加噪声到 xt
    Args:
        x_0: 输入数据 (batch_size, channels, height, width)
        t: 时间步 (batch_size,) (可以是一个标量或批量时间步)
        sqrt_alphas_cumprod: 预计算的 alpha 累积乘积的平方根 (T,)
        sqrt_oneminus_alphas_cumprod: 预计算的 (1-alpha) 累积乘积的平方根 (T,)
    Returns:
        xt: 添加噪声后的图像
        noise: 添加的噪声
    """
    noise = torch.randn_like(x_0)  # 生成与 x_0 相同形状的噪声
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # 形状 (batch_size, 1, 1, 1)
    sqrt_oneminus_alphas_cumprod_t = sqrt_oneminus_alphas_cumprod[t].view(-1, 1, 1, 1)

    xt = sqrt_alphas_cumprod_t * x_0 + sqrt_oneminus_alphas_cumprod_t * noise
    return xt, noise



@torch.no_grad()
def backward_diffusion(x_t, t, predicted_noise, alphas, sqrt_alphas_cumprod, sqrt_oneminus_alphas_cumprod, beta_schedule, posterior_variance):
    """
    反向扩散过程：从 x_t 预测 x_{t-1}
    Args:
        x_t: 当前时间步的图像 (batch_size, channels, height, width)
        t: 当前时间步 (标量或批量时间步)
        predicted_noise: 当前时间步模型预测的噪声
        sqrt_alphas_cumprod: 预计算的 alpha 累积乘积的平方根 (T,)
        sqrt_oneminus_alphas_cumprod: 预计算的 (1-alpha) 累积乘积的平方根 (T,)
        beta_schedule: Beta 调度表 (T,)
        posterior_variance: 后验方差 (T,)
    Returns:
        x_prev: 预测的 x_{t-1}
    """

    # 获取 beta_t 和 alpha_t 的值
    beta_t = beta_schedule[t].view(-1, 1, 1, 1)  # 形状 (batch_size, 1, 1, 1)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_oneminus_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_recip_alpha_t = torch.sqrt(1 / alphas[t]).view(-1, 1, 1, 1)

    # 计算均值 (model mean)
    model_mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)

    # 获取后验方差
    posterior_variance_t = posterior_variance[t].view(-1, 1, 1, 1)

    # 如果 t > 0，添加噪声项；否则，直接返回均值

    noise = torch.randn_like(x_t)
    x_prev = model_mean + (posterior_variance_t) * noise


    return x_prev





alphas = 1. - beta_schedule
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 将首个值填充为1
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 计算后验方差
posterior_variance = beta_schedule.sqrt() * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def generate_random_square_mask(image_size, mask_size=16):
    """
    生成中心方形遮罩
    Args:
        image_size: 图像尺寸 (batch_size, channels, height, width)
        mask_size: 方形遮罩的大小
    Returns:
        mask: 中心方形遮罩
    """
    batch_size, c, h, w = image_size

    # 确保 mask_size 不超过图像尺寸
    assert mask_size < h and mask_size < w, "遮罩大小不能超过图像尺寸"

    # 计算遮罩的左上角坐标，使其位于图像中心
    top_left_x = (w - mask_size) // 2
    top_left_y = (h - mask_size) // 2

    # 创建一个遮罩，所有像素初始化为 0
    mask = torch.zeros(image_size, device=device)

    # 在图像的中心区域设置为 1
    mask[:, :, top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = torch.ones(1,c,mask_size,mask_size)

    return mask


# 随机生成随机方形遮罩
def generate_random_square_mask0(image_size, mask_size=64):
    batch_size, c, h, w = image_size
    radius = w // 2  # 半径为图像宽度的一半

    # 圆心位置
    center_x = w // 2
    center_y = h // 2

    mask = torch.zeros(image_size, device=device)

    for i in range(batch_size):
        # 随机生成遮挡块的位置，限制在圆内
        while True:
            top_left_x = torch.randint(0, w - mask_size, (1,)).item()
            top_left_y = torch.randint(0, h - mask_size, (1,)).item()

            # 计算遮挡块中心点的位置
            mask_center_x = top_left_x + mask_size // 2
            mask_center_y = top_left_y + mask_size // 2

            # 检查遮挡块中心点是否在圆内
            if (mask_center_x - center_x) ** 2 + (mask_center_y - center_y) ** 2 <= radius ** 2:
                break

        # 创建一个遮挡区域的掩码

        mask[i,:, top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = torch.ones(1,c,mask_size,mask_size)

    return mask

if __name__ == "__main__":
    model_path="model/diffusion_model256.pth"
    optimizer_path = "model/diffusion_optimizer256.pth"
    # 如果所有模型和优化器的权重文件都存在，则加载
    if os.path.exists(model_path)&os.path.exists(optimizer_path):
        print("发现已有模型和优化器权重文件，正在加载...")
        diffusion_model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        print("未发现模型权重文件，将从头开始训练。")
    # 训练循环
    for epoch in range(epochs):
        diffusion_model.train()
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)

            # 随机生成随机方形遮罩
            mask = generate_random_square_mask(image_size=imgs.size())

            # 随机时间步长 t
            t = torch.randint(0, T, (imgs.size(0),)).to(device)
            # 偏向于采样较大的时间步
            #time_distribution = torch.linspace(0, 1, T) ** 2 # 调整分布权重
            #time_steps = torch.multinomial(time_distribution, num_samples=imgs.size(0), replacement=True)
            #t = time_steps.to(device)
            # 正向扩散（仅遮罩部分加噪声）
            xt, noise = forward_diffusion(imgs, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            #xt = xt / (xt.max()-xt.min())

            xt = xt * (1 - mask)
            noise = noise * (1 - mask)



            # 预测噪声
            optimizer.zero_grad()
            predicted_noise = diffusion_model(xt + imgs * mask, t.float())
            predicted_noise = predicted_noise * (1 - mask)


            # 假设 weights 是根据时间步 t 计算的
            loss = F.mse_loss(predicted_noise, noise)
            #weights = torch.exp(2*t.float() / T)  # 对大时间步给予更大的权重
            #loss = (weights.view(-1,1,1,1) * (predicted_noise - noise).pow(2)).mean()
            loss.backward()
            optimizer.step()

            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] [Loss: {loss.item()}]")
            fake_imgs=backward_diffusion(xt[0], t[0], predicted_noise[0],alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,beta_schedule,posterior_variance)
            print(f"Step {t[1]}: xt min {fake_imgs.min().item()}, max {fake_imgs.max().item()}")

        save_image(fake_imgs*(1-mask[0])+imgs[0]*mask[0], f"images/{epoch + 1}.png",  normalize=False)
        #save_image(predicted_noise[:25]-noise[:25], f"images/{epoch + 1}noise.png", nrow=5, normalize=False)

        # 每 10 个 epoch 保存模型
        if epoch % 100 == 0:
            torch.save(diffusion_model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print("模型已保存。")

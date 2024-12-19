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

from DiT import DiT_L_4, DiT_B_4, DiT_L_8
from Transformer_model import UNetTransformer
from UNet import UNet


# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
lr = 1e-4
epochs = 1001
T = 256  # 扩散过程的总时间步长
image_size=64

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
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 加载数据集
dataset = CustomDataset(root_dir="ADS_smith_chart_plots", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Diffusion 模型
import torch
import torch.nn as nn




# 初始化模型
diffusion_model = DiT_B_4(
        input_size=image_size,  # 输入图像的大小，例如 32x32
        in_channels=3,  # 输入图像通道数（RGB 图像为 3）
        num_classes=1,  # 类别数量
        learn_sigma=False  # 是否学习噪声方差
    ).cuda()  # 使用 GPU

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
def backward_diffusion(x_t, t, predicted_noise, alphas, sqrt_alphas_cumprod, sqrt_oneminus_alphas_cumprod, beta_schedule, sigma):
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


    # 如果 t > 0，添加噪声项；否则，直接返回均值

    noise = torch.randn_like(x_t)
    x_prev = model_mean + predicted_noise * noise


    return x_prev





alphas = 1. - beta_schedule
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 将首个值填充为1
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 计算后验方差
posterior_variance = beta_schedule.sqrt() * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def generate_random_square_mask(image_size, mask_size=int(image_size/4)):
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

model_path=f"model/diffusion_model_transformer{image_size}.pth"
optimizer_path = f"model/diffusion_optimizer_transformer{image_size}.pth"

if __name__ == "__main__":

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

            labels = torch.zeros((imgs.size(0),),dtype=torch.long).cuda()  # 随机分类标签

            # 预测噪声
            optimizer.zero_grad()
            predicted_noise = diffusion_model(xt + imgs * mask, t,labels)



            predicted_noise = predicted_noise * (1 - mask)




            # 计算 NLL 损失
            loss = F.mse_loss(predicted_noise,noise)



            #weights = torch.exp(2*t.float() / T)  # 对大时间步给予更大的权重
            #loss = (weights.view(-1,1,1,1) * (predicted_noise - noise).pow(2)).mean()
            loss.backward()
            optimizer.step()

            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] [Loss: {loss.item()}]")
            #fake_imgs=backward_diffusion(xt[0], t[0], predicted_noise[0],alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,beta_schedule,posterior_variance)
            #print(f"Step {t[1]}: xt min {fake_imgs.min().item()}, max {fake_imgs.max().item()}")


        #save_image(predicted_noise[:25]-noise[:25], f"images/{epoch + 1}noise.png", nrow=5, normalize=False)

        # 每 10 个 epoch 保存模型
        if epoch % 10 == 0:
            torch.save(diffusion_model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print("模型已保存。")
            fake_imgs = backward_diffusion(xt, t, predicted_noise, alphas, sqrt_alphas_cumprod,
                                           sqrt_one_minus_alphas_cumprod, beta_schedule, posterior_variance)
            save_image(fake_imgs * (1 - mask) + imgs * mask, f"images/{epoch + 1}.png",nrow=5, normalize=False)

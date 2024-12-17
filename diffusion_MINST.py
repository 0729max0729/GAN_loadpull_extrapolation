import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from diffusion_train import DiffusionModel as SimpleUNet
import matplotlib.pyplot as plt

# === Step 1: Define Beta Schedule ===
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# === Step 2: Diffusion Model ===
class DiffusionModel:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

# === Step 3: Define the Model ===
# SimpleUNet already imported

# === Step 4: Loss Function ===
def diffusion_loss(model, x0, t, diffusion):
    xt, noise = diffusion.forward_diffusion(x0, t)
    predicted_noise = model(xt, t.float())
    return nn.MSELoss()(predicted_noise, noise)

# === Step 5: Training Function ===
def train_diffusion_model(model, diffusion, dataloader, optimizer, epochs=10, sample_shape=(1, 3, 256, 256)):
    model.train()
    for epoch in range(epochs):
        for x0 in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, diffusion.timesteps, (x0.size(0),)).to(device)  # Random time steps
            optimizer.zero_grad()
            loss = diffusion_loss(model, x0, t, diffusion)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f"model_epoch_{epoch + 1}.pth")

        # Generate and save sample image
        samples = sample(model, diffusion, shape=sample_shape)
        sample_image = samples[0].cpu().permute(1, 2, 0).numpy()
        sample_image = (sample_image * 0.5 + 0.5)  # De-normalize to [0, 1]
        plt.imshow(sample_image)
        plt.axis("off")
        plt.savefig(f"images/sample_epoch_{epoch + 1}.png")
        plt.close()

# === Step 6: Sampling Function ===
@torch.no_grad()
def sample(model, diffusion, shape):
    model.eval()
    x = torch.randn(shape).to(device)  # Start from Gaussian noise
    for t in reversed(range(diffusion.timesteps)):
        alpha_t = diffusion.alphas[t]
        beta_t = diffusion.betas[t]
        alpha_cumprod_t = diffusion.alpha_cumprod[t]
        noise = torch.randn_like(x) if t > 0 else 0
        predicted_noise = model(x, torch.tensor([t]).float().to(device))
        x = (x - beta_t / (1 - alpha_cumprod_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        x += beta_t.sqrt() * noise
    return x

# === Step 7: Inpainting Function ===
@torch.no_grad()
def inpaint(model, diffusion, masked_image, mask):
    model.eval()
    x = masked_image.clone().to(device)
    mask = mask.to(device)
    for t in reversed(range(diffusion.timesteps)):
        alpha_t = diffusion.alphas[t]
        beta_t = diffusion.betas[t]
        alpha_cumprod_t = diffusion.alpha_cumprod[t]
        noise = torch.randn_like(x) if t > 0 else 0
        predicted_noise = model(x, torch.tensor([t]).float().to(device))
        x = (x - beta_t / (1 - alpha_cumprod_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        x += beta_t.sqrt() * noise
        x = x * (1 - mask) + masked_image * mask  # Preserve known regions
    return x

# === Custom Dataset Class ===
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


def generate_random_square_mask(image_size, mask_size=64):
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
    assert mask_size <= h and mask_size <= w, "遮罩大小不能超过图像尺寸"

    # 计算遮罩的左上角坐标，使其位于图像中心
    top_left_x = (w - mask_size) // 2
    top_left_y = (h - mask_size) // 2

    # 创建一个遮罩，所有像素初始化为 0
    mask = torch.zeros(image_size, device=device)

    # 在图像的中心区域设置为 1
    mask[:, :, top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = torch.ones(1,c,mask_size,mask_size)

    return mask

# === Step 8: Main Script ===
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    timesteps = 1000
    epochs = 10
    batch_size = 64
    learning_rate = 1e-4

    # Initialize diffusion model and network
    diffusion = DiffusionModel(timesteps)
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_path = "model/diffusion_model.pth"
    optimizer_path = "model/diffusion_optimizer.pth"

    # 如果所有模型和优化器的权重文件都存在，则加载
    if os.path.exists(model_path) & os.path.exists(optimizer_path):
        print("发现已有模型和优化器权重文件，正在加载...")
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        print("未发现模型权重文件，将从头开始训练。")

    # Load custom dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    dataset = CustomDataset(root_dir="ADS_smith_chart_plots", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_diffusion_model(model, diffusion, dataloader, optimizer, epochs=epochs, sample_shape=(1, 3, 256, 256))

    # Generate samples
    samples = sample(model, diffusion, shape=(1, 3, 256, 256))

    # Visualize results
    samples = samples.cpu().numpy()
    sample_image = samples[0].transpose(1, 2, 0)
    plt.imshow(sample_image)
    plt.axis("off")
    plt.show()

    # Example: Inpainting
    mask = generate_random_square_mask(samples[0].shape)  # Define a mask for inpainting
    mask[:, 10:20, 10:20] = 0  # Example masked region
    inpainted_image = inpaint(model, diffusion, samples[0], mask)
    plt.imshow(inpainted_image.cpu().numpy()[0], cmap="gray")
    plt.axis("off")
    plt.show()

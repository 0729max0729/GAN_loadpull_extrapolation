import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import os

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
lr = 0.0002
epochs = 300

# 图像修复中的遮挡函数
def mask_image(img, mask_size=64):
    """在图像上添加遮挡块并用高斯噪声填充，遮挡区域限制在圆内"""
    _, h, w = img.size()
    radius = w // 2  # 半径为图像宽度的一半

    # 圆心位置
    center_x = w // 2
    center_y = h // 2

    # 随机生成遮挡块的位置，限制在圆内
    while True:
        top_left_x = torch.randint(0, w - mask_size, (1,)).item()
        top_left_y = torch.randint(0, h - mask_size, (1,)).item()

        # 计算遮挡块中心点的位置
        mask_center_x = top_left_x + mask_size // 2
        mask_center_y = top_left_y + mask_size // 2

        # 检查遮挡块中心点是否在圆内
        if (mask_center_x - center_x)**2 + (mask_center_y - center_y)**2 <= radius**2:
            break

    # 创建一个遮挡区域的掩码
    mask = torch.zeros_like(img)
    mask[:, top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = 1

    # 在遮挡区域内添加高斯噪声
    noise = torch.normal(mean=0, std=1, size=img.size()).to(img.device)
    img = img * mask + noise# * (1-mask)  # 原图与噪声混合
    return img


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        self.enc6 = self.conv_block(1024, 2048)
        self.enc7 = self.conv_block(2048, 4096)  # 增加深度

        # 解码器部分
        self.dec7 = self.upconv_block(4096, 2048)
        self.dec6 = self.upconv_block(2048 + 2048, 1024)
        self.dec5 = self.upconv_block(1024 + 1024, 512)
        self.dec4 = self.upconv_block(512 + 512, 256)
        self.dec3 = self.upconv_block(256 + 256, 128)
        self.dec2 = self.upconv_block(128 + 128, 64)
        self.dec1 = self.upconv_block(64 + 64, 64)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh())


    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """卷积块：Conv2d -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        """反卷积块：ConvTranspose2d -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        # 编码器
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # 解码器
        d7 = self.dec7(e7)
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # 输出
        return self.output_conv(d1)






# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2048, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加全局平均池化层
            nn.Flatten(),  # 压平输出
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 加载图像并转换为 RGB
        if self.transform:
            image = self.transform(image)
        return image




# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 图像归一化到 [-1, 1]
])

# 加载自定义数据集
dataset = CustomDataset(root_dir="ADS_smith_chart_plots", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
reconstruction_criterion = nn.L1Loss()  # 使用 L1 损失
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


import os

# 检查文件是否存在
generator_path = "model/generator_UNET.pth"
discriminator_path = "model/discriminator_UNET.pth"
optimizer_G_path = "model/optimizer_G_UNET.pth"
optimizer_D_path = "model/optimizer_D_UNET.pth"

# 如果所有模型和优化器的权重文件都存在，则加载
if all(os.path.exists(path) for path in [generator_path, discriminator_path, optimizer_G_path, optimizer_D_path]):
    print("发现已有模型和优化器权重文件，正在加载...")
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))
    optimizer_G.load_state_dict(torch.load(optimizer_G_path))
    optimizer_D.load_state_dict(torch.load(optimizer_D_path))
else:
    print("未发现模型权重文件，将从头开始训练。")







if __name__ == "__main__":
    # 实时绘图设置
    plt.ion()
    fig, ax = plt.subplots()
    g_loss_history = []
    d_loss_history = []
    line_g, = ax.plot([], [], label="Generator Loss")
    line_d, = ax.plot([], [], label="Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


    def update_plot():
        line_g.set_data(range(len(g_loss_history)), g_loss_history)
        line_d.set_data(range(len(d_loss_history)), d_loss_history)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)



    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            masked_imgs = torch.ones(batch_size, 3, imgs.size(2), imgs.size(3)).to(device)
            for j in range(batch_size):
                masked_imgs[j] = mask_image(imgs[j])

            real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
            real_labels = real_labels.view(-1, 1)
            fake_labels = fake_labels.view(-1, 1)

            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs), real_labels)
            fake_imgs = generator(masked_imgs)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            reconstruction_loss = reconstruction_criterion(fake_imgs, imgs)
            total_g_loss = g_loss + 50 * reconstruction_loss
            total_g_loss.backward()
            optimizer_G.step()



            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Reconstruction loss: {reconstruction_loss.item()}]")

            g_loss_history.append(g_loss.item())
            d_loss_history.append(d_loss.item())
            update_plot()
        # 保存生成的图像
        save_image(fake_imgs[:25], f"images/{epoch + 1}.png", nrow=5, normalize=True)
        if epoch % 20 == 19:
            torch.save(generator.state_dict(), f"model/generator_UNET.pth")
            torch.save(discriminator.state_dict(), f"model/discriminator_UNET.pth")
            torch.save(optimizer_G.state_dict(), f"model/optimizer_G_UNET.pth")
            torch.save(optimizer_D.state_dict(), f"model/optimizer_D_UNET.pth")

    plt.ioff()
    plt.show()
    plt.savefig("test_results/loss.png")
    print("训练完成！")




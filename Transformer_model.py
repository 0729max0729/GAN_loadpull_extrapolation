import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, t):
        t = t.view(-1, 1)  # 確保 t 是 (batch_size, 1)
        return self.time_embedding(t)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, embed_dim)
        return x, H // self.patch_size, W // self.patch_size


class LocalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LocalAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_ln = self.ln(x)
        attention_output, _ = self.mha(x_ln, x_ln, x_ln)
        return attention_output + x  # 殘差連接


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([LocalAttention(embed_dim) for _ in range(num_layers)])

    def forward(self, x, time_emb):
        for layer in self.layers:
            x = layer(x)
            x = x + time_emb.unsqueeze(1)  # 加入時間嵌入
        return x


class UNetTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embed_dim=64, patch_size=4, time_dim=128):
        super(UNetTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.time_embedding = TimeEmbedding(embed_dim)

        # 编码器
        self.encoder1 = TransformerEncoder(embed_dim, num_layers=2)
        self.encoder2 = TransformerEncoder(embed_dim * 2, num_layers=2)
        self.encoder3 = TransformerEncoder(embed_dim * 4, num_layers=2)

        # 通道调整
        self.adjust_skip1 = nn.Linear(embed_dim, embed_dim * 2)
        self.adjust_skip2 = nn.Linear(embed_dim * 2, embed_dim * 4)

        # 瓶颈层
        self.bottleneck = TransformerEncoder(embed_dim * 4, num_layers=4)

        # 解码器
        self.upconv1 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.decoder1 = TransformerEncoder(embed_dim * 2, num_layers=2)

        self.upconv2 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.decoder2 = TransformerEncoder(embed_dim, num_layers=2)

        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, t):
        B, C, H, W = x.shape

        # 时间嵌入
        t_emb = self.time_embedding(t)  # (B, embed_dim)

        # Patchify and encode
        x, H_p, W_p = self.patch_embed(x)  # (B, num_patches, embed_dim)
        skip1 = self.encoder1(x, t_emb)

        # 通道调整
        skip1 = self.adjust_skip1(skip1)

        skip2 = self.encoder2(skip1, t_emb)
        skip2 = self.adjust_skip2(skip2)

        # 瓶颈层
        x = self.bottleneck(skip2, t_emb)

        # 解码器
        x = x + skip2
        x = x.permute(0, 2, 1).view(B, -1, H_p, W_p)
        x = self.upconv1(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.decoder1(x, t_emb)

        x = x + skip1
        x = x.permute(0, 2, 1).view(B, -1, H_p * 2, W_p * 2)
        x = self.upconv2(x)
        x = self.decoder2(x.flatten(2).permute(0, 2, 1), t_emb)

        x = x.permute(0, 2, 1).view(B, -1, H_p * 4, W_p * 4)
        x = self.output_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



# 測試模型
if __name__ == "__main__":
    from torchvision.transforms import v2

    H, W = 32, 32
    img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transforms(img)

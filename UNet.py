import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_c),  # equivalent with LayerNorm
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_c),  # equivalent with LayerNorm
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, t):
        t = t.view(-1, 1)  # 保證 t 是 (batch_size, 1)
        return self.time_embedding(t)

class Down(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, out_c),
        )

    def forward(self, x, t):
        x = self.down(x)
        # Generate t_emb with correct dimensions
        t_emb = self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, out_c, 1, 1)
        t_emb = t_emb.repeat(1, 1, x.shape[-2], x.shape[-1])  # Match spatial dimensions
        return x + t_emb


class Up(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_c, out_c)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_c),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)

        # Generate t_emb with correct dimensions
        t_emb = self.emb_layer(t).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, out_c, 1, 1)
        t_emb = t_emb.repeat(1, 1, x.shape[-2], x.shape[-1])  # Match spatial dimensions
        return x + t_emb


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels

        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        # Attention output projection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(batch_size, -1, height * width)  # (B, C//8, H*W)
        value = self.value(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # Compute attention map
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)  # Normalize along the last dimension

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)  # Reshape to (B, C, H, W)

        # Apply learnable scaling factor and residual connection
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)  # (b,3,64,64) -> (b,64,64,64)

        self.down1 = Down(64, 128)  # (b,64,64,64) -> (b,128,32,32)
        self.sa1 = SelfAttention(128)  # (b,128,32,32) -> (b,128,32,32)
        self.down2 = Down(128, 256)  # (b,128,32,32) -> (b,256,16,16)
        self.sa2 = SelfAttention(256)  # (b,256,16,16) -> (b,256,16,16)
        self.down3 = Down(256, 256)  # (b,256,16,16) -> (b,256,8,8)
        self.sa3 = SelfAttention(256)  # (b,256,8,8) -> (b,256,8,8)

        self.bot1 = DoubleConv(256, 512)  # (b,256,8,8) -> (b,512,8,8)
        self.bot2 = DoubleConv(512, 512)  # (b,512,8,8) -> (b,512,8,8)
        self.bot3 = DoubleConv(512, 256)  # (b,512,8,8) -> (b,256,8,8)

        self.up1 = Up(512, 128)  # (b,512,8,8) -> (b,128,16,16) because the skip_x
        self.sa4 = SelfAttention(128)  # (b,128,16,16) -> (b,128,16,16)
        self.up2 = Up(256, 64)  # (b,256,16,16) -> (b,64,32,32)
        self.sa5 = SelfAttention(64)  # (b,64,32,32) -> (b,64,32,32)
        self.up3 = Up(128, 64)  # (b,128,32,32) -> (b,64,64,64)
        self.sa6 = SelfAttention(64)  # (b,64,64,64) -> (b,64,64,64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)  # (b,64,64,64) -> (b,3,64,64)

    def pos_encoding(self, t, channels):
        #t = torch.tensor([t])
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2).float() / channels)
        ).to(device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # (bs,) -> (bs, time_dim)
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # initial conv
        x1 = self.inc(x)

        # Down
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottle neck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Up
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        # Output
        output = self.outc(x)
        return output

if __name__ == "__main__":
    device = torch.device('cpu')
    T=300
    sample = torch.randn((8, 3, 128, 128))
    t = torch.randint(0, T, (8,))

    model = UNet()
    print(model(sample, t).shape,t.shape)

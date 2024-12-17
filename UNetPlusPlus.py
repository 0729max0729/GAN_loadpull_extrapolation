import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):
        x1 = self.up(x1)
        x = torch.cat([x4, x3, x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, target_size):
        diffX = target_size[2] - x.size(2)
        diffY = target_size[3] - x.size(3)
        x = F.pad(x, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
        return self.conv(x)


class Unet_2D(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, mode='train'):
        super(Unet_2D, self).__init__()
        cc = 32  # 可调整为更小的值以优化速度

        # 时间嵌入层
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_channels),  # 将时间嵌入调整到与输入通道数匹配
            nn.ReLU(inplace=True)
        )

        # 编码器部分
        self.inconv = double_conv2(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)

        # 解码器部分
        self.up1 = up(12 * cc, 4 * cc)
        self.up20 = up(6 * cc, 2 * cc)
        self.up2 = up3(8 * cc, 2 * cc)
        self.up30 = up(3 * cc, cc)
        self.up31 = up3(4 * cc, cc)
        self.up3 = up4(5 * cc, cc)

        # 输出层
        self.outconv = outconv(cc, n_classes)
        self.mode = mode

    def forward(self, x, t):
        target_size = x.size()  # 保存输入大小

        # 处理时间步 t
        t_embedding = self.time_embedding(t.view(-1, 1))
        t_embedding = t_embedding.view(-1, t_embedding.size(1), 1, 1)
        t_embedding = t_embedding.expand(-1, -1, x.size(2), x.size(3))

        # 将时间嵌入与输入图像相加
        x = x + t_embedding

        if self.mode == 'train':
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

            x = self.up1(x4, x3)
            x21 = self.up20(x3, x2)
            x = self.up2(x, x21, x2)
            x11 = self.up30(x2, x1)
            x12 = self.up31(x21, x11, x1)
            x = self.up3(x, x12, x11, x1)

            y2 = self.outconv(x, target_size)
            y0 = self.outconv(x11, target_size)
            y1 = self.outconv(x12, target_size)

            return y0, y1, y2
        '''
        else:
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x11 = self.up30(x2, x1)
            y0 = self.outconv(x11, target_size)
            return y0
        '''


if __name__ == "__main__":
    model = Unet_2D(n_channels=3, n_classes=3, mode='train')
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([10.0])  # 示例时间步
    out0,out1,out2 = model(x, t)
    print(f"Output shape: {out1.shape}")

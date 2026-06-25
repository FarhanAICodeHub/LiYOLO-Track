import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.conv(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, c1=None, reduction=16, kernel_size=7):
        super().__init__()
        self.r = reduction
        self.k = kernel_size
        self.ca = None
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        if self.ca is None:
            self.ca = ChannelAttention(x.shape[1], self.r)
            if next(self.sa.parameters()).is_cuda:
                self.ca = self.ca.cuda()
        return self.sa(self.ca(x))

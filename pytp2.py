import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import tqdm

device = torch.device("cpu")

# ==================================
# Dataset: CIFAR-10
# ==================================

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # scales to [0,1] tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # scales to [-1,1]
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ==================================
# Forward Diffusion
# ==================================


# determines the timesteps
def make_beta_schedule(T=100, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    return betas


# determines the accumulated timesteps
def compute_alphas(betas):
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    return (alphas, alpha_cumprod)


# distorts the image and gives the noise
def forward_diffusion_sample(x0, t, alpha_cumprod):
    noise = torch.randn_like(x0)

    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1, 1)

    xt = x0 * sqrt_alpha_cumprod + sqrt_one_minus_alpha_cumprod * noise

    return (xt, noise)


# returns random timesteps uniformly from [0, T-1] for each image
def sample_timesteps(batch_size, T):
    return torch.randint(0, T, (batch_size,), dtype=torch.long)


# ==================================
# U-NET
# ==================================


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.silu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


# ---- Down block: maxpool + DoubleConv ----
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ---- Up block: upsample + DoubleConv ----
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad so shapes match before concatenation
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---- Final output conv ----
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---- The full UNet ----
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)  # Initial conv
        self.down1 = Down(64, 128)  # Down block 1
        self.down2 = Down(128, 256)  # Down block 2
        self.down3 = Down(256, 512)  # Down block 3
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # Down block 4
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # Final conv

    def forward(self, x):
        x1 = self.inc(x)  # First conv
        x2 = self.down1(x1)  # Downsample 1
        x3 = self.down2(x2)  # Downsample 2
        x4 = self.down3(x3)  # Downsample 3
        x5 = self.down4(x4)  # Bottleneck
        x = self.up1(x5, x4)  # Upsample 1
        x = self.up2(x, x3)  # Upsample 2
        x = self.up3(x, x2)  # Upsample 3
        x = self.up4(x, x1)  # Upsample 4
        logits = self.outc(x)  # Output
        return logits


# ==================================
# Loss + Loop
# ==================================


# ==================================
# Sampling
# ==================================


# ==================================
# Visualizations
# ==================================

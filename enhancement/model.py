"""
LIVENet: A Novel Network for Real-World Low-Light Image Denoising and Enhancement
WACV 2024 - Makwana et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _match(x, ref):
    """Crop x to ref's spatial size — handles odd-dimension edge cases."""
    if x.shape[2] != ref.shape[2] or x.shape[3] != ref.shape[3]:
        x = x[:, :, :ref.shape[2], :ref.shape[3]]
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)


class ResidualSEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(self.se(out) + x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = _match(x, skip)          # handles odd-size inputs
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# LSDB — Latent Subspace Denoising Block
# ─────────────────────────────────────────────────────────────────────────────

class LSDB(nn.Module):
    def __init__(self, channels, rank=32):
        super().__init__()
        self.rank = rank
        self.conv_shallow = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.branch_a = nn.Sequential(nn.Conv2d(channels, rank, 1, bias=False), nn.GELU())
        self.branch_b = nn.Sequential(nn.Conv2d(channels, rank, 1, bias=False), nn.GELU())
        self.proj = nn.Conv2d(rank, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        shallow  = self.conv_shallow(x)
        fa       = self.branch_a(shallow).view(B, self.rank, H * W)
        fb       = self.branch_b(shallow).view(B, self.rank, H * W)
        cov      = torch.bmm(fa, fb.transpose(1, 2)) / (H * W)
        low_rank = torch.bmm(cov, fa).view(B, self.rank, H, W)
        return self.proj(low_rank) + x


# ─────────────────────────────────────────────────────────────────────────────
# GSIP — GrayScale Image Prediction
# ─────────────────────────────────────────────────────────────────────────────

class GSIP(nn.Module):
    """
    Encoder-decoder with 6 ResidualSE blocks + LSDB bottleneck.
    Skip connections: up3←e2, up2←e1, up1←e0  (correct U-Net order)
    Input : (B, 3, H, W)  →  Output: (B, 1, H, W) noise-free grayscale
    """
    def __init__(self, base_ch=32, rank=32):
        super().__init__()
        C = base_ch
        self.enc0  = nn.Conv2d(3, C, 3, padding=1)
        self.res0  = ResidualSEBlock(C)
        self.down1 = DownBlock(C,   C*2)
        self.res1  = ResidualSEBlock(C*2)
        self.down2 = DownBlock(C*2, C*4)
        self.res2  = ResidualSEBlock(C*4)
        self.down3 = DownBlock(C*4, C*8)
        self.res3  = ResidualSEBlock(C*8)
        self.res4  = ResidualSEBlock(C*8)
        self.lsdb  = LSDB(C*8, rank=rank)
        self.res5  = ResidualSEBlock(C*8)
        # Correct skip sizes: bottleneck(C*8) + e2(C*4) → C*4
        self.up3   = UpBlock(C*8 + C*4, C*4)
        self.up2   = UpBlock(C*4 + C*2, C*2)
        self.up1   = UpBlock(C*2 + C,   C)
        self.out   = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        e0 = self.res0(self.enc0(x))           # full res
        e1 = self.res1(self.down1(e0))         # /2
        e2 = self.res2(self.down2(e1))         # /4
        e3 = self.res3(self.down3(e2))         # /8
        b  = self.res5(self.lsdb(self.res4(e3)))
        d2 = self.up3(b,  e2)                  # /8→/4, cat e2
        d1 = self.up2(d2, e1)                  # /4→/2, cat e1
        d0 = self.up1(d1, e0)                  # /2→full, cat e0
        return self.out(d0)


# ─────────────────────────────────────────────────────────────────────────────
# CMIG — Coarse Map Image Generation
# ─────────────────────────────────────────────────────────────────────────────

class CMIG(nn.Module):
    """
    Predicts transmission map t (B,1,H,W) and atmospheric light A (B,3,1,1).
    """
    def __init__(self, base_ch=32):
        super().__init__()
        C = base_ch
        self.enc0  = nn.Conv2d(3, C, 3, padding=1)
        self.res0  = ResidualSEBlock(C)
        self.down1 = DownBlock(C,   C*2)
        self.res1  = ResidualSEBlock(C*2)
        self.down2 = DownBlock(C*2, C*4)
        self.res2  = ResidualSEBlock(C*4)
        self.down3 = DownBlock(C*4, C*8)
        self.res3  = ResidualSEBlock(C*8)
        self.res4  = ResidualSEBlock(C*8)
        self.res5  = ResidualSEBlock(C*8)
        self.t_up3 = UpBlock(C*8 + C*4, C*4)
        self.t_up2 = UpBlock(C*4 + C*2, C*2)
        self.t_up1 = UpBlock(C*2 + C,   C)
        self.t_out = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1), nn.Sigmoid()
        )
        self.atm_pool = nn.AdaptiveAvgPool2d(1)
        self.atm_fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*8, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 3),   nn.Sigmoid()
        )

    def forward(self, x):
        e0 = self.res0(self.enc0(x))
        e1 = self.res1(self.down1(e0))
        e2 = self.res2(self.down2(e1))
        e3 = self.res3(self.down3(e2))
        b  = self.res5(self.res4(e3))
        t  = self.t_up3(b,  e2)   # /8→/4, cat e2
        t  = self.t_up2(t,  e1)   # /4→/2, cat e1
        t  = self.t_up1(t,  e0)   # /2→full, cat e0
        t  = self.t_out(t)
        atm = self.atm_fc(self.atm_pool(b)).view(-1, 3, 1, 1)
        return t, atm


# ─────────────────────────────────────────────────────────────────────────────
# Atmospheric Scattering Model + YCbCr helpers
# ─────────────────────────────────────────────────────────────────────────────

def atmospheric_scattering_model(ilow, t, atm, eps=1e-6):
    return ((ilow - atm) / t.clamp(min=eps) + atm).clamp(0.0, 1.0)

def rgb_to_ycbcr(img):
    r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
    y  =  0.299*r + 0.587*g + 0.114*b
    cb = -0.1687*r - 0.3313*g + 0.5*b + 0.5
    cr =  0.5*r - 0.4187*g - 0.0813*b + 0.5
    return torch.cat([y, cb, cr], dim=1)

def ycbcr_to_rgb(img):
    y, cb, cr = img[:,0:1], img[:,1:2]-0.5, img[:,2:3]-0.5
    r = y + 1.402*cr
    g = y - 0.34414*cb - 0.71414*cr
    b = y + 1.772*cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SFT Layer + Refinement Stage
# ─────────────────────────────────────────────────────────────────────────────

class SFTLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.beta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, feat, cond):
        cond = _match(cond, feat)
        return self.gamma(cond) * feat + self.beta(cond)


class RefinementStage(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        C = base_ch
        self.a_enc0  = nn.Sequential(nn.Conv2d(3, C, 3, padding=1), nn.ReLU(True))
        self.a_down1 = DownBlock(C,   C*2)
        self.a_down2 = DownBlock(C*2, C*4)
        self.a_down3 = DownBlock(C*4, C*8)
        self.b_enc0  = nn.Sequential(nn.Conv2d(1, C, 3, padding=1), nn.ReLU(True))
        self.b_down1 = DownBlock(C,   C*2)
        self.b_down2 = DownBlock(C*2, C*4)
        self.b_down3 = DownBlock(C*4, C*8)
        self.sft0 = SFTLayer(C)
        self.sft1 = SFTLayer(C*2)
        self.sft2 = SFTLayer(C*4)
        self.sft3 = SFTLayer(C*8)
        self.bottleneck = ResidualSEBlock(C*8)
        self.up3 = UpBlock(C*8 + C*4, C*4)
        self.up2 = UpBlock(C*4 + C*2, C*2)
        self.up1 = UpBlock(C*2 + C,   C)
        self.out = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, 3, 1), nn.Sigmoid()
        )

    def forward(self, coarse_map, gray_img):
        a0 = self.a_enc0(coarse_map);  b0 = self.b_enc0(gray_img)
        a1 = self.a_down1(a0);         b1 = self.b_down1(b0)
        a2 = self.a_down2(a1);         b2 = self.b_down2(b1)
        a3 = self.a_down3(a2);         b3 = self.b_down3(b2)
        a0 = self.sft0(a0, b0);  a1 = self.sft1(a1, b1)
        a2 = self.sft2(a2, b2);  a3 = self.sft3(a3, b3)
        bot = self.bottleneck(a3)
        d2  = self.up3(bot, a2)   # /8→/4, cat a2
        d1  = self.up2(d2,  a1)   # /4→/2, cat a1
        d0  = self.up1(d1,  a0)   # /2→full, cat a0
        return self.out(d0)


# ─────────────────────────────────────────────────────────────────────────────
# Full LIVENet
# ─────────────────────────────────────────────────────────────────────────────

class LIVENet(nn.Module):
    def __init__(self, base_ch=32, lsdb_rank=32):
        super().__init__()
        self.gsip   = GSIP(base_ch=base_ch, rank=lsdb_rank)
        self.cmig   = CMIG(base_ch=base_ch)
        self.refine = RefinementStage(base_ch=base_ch)

    def forward(self, ilow):
        gray_pred      = self.gsip(ilow)
        t, atm         = self.cmig(ilow)
        coarse_rgb     = atmospheric_scattering_model(ilow, t, atm)
        coarse_ycbcr   = rgb_to_ycbcr(coarse_rgb)
        denoised_ycbcr = torch.cat([gray_pred,
                                     coarse_ycbcr[:, 1:2],
                                     coarse_ycbcr[:, 2:3]], dim=1)
        denoised_rgb   = ycbcr_to_rgb(denoised_ycbcr)
        enhanced       = self.refine(denoised_rgb, gray_pred)
        return {
            'gray':         gray_pred,
            'trans_map':    t,
            'atm_light':    atm,
            'coarse_rgb':   coarse_rgb,
            'denoised_rgb': denoised_rgb,
            'enhanced':     enhanced,
        }


if __name__ == '__main__':
    model = LIVENet(base_ch=32, lsdb_rank=32)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LIVENet parameters: {total/1e6:.2f} M')
    for H in [128, 100, 256, 300]:
        x = torch.randn(1, 3, H, H)
        with torch.no_grad():
            out = model(x)
        print(f'  H={H}: enhanced={tuple(out["enhanced"].shape)} ✓')
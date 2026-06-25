"""
Evaluation metrics for LIVENet:
  PSNR, SSIM, MAE, LPIPS

All functions expect tensors of shape (B, 3, H, W) in [0, 1].
"""

import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    with torch.no_grad():
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10.0 * math.log10(max_val ** 2 / mse.item())


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_window(size: int, sigma: float, channels: int, device):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel_2d = kernel.outer(kernel)
    window = kernel_2d.view(1, 1, size, size).expand(channels, 1, size, size)
    return window.contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> float:
    with torch.no_grad():
        C = pred.shape[1]
        win = _gaussian_window(window_size, sigma, C, pred.device)
        pad = window_size // 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2

        mu1 = F.conv2d(pred,   win, padding=pad, groups=C)
        mu2 = F.conv2d(target, win, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2

        s1 = F.conv2d(pred*pred,     win, padding=pad, groups=C) - mu1_sq
        s2 = F.conv2d(target*target, win, padding=pad, groups=C) - mu2_sq
        s12= F.conv2d(pred*target,   win, padding=pad, groups=C) - mu12

        ssim_map = ((2*mu12 + C1)*(2*s12 + C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
        return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# MAE
# ─────────────────────────────────────────────────────────────────────────────

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return F.l1_loss(pred, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS (lightweight VGG-based approximation; install lpips for full version)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import lpips as lpips_lib

    _lpips_net = None

    def lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net='alex').to(pred.device)
            _lpips_net.eval()
        # lpips expects [-1, 1]
        with torch.no_grad():
            pred_n   = pred   * 2 - 1
            target_n = target * 2 - 1
            score = _lpips_net(pred_n, target_n)
        return score.mean().item()

except ImportError:

    def lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Fallback: mean squared error in feature space (not a true LPIPS)."""
        with torch.no_grad():
            return F.mse_loss(pred, target).item()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all four metrics for a single (B, 3, H, W) pair."""
    return {
        'psnr':  psnr(pred, target),
        'ssim':  ssim(pred, target),
        'mae':   mae(pred, target),
        'lpips': lpips(pred, target),
    }

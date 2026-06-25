"""
LIVENet Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels    = channels
        # Register as buffer so it moves with .to(device) and .half()
        self.register_buffer('window', self._make_window(window_size, sigma, channels))

    @staticmethod
    def _gaussian_1d(size, sigma):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        k = torch.exp(-x ** 2 / (2 * sigma ** 2))
        return k / k.sum()

    def _make_window(self, size, sigma, ch):
        k1d = self._gaussian_1d(size, sigma)
        k2d = k1d.outer(k1d).unsqueeze(0).unsqueeze(0)   # (1,1,ws,ws)
        return k2d.expand(ch, 1, size, size).contiguous()

    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2

        # Cast window to same dtype as input (handles FP16 autocast)
        win = self.window.to(dtype=pred.dtype)

        mu1 = F.conv2d(pred,   win, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, win, padding=pad, groups=self.channels)
        mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        s1  = F.conv2d(pred   * pred,   win, padding=pad, groups=self.channels) - mu1s
        s2  = F.conv2d(target * target, win, padding=pad, groups=self.channels) - mu2s
        s12 = F.conv2d(pred   * target, win, padding=pad, groups=self.channels) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
                   ((mu1s + mu2s + C1) * (s1 + s2 + C2))
        return 1.0 - ssim_map.mean()


class LIVENetLoss(nn.Module):
    """
    Total loss:
      L_gray   = lambda_l1 * L1(gray_pred, gray_gt) + lambda_ssim * SSIM(gray_pred, gray_gt)
      L_refine = lambda_l1 * L1(enhanced,  normal)  + lambda_ssim * SSIM(enhanced,  normal)
      total    = w_gray * L_gray + w_refine * L_refine
    """
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5,
                 w_gray=0.5, w_refine=1.0):
        super().__init__()
        self.l1          = nn.L1Loss()
        self.ssim_gray   = SSIMLoss(channels=1)
        self.ssim_rgb    = SSIMLoss(channels=3)
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.w_gray      = w_gray
        self.w_refine    = w_refine

    def forward(self, preds, targets):
        normal  = targets['normal']
        gray_gt = targets['gray_gt']

        l_gray = (self.lambda_l1   * self.l1(preds['gray'], gray_gt) +
                  self.lambda_ssim * self.ssim_gray(preds['gray'], gray_gt))

        l_refine = (self.lambda_l1   * self.l1(preds['enhanced'], normal) +
                    self.lambda_ssim * self.ssim_rgb(preds['enhanced'], normal))

        total = self.w_gray * l_gray + self.w_refine * l_refine

        return {
            'total':    total,
            'l_gray':   l_gray,
            'l_refine': l_refine,
        }
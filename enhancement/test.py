"""
LIVENet Test / Inference Script

Usage:
    # Evaluate on LoLv1 test split (paired – prints metrics):
    python test.py --mode eval \
                   --val_root data/eval15 \
                   --ckpt checkpoints/best.pth \
                   --save_dir results/eval15

    # Inference on unpaired dark images in a folder:
    python test.py --mode infer \
                   --input_dir my_dark_images/ \
                   --ckpt checkpoints/best.pth \
                   --save_dir results/infer
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from model   import LIVENet
from dataset import LolDataset, SingleImageDataset
from metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device, base_ch=32, lsdb_rank=32) -> LIVENet:
    model = LIVENet(base_ch=base_ch, lsdb_rank=lsdb_rank).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)   # support both full-dict and bare state_dict
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def save_image(tensor: torch.Tensor, path: str):
    """tensor: (3, H, W) in [0, 1]"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    vutils.save_image(tensor, path)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation mode  (paired – computes PSNR / SSIM / MAE / LPIPS)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_root: str, save_dir: str, device):
    ds     = LolDataset(val_root, split='val', patch_size=None, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    psnr_sum = ssim_sum = mae_sum = lpips_sum = 0.0
    n = 0

    print(f"\n{'Image':<30} {'PSNR':>7} {'SSIM':>7} {'MAE':>7} {'LPIPS':>7}")
    print('-' * 62)

    for batch in loader:
        low    = batch['low'].to(device)
        normal = batch['normal'].to(device)
        name   = Path(batch['low_path'][0]).name

        preds = model(low)
        enh   = preds['enhanced']

        m = compute_metrics(enh, normal)
        psnr_sum  += m['psnr']
        ssim_sum  += m['ssim']
        mae_sum   += m['mae']
        lpips_sum += m['lpips']
        n += 1

        print(f"{name:<30} {m['psnr']:7.2f} {m['ssim']:7.4f} "
              f"{m['mae']:7.4f} {m['lpips']:7.4f}")

        if save_dir:
            save_image(enh[0].cpu(), os.path.join(save_dir, name))

    avg = {k: v / n for k, v in
           [('PSNR', psnr_sum), ('SSIM', ssim_sum),
            ('MAE', mae_sum), ('LPIPS', lpips_sum)]}

    print('-' * 62)
    print(f"{'AVERAGE':<30} {avg['PSNR']:7.2f} {avg['SSIM']:7.4f} "
          f"{avg['MAE']:7.4f} {avg['LPIPS']:7.4f}\n")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Inference mode  (no GT required)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, input_dir: str, save_dir: str, device,
          save_intermediate: bool = False):
    ds     = SingleImageDataset(input_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    os.makedirs(save_dir, exist_ok=True)

    for batch in loader:
        low  = batch['low'].to(device)
        name = Path(batch['path'][0]).name
        stem = Path(name).stem

        preds = model(low)

        # Save final enhanced image
        save_image(preds['enhanced'][0].cpu(),
                   os.path.join(save_dir, f'{stem}_enhanced.png'))

        if save_intermediate:
            inter_dir = os.path.join(save_dir, 'intermediate')
            os.makedirs(inter_dir, exist_ok=True)
            save_image(preds['gray'][0].repeat(3, 1, 1).cpu(),
                       os.path.join(inter_dir, f'{stem}_gray.png'))
            save_image(preds['trans_map'][0].repeat(3, 1, 1).cpu(),
                       os.path.join(inter_dir, f'{stem}_trans.png'))
            save_image(preds['coarse_rgb'][0].cpu(),
                       os.path.join(inter_dir, f'{stem}_coarse.png'))
            save_image(preds['denoised_rgb'][0].cpu(),
                       os.path.join(inter_dir, f'{stem}_denoised.png'))

        print(f"Saved: {stem}_enhanced.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('LIVENet Test / Inference')
    p.add_argument('--mode',      choices=['eval', 'infer'], default='eval')
    p.add_argument('--ckpt',      required=True, help='Path to checkpoint (.pth)')
    p.add_argument('--val_root',  default='data/eval15', help='[eval mode] dataset root')
    p.add_argument('--input_dir', default='input/',      help='[infer mode] folder of dark images')
    p.add_argument('--save_dir',  default='results/',    help='Folder to save output images')
    p.add_argument('--base_ch',   type=int, default=32)
    p.add_argument('--lsdb_rank', type=int, default=32)
    p.add_argument('--save_intermediate', action='store_true',
                   help='[infer mode] also save gray/trans/coarse intermediate maps')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.ckpt, device, args.base_ch, args.lsdb_rank)

    if args.mode == 'eval':
        evaluate(model, args.val_root, args.save_dir, device)
    else:
        infer(model, args.input_dir, args.save_dir, device,
              save_intermediate=args.save_intermediate)


if __name__ == '__main__':
    main()

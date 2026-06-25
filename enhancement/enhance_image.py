"""
enhance_image.py – Quickstart: enlighten a single dark image with LIVENet.

Usage:
    python enhance_image.py --input dark_photo.jpg \
                            --ckpt  checkpoints/best.pth \
                            --output enhanced.png

No dataset or training required. Just pass a checkpoint and an image.
"""

import argparse
import os

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from model import LIVENet


def enhance_image(input_path: str, ckpt_path: str, output_path: str,
                  base_ch: int = 32, lsdb_rank: int = 32,
                  tile_size: int = None):
    """
    Enhance a single dark image.

    Args:
        input_path  : path to input low-light image
        ckpt_path   : path to trained LIVENet checkpoint
        output_path : where to save the enhanced image
        base_ch     : model base channels (must match checkpoint)
        lsdb_rank   : LSDB rank (must match checkpoint)
        tile_size   : process in tiles if not None (for large images / low GPU VRAM)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = LIVENet(base_ch=base_ch, lsdb_rank=lsdb_rank).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from: {ckpt_path}")

    # Load image
    img_pil = Image.open(input_path).convert('RGB')
    w, h    = img_pil.size
    print(f"Input image: {input_path}  ({w}×{h})")

    to_tensor = T.ToTensor()
    img_t     = to_tensor(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Inference
    with torch.no_grad():
        if tile_size is None:
            preds   = model(img_t)
            enhanced = preds['enhanced']
        else:
            enhanced = _tile_inference(model, img_t, tile_size, device)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    vutils.save_image(enhanced.cpu(), output_path)
    print(f"Enhanced image saved to: {output_path}")
    return enhanced


def _tile_inference(model, img_t, tile_size: int, device, overlap: int = 64):
    """
    Tile-based inference for large images to avoid OOM.
    Overlapping tiles are blended with a linear ramp at edges.
    """
    B, C, H, W = img_t.shape
    output      = torch.zeros_like(img_t)
    weight_map  = torch.zeros(1, 1, H, W, device=device)

    stride = tile_size - overlap

    def ramp(n, device):
        r = torch.ones(n, device=device)
        fade = overlap // 2
        for i in range(fade):
            v = (i + 1) / (fade + 1)
            r[i] = v
            r[n - 1 - i] = v
        return r

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)
            tile   = img_t[:, :, y1:y2, x1:x2]
            preds  = model(tile)
            out_tile = preds['enhanced']

            # Compute blend weight
            ry = ramp(y2 - y1, device)
            rx = ramp(x2 - x1, device)
            w  = ry[:, None] * rx[None, :]   # (th, tw)
            w  = w.view(1, 1, y2 - y1, x2 - x1)

            output[:, :, y1:y2, x1:x2] += out_tile * w
            weight_map[:, :, y1:y2, x1:x2] += w

    output = output / weight_map.clamp(min=1e-6)
    return output.clamp(0, 1)


def parse_args():
    p = argparse.ArgumentParser('LIVENet – single image enhancement')
    p.add_argument('--input',     required=True,  help='Path to dark input image')
    p.add_argument('--ckpt',      required=True,  help='Path to checkpoint (.pth)')
    p.add_argument('--output',    default='enhanced.png', help='Output path')
    p.add_argument('--base_ch',   type=int, default=32)
    p.add_argument('--lsdb_rank', type=int, default=32)
    p.add_argument('--tile_size', type=int, default=None,
                   help='Use tiled inference (e.g. 512) for large images')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    enhance_image(
        input_path=args.input,
        ckpt_path=args.ckpt,
        output_path=args.output,
        base_ch=args.base_ch,
        lsdb_rank=args.lsdb_rank,
        tile_size=args.tile_size,
    )

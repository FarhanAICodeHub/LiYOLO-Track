"""
LIVENet Training Script

Usage:
    python train.py --train_root data/ours485 --val_root data/eval15 --epochs 200 --batch_size 8 --patch_size 256 --lr 2e-4
"""

import argparse
import logging
import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model   import LIVENet
from losses  import LIVENetLoss
from dataset import build_dataloaders
from metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('livenet')
    logger.setLevel(logging.INFO)
    fh  = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    ch  = logging.StreamHandler()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt  = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('best_psnr', 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch, logger, writer):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(loader):
        low     = batch['low'].to(device)
        normal  = batch['normal'].to(device)
        gray_gt = batch['gray_gt'].to(device)

        optimizer.zero_grad()

        # Fixed: use torch.amp.autocast instead of deprecated torch.cuda.amp.autocast
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds  = model(low)
            losses = criterion(preds, {'normal': normal, 'gray_gt': gray_gt})
            loss   = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (i + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch} | Iter {i+1}/{len(loader)} | "
                f"Loss={loss.item():.4f} "
                f"(gray={losses['l_gray'].item():.4f}, "
                f"refine={losses['l_refine'].item():.4f})"
            )

    avg_loss = total_loss / len(loader)
    elapsed  = time.time() - t0
    logger.info(f"Epoch {epoch} done | AvgLoss={avg_loss:.4f} | Time={elapsed:.1f}s")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, epoch, logger, writer):
    model.eval()
    psnr_sum = ssim_sum = mae_sum = lpips_sum = 0.0
    n = 0

    for batch in loader:
        low    = batch['low'].to(device)
        normal = batch['normal'].to(device)
        preds  = model(low)
        m = compute_metrics(preds['enhanced'], normal)
        psnr_sum  += m['psnr']
        ssim_sum  += m['ssim']
        mae_sum   += m['mae']
        lpips_sum += m['lpips']
        n += 1

    avg = {
        'psnr':  psnr_sum  / n,
        'ssim':  ssim_sum  / n,
        'mae':   mae_sum   / n,
        'lpips': lpips_sum / n,
    }

    logger.info(
        f"[Val] Epoch {epoch} | "
        f"PSNR={avg['psnr']:.2f} | SSIM={avg['ssim']:.4f} | "
        f"MAE={avg['mae']:.4f}  | LPIPS={avg['lpips']:.4f}"
    )
    for k, v in avg.items():
        writer.add_scalar(f'Val/{k.upper()}', v, epoch)

    return avg['psnr']


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('LIVENet Training')
    p.add_argument('--train_root',  default='data/ours485')
    p.add_argument('--val_root',    default='data/eval15')
    p.add_argument('--ckpt_dir',    default='checkpoints')
    p.add_argument('--log_dir',     default='logs')
    p.add_argument('--resume',      default=None)
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--patch_size',  type=int,   default=256)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--lr',          type=float, default=2e-4)
    p.add_argument('--base_ch',     type=int,   default=32)
    p.add_argument('--lsdb_rank',   type=int,   default=32)
    p.add_argument('--val_every',   type=int,   default=5)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # Data
    train_loader, val_loader = build_dataloaders(
        args.train_root, args.val_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Model
    model     = LIVENet(base_ch=args.base_ch, lsdb_rank=args.lsdb_rank).to(device)
    criterion = LIVENetLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Fixed: use torch.amp.GradScaler instead of deprecated torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    start_epoch = 1
    best_psnr   = 0.0

    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch-1}, best PSNR {best_psnr:.2f})")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params / 1e6:.2f} M")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer,
                        scaler, device, epoch, logger, writer)
        scheduler.step()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            psnr_val = validate(model, val_loader, device, epoch, logger, writer)

            if psnr_val > best_psnr:
                best_psnr = psnr_val
                save_checkpoint(
                    {'epoch': epoch, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr},
                    os.path.join(args.ckpt_dir, 'best.pth')
                )
                logger.info(f"  ★ New best PSNR: {best_psnr:.2f} → saved best.pth")

        if epoch % 20 == 0:
            save_checkpoint(
                {'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr},
                os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pth')
            )

    writer.close()
    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
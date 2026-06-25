"""
Dataset loaders for LIVENet.

Supports LoLv1 structure:
  data/
    ours485/
      low/   <- low-light training images
      high/  <- normal-light (ground truth) training images
    eval15/
      low/
      high/
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def make_grayscale_gt(normal_img: torch.Tensor) -> torch.Tensor:
    """Convert normal-light RGB tensor (C,H,W) -> grayscale (1,H,W)."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=normal_img.device)
    weights = weights.view(3, 1, 1)
    gray = (normal_img * weights).sum(0, keepdim=True)
    return gray


class LolDataset(Dataset):
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}

    def __init__(self, root: str, split: str = 'train',
                 patch_size: int = 256, augment: bool = True):
        super().__init__()
        self.split      = split
        self.patch_size = patch_size
        self.augment    = augment and (split == 'train')

        low_dir  = Path(root) / 'low'
        high_dir = Path(root) / 'high'

        low_files  = sorted([f for f in low_dir.iterdir()
                              if f.suffix.lower() in self.IMG_EXTS])
        high_files = sorted([f for f in high_dir.iterdir()
                              if f.suffix.lower() in self.IMG_EXTS])

        assert len(low_files) == len(high_files), (
            f"Mismatch: {len(low_files)} low vs {len(high_files)} high images.")

        self.pairs     = list(zip(low_files, high_files))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low  = Image.open(low_path).convert('RGB')
        high = Image.open(high_path).convert('RGB')

        # Paired random crop
        if self.patch_size is not None:
            w, h = low.size
            ps   = self.patch_size
            if w >= ps and h >= ps:
                x = random.randint(0, w - ps)
                y = random.randint(0, h - ps)
                low  = low.crop((x, y, x + ps, y + ps))
                high = high.crop((x, y, x + ps, y + ps))

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                low  = low.transpose(Image.FLIP_LEFT_RIGHT)
                high = high.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                low  = low.transpose(Image.FLIP_TOP_BOTTOM)
                high = high.transpose(Image.FLIP_TOP_BOTTOM)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                low  = low.rotate(angle)
                high = high.rotate(angle)

        low_t  = self.to_tensor(low)
        high_t = self.to_tensor(high)
        gray_t = make_grayscale_gt(high_t)

        return {
            'low':       low_t,
            'normal':    high_t,
            'gray_gt':   gray_t,
            'low_path':  str(low_path),
            'high_path': str(high_path),
        }


def build_dataloaders(train_root: str, val_root: str,
                      patch_size: int = 256,
                      batch_size: int = 8,
                      num_workers: int = 4):

    # pin_memory only when CUDA is available
    pin = torch.cuda.is_available()

    train_ds = LolDataset(train_root, split='train',
                          patch_size=patch_size, augment=True)
    val_ds   = LolDataset(val_root,   split='val',
                          patch_size=None,      augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,          # only True when GPU is available
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


class SingleImageDataset(Dataset):
    """Inference-only dataset — loads all images from a folder."""
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}

    def __init__(self, folder: str):
        self.files     = sorted([f for f in Path(folder).iterdir()
                                  if f.suffix.lower() in self.IMG_EXTS])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img  = Image.open(path).convert('RGB')
        return {
            'low':  self.to_tensor(img),
            'path': str(path),
        }
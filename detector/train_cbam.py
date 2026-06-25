"""
train_cbam.py
Training script for YOLOv11n-CBAM modified architecture.

Usage:
    python train_cbam.py

Make sure cbam.py is in the same directory, and your
dataset YAML path is updated below.
"""

import sys
import os
from pathlib import Path

# ?? Register CBAM with Ultralytics ??????????????????????????
# This must happen BEFORE importing YOLO so Ultralytics
# knows about the CBAM module when parsing the YAML.
from ultralytics.nn.modules import conv
from cbam import CBAM, ChannelAttention, SpatialAttention

# Patch CBAM into Ultralytics module registry
import ultralytics.nn.modules as ult_modules
ult_modules.CBAM            = CBAM
ult_modules.ChannelAttention = ChannelAttention
ult_modules.SpatialAttention = SpatialAttention

# Also patch the tasks parser
import ultralytics.nn.tasks as tasks
tasks.CBAM            = CBAM
tasks.ChannelAttention = ChannelAttention
tasks.SpatialAttention = SpatialAttention

from ultralytics import YOLO

# ?? Config ??????????????????????????????????????????????????
YAML_MODEL  = "yolo11n-cbam.yaml"          # architecture file
DATA_YAML   = "dataset.yaml"                  # ??UPDATE to your dataset path
EPOCHS      = 100
IMGSZ       = 640
BATCH       = 8                           # reduce to 8 if VRAM tight on edge
WORKERS     = 8
DEVICE      = 0                            # GPU index, or "cpu"
PROJECT     = "runs/detect/results"
NAME        = "cbam_modified"
PRETRAINED  = True                         # use YOLO11n pretrained backbone weights


def main():
    print("=" * 60)
    print("  YOLOv11n-CBAM  ?? Small Drone Detection")
    print("=" * 60)

    # Build model from YAML (with pretrained backbone)
    if PRETRAINED:
        # Load pretrained YOLO11n then transfer weights to our modified arch
        model = YOLO(YAML_MODEL)
        print("\n[INFO] Training from YAML with random init.")
        print("[TIP]  For better results, manually transfer backbone weights")
        print("       from yolo11n.pt after first verifying arch compatibility.\n")
    else:
        model = YOLO(YAML_MODEL)

    # Print model summary
    model.info(verbose=True)

    # ?? Training ????????????????????????????????????????????
    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = BATCH,
        workers    = WORKERS,
        device     = DEVICE,
        project    = PROJECT,
        name       = NAME,

        # Optimizer
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,

        # Augmentation ??important for drone detection robustness
        hsv_h      = 0.015,   # hue variation (helps night/low-light)
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 10.0,    # rotation (drones fly at angles)
        translate  = 0.1,
        scale      = 0.9,     # aggressive scale aug for small drones
        shear      = 2.0,
        perspective = 0.0,
        flipud     = 0.1,
        fliplr     = 0.5,
        mosaic     = 1.0,     # mosaic helps small object detection
        mixup      = 0.1,
        copy_paste = 0.1,     # copy-paste aug ??great for small drones

        # Multi-scale training
        multi_scale = False,   # vary input size during training

        # Save
        save       = True,
        save_period = 10,
        val        = True,
        plots      = True,
        verbose    = True,
    )

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Results saved to: {PROJECT}/{NAME}")
    print(f"  Best weights: {PROJECT}/{NAME}/weights/best.pt")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()

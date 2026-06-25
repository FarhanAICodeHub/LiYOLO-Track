# LiYOLO-Track

LiYOLO-Track is a day-and-night UAV detection and tracking framework that combines low-light image enhancement, a modified YOLOv11n detector, and BoT-SORT tracking.

## Main Components

* **LivNet enhancement:** improves visibility in low-light and nighttime UAV images.
* **LiYOLO detector:** modified YOLOv11n with CBAM attention and a P2 detection head for small UAV detection.
* **BoT-SORT tracking:** maintains UAV identities across video frames.
* **Day/night processing:** supports daytime frames and enhanced nighttime frames.

## Repository Structure

```text
LiYOLO-Track/
戍式式 assets/
戍式式 configs/
戍式式 detector/
戍式式 enhancement/
戍式式 scripts/
戍式式 tracking/
戍式式 weights/
戍式式 .gitignore
戍式式 README.md
戌式式 requirements.txt
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/LiYOLO-Track.git
cd LiYOLO-Track
pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with your GitHub username.

## Required Weights

Place the following trained model files inside the `weights/` directory:

```text
weights/
戍式式 livenet_best.pth
戍式式 liyolo_best.pt
戌式式 yolo11n_baseline.pt
```

The trained weights are not included in the repository.

## Dataset Structure

```text
data/
戍式式 train/
弛   戍式式 img/
弛   戌式式 labels/
戍式式 val/
弛   戍式式 img/
弛   戌式式 labels/
戌式式 test/
    戍式式 img/
    戌式式 labels/
```

The dataset path is configured in `configs/dataset.yaml`.

## Detector Training

```bash
python detector/train_cbam.py
```

For ablation experiments:

```bash
python detector/train_ablation.py
```

## LivNet Training

```bash
cd enhancement
python train.py --train_root data/ours485 --val_root data/eval15
```

## Low-Light Enhancement

```bash
cd enhancement
python enhance_image.py --input INPUT_IMAGE --ckpt ../weights/livenet_best.pth
```

## Nighttime UAV Detection

```bash
python scripts/night_detect.py --help
```

## UAV Tracking

```bash
python tracking/track_perframe.py
```

## Detector Configurations

The repository includes:

* YOLOv11n with CBAM and P2 detection head
* YOLOv11n with CBAM only
* YOLOv11n with P2 detection head only

## Citation

Citation information will be added after publication.

## License

A license has not yet been selected.

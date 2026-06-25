import sys
sys.path.insert(0, '.')
from cbam import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks as tasks
ult_modules.CBAM = CBAM
ult_modules.ChannelAttention = ChannelAttention
ult_modules.SpatialAttention = SpatialAttention
tasks.CBAM = CBAM
tasks.ChannelAttention = ChannelAttention
tasks.SpatialAttention = SpatialAttention
from ultralytics import YOLO

def train_variant(yaml_model, name):
    print('='*60)
    print('  Training: ' + name)
    print('='*60)
    model = YOLO(yaml_model)
    model.train(
        data='dataset.yaml', epochs=100, imgsz=640,
        batch=8, device=0, project='runs/detect/results', name=name,
        optimizer='AdamW', lr0=0.001, lrf=0.01,
        momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, mosaic=1.0, mixup=0.1,
        copy_paste=0.1, multi_scale=False,
        save=True, val=True, plots=True, verbose=True,
    )
    print('Done: ' + name)

if __name__ == '__main__':
    train_variant('yolo11n-cbam-only.yaml', 'ablation_cbam_only')
    train_variant('yolo11n-p2-only.yaml',   'ablation_p2_only')

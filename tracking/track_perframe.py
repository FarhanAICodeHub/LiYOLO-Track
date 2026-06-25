import argparse, os, time, cv2, numpy as np, csv
from pathlib import Path
from ultralytics import YOLO

BASELINE_PT = "runs/detect/results/baseline5/weights/best.pt"
CBAM_PT     = "runs/detect/runs/detect/results/cbam_modified2/weights/best.pt"
TRACKER     = "botsort.yaml"
IMGSZ       = 640
CONF        = 0.25
IOU         = 0.5
DEVICE      = 0
OUTPUT_DIR  = "runs/tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_tracking(model, source, name):
    records = []
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {name} - {total} frames")
    prev_ids = set()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        results = model.track(frame, tracker=TRACKER, imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, persist=True, verbose=False)
        inf_ms = (time.time() - t0) * 1000
        boxes = results[0].boxes
        n_dets = len(boxes) if boxes is not None else 0
        avg_conf = float(boxes.conf.mean().cpu()) if boxes is not None and len(boxes) > 0 else 0.0
        curr_ids = set()
        if boxes is not None and boxes.id is not None:
            curr_ids = set(boxes.id.cpu().numpy().astype(int).tolist())
        id_switch = min(len(curr_ids - prev_ids), len(prev_ids - curr_ids)) if frame_idx > 0 else 0
        records.append({'frame': frame_idx+1, 'detections': n_dets, 'avg_conf': round(avg_conf,4), 'track_ids': len(curr_ids), 'id_switch': id_switch, 'inf_ms': round(inf_ms,2)})
        prev_ids = curr_ids
        frame_idx += 1
        if frame_idx % 30 == 0: print(f"  {frame_idx}/{total} frames...")
    cap.release()
    csv_path = os.path.join(OUTPUT_DIR, f"perframe_{name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {csv_path}")
    return records

def run(source):
    print("="*60)
    print("  Per-frame tracking data collection")
    print("="*60)
    model_b = YOLO(BASELINE_PT)
    model_c = YOLO(CBAM_PT)
    print("\n[1/2] Running Baseline...")
    run_tracking(model_b, source, "Baseline")
    print("\n[2/2] Running CBAM...")
    run_tracking(model_c, source, "CBAM")
    print("\nDone! CSV files saved in runs/tracking/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()
    run(args.source)

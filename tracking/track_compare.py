import argparse, os, time, cv2, numpy as np
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

class TrackingMetrics:
    def __init__(self, name):
        self.name=name; self.total_dets=0; self.total_tracks=0
        self.id_switches=0; self.frames=0; self.track_ids=set()
        self.prev_ids=set(); self.inference_times=[]; self.track_lengths={}
    def update(self, results, inf_time):
        self.frames+=1; self.inference_times.append(inf_time)
        if results[0].boxes is None:
            self.prev_ids=set(); return
        boxes=results[0].boxes
        self.total_dets+=len(boxes)
        if boxes.id is not None:
            curr_ids=set(boxes.id.cpu().numpy().astype(int).tolist())
            self.track_ids.update(curr_ids)
            self.total_tracks=len(self.track_ids)
            if self.frames>1:
                self.id_switches+=min(len(curr_ids-self.prev_ids),len(self.prev_ids-curr_ids))
            for tid in curr_ids:
                self.track_lengths[tid]=self.track_lengths.get(tid,0)+1
            self.prev_ids=curr_ids
    def summary(self):
        avg_inf=np.mean(self.inference_times)*1000 if self.inference_times else 0
        avg_len=np.mean(list(self.track_lengths.values())) if self.track_lengths else 0
        return {'Model':self.name,'Frames':self.frames,'Total Detections':self.total_dets,
                'Unique Track IDs':self.total_tracks,'ID Switches':self.id_switches,
                'Avg Track Length':round(avg_len,1),'Avg Inference ms':round(avg_inf,2),
                'FPS':round(1000/avg_inf if avg_inf>0 else 0,1)}

def draw_tracks(frame, results, color, label):
    if results[0].boxes is not None:
        boxes=results[0].boxes
        for i,box in enumerate(boxes.xyxy.cpu().numpy()):
            x1,y1,x2,y2=map(int,box)
            conf=float(boxes.conf[i].cpu())
            tid=int(boxes.id[i].cpu()) if boxes.id is not None else -1
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            txt=f"ID:{tid} {conf:.2f}" if tid>=0 else f"{conf:.2f}"
            cv2.putText(frame,txt,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
    cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    return frame

def run(source):
    print("="*60)
    print("  BoT-SORT Tracking: Baseline vs CBAM")
    print("="*60)
    model_b=YOLO(BASELINE_PT); model_c=YOLO(CBAM_PT)
    metrics_b=TrackingMetrics("YOLOv11n Baseline")
    metrics_c=TrackingMetrics("CBAM Modified")
    source=Path(source)

    # ── handle video or image folder
    is_video = source.is_file() and source.suffix.lower() in ['.mp4','.avi','.mov','.mkv']
    if is_video:
        cap=cv2.VideoCapture(str(source))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video: {total} frames @ {fps_in:.1f}fps | {w}x{h}")
        out_path=os.path.join(OUTPUT_DIR,"tracking_comparison.mp4")
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter(out_path,fourcc,fps_in,(w*2,h))
    else:
        exts=['.jpg','.jpeg','.png','.bmp']
        frames_paths=sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
        print(f"[INFO] Found {len(frames_paths)} images in {source}")
        total=len(frames_paths)

    frame_idx=0
    while True:
        if is_video:
            ret,frame=cap.read()
            if not ret: break
        else:
            if frame_idx>=total: break
            frame=cv2.imread(str(frames_paths[frame_idx]))
            if frame is None: frame_idx+=1; continue

        t0=time.time()
        res_b=model_b.track(frame,tracker=TRACKER,imgsz=IMGSZ,conf=CONF,iou=IOU,device=DEVICE,persist=True,verbose=False)
        metrics_b.update(res_b,time.time()-t0)

        t0=time.time()
        res_c=model_c.track(frame,tracker=TRACKER,imgsz=IMGSZ,conf=CONF,iou=IOU,device=DEVICE,persist=True,verbose=False)
        metrics_c.update(res_c,time.time()-t0)

        if is_video:
            vis_b=draw_tracks(frame.copy(),res_b,(0,140,255),"YOLOv11n Baseline + BoT-SORT")
            vis_c=draw_tracks(frame.copy(),res_c,(0,200,100),"CBAM Modified + BoT-SORT")
            out.write(np.hstack([vis_b,vis_c]))

        frame_idx+=1
        if frame_idx%50==0:
            print(f"  Processed {frame_idx}/{total} frames...")

    if is_video:
        cap.release(); out.release()
        print(f"\n[INFO] Output video saved: {out_path}")

    print("\n"+"="*60)
    print("  TRACKING RESULTS SUMMARY")
    print("="*60)
    summary_path=os.path.join(OUTPUT_DIR,"tracking_summary.txt")
    with open(summary_path,'w') as f:
        for metrics in [metrics_b,metrics_c]:
            s=metrics.summary()
            print(f"\n  {s['Model']}")
            f.write(f"\n{s['Model']}\n")
            for k,v in s.items():
                print(f"    {k}: {v}")
                f.write(f"  {k}: {v}\n")
    print(f"\n  Summary saved: {summary_path}")
    print("="*60)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=str,required=True)
    args=parser.parse_args()
    run(args.source)

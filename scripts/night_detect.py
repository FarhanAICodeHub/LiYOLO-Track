import argparse, os, sys, time, cv2, numpy as np, torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

LIVENET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "enhancement"))
LIVENET_CKPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "livenet_best.pth"))
CBAM_PT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "liyolo_best.pt"))
BASELINE_PT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11n_baseline.pt"))
OUTPUT_DIR   = "runs/night_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, LIVENET_DIR)
CONF=0.10; IOU=0.45; IMGSZ=640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_livenet():
    print(f"[INFO] Loading LIVENet from: {LIVENET_CKPT}")
    from model import LIVENet
    model = LIVENet().to(DEVICE)
    ckpt = torch.load(LIVENET_CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[INFO] LIVENet loaded!")
    return model

def enhance_frame(livenet, frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = TF.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = livenet(tensor)
    if isinstance(out, dict):
        out = out['enhanced']
    elif isinstance(out, (list, tuple)):
        out = out[-1]
    out_np = out.squeeze(0).cpu().clamp(0,1).permute(1,2,0).numpy()
    out_bgr = cv2.cvtColor((out_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

def draw_detections(frame, results, color):
    if results[0].boxes is None:
        return frame, 0
    boxes = results[0].boxes
    n = 0
    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
        x1,y1,x2,y2 = map(int,box)
        conf = float(boxes.conf[i].cpu())
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label = f"Drone {conf:.2f}"
        cv2.putText(frame,label,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        n+=1
    return frame, n

def add_label(frame, text, color):
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),3)
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

def run(source, model_type="cbam"):
    print("="*60)
    print("  Night Vision Antidrone Pipeline")
    print("  LIVENet Enhancement + YOLO Detection")
    print("="*60)
    livenet = load_livenet()
    from ultralytics import YOLO
    pt = CBAM_PT if model_type=="cbam" else BASELINE_PT
    print(f"[INFO] Loading YOLO from: {pt}")
    yolo = YOLO(pt)
    print("[INFO] YOLO loaded!")
    source = Path(source)
    is_image = source.is_file() and source.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']
    is_video = source.is_file() and source.suffix.lower() in ['.mp4','.avi','.mov','.mkv']
    is_folder = source.is_dir()
    stats = {'frames':0,'enhanced_dets':0,'raw_dets':0,'enhance_ms':[],'detect_ms':[]}
    if is_video:
        cap=cv2.VideoCapture(str(source))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video: {total} frames @ {fps_in:.1f}fps")
        out_path=os.path.join(OUTPUT_DIR,"night_detection.mp4")
        out=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'mp4v'),fps_in,(w*3,h))
    if is_folder:
        exts=['.jpg','.jpeg','.png','.bmp']
        frame_paths=sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
        total=len(frame_paths)
        print(f"[INFO] Found {total} images in {source}")
    frame_idx=0
    while True:
        if is_video:
            ret,frame=cap.read()
            if not ret: break
        elif is_image:
            frame=cv2.imread(str(source))
            if frame is None: break
        elif is_folder:
            if frame_idx>=total: break
            frame=cv2.imread(str(frame_paths[frame_idx]))
            if frame is None: frame_idx+=1; continue
        else:
            break
        stats['frames']+=1
        t0=time.time()
        enhanced=enhance_frame(livenet,frame)
        stats['enhance_ms'].append((time.time()-t0)*1000)
        t0=time.time()
        res_raw=yolo(frame,imgsz=IMGSZ,conf=CONF,iou=IOU,device=DEVICE,verbose=False)
        stats['detect_ms'].append((time.time()-t0)*1000)
        res_enh=yolo(enhanced,imgsz=IMGSZ,conf=CONF,iou=IOU,device=DEVICE,verbose=False)
        vis_raw=frame.copy()
        vis_raw,n_raw=draw_detections(vis_raw,res_raw,(0,100,255))
        stats['raw_dets']+=n_raw
        vis_enh=enhanced.copy()
        vis_enh,n_enh=draw_detections(vis_enh,res_enh,(0,220,100))
        stats['enhanced_dets']+=n_enh
        add_label(vis_raw,f"Dark Input | Dets:{n_raw}",(0,100,255))
        add_label(enhanced,"LIVENet Enhanced",(200,200,200))
        add_label(vis_enh,f"Enhanced+Detect | Dets:{n_enh}",(0,220,100))
        if is_image:
            out_img=np.hstack([vis_raw,enhanced,vis_enh])
            out_path=os.path.join(OUTPUT_DIR,"night_result.jpg")
            cv2.imwrite(out_path,out_img)
            print(f"[INFO] Result saved: {out_path}")
            break
        if is_video:
            combined=np.hstack([vis_raw,enhanced,vis_enh])
            out.write(combined)
        if is_folder:
            out_img=np.hstack([vis_raw,enhanced,vis_enh])
            cv2.imwrite(os.path.join(OUTPUT_DIR,f"result_{frame_idx:04d}.jpg"),out_img)
        frame_idx+=1
        if frame_idx%5==0: print(f"  Processed {frame_idx}/{total}...")
    if is_video:
        cap.release(); out.release()
        print(f"\n[INFO] Video saved: {out_path}")
    print("\n"+"="*60)
    print("  NIGHT VISION DETECTION SUMMARY")
    print("="*60)
    print(f"  Frames processed      : {stats['frames']}")
    print(f"  Detections (dark)     : {stats['raw_dets']}")
    print(f"  Detections (enhanced) : {stats['enhanced_dets']}")
    det_gain=stats['enhanced_dets']-stats['raw_dets']
    print(f"  Detection gain        : +{det_gain}")
    print(f"  Avg enhance time      : {np.mean(stats['enhance_ms']):.1f} ms")
    print(f"  Avg detect time       : {np.mean(stats['detect_ms']):.1f} ms")
    total_ms=np.mean(stats['enhance_ms'])+np.mean(stats['detect_ms'])
    print(f"  Pipeline FPS          : {1000/total_ms:.1f}")
    print("="*60)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=str,required=True)
    parser.add_argument("--model",type=str,default="cbam",choices=["cbam","baseline"])
    args=parser.parse_args()
    run(args.source,args.model)



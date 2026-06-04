import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T
import json, time, io, sys
from pathlib import Path

BASE      = "/home/asistencia/tesis"
DATA_DIR  = f"{BASE}/dataset/coco"
SAVE_DIR  = f"{BASE}/modelos/v2/fasterrcnn"
SAVE_JSON = f"{BASE}/modelos/v2/fasterrcnn_full.json"
DEVICE    = torch.device("cuda:1")
ALL_SEEDS = [42, 123, 456, 789, 1000, 1234, 2024, 2025, 3141, 9999]
CLASSES   = ["libre", "no_disponible", "ocupado"]
NUM_CLASS = 4

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Auto-resume
try:
    with open(SAVE_JSON) as f:
        existing = json.load(f)
    runs = existing.get("runs", [])
    seeds_done = [r["seed"] for r in runs]
    SEEDS = [s for s in ALL_SEEDS if s not in seeds_done]
    print(f"Continuando desde {len(runs)} runs existentes...")
    print(f"Seeds faltantes: {SEEDS}")
except:
    runs  = []
    SEEDS = ALL_SEEDS
    print("Iniciando desde cero...")

def collate_fn(batch):
    return tuple(zip(*batch))

class CocoParking(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = T.ToTensor()(img)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x+w, y+h])
                labels.append(ann["category_id"])
        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0,4)),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }
        return img, target

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASS)
    return model.to(DEVICE)

def iou_bbox(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[0]+b1[2],b2[0]+b2[2]); y2=min(b1[1]+b1[3],b2[1]+b2[3])
    if x2<x1 or y2<y1: return 0
    inter=(x2-x1)*(y2-y1)
    return inter/(b1[2]*b1[3]+b2[2]*b2[3]-inter+1e-8)

def evaluar(model, split="test"):
    ds = CocoDetection(
        root=f"{DATA_DIR}/{split}",
        annFile=f"{DATA_DIR}/{split}/_annotations.coco.json",
        transforms=None
    )
    coco_gt = COCO(f"{DATA_DIR}/{split}/_annotations.coco.json")
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            img, _ = ds[i]
            img_id = ds.ids[i]
            img_t  = T.ToTensor()(img).to(DEVICE)
            preds  = model([img_t])[0]
            for box, score, label in zip(
                preds["boxes"].cpu(), preds["scores"].cpu(), preds["labels"].cpu()
            ):
                if score > 0.05:
                    x1,y1,x2,y2 = box.tolist()
                    results.append({
                        "image_id":    img_id,
                        "category_id": int(label),
                        "bbox":  [x1,y1,x2-x1,y2-y1],
                        "score": float(score)
                    })
    if not results:
        return 0,0,0,0,{}

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate()
    old=sys.stdout; sys.stdout=io.StringIO(); ev.summarize(); sys.stdout=old

    per_class = {}
    for cat_id, cls_name in [(1,"libre"),(2,"no_disponible"),(3,"ocupado")]:
        ev2 = COCOeval(coco_gt, coco_dt, "bbox")
        ev2.params.catIds = [cat_id]
        ev2.evaluate(); ev2.accumulate()
        old=sys.stdout; sys.stdout=io.StringIO(); ev2.summarize(); sys.stdout=old
        per_class[cls_name] = {
            "mAP50":    round(float(ev2.stats[1]), 4),
            "mAP50_95": round(float(ev2.stats[0]), 4),
        }

    tp,fp,fn = 0,0,0
    pred_by = {}
    for r in results: pred_by.setdefault(r["image_id"],[]).append(r)
    gt_by = {
        img_id: coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        for img_id in coco_gt.getImgIds()
    }
    for img_id,gts in gt_by.items():
        preds_img = pred_by.get(img_id,[])
        matched = set()
        for pred in preds_img:
            best,best_idx = 0,-1
            for gi,gt in enumerate(gts):
                if gi in matched: continue
                v = iou_bbox(pred["bbox"],gt["bbox"])
                if v>best: best,best_idx=v,gi
            if best>0.5: tp+=1; matched.add(best_idx)
            else: fp+=1
        fn+=len(gts)-len(matched)
    prec = tp/(tp+fp+1e-8)
    rec  = tp/(tp+fn+1e-8)

    for cat_id, cls_name in [(1,"libre"),(2,"no_disponible"),(3,"ocupado")]:
        tp2,fp2,fn2=0,0,0
        for img_id,gts in gt_by.items():
            gts_cls   = [g for g in gts if g["category_id"]==cat_id]
            preds_cls = [p for p in pred_by.get(img_id,[]) if p["category_id"]==cat_id]
            matched=set()
            for pred in preds_cls:
                best,best_idx=0,-1
                for gi,gt in enumerate(gts_cls):
                    if gi in matched: continue
                    v=iou_bbox(pred["bbox"],gt["bbox"])
                    if v>best: best,best_idx=v,gi
                if best>0.5: tp2+=1; matched.add(best_idx)
                else: fp2+=1
            fn2+=len(gts_cls)-len(matched)
        per_class[cls_name]["precision"] = round(tp2/(tp2+fp2+1e-8), 4)
        per_class[cls_name]["recall"]    = round(tp2/(tp2+fn2+1e-8), 4)

    return float(ev.stats[1]),float(ev.stats[0]),float(prec),float(rec),per_class

for i, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"Faster R-CNN v2 — Run {len(runs)+1}/10 (seed={seed})")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CocoParking(
        root=f"{DATA_DIR}/train",
        annFile=f"{DATA_DIR}/train/_annotations.coco.json"
    )
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )

    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_loss, patience_counter, best_epoch = float("inf"), 0, 0

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs    = [im.to(DEVICE) for im in imgs]
            targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"  Epoch {epoch}/100 loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_seed{seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"  Early stopping en epoch {epoch}")
                break

    print(f"  Evaluando best model (epoch {best_epoch})...")
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_seed{seed}.pt"))
    m50, m5095, prec, rec, per_class = evaluar(model, "test")
    f1 = 2*prec*rec/(prec+rec+1e-8)

    dummy = torch.rand(1,3,640,640).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model([dummy[0]])
    lats = []
    with torch.no_grad():
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model([dummy[0]])
            torch.cuda.synchronize()
            lats.append((time.perf_counter()-t0)*1000)

    lat_mean = float(np.mean(lats))
    lat_p50  = float(np.percentile(lats, 50))
    lat_p95  = float(np.percentile(lats, 95))
    fps      = 1000/lat_mean
    gpu_mb   = torch.cuda.max_memory_allocated()/1024**2
    torch.cuda.reset_peak_memory_stats()

    Path(f"{SAVE_DIR}/best_seed{seed}.pt").unlink(missing_ok=True)

    run_data = {
        "seed":       seed,
        "best_epoch": best_epoch,
        "final": {
            "mAP50":     round(m50,     4),
            "mAP50_95":  round(m5095,   4),
            "precision": round(prec,    4),
            "recall":    round(rec,     4),
            "f1":        round(f1,      4),
            "fps":       round(fps,     1),
            "lat_mean":  round(lat_mean,2),
            "lat_p50":   round(lat_p50, 2),
            "lat_p95":   round(lat_p95, 2),
            "gpu_mb":    round(gpu_mb,  0),
            "per_class": per_class,
        }
    }
    runs.append(run_data)

    print(f"\n✅ Run {len(runs)}/10 completado:")
    print(f"   mAP50={m50:.4f} mAP50-95={m5095:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print(f"   FPS={fps:.1f} lat_p50={lat_p50:.2f}ms lat_p95={lat_p95:.2f}ms GPU={gpu_mb:.0f}MB")
    for cls,v in per_class.items():
        print(f"   {cls}: mAP50={v['mAP50']:.4f} mAP50-95={v['mAP50_95']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")

    with open(SAVE_JSON, "w") as f:
        json.dump({"model": "Faster R-CNN", "runs": runs}, f, indent=2)
    print(f"   💾 Guardado: {SAVE_JSON}")

keys = ["mAP50","mAP50_95","precision","recall","f1","fps","lat_p50","lat_p95","gpu_mb"]
summary = {}
print(f"\n{'='*60}\nSUMMARY Faster R-CNN v2 — 10 runs\n{'='*60}")
for k in keys:
    vals = [r["final"][k] for r in runs]
    summary[k] = {
        "mean":   round(float(np.mean(vals)),        4),
        "std":    round(float(np.std(vals, ddof=1)), 4),
        "values": vals,
    }
    print(f"  {k:12}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

cls_summary = {}
for cls in CLASSES:
    cls_summary[cls] = {}
    for k in ["mAP50","mAP50_95","precision","recall"]:
        vals = [r["final"]["per_class"][cls][k] for r in runs]
        cls_summary[cls][k] = {
            "mean": round(float(np.mean(vals)),        4),
            "std":  round(float(np.std(vals, ddof=1)), 4),
        }
    print(f"\n  {cls}:")
    for k,v in cls_summary[cls].items():
        print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

with open(SAVE_JSON, "w") as f:
    json.dump({
        "model": "Faster R-CNN", "dataset": "Smart Parking UPeU v4",
        "runs": runs, "summary": summary, "cls_summary": cls_summary,
    }, f, indent=2)
print(f"\n✅ JSON final: {SAVE_JSON}")

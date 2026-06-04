import numpy as np
import torch
import json
import time
from pathlib import Path
from ultralytics import YOLO

BASE      = "/home/asistencia/tesis"
DATA_YAML = f"{BASE}/dataset/ultra/data.yaml"
SAVE_DIR  = f"{BASE}/modelos/v2/rtdetr"
SAVE_JSON = f"{BASE}/modelos/v2/rtdetr_full.json"
DEVICE    = "cuda:0"
SEEDS     = [789, 1000, 1234, 2024, 2025, 3141, 9999]
CLASSES   = ["libre", "no_disponible", "ocupado"]

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

with open(SAVE_JSON) as f:
    existing = json.load(f)
runs = existing.get("runs", [])
print(f"Continuando desde {len(runs)} runs existentes...")

for i, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"RT-DETR-L v2 — Run {len(runs)+1}/10 (seed={seed})")
    print(f"{'='*60}")

    model = YOLO("rtdetr-l.pt")
    model.train(
        data=DATA_YAML, epochs=100, imgsz=640, batch=8,
        device=DEVICE, seed=seed, deterministic=True,
        patience=20, project=SAVE_DIR, name=f"run_{seed}",
        verbose=False, plots=False, save=True, exist_ok=True,
    )

    best_pt = f"{SAVE_DIR}/run_{seed}/weights/best.pt"
    model   = YOLO(best_pt)
    metrics = model.val(
        data=DATA_YAML, device=DEVICE,
        split="test", verbose=False, plots=False,
    )

    map50   = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    prec    = float(metrics.box.mp)
    rec     = float(metrics.box.mr)
    f1      = 2*prec*rec/(prec+rec+1e-8)

    per_class = {}
    for ci, cls_name in enumerate(CLASSES):
        per_class[cls_name] = {
            "mAP50":    round(float(metrics.box.ap50[ci]), 4),
            "mAP50_95": round(float(metrics.box.ap[ci]),   4),
            "precision":round(float(metrics.box.p[ci]),    4),
            "recall":   round(float(metrics.box.r[ci]),    4),
        }

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False, device=DEVICE)
    lats = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy, verbose=False, device=DEVICE)
        torch.cuda.synchronize()
        lats.append((time.perf_counter()-t0)*1000)

    lat_mean = float(np.mean(lats))
    lat_p50  = float(np.percentile(lats, 50))
    lat_p95  = float(np.percentile(lats, 95))
    fps      = 1000/lat_mean
    gpu_mb   = torch.cuda.max_memory_allocated()/1024**2
    torch.cuda.reset_peak_memory_stats()

    run_data = {
        "seed": seed,
        "final": {
            "mAP50":     round(map50,    4),
            "mAP50_95":  round(map5095,  4),
            "precision": round(prec,     4),
            "recall":    round(rec,      4),
            "f1":        round(f1,       4),
            "fps":       round(fps,      1),
            "lat_mean":  round(lat_mean, 2),
            "lat_p50":   round(lat_p50,  2),
            "lat_p95":   round(lat_p95,  2),
            "gpu_mb":    round(gpu_mb,   0),
            "per_class": per_class,
        }
    }
    runs.append(run_data)

    print(f"\n✅ Run {len(runs)}/10 completado:")
    print(f"   mAP50={map50:.4f} mAP50-95={map5095:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print(f"   FPS={fps:.1f} lat_p50={lat_p50:.2f}ms lat_p95={lat_p95:.2f}ms GPU={gpu_mb:.0f}MB")
    for cls, v in per_class.items():
        print(f"   {cls}: mAP50={v['mAP50']:.4f} mAP50-95={v['mAP50_95']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")

    with open(SAVE_JSON, "w") as f:
        json.dump({"model": "RT-DETR-L", "runs": runs}, f, indent=2)
    print(f"   💾 Guardado: {SAVE_JSON}")

keys = ["mAP50","mAP50_95","precision","recall","f1","fps","lat_p50","lat_p95","gpu_mb"]
summary = {}
print(f"\n{'='*60}\nSUMMARY RT-DETR-L v2 — 10 runs\n{'='*60}")
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
    for k, v in cls_summary[cls].items():
        print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

with open(SAVE_JSON, "w") as f:
    json.dump({
        "model": "RT-DETR-L", "dataset": "Smart Parking UPeU v4",
        "runs": runs, "summary": summary, "cls_summary": cls_summary,
    }, f, indent=2)
print(f"\n✅ JSON final: {SAVE_JSON}")

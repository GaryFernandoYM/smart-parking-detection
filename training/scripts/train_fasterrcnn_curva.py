import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T
import json, io, sys
from pathlib import Path

BASE     = "/home/asistencia/tesis"
DATA_DIR = f"{BASE}/dataset/coco"
SAVE_PATH= f"{BASE}/modelos/v2/fasterrcnn_map_curve_789.json"
DEVICE   = torch.device("cuda:1")
SEED     = 789
NUM_CLASS= 4

def collate_fn(batch): return tuple(zip(*batch))

class CocoParking(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = T.ToTensor()(img)
        boxes, labels = [], []
        for ann in anns:
            x,y,w,h = ann["bbox"]
            if w>0 and h>0:
                boxes.append([x,y,x+w,y+h])
                labels.append(ann["category_id"])
        return img, {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0,4)),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }

def get_map50(model, split="valid"):
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
            img_t  = T.ToTensor()(img).to(DEVICE)
            preds  = model([img_t])[0]
            for box, score, label in zip(
                preds["boxes"].cpu(), preds["scores"].cpu(), preds["labels"].cpu()
            ):
                if score > 0.05:
                    x1,y1,x2,y2 = box.tolist()
                    results.append({
                        "image_id": ds.ids[i],
                        "category_id": int(label),
                        "bbox": [x1,y1,x2-x1,y2-y1],
                        "score": float(score)
                    })
    if not results: return 0.0
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate()
    old=sys.stdout; sys.stdout=io.StringIO(); ev.summarize(); sys.stdout=old
    return float(ev.stats[1])

torch.manual_seed(SEED)
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASS)
model.to(DEVICE)

train_ds = CocoParking(
    root=f"{DATA_DIR}/train",
    annFile=f"{DATA_DIR}/train/_annotations.coco.json"
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

history = []
best_loss = float("inf")

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

    map50 = get_map50(model, "valid")
    history.append({
        "epoch": epoch,
        "loss":  round(avg_loss, 4),
        "mAP50": round(map50, 4)
    })
    print(f"Epoch {epoch}/100 loss={avg_loss:.4f} mAP50={map50:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss

    with open(SAVE_PATH, "w") as f:
        json.dump(history, f, indent=2)

print(f"\n✅ Guardado: {SAVE_PATH}")

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import json

BASE      = "/home/asistencia/tesis"
DATA_DIR  = f"{BASE}/dataset/coco"
JSON_PATH = f"{BASE}/modelos/v2/fasterrcnn_full.json"
DEVICE    = torch.device("cuda:1")
SEEDS     = [42, 123, 456, 789, 1000, 1234, 2024, 2025, 3141, 9999]
NUM_CLASS = 4

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

train_ds = CocoParking(
    root=f"{DATA_DIR}/train",
    annFile=f"{DATA_DIR}/train/_annotations.coco.json"
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)

gpu_values = []
print("Midiendo GPU memory durante entrenamiento (2 epochs por seed)...\n")

for seed in SEEDS:
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASS)
    model.to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(2):
        for imgs, targets in train_loader:
            imgs    = [im.to(DEVICE) for im in imgs]
            targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  seed={seed} epoch={epoch+1}/2")

    gpu_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    gpu_values.append(round(gpu_mb, 0))
    print(f"  seed={seed}: GPU peak = {gpu_mb:.0f} MB\n")

    del model, optimizer
    torch.cuda.empty_cache()

mean = np.mean(gpu_values)
std  = np.std(gpu_values, ddof=1)
print(f"\nGPU MB: {mean:.2f} ± {std:.2f}")
print(f"Valores: {gpu_values}")

with open(JSON_PATH) as f:
    d = json.load(f)

for i, run in enumerate(d['runs']):
    run['final']['gpu_mb'] = gpu_values[i]

d['summary']['gpu_mb'] = {
    "mean":   round(float(mean), 2),
    "std":    round(float(std),  2),
    "values": gpu_values
}

with open(JSON_PATH, 'w') as f:
    json.dump(d, f, indent=2)

print(f"\n✅ JSON actualizado: {JSON_PATH}")

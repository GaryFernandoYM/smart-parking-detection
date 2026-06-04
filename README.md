# Smart Parking Detection — UPeU

> Real-time parking occupancy detection using YOLOv8s, FastAPI, and Next.js

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-fine--tuned-purple)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## Overview

**Smart Parking UPeU** is a computer vision system that monitors a 15-space campus parking lot in real time. A fine-tuned YOLOv8s model processes video frames (local MP4 or RTSP stream) and classifies each parking space as:

| State | Label | Color |
|-------|-------|-------|
| Available | `libre` | Green |
| Occupied | `ocupado` | Red |
| Unavailable | `no_disponible` | Orange |

Results are streamed live over WebSocket to a Next.js dashboard. Developed as a thesis project (2026) by **Fernando Yunganina Mamani** at Universidad Peruana Unión (UPeU).

---

## Features

- **Real-time inference** — ~17–20 FPS on Apple M4 (MPS); also supports CUDA and CPU
- **Hysteresis stabilizer** — asymmetric threshold prevents state flickering between frames
- **ByteTrack vehicle tracking** — tracks vehicle IDs and logs occupancy duration per space
- **Live WebSocket dashboard** — Next.js frontend updates every second without polling
- **Field evaluation tool** — click-to-annotate UI for manual ground-truth collection
- **CSV session recorder** — logs predictions at 1-minute intervals for post-analysis
- **5-model comparative study** — YOLOv8s, YOLOv11s, YOLOv12s, RT-DETR, Faster R-CNN; each trained over 10 random seeds for statistical robustness

---

## Architecture

```
Video source (MP4 / RTSP)
        │
        ▼
 ┌─────────────────┐        ┌──────────────────┐
 │  FastAPI backend │◄──────►│  YOLOv8s model   │
 │  (port 8000)    │        │  best.pt weights  │
 └────────┬────────┘        └──────────────────┘
          │  WebSocket (/ws)
          ▼
 ┌─────────────────┐
 │ Next.js frontend│
 │  (port 3000)    │
 └─────────────────┘
```

**Backend:** FastAPI · Ultralytics · OpenCV · Shapely · ByteTrack  
**Frontend:** Next.js 14 · TypeScript · Tailwind CSS

---

## Model Comparison

All models trained 10 times with different random seeds (42, 123, 456, …). Values are mean ± std.

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|----|---------|-------------|
| **YOLOv8s** ✓ | 0.9964 ± 0.0015 | 0.9971 ± 0.0009 | 0.9967 ± 0.0009 | **0.9948 ± 0.0002** | **0.9916 ± 0.0003** |
| YOLOv11s | 0.9964 ± 0.0009 | 0.9975 ± 0.0007 | 0.9970 ± 0.0005 | 0.9947 ± 0.0001 | 0.9910 ± 0.0004 |
| YOLOv12s | 0.9963 ± 0.0008 | 0.9975 ± 0.0006 | 0.9969 ± 0.0004 | 0.9946 ± 0.0001 | 0.9906 ± 0.0003 |
| RT-DETR | 0.9962 ± 0.0008 | 0.9972 ± 0.0007 | 0.9967 ± 0.0004 | 0.9946 ± 0.0002 | 0.9907 ± 0.0006 |
| Faster R-CNN | 0.9742 ± 0.0228 | 1.0000 ± 0.0000 | 0.9868 ± 0.0118 | 0.9935 ± 0.0010 | 0.9858 ± 0.0042 |

YOLOv8s was selected for production: best mAP@0.5, tightest variance, and lowest inference latency (~45 ms/frame).

---

## Project Structure

```
smart-parking-detection/
├── backend/
│   ├── main.py                        # FastAPI server (REST + WebSocket)
│   ├── smart_parking_detector.py      # Core detection pipeline
│   ├── requirements.txt
│   ├── data/
│   │   └── espacios.json              # 15 parking space polygon definitions
│   └── modelos/yolov8s/resultados/
│       └── weights/
│           ├── best.pt                # Fine-tuned weights (production)
│           └── last.pt
│
├── frontend/
│   └── src/
│       ├── app/                       # Next.js app router
│       └── components/
│           ├── MapaEstacionamiento.tsx # Main parking lot dashboard
│           ├── EvaluacionPanel.tsx     # Metrics & evaluation tabs
│           ├── EvaluacionCampo.tsx     # Manual ground-truth tool
│           └── SesionCSV.tsx           # CSV session recorder
│
└── training/
    ├── scripts/                        # Training scripts (5 models)
    ├── dataset/Ultralytics/data.yaml   # YOLO dataset config
    └── modelos/
        ├── *.json                      # Per-model aggregate stats
        ├── tabla_estadistica_modelos.csv
        └── Analisis_Comparativo_v8s_v11s_v12s.ipynb
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Fine-tuned model weights at `backend/modelos/yolov8s/resultados/weights/best.pt`

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Configuration

Open `backend/main.py` and adjust at the top of the file:

```python
# Local video file (default)
VIDEO_SOURCE = str(BASE / "data/video_estacionamiento.mp4")

# Or an RTSP camera stream
VIDEO_SOURCE = "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
```

The inference device is auto-detected (MPS → CUDA → CPU). Override in `SmartParkingService.__init__` if needed.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Current state of all 15 spaces + metrics |
| `GET` | `/snapshot` | Annotated frame as base64 JPEG |
| `GET` | `/snapshot/raw` | Raw input frame |
| `GET` | `/metricas-modelo` | Model mAP scores from results.csv |
| `POST` | `/evaluar/iniciar` | Start 30-second evaluation session |
| `GET` | `/evaluar/estado` | Evaluation progress + FPS/latency |
| `POST` | `/campo/muestra` | Submit manual ground-truth snapshot |
| `GET` | `/campo/reporte` | Aggregate field evaluation report |
| `POST` | `/sesion/iniciar` | Start CSV recording session |
| `POST` | `/sesion/detener` | End session and save CSV |
| `GET` | `/sesion/metricas` | Calculate accuracy from CSV |
| `WS` | `/ws` | WebSocket — live status stream (1 Hz) |

### WebSocket payload (JSON, every 1 s)

```json
{
  "estados":  { "espacio_Q": "libre", "espacio_R": "ocupado" },
  "track_ids": { "espacio_Q": null,    "espacio_R": 42 },
  "fps": 17.3,
  "frame": 1234,
  "total": 5000,
  "latencia_ms": 45.2,
  "conf_promedio": 0.87,
  "detecciones": 3,
  "tiempos_s": { "espacio_R": 120.5 }
}
```

---

## Training

Training scripts are in `training/scripts/`. Each script runs the corresponding model 10 times with different seeds and saves results as JSON.

```bash
cd training/scripts

# Train YOLOv8s (10 seeds)
python train_yolov8s_v2.py

# Train other models
python train_yolov11s_v2.py
python train_yolov12s_v2.py
python train_rtdetr_v2.py
python train_fasterrcnn_v2.py
```

Dataset configuration: `training/dataset/Ultralytics/data.yaml`  
Analysis notebook: `training/modelos/Analisis_Comparativo_v8s_v11s_v12s.ipynb`

---

## Resources

- Dataset: [Roboflow Universe](https://universe.roboflow.com/fernando-qzldu/smart-parking-upeu/4/...)
- Weights: [Hugging Face](https://huggingface.co/GaryFer/smart-parking-weights/...)
- Validation videos: [Google Drive](https://drive.google.com/...)

---

## License

Distributed under the [Apache License 2.0](LICENSE).

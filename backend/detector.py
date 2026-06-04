import base64
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from ultralytics import YOLO

def _iou(det_box: Polygon, space_poly: Polygon) -> float:
    try:
        inter = det_box.intersection(space_poly).area
        union = det_box.union(space_poly).area
        return inter / union if union else 0.0
    except Exception:
        return 0.0


def _space_to_shapely(poligono: list) -> Polygon:
    return Polygon([(p["x"], p["y"]) for p in poligono])


class ParkingDetector:
    IOU_THRESHOLD = 0.10

    def __init__(self, model_path: str, espacios_path: str, rtsp_url: str, conf: float = 0.5):
        self.model = YOLO(model_path)
        self._class_names: dict[int, str] = self.model.names
        self.conf = conf
        self.rtsp_url = rtsp_url

        with open(espacios_path) as f:
            data = json.load(f)
        self._spaces = data["espacios"]

        self._state: dict[str, str] = {s["id"]: s["clase"] for s in self._spaces}
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_status(self) -> dict[str, str]:
        with self._lock:
            return dict(self._state)

    def get_snapshot(self) -> str | None:
        with self._lock:
            if self._frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buf).decode()

    def _loop(self) -> None:
        space_polys = [(s["id"], _space_to_shapely(s["poligono"])) for s in self._spaces]

        cap = cv2.VideoCapture(self.rtsp_url)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                if self.rtsp_url.startswith("rtsp://"):
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.rtsp_url)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = self.model.predict(frame, conf=self.conf, imgsz=2560, verbose=False)

            det_polys: list[tuple[Polygon, str]] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_idx = int(box.cls[0])
                    cls_name = self._class_names.get(cls_idx, "libre")
                    det_polys.append((shapely_box(x1, y1, x2, y2), cls_name))

            with self._lock:
                for sid, space_poly in space_polys:
                    best_iou = 0.0
                    best_cls: str | None = None
                    for det_box, cls_name in det_polys:
                        iou = _iou(det_box, space_poly)
                        if iou > best_iou:
                            best_iou = iou
                            best_cls = cls_name
                    if best_cls is not None and best_iou >= self.IOU_THRESHOLD:
                        self._state[sid] = best_cls

                self._frame = frame.copy()

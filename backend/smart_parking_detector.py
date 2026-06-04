"""
Smart Parking UPeU — Detector de Ocupación en Tiempo Real
─────────────────────────────────────────────────────────
Utiliza YOLOv8s fine-tuned sobre dataset Smart Parking UPeU v2
para detectar el estado de 15 espacios de estacionamiento.

Uso:
    python smart_parking_detector.py                          # video local
    python smart_parking_detector.py --source rtsp://...      # cámara RTSP
    python smart_parking_detector.py --model path/to/best.pt  # otro modelo

Controles:
    Q / ESC  →  Salir
    ESPACIO  →  Pausar / Reanudar
    S        →  Guardar captura actual

Autor: Fernando Yunganina Mamani
Proyecto: Smart Parking UPeU — Tesis 2026
"""

import argparse
import time
import cv2
import json
import numpy as np
from collections import deque, Counter
from pathlib import Path
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "model":       "modelos/yolov8s/resultados/weights/best.pt",
    "source":      "",
    "espacios":    "data/espacios.json",
    "conf":        0.35,
    "iou_nms":     0.3,
    "min_bbox":    50,
    "iou_umbral":  0.3,     
    "alpha":       0.30, 
    "display_w":   1280,
    "display_h":   720,
    "device":      "mps",   # mps para Mac M4, cuda:0 para NVIDIA
}

COLORES = {
    "libre":         (46, 204, 113),   # verde esmeralda
    "ocupado":       (52, 73, 235),    # rojo coral
    "no_disponible": (39, 174, 245),   # naranja ámbar
}

COLORES_BORDE = {
    "libre":         (39, 174, 96),
    "ocupado":       (41, 58, 189),
    "no_disponible": (31, 139, 196),
}


# ══════════════════════════════════════════════════════════════════════
# CLASES
# ══════════════════════════════════════════════════════════════════════

class StabilizadorEstados:
    """Histeresis asimétrica: fácil detectar ocupación, difícil volver a libre.

    libre → ocupado: basta con que umbral_ocupar  fracción de frames detecten algo
    ocupado → libre: requiere que umbral_liberar fracción de frames no detecten nada
    """

    def __init__(self, ventana: int = 10,
                 umbral_ocupar: float = 0.30,
                 umbral_liberar: float = 0.80):
        self._historial: dict[str, deque] = {}
        self._ventana        = ventana
        self._umbral_ocupar  = umbral_ocupar
        self._umbral_liberar = umbral_liberar
        self._estable: dict[str, str] = {}

    def actualizar(self, estados_raw: dict[str, str]) -> dict[str, str]:
        for eid, estado in estados_raw.items():
            if eid not in self._historial:
                self._historial[eid] = deque(maxlen=self._ventana)
                self._estable[eid]   = estado

            self._historial[eid].append(estado)
            n      = len(self._historial[eid])
            conteo = Counter(self._historial[eid])

            actual     = self._estable.get(eid, "libre")
            frac_libre = conteo.get("libre", 0) / n

            if actual == "libre":
                frac_no_libre = 1.0 - frac_libre
                if frac_no_libre >= self._umbral_ocupar:
                    candidato = max(
                        (k for k in conteo if k != "libre"),
                        key=lambda k: conteo[k],
                        default=None,
                    )
                    if candidato:
                        self._estable[eid] = candidato
            else:
                if frac_libre >= self._umbral_liberar:
                    self._estable[eid] = "libre"

        return dict(self._estable)


class RegistroOcupacion:
    """Mide cuánto tiempo lleva cada vehículo en cada espacio."""

    def __init__(self, max_history: int = 200):
        self._activos:   dict[str, dict] = {}
        self._historial: list[dict]      = []
        self._max_history = max_history

    def actualizar(self, estados: dict, track_ids: dict) -> None:
        ahora = time.time()
        for eid, estado in estados.items():
            tid    = track_ids.get(eid)
            activo = self._activos.get(eid)

            if estado in ("libre", "no_disponible"):
                if activo is not None:
                    self._cerrar(eid, activo, ahora)
            else:  # ocupado
                if activo is None:
                    self._activos[eid] = {"track_id": tid if tid is not None else -1, "inicio": ahora}
                else:
                    if tid is not None and activo["track_id"] != -1 and tid != activo["track_id"]:
                        self._cerrar(eid, activo, ahora)
                        self._activos[eid] = {"track_id": tid, "inicio": ahora}
                    elif tid is not None and activo["track_id"] == -1:
                        activo["track_id"] = tid

    def tiempos_s(self) -> dict[str, float]:
        ahora = time.time()
        return {eid: round(ahora - r["inicio"], 1) for eid, r in self._activos.items()}

    def reset(self) -> None:
        self._activos.clear()

    def _cerrar(self, eid: str, rec: dict, fin: float) -> None:
        self._historial.append({
            "espacio_id": eid,
            "track_id":   rec["track_id"],
            "inicio_ts":  rec["inicio"],
            "fin_ts":     fin,
            "duracion_s": round(fin - rec["inicio"], 2),
        })
        if len(self._historial) > self._max_history:
            self._historial.pop(0)
        del self._activos[eid]


class EspacioManager:
    """Gestiona los espacios del estacionamiento y sus polígonos."""

    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.metadata  = data["metadata"]
        self.img_h     = data["metadata"]["alto_imagen"]
        self.img_w     = data["metadata"]["ancho_imagen"]
        self.espacios  = {e["id"]: e for e in data["espacios"]}
        self._masks    = self._precomputar_masks()

    def _precomputar_masks(self) -> dict:
        """Pre-computa las máscaras de cada polígono UNA vez."""
        masks = {}
        for eid, esp in self.espacios.items():
            pts = np.array(
                [[p["x"], p["y"]] for p in esp["poligono"]],
                dtype=np.int32
            )
            mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            area = int(np.sum(mask))
            centro = (
                int(np.mean([p["x"] for p in esp["poligono"]])),
                int(np.mean([p["y"] for p in esp["poligono"]])),
            )
            masks[eid] = {
                "pts":    pts,
                "mask":   mask,
                "area":   area,
                "centro": centro,
            }
        return masks

    def calcular_iou(self, x1: int, y1: int, x2: int, y2: int, eid: str) -> float:
        """IoU entre un bbox y un polígono pre-computado."""
        m = self._masks[eid]
        roi = m["mask"][y1:y2, x1:x2]
        inter = int(np.sum(roi))
        return inter / (m["area"] + 1e-8)

    def bbox_intersecta_poligono(self, x1: int, y1: int, x2: int, y2: int, eid: str) -> bool:
        """True si cualquier punto clave de la bbox cae dentro del polígono."""
        m  = self._masks[eid]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        for px, py in [(cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            if 0 <= py < self.img_h and 0 <= px < self.img_w:
                if m["mask"][py, px]:
                    return True
        return False

    @property
    def ids(self):
        return list(self.espacios.keys())

    def poligono_pts(self, eid: str) -> np.ndarray:
        return self._masks[eid]["pts"]

    def centro(self, eid: str) -> tuple:
        return self._masks[eid]["centro"]


class Detector:
    """Encapsula el modelo YOLO y la lógica de detección."""

    def __init__(self, model_path: str, device: str, conf: float,
                 iou_nms: float, min_bbox: int):
        self.model    = YOLO(model_path)
        self.device   = device
        self.conf     = conf
        self.iou_nms  = iou_nms
        self.min_bbox = min_bbox

    def inferir(self, frame: np.ndarray) -> list:
        """Ejecuta inferencia y retorna boxes filtrados con track_id."""
        results = self.model.track(
            frame, conf=self.conf, iou=self.iou_nms,
            device=self.device, verbose=False,
            persist=True, tracker="bytetrack.yaml",
        )[0]

        boxes = []
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if (x2 - x1) < self.min_bbox or (y2 - y1) < self.min_bbox:
                continue
            track_id = int(b.id.item()) if b.id is not None else None
            boxes.append({
                "clase":    self.model.names[int(b.cls)],
                "conf":     float(b.conf),
                "coords":   (x1, y1, x2, y2),
                "track_id": track_id,
            })
        return boxes


class Renderer:
    """Renderiza el frame con polígonos, bboxes y panel HUD."""

    def __init__(self, espacios_mgr: EspacioManager, alpha: float):
        self.mgr   = espacios_mgr
        self.alpha = alpha

    def dibujar(self, frame: np.ndarray, estados: dict,
                boxes: list, info: dict) -> np.ndarray:
        """Dibuja todo en el frame: polígonos, bboxes, panel."""
        frame = self._dibujar_poligonos(frame, estados)
        frame = self._dibujar_bboxes(frame, boxes)
        frame = self._dibujar_panel(frame, estados, info)
        return frame

    def _dibujar_poligonos(self, frame: np.ndarray,
                           estados: dict) -> np.ndarray:
        # Relleno — un solo overlay, sin parpadeo
        overlay = frame.copy()
        for eid in self.mgr.ids:
            estado = estados.get(eid, "libre")
            color  = COLORES.get(estado, (200, 200, 200))
            cv2.fillPoly(overlay, [self.mgr.poligono_pts(eid)], color)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # Bordes + etiquetas
        for eid in self.mgr.ids:
            estado = estados.get(eid, "libre")
            color  = COLORES_BORDE.get(estado, (150, 150, 150))
            cv2.polylines(frame, [self.mgr.poligono_pts(eid)], True, color, 2)
            cx, cy = self.mgr.centro(eid)
            label  = eid.replace("espacio_", "").upper()
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.putText(
                frame, label,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
        return frame

    def _dibujar_bboxes(self, frame: np.ndarray,
                        boxes: list) -> np.ndarray:
        for b in boxes:
            x1, y1, x2, y2 = b["coords"]
            color = COLORES.get(b["clase"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            txt = f'{b["clase"]} {b["conf"]:.0%}'
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                frame, txt, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
        return frame

    def _dibujar_panel(self, frame: np.ndarray, estados: dict,
                       info: dict) -> np.ndarray:
        total  = len(self.mgr.ids)
        conteo = {"libre": 0, "ocupado": 0, "no_disponible": 0}
        for e in estados.values():
            conteo[e] = conteo.get(e, 0) + 1

        # Fondo del panel
        panel_h, panel_w = 200, 380
        sub = frame[12:12 + panel_h, 12:12 + panel_w].copy()
        dark = np.zeros_like(sub)
        cv2.addWeighted(dark, 0.75, sub, 0.25, 0, sub)
        frame[12:12 + panel_h, 12:12 + panel_w] = sub

        # Borde
        cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h),
                       (80, 80, 80), 1, cv2.LINE_AA)

        x0, y0 = 28, 12
        font   = cv2.FONT_HERSHEY_SIMPLEX

        # Título
        cv2.putText(frame, "SMART PARKING UPeU",
                     (x0, y0 + 30), font, 0.65, (255, 255, 255), 2,
                     cv2.LINE_AA)
        cv2.line(frame, (x0 - 4, y0 + 38), (x0 + 340, y0 + 38),
                  (80, 80, 80), 1, cv2.LINE_AA)

        # Contadores
        cv2.putText(frame, f"Total espacios:   {total}",
                     (x0, y0 + 65), font, 0.55, (220, 220, 220), 1,
                     cv2.LINE_AA)

        cv2.circle(frame, (x0 + 5, y0 + 85), 5, COLORES["libre"], -1)
        cv2.putText(frame, f"Libres:  {conteo['libre']}",
                     (x0 + 18, y0 + 90), font, 0.55, COLORES["libre"], 1,
                     cv2.LINE_AA)

        cv2.circle(frame, (x0 + 5, y0 + 110), 5, COLORES["ocupado"], -1)
        cv2.putText(frame, f"Ocupados:  {conteo['ocupado']}",
                     (x0 + 18, y0 + 115), font, 0.55, COLORES["ocupado"], 1,
                     cv2.LINE_AA)

        cv2.circle(frame, (x0 + 5, y0 + 135), 5,
                    COLORES["no_disponible"], -1)
        cv2.putText(frame, f"No disponible:  {conteo['no_disponible']}",
                     (x0 + 18, y0 + 140), font, 0.5,
                     COLORES["no_disponible"], 1, cv2.LINE_AA)

        # Info
        cv2.line(frame, (x0 - 4, y0 + 152), (x0 + 340, y0 + 152),
                  (60, 60, 60), 1, cv2.LINE_AA)
        fps_txt = f'FPS: {info.get("fps", 0):.1f}'
        frm_txt = f'Frame: {info.get("frame", 0)}/{info.get("total", 0)}'
        cv2.putText(frame, fps_txt, (x0, y0 + 172), font, 0.42,
                     (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, frm_txt, (x0 + 140, y0 + 172), font, 0.42,
                     (160, 160, 160), 1, cv2.LINE_AA)

        if info.get("pausado"):
            cv2.putText(frame, "[ PAUSA ]", (x0 + 120, y0 + 30), font,
                         0.55, (0, 200, 255), 1, cv2.LINE_AA)

        return frame


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def asignar_estados(boxes: list, mgr: EspacioManager,
                    iou_umbral: float) -> tuple[dict, dict]:
    """Asigna estado e ID de tracking dominante por espacio."""
    estados   = {}
    track_ids = {}
    for eid in mgr.ids:
        mejor_score = 0.0
        mejor_clase = "libre"
        mejor_tid   = None
        for b in boxes:
            x1, y1, x2, y2 = b["coords"]
            iou     = mgr.calcular_iou(x1, y1, x2, y2, eid)
            en_poly = mgr.bbox_intersecta_poligono(x1, y1, x2, y2, eid)
            if iou >= iou_umbral or en_poly:
                score = iou if iou > 0 else 0.001
                if score > mejor_score:
                    mejor_score = score
                    mejor_clase = b["clase"]
                    mejor_tid   = b.get("track_id")
        estados[eid]   = mejor_clase
        track_ids[eid] = mejor_tid if mejor_clase != "libre" else None
    return estados, track_ids


def run(cfg: dict):
    """Loop principal de detección."""
    base = Path(cfg.get("base", "."))

    mgr      = EspacioManager(str(base / cfg["espacios"]))
    detector = Detector(
        str(base / cfg["model"]),
        cfg["device"], cfg["conf"], cfg["iou_nms"], cfg["min_bbox"],
    )
    renderer = Renderer(mgr, cfg["alpha"])

    source = str(base / cfg["source"]) if not cfg["source"].startswith("rtsp") else cfg["source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 20
    print(f"Fuente: {source}")
    print(f"Frames: {total_frames}  FPS: {fps_video:.1f}")
    print("─" * 45)
    print("Q/ESC → salir | ESPACIO → pausar | S → captura")
    print("─" * 45)

    frame_n       = 0
    pausado       = False
    ultimo_estado = {eid: "libre" for eid in mgr.ids}
    ultimo_boxes  = []
    t_prev        = time.time()
    fps_display   = 0.0

    while cap.isOpened():
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1

            # Inferencia
            boxes         = detector.inferir(frame)
            ultimo_boxes  = boxes
            ultimo_estado = asignar_estados(
                boxes, mgr, cfg["iou_umbral"]
            )

            # FPS
            t_now       = time.time()
            fps_display = 1.0 / (t_now - t_prev + 1e-8)
            t_prev      = t_now

        # Render
        info = {
            "fps":     fps_display,
            "frame":   frame_n,
            "total":   total_frames,
            "pausado": pausado,
        }
        vis = renderer.dibujar(
            frame.copy(), ultimo_estado, ultimo_boxes, info
        )

        display = cv2.resize(vis, (cfg["display_w"], cfg["display_h"]),
                              interpolation=cv2.INTER_AREA)
        cv2.imshow("Smart Parking UPeU", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            pausado = not pausado
        elif key == ord('s'):
            out = f"captura_{frame_n}.jpg"
            cv2.imwrite(out, vis)
            print(f"Captura guardada: {out}")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen final
    conteo = {"libre": 0, "ocupado": 0, "no_disponible": 0}
    for e in ultimo_estado.values():
        conteo[e] += 1
    print(f"\nFrames procesados: {frame_n}")
    print(f"Libres: {conteo['libre']}  |  "
          f"Ocupados: {conteo['ocupado']}  |  "
          f"No disponible: {conteo['no_disponible']}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Parking UPeU — Detector de Ocupación"
    )
    parser.add_argument("--base",    default="/smartPv1/backend")
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"])
    parser.add_argument("--source",  default=DEFAULT_CONFIG["source"])
    parser.add_argument("--espacios", default=DEFAULT_CONFIG["espacios"])
    parser.add_argument("--conf",    type=float, default=DEFAULT_CONFIG["conf"])
    parser.add_argument("--device",  default=DEFAULT_CONFIG["device"])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["base"]     = args.base
    cfg["model"]    = args.model
    cfg["source"]   = args.source
    cfg["espacios"] = args.espacios
    cfg["conf"]     = args.conf
    cfg["device"]   = args.device

    run(cfg)

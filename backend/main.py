import asyncio
import base64
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from smart_parking_detector import (
    EspacioManager, Detector, Renderer, asignar_estados,
    StabilizadorEstados, RegistroOcupacion, DEFAULT_CONFIG,
)

BASE          = Path(__file__).parent
MODEL_PATH    = BASE / "modelos/yolov8s/best.pt"
ESPACIOS_PATH = BASE / "data/espacios.json"
SESIONES_DIR  = BASE / "sesiones"
SESIONES_DIR.mkdir(exist_ok=True)

# ── Fuente de video ────────────────────────────────────────────────────────────
# Video local (actual):
VIDEO_SOURCE = str(BASE / "data/video_estacionamiento.mp4")
#VIDEO_SOURCE = "rtsp://admin:password@172.20.20.79:554/Streaming/Channels/101"

# True  → archivo de video (reinicia al terminar, sin FrameGrabber)
# False → cámara en vivo (RTSP/USB): siempre el frame más reciente, sin lag
IS_VIDEO_FILE = not VIDEO_SOURCE.startswith("rtsp://") and not VIDEO_SOURCE.startswith("http")


class FrameGrabber:
    """Hilo dedicado a leer frames de la cámara, descarta los viejos.

    Evita el lag acumulado en streams RTSP: YOLO siempre procesa
    el frame más reciente en lugar de frames de hace varios segundos.
    """
    def __init__(self, source: str):
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # buffer mínimo del SO
        self._frame: cv2.typing.MatLike | None = None
        self._lock  = threading.Lock()
        self._ok    = False
        t = threading.Thread(target=self._grab, daemon=True)
        t.start()

    def _grab(self) -> None:
        while True:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok    = ok
                self._frame = frame if ok else self._frame

    def latest(self) -> tuple[bool, cv2.typing.MatLike | None]:
        with self._lock:
            return self._ok, self._frame

    def release(self) -> None:
        self._cap.release()

CFG = {
    **DEFAULT_CONFIG,
    "conf":       0.20,
    "iou_nms":    0.3,
    "min_bbox":   50,
    "iou_umbral": 0.45,
    "alpha":                0.30,
    "device":               "mps",
    "estab_ventana":        25,    # ~1.5s a 17fps — ventana amplia para evitar parpadeos
    "estab_umbral_ocupar":  0.50,  # 12/25 frames con detección → ocupado (detección rápida)
    "estab_umbral_liberar": 0.90,  # 22/25 frames sin detección → libre (liberación muy lenta)
}


def _fit_frame(frame: np.ndarray, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    """Redimensiona preservando aspect ratio — no fuerza 16:9."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


class SmartParkingService:
    def __init__(self):
        self.mgr      = EspacioManager(str(ESPACIOS_PATH))
        self.detector = Detector(
            str(MODEL_PATH), CFG["device"],
            CFG["conf"], CFG["iou_nms"], CFG["min_bbox"],
        )
        self.renderer     = Renderer(self.mgr, CFG["alpha"])
        self.estabilizador = StabilizadorEstados(
            CFG["estab_ventana"],
            CFG["estab_umbral_ocupar"],
            CFG["estab_umbral_liberar"],
        )
        self.registro      = RegistroOcupacion(max_history=200)
        self._tiempos_s:   dict[str, float] = {}
        self._track_ids:   dict[str, int | None] = {}

        self._estados     = {eid: "libre" for eid in self.mgr.ids}
        self._boxes       = []
        self._frame_raw   = None
        self._annotated   = None
        self._fps         = 0.0
        self._frame_n     = 0
        self._total       = 0
        self._latencia_ms  = 0.0
        self._conf_promedio = 0.0
        self._detecciones   = 0

        self._eval_activa   = False
        self._eval_inicio   = 0.0
        self._eval_dur      = 30
        self._eval_fps_list:  list[float] = []
        self._eval_lat_list:  list[float] = []
        self._eval_conf_list: list[float] = []
        self._eval_cambios  = 0
        self._eval_prev_est: dict[str, str] = {}

        self._muestras: list[dict] = []   # evaluación de campo manual

        # sesión automática de evaluación
        self._auto_activa    = False
        self._auto_inicio    = 0.0
        self._auto_duracion  = 1800
        self._auto_intervalo = 60
        self._auto_ultima    = 0.0
        self._auto_capturas: list[dict] = []
        self._auto_fps_list: list[float] = []
        self._auto_lat_list: list[float] = []

        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None

        # ── Sesión CSV ─────────────────────────────────────────────────────────
        self._csv_activa:    bool       = False
        self._csv_nombre:    str        = ""
        self._csv_condicion: str        = ""
        self._csv_duracion:  int        = 1800
        self._csv_inicio:    float      = 0.0
        self._csv_ultima:    float      = 0.0
        self._csv_minuto:    int        = 0
        self._csv_filas:     list[dict] = []
        self._csv_ruta:      Path | None = None

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def captura_actual(self) -> dict:
        """Devuelve predicción actual + snapshot para la evaluación de campo."""
        with self._lock:
            estados   = dict(self._estados)
            frame_n   = self._frame_n
            annotated = self._annotated  # solo referencia, sin codificar dentro del lock

        # Codificación FUERA del lock — no bloquea _loop ni /snapshot
        snapshot = None
        if annotated is not None:
            small = _fit_frame(annotated)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
            snapshot = base64.b64encode(buf).decode()

        return {"estados": estados, "snapshot": snapshot, "frame": frame_n}

    def registrar_muestra(self, ground_truth: dict[str, str]) -> dict:
        """Compara ground_truth con la predicción actual y acumula el resultado."""
        with self._lock:
            pred = dict(self._estados)

        ids = list(pred.keys())
        tp = tn = fp = fn = 0
        errores = []
        for eid in ids:
            p = pred.get(eid, "libre") != "libre"   # True = sistema dice ocupado
            g = ground_truth.get(eid, "libre") != "libre"  # True = realmente ocupado
            if p and g:
                tp += 1
            elif not p and not g:
                tn += 1
            elif p and not g:
                fp += 1
                errores.append(eid)
            else:
                fn += 1
                errores.append(eid)

        total    = tp + tn + fp + fn
        accuracy  = round((tp + tn) / total * 100, 1) if total else 0
        precision = round(tp / (tp + fp) * 100, 1)   if (tp + fp) else 0
        recall    = round(tp / (tp + fn) * 100, 1)   if (tp + fn) else 0
        f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) else 0

        muestra = {
            "timestamp": time.strftime("%H:%M:%S"),
            "prediccion": pred,
            "ground_truth": ground_truth,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "errores": errores,
        }
        with self._lock:
            self._muestras.append(muestra)
        return muestra

    def get_reporte_campo(self) -> dict:
        with self._lock:
            muestras = list(self._muestras)
        if not muestras:
            return {"n": 0, "muestras": []}
        acc_list  = [m["accuracy"]  for m in muestras]
        prec_list = [m["precision"] for m in muestras]
        rec_list  = [m["recall"]    for m in muestras]
        f1_list   = [m["f1"]        for m in muestras]
        return {
            "n":            len(muestras),
            "accuracy_prom":  round(sum(acc_list)  / len(acc_list),  1),
            "precision_prom": round(sum(prec_list) / len(prec_list), 1),
            "recall_prom":    round(sum(rec_list)  / len(rec_list),  1),
            "f1_prom":        round(sum(f1_list)   / len(f1_list),   1),
            "muestras":     muestras,
        }

    def limpiar_muestras(self) -> None:
        with self._lock:
            self._muestras.clear()

    # ── Sesión automática ────────────────────────────────────────────────────

    def iniciar_auto(self, minutos: int = 30, intervalo_s: int = 60) -> None:
        with self._lock:
            self._auto_capturas.clear()
            self._auto_fps_list.clear()
            self._auto_lat_list.clear()
            self._auto_duracion  = minutos * 60
            self._auto_intervalo = intervalo_s
            self._auto_inicio    = time.time()
            self._auto_ultima    = 0.0
            self._auto_activa    = True

    def get_auto_estado(self) -> dict:
        with self._lock:
            activa    = self._auto_activa
            elapsed   = time.time() - self._auto_inicio if self._auto_inicio else 0
            total_cap = int(self._auto_duracion / self._auto_intervalo)
            n_cap     = len(self._auto_capturas)
            prox      = max(0, self._auto_intervalo - (time.time() - self._auto_ultima))
            pendientes = sum(1 for c in self._auto_capturas if c["anotacion"] is None)
        return {
            "activa":            activa,
            "progreso":          min(100, round(elapsed / max(self._auto_duracion, 1) * 100)),
            "elapsed_s":         int(elapsed),
            "duracion_s":        self._auto_duracion,
            "capturas_tomadas":  n_cap,
            "capturas_total":    total_cap,
            "proxima_en_s":      int(prox) if activa else 0,
            "pendientes_anotar": pendientes,
        }

    def get_auto_captura(self, idx: int) -> dict | None:
        with self._lock:
            if 0 <= idx < len(self._auto_capturas):
                return dict(self._auto_capturas[idx])
        return None

    def anotar_captura(self, idx: int, ground_truth: dict[str, str]) -> dict:
        with self._lock:
            if not (0 <= idx < len(self._auto_capturas)):
                return {"error": "índice inválido"}
            pred = dict(self._auto_capturas[idx]["estados"])

        ids = list(pred.keys())
        tp = tn = fp = fn = 0
        errores = []
        for eid in ids:
            p = pred.get(eid, "libre") != "libre"
            g = ground_truth.get(eid, "libre") != "libre"
            if   p and g:     tp += 1
            elif not p and not g: tn += 1
            elif p and not g: fp += 1; errores.append(eid)
            else:             fn += 1; errores.append(eid)

        total    = tp + tn + fp + fn
        accuracy  = round((tp + tn) / total * 100, 1) if total else 0
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) else 100
        recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 100
        f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) else 0

        resultado = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "errores": errores,
        }
        with self._lock:
            self._auto_capturas[idx]["anotacion"] = resultado
            self._auto_capturas[idx]["ground_truth"] = ground_truth
        return resultado

    def get_reporte_auto(self) -> dict:
        with self._lock:
            capturas  = [c for c in self._auto_capturas if c["anotacion"]]
            total_cap = list(self._auto_capturas)
            fps_l     = list(self._auto_fps_list)
            lat_l     = list(self._auto_lat_list)
        if not capturas:
            return {"n": 0, "detalle": []}
        acc  = [c["anotacion"]["accuracy"]  for c in capturas]
        prec = [c["anotacion"]["precision"] for c in capturas]
        rec  = [c["anotacion"]["recall"]    for c in capturas]
        f1s  = [c["anotacion"]["f1"]        for c in capturas]
        detalle = [{"idx": c["idx"], "timestamp": c["timestamp"], **c["anotacion"]}
                   for c in capturas]
        return {
            "n":              len(capturas),
            "pendientes":     sum(1 for c in total_cap if c["anotacion"] is None),
            "accuracy_prom":  round(sum(acc)  / len(acc),  1),
            "precision_prom": round(sum(prec) / len(prec), 1),
            "recall_prom":    round(sum(rec)  / len(rec),  1),
            "f1_prom":        round(sum(f1s)  / len(f1s),  1),
            "fps_promedio":   round(sum(fps_l) / len(fps_l), 1) if fps_l else 0,
            "fps_min":        round(min(fps_l), 1) if fps_l else 0,
            "fps_max":        round(max(fps_l), 1) if fps_l else 0,
            "latencia_prom":  round(sum(lat_l) / len(lat_l), 1) if lat_l else 0,
            "latencia_min":   round(min(lat_l), 1) if lat_l else 0,
            "latencia_max":   round(max(lat_l), 1) if lat_l else 0,
            "detalle":        detalle,
        }

    def limpiar_auto(self) -> None:
        with self._lock:
            self._auto_capturas.clear()
            self._auto_activa = False

    def iniciar_eval(self, segundos: int = 30) -> None:
        with self._lock:
            self._eval_fps_list.clear()
            self._eval_lat_list.clear()
            self._eval_conf_list.clear()
            self._eval_cambios  = 0
            self._eval_prev_est = {}
            self._eval_dur      = segundos
            self._eval_inicio   = time.time()
            self._eval_activa   = True

    def get_eval_estado(self) -> dict:
        with self._lock:
            activa   = self._eval_activa
            elapsed  = time.time() - self._eval_inicio if self._eval_inicio else 0
            progreso = min(100, round(elapsed / max(self._eval_dur, 1) * 100))
            fps_l    = list(self._eval_fps_list)
            lat_l    = list(self._eval_lat_list)
        with self._lock:
            conf_l = list(self._eval_conf_list)
        return {
            "activa":            activa,
            "progreso":          progreso if activa else 100,
            "frames_evaluados":  len(fps_l),
            "fps_promedio":      round(sum(fps_l) / len(fps_l), 1) if fps_l else 0,
            "fps_min":           round(min(fps_l), 1) if fps_l else 0,
            "fps_max":           round(max(fps_l), 1) if fps_l else 0,
            "latencia_promedio": round(sum(lat_l) / len(lat_l), 1) if lat_l else 0,
            "latencia_min":      round(min(lat_l), 1) if lat_l else 0,
            "latencia_max":      round(max(lat_l), 1) if lat_l else 0,
            "cambios_estado":    self._eval_cambios,
            "conf_promedio":     round(sum(conf_l) / len(conf_l), 1) if conf_l else 0,
        }

    def get_status(self) -> dict:
        with self._lock:
            return {
                "estados":       dict(self._estados),
                "track_ids":     dict(self._track_ids),
                "fps":           round(self._fps, 1),
                "frame":         self._frame_n,
                "total":         self._total,
                "latencia_ms":   round(self._latencia_ms, 2),
                "conf_promedio": self._conf_promedio,
                "detecciones":   self._detecciones,
                "tiempos_s":     dict(self._tiempos_s),
            }

    def get_snapshot(self) -> str | None:
        with self._lock:
            if self._annotated is None:
                return None
            small = _fit_frame(self._annotated)
            _, buf = cv2.imencode(
                ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78]
            )
            return base64.b64encode(buf).decode()

    def get_raw_snapshot(self) -> str | None:
        with self._lock:
            if self._frame_raw is None:
                return None
            small = _fit_frame(self._frame_raw)
            _, buf = cv2.imencode(
                ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78]
            )
            return base64.b64encode(buf).decode()

    def _loop(self) -> None:
        if IS_VIDEO_FILE:
            cap         = cv2.VideoCapture(VIDEO_SOURCE)
            self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            grabber     = None
        else:
            cap         = None
            grabber     = FrameGrabber(VIDEO_SOURCE)
            self._total = 0   # stream en vivo, sin total

        t_prev  = time.time()
        frame_n = 0

        while self._running:
            if IS_VIDEO_FILE:
                ok, frame = cap.read()          # type: ignore[union-attr]
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[union-attr]
                    self.registro.reset()
                    continue
                frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # type: ignore[union-attr]
            else:
                ok, frame = grabber.latest()    # type: ignore[union-attr]
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                frame_n += 1

            t_inf       = time.time()
            boxes       = self.detector.inferir(frame)
            lat_ms      = (time.time() - t_inf) * 1000
            estados_raw, track_ids_raw = asignar_estados(boxes, self.mgr, CFG["iou_umbral"])
            estados = self.estabilizador.actualizar(estados_raw)

            t_now  = time.time()
            fps    = 1.0 / (t_now - t_prev + 1e-8)
            t_prev = t_now

            info = {
                "fps":     fps,
                "frame":   frame_n,
                "total":   self._total,
                "pausado": False,
            }
            annotated = self.renderer.dibujar(
                frame.copy(), estados, boxes, info
            )

            with self._lock:
                self._estados      = estados
                self._boxes        = boxes
                self._frame_raw    = frame
                self._annotated    = annotated
                self._fps          = fps
                self._frame_n      = frame_n
                self._latencia_ms  = lat_ms
                self._detecciones  = len(boxes)
                self._conf_promedio = (
                    round(sum(b["conf"] for b in boxes) / len(boxes) * 100, 1)
                    if boxes else 0.0
                )
                self.registro.actualizar(estados, track_ids_raw)
                self._tiempos_s = self.registro.tiempos_s()
                self._track_ids = dict(track_ids_raw)

                if self._eval_activa:
                    self._eval_fps_list.append(fps)
                    self._eval_lat_list.append(lat_ms)
                    if self._conf_promedio > 0:
                        self._eval_conf_list.append(self._conf_promedio)
                    self._eval_cambios += sum(
                        1 for eid in estados
                        if estados[eid] != self._eval_prev_est.get(eid, estados[eid])
                    )
                    self._eval_prev_est = dict(estados)
                    if time.time() - self._eval_inicio >= self._eval_dur:
                        self._eval_activa = False

            # ── captura automática (fuera del lock principal) ──────────────────
            with self._lock:
                auto_activa  = self._auto_activa
                auto_ultima  = self._auto_ultima
                auto_int     = self._auto_intervalo
                auto_inicio  = self._auto_inicio
                auto_dur     = self._auto_duracion
                snap_estados = dict(self._estados)
                snap_frame   = self._annotated

            ahora = time.time()
            # Recolectar FPS/latencia durante sesión auto
            if auto_activa:
                with self._lock:
                    self._auto_fps_list.append(fps)
                    self._auto_lat_list.append(lat_ms)

            if auto_activa and (ahora - auto_ultima) >= auto_int:
                snap64 = None
                if snap_frame is not None:
                    small = _fit_frame(snap_frame)
                    _, buf = cv2.imencode(".jpg", small,
                                         [cv2.IMWRITE_JPEG_QUALITY, 72])
                    snap64 = base64.b64encode(buf).decode()
                captura = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "estados":   snap_estados,
                    "snapshot":  snap64,
                    "anotacion": None,
                }
                with self._lock:
                    captura["idx"] = len(self._auto_capturas)
                    self._auto_capturas.append(captura)
                    self._auto_ultima = ahora
                    if ahora - auto_inicio >= auto_dur:
                        self._auto_activa = False

            # ── captura CSV sesión ─────────────────────────────────────────────
            with self._lock:
                csv_activa    = self._csv_activa
                csv_ultima    = self._csv_ultima
                csv_inicio    = self._csv_inicio
                csv_dur       = self._csv_duracion

            if csv_activa and (ahora - csv_ultima) >= 60:
                guardar_ahora = False
                with self._lock:
                    self._csv_minuto += 1
                    minuto = self._csv_minuto
                    for eid, estado in snap_estados.items():
                        self._csv_filas.append({
                            "sesion":       self._csv_nombre,
                            "condicion":    self._csv_condicion,
                            "minuto":       minuto,
                            "plaza":        eid,
                            "prediccion":   estado,
                            "ground_truth": "",
                        })
                    self._csv_ultima = ahora
                    if ahora - csv_inicio >= csv_dur:
                        self._csv_activa = False
                        guardar_ahora = True
                        nombre_snap    = self._csv_nombre
                        condicion_snap = self._csv_condicion
                        filas_snap     = list(self._csv_filas)
                if guardar_ahora:
                    ruta = self._guardar_csv_disco(nombre_snap, condicion_snap, filas_snap)
                    with self._lock:
                        self._csv_ruta = ruta

        if cap:
            cap.release()
        if grabber:
            grabber.release()


    # ── Sesión CSV ────────────────────────────────────────────────────────────

    def iniciar_csv_sesion(self, nombre: str, condicion: str, minutos: int = 30) -> None:
        with self._lock:
            self._csv_filas.clear()
            self._csv_nombre    = nombre
            self._csv_condicion = condicion
            self._csv_duracion  = minutos * 60
            self._csv_inicio    = time.time()
            self._csv_ultima    = 0.0
            self._csv_minuto    = 0
            self._csv_activa    = True

    def get_csv_estado(self) -> dict:
        with self._lock:
            elapsed = time.time() - self._csv_inicio if self._csv_inicio else 0
            return {
                "activa":              self._csv_activa,
                "nombre":              self._csv_nombre,
                "condicion":           self._csv_condicion,
                "minutos_registrados": self._csv_minuto,
                "filas_acumuladas":    len(self._csv_filas),
                "progreso":            min(100, round(elapsed / max(self._csv_duracion, 1) * 100)),
            }

    def detener_csv_sesion(self) -> Path:
        with self._lock:
            self._csv_activa = False
            nombre    = self._csv_nombre
            condicion = self._csv_condicion
            filas     = list(self._csv_filas)
        ruta = self._guardar_csv_disco(nombre, condicion, filas)
        with self._lock:
            self._csv_ruta = ruta
        return ruta

    def _guardar_csv_disco(self, nombre: str, condicion: str, filas: list[dict]) -> Path:
        import csv
        ts   = time.strftime("%Y%m%d_%H%M%S")
        ruta = SESIONES_DIR / f"{nombre}_{condicion}_{ts}.csv"
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sesion", "condicion", "minuto", "plaza", "prediccion", "ground_truth"],
            )
            writer.writeheader()
            writer.writerows(filas)
        return ruta

    def get_csv_ruta(self) -> Path | None:
        with self._lock:
            return self._csv_ruta


def _calcular_metricas(csv_text: str) -> dict:
    import io
    import csv as csv_mod
    reader = csv_mod.DictReader(io.StringIO(csv_text))
    filas = list(reader)
    if not filas:
        return {"error": "CSV vacío"}

    total     = len(filas)
    correctas = sum(1 for f in filas if f["prediccion"] == f["ground_truth"])
    slot_accuracy = correctas / total

    libres  = [f for f in filas if f["ground_truth"] == "libre"]
    fpr = (
        sum(1 for f in libres if f["prediccion"] != "libre") / len(libres)
        if libres else 0.0
    )

    ocupados = [f for f in filas if f["ground_truth"] == "ocupado"]
    fnr = (
        sum(1 for f in ocupados if f["prediccion"] != "ocupado") / len(ocupados)
        if ocupados else 0.0
    )

    return {
        "slot_accuracy": round(slot_accuracy, 4),
        "fpr":           round(fpr, 4),
        "fnr":           round(fnr, 4),
        "total_filas":   total,
        "n_libres":      len(libres),
        "n_ocupados":    len(ocupados),
    }


service: SmartParkingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = SmartParkingService()
    service.start()
    yield
    service.stop()


app = FastAPI(title="Smart Parking API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def get_status():
    return service.get_status()


@app.get("/snapshot")
def get_snapshot():
    return {"image": service.get_snapshot()}


@app.get("/snapshot/raw")
def get_raw_snapshot():
    return {"image": service.get_raw_snapshot()}


@app.post("/evaluar/iniciar")
def iniciar_evaluacion(segundos: int = 30):
    service.iniciar_eval(segundos)
    return {"ok": True, "duracion": segundos}


@app.get("/evaluar/estado")
def estado_evaluacion():
    return service.get_eval_estado()


@app.post("/auto/iniciar")
def auto_iniciar(minutos: int = 30, intervalo: int = 60):
    service.iniciar_auto(minutos, intervalo)
    return {"ok": True, "minutos": minutos, "intervalo": intervalo}

@app.get("/auto/estado")
def auto_estado():
    return service.get_auto_estado()

@app.get("/auto/captura/{idx}")
def auto_captura(idx: int):
    c = service.get_auto_captura(idx)
    if c is None:
        from fastapi import HTTPException
        raise HTTPException(404, "Captura no encontrada")
    return c

@app.post("/auto/anotar/{idx}")
def auto_anotar(idx: int, ground_truth: dict):
    return service.anotar_captura(idx, ground_truth)

@app.get("/auto/reporte")
def auto_reporte():
    return service.get_reporte_auto()

@app.delete("/auto")
def auto_limpiar():
    service.limpiar_auto()
    return {"ok": True}


@app.get("/campo/captura")
def campo_captura():
    return service.captura_actual()


@app.post("/campo/muestra")
def campo_muestra(ground_truth: dict):
    return service.registrar_muestra(ground_truth)


@app.get("/campo/reporte")
def campo_reporte():
    return service.get_reporte_campo()


@app.delete("/campo/muestras")
def campo_limpiar():
    service.limpiar_muestras()
    return {"ok": True}


@app.get("/metricas-modelo")
def get_metricas_modelo():
    results_path = BASE / "modelos/yolov8s/resultados/results.csv"
    try:
        with open(results_path) as f:
            lines = f.readlines()
        headers  = [h.strip() for h in lines[0].split(",")]
        last_row = [v.strip() for v in lines[-1].split(",")]
        data = dict(zip(headers, last_row))
        return {
            "map50":    round(float(data["metrics/mAP50(B)"]), 4),
            "map50_95": round(float(data["metrics/mAP50-95(B)"]), 4),
        }
    except Exception:
        return {"map50": None, "map50_95": None}


# ── Sesión CSV ──────────────────────────────────────────────────────────────

@app.post("/sesion/iniciar")
def iniciar_sesion(nombre: str, condicion: str, minutos: int = 30):
    service.iniciar_csv_sesion(nombre, condicion, minutos)
    return {"ok": True, "nombre": nombre, "condicion": condicion, "minutos": minutos}


@app.get("/sesion/estado")
def estado_sesion():
    return service.get_csv_estado()


@app.post("/sesion/detener")
def detener_sesion():
    ruta = service.detener_csv_sesion()
    return {"ok": True, "archivo": str(ruta)}


@app.get("/sesion/metricas")
def metricas_sesion():
    ruta = service.get_csv_ruta()
    if ruta is None or not ruta.exists():
        from fastapi import HTTPException
        raise HTTPException(404, "No hay CSV guardado. Inicia y detén una sesión primero.")
    return _calcular_metricas(ruta.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(service.get_status())
            await asyncio.sleep(1)
    except (WebSocketDisconnect, Exception):
        pass

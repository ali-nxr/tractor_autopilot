"""
yolo_detector.py — lightweight object detection (YOLO26n), GPU-only,
informational overlay ONLY. Never touches braking, steering, or the
perception watchdog — those remain pure-geometry decisions from
ground_segmentation.py / obstacle_decision.py, completely unchanged. This
exists purely to draw "here's a person/animal/vehicle" boxes on the video
feed using spare GPU capacity that the rest of this app (CPU-bound
segmentation/obstacle math) never touches at all.

Loading order, falling back automatically at ANY failure (never raises out
of AgriYoloDetector.__init__ — .available tells the caller whether it
worked):
  1. A cached TensorRT engine (fastest — GPU-specific, built once)
  2. Exporting a fresh TensorRT engine from the .pt weights (slow,
     one-time — only happens if no cached engine exists yet)
  3. Plain CUDA inference on the .pt weights (still GPU, just without the
     extra TensorRT speedup)
  4. CPU (last resort — functional, just slow)

Precision handling — this matters, and got it wrong once before in this
project's history: TensorRT engines bake their precision in AT EXPORT TIME
(half=True passed to export(), never to predict() afterward — the engine's
precision is already fixed by the time it's loaded). The plain-CUDA .pt
fallback path is different and needs half=True passed to EVERY predict()
call instead — and critically must NOT ALSO separately .half() the model
object first. Doing both at once left some layer/buffer still FP32 while
predict() assumed the whole model was FP16, producing exactly this crash,
confirmed from a real log earlier in this project:
    "expected mat1 and mat2 to have the same dtype... c10::Half != float"
Only ONE of those two precision paths is ever used per mode here, never
both at once.
"""

import os

import numpy as np
import cv2

import config

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _cuda_available():
    if not _TORCH_AVAILABLE:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _limit_torch_cpu_threads():
    """
    Caps PyTorch's own internal CPU threading (see config.YOLO_TORCH_CPU_THREADS)
    — measured on real hardware to matter: PyTorch defaults to using every
    available core for its own CPU-side ops (preprocessing/postprocessing
    around the actual GPU inference), competing with the main vision
    loop's numpy/OpenCV work for real CPU cycles. Best-effort — never
    raises, since this is a performance tweak, not something that should
    be able to prevent detection from loading at all if it fails.
    """
    if not _TORCH_AVAILABLE:
        return
    try:
        torch.set_num_threads(config.YOLO_TORCH_CPU_THREADS)
        print(f"[yolo] torch.set_num_threads({config.YOLO_TORCH_CPU_THREADS})")
    except Exception as e:
        print(f"[yolo] Could not limit torch CPU threads (non-fatal): {e}")


def _engine_path_for(weights_path):
    """The path ultralytics' own export(format='engine') writes to by
    default: same directory, same filename stem, .engine extension. Derived
    from YOLO_WEIGHTS_PATH rather than a second, separately-maintained
    config constant, so the "does a cached engine already exist" check can
    never silently drift out of sync with where export() actually put it."""
    stem, _ = os.path.splitext(weights_path)
    return stem + ".engine"


class AgriYoloDetector:
    def __init__(self):
        self.available = False
        self.error = None
        self.model = None
        self.mode = "unavailable"     # "tensorrt" | "cuda" | "cpu" | "unavailable"
        self.device = "cpu"
        self._half = False

        if not config.YOLO_ENABLED:
            self.error = "disabled in config (YOLO_ENABLED=False)"
            return
        if not _ULTRALYTICS_AVAILABLE:
            self.error = "ultralytics not installed — run: pip install ultralytics"
            return

        _limit_torch_cpu_threads()

        using_cuda = _cuda_available()

        if using_cuda and config.YOLO_USE_TENSORRT:
            if self._try_load_tensorrt():
                return  # success — self.available already set inside

        # TensorRT unavailable, disabled, or its load/export failed for any
        # reason — fall back to plain CUDA (or CPU if no GPU at all) on the
        # original .pt weights. A FULL independent attempt, not a
        # continuation of a half-failed TensorRT state.
        self._try_load_plain(using_cuda)

    def _try_load_tensorrt(self):
        try:
            engine_path = _engine_path_for(config.YOLO_WEIGHTS_PATH)
            if os.path.exists(engine_path):
                print(f"[yolo] Loading cached TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
            else:
                print("=" * 72)
                print("[yolo] No cached TensorRT engine found — building one now.")
                print("[yolo] This is a ONE-TIME, GPU-specific compile step: roughly")
                print("[yolo] 1-3 minutes, and TensorRT's own compiler prints NOTHING")
                print("[yolo] for long stretches in the middle of it — that is normal,")
                print("[yolo] not a hang. PLEASE DO NOT PRESS CTRL+C OR CLOSE THIS")
                print("[yolo] WINDOW during this step: interrupting it partway through")
                print("[yolo] means the finished engine never gets saved, and the next")
                print("[yolo] run has to redo the entire build from scratch.")
                print("=" * 72)
                base_model = YOLO(config.YOLO_WEIGHTS_PATH)
                exported_path = base_model.export(
                    format="engine", device=0, half=config.YOLO_USE_HALF_PRECISION,
                )
                print(f"[yolo] TensorRT engine exported and saved: {exported_path}")
                self.model = YOLO(exported_path)

            # Warm-up inference — first real call on a fresh engine/context
            # pays a one-time init cost; pay it here, not on frame 1.
            dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            self.model.predict(dummy, device=0, verbose=False)

            self.mode = "tensorrt"
            self.device = "cuda"
            self.available = True
            print("[yolo] Running on TensorRT (GPU)")
            return True
        except KeyboardInterrupt:
            # Deliberately NOT caught by "except Exception" below (it
            # doesn't inherit from Exception, on purpose — Python doesn't
            # silently swallow Ctrl+C). Re-raised, stopping the app as the
            # person actually asked — but with a clear explanation first,
            # since a raw traceback here looks like a crash when it isn't
            # one: this is confirmed, from a real run, to happen most
            # often by interrupting the TensorRT build partway through
            # (it goes quiet for over a minute mid-compile, which reads as
            # "stuck" even when it's working) — the fix is waiting it out
            # next time, not code.
            print("[yolo] Interrupted (Ctrl+C) during the TensorRT build.")
            print("[yolo] If the engine build was still in progress, it did NOT")
            print("[yolo] finish saving — the next run has to redo it from")
            print("[yolo] scratch. This is not a crash in this application.")
            raise
        except Exception as e:
            print(f"[yolo] TensorRT path failed ({e}) — falling back to plain CUDA.")
            self.model = None
            return False

    def _try_load_plain(self, using_cuda):
        try:
            self.model = YOLO(config.YOLO_WEIGHTS_PATH)
            self.device = "cuda" if using_cuda else "cpu"
            self.model.to(self.device)
            # Deliberately NOT calling self.model.model.half() here — see
            # this module's docstring. half=True goes to predict() alone,
            # every call; that combination is the one that doesn't crash.
            self._half = using_cuda and config.YOLO_USE_HALF_PRECISION

            dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, half=self._half, verbose=False)

            self.mode = "cuda" if using_cuda else "cpu"
            self.available = True
            print(f"[yolo] Running on {'CUDA (no TensorRT)' if using_cuda else 'CPU'}")
        except Exception as e:
            self.error = f"failed to load '{config.YOLO_WEIGHTS_PATH}': {e}"
            self.model = None
            self.available = False

    def detect(self, frame_bgr):
        """
        Returns a list of {class_name, confidence, bbox=(x, y, w, h)},
        filtered to config.YOLO_ALLOWED_CLASSES. Empty list (never an
        exception) on ANY inference-time failure — a detection hiccup must
        never be able to take down the caller reading this.
        """
        if not self.available:
            return []
        try:
            if self.mode == "tensorrt":
                # No half= here — see this module's docstring: the
                # engine's precision was fixed at export time.
                results = self.model.predict(
                    frame_bgr, conf=config.YOLO_CONF_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD,
                    device=0, verbose=False,
                )
            else:
                results = self.model.predict(
                    frame_bgr, conf=config.YOLO_CONF_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD,
                    device=self.device, half=self._half, verbose=False,
                )
        except Exception as e:
            print(f"[yolo] inference error (skipping this frame): {e}")
            return []

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if class_name not in config.YOLO_ALLOWED_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            })
        return detections


def draw_detections(frame, detections):
    """Draws YOLO boxes directly onto frame, in place — a color distinct
    from every other overlay element (green land, red/orange obstacles,
    amber corridor) so detections read as a clearly separate layer."""
    for det in detections:
        x, y, w, h = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_YOLO_BOX, 2)
        cv2.putText(frame, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    config.COLOR_YOLO_BOX, 1, cv2.LINE_AA)
    return frame

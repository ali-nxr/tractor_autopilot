"""
segformer_onnx.py — SegFormer semantic segmentation on onnxruntime,
producing a per-pixel "driveable ground" probability map that guides
ground_segmentation.py (see config_segformer.py for how it is used).

The model file comes from tools/export_segformer_onnx.py. Preprocessing,
softmax and the driveable-class reduction all live inside that graph, so
per frame this module does one cv2.resize in, one session.run, and one
cv2.resize out.

Provider order, falling back automatically (never raises out of
SegformerOnnx.__init__ — .available tells the caller whether it worked,
same contract as yolo_detector.AgriYoloDetector):
  1. TensorRT (fastest; engine built + cached on first run)
  2. CUDA
  3. CPU
"""

import ctypes.util
import json
import os
import time

import numpy as np
import cv2

import config_segformer as scfg

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


def _preload_gpu_dlls():
    """
    onnxruntime-gpu needs the CUDA/cuDNN DLLs on the search path, and on
    Windows they are usually NOT there unless a CUDA toolkit is installed
    system-wide. ort.preload_dlls() (ORT >= 1.21) finds them in the
    nvidia-* pip packages / torch install instead. Best-effort: without it
    ORT simply falls back to the CPU provider, which _create_session reports.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as e:
        print(f"[segformer] ort.preload_dlls() failed (non-fatal): {e}")


def _tensorrt_libs_present():
    """
    onnxruntime-gpu always LISTS the TensorRT provider, but it only works
    if NVIDIA's TensorRT runtime (nvinfer) is installed separately. Without
    this check, ORT prints a long DLL-load error on every start before
    falling back to CUDA on its own.
    """
    return any(ctypes.util.find_library(name) for name in ("nvinfer_10", "nvinfer"))


def _provider_list():
    available = set(ort.get_available_providers())
    providers = []
    if (scfg.SEGFORMER_USE_TENSORRT and "TensorrtExecutionProvider" in available
            and not _tensorrt_libs_present()):
        print("[segformer] TensorRT runtime (nvinfer) not found on PATH — skipping TensorRT, "
              "using CUDA. See requirements.txt to enable it.")
    elif scfg.SEGFORMER_USE_TENSORRT and "TensorrtExecutionProvider" in available:
        os.makedirs(scfg.SEGFORMER_TRT_CACHE_DIR, exist_ok=True)
        providers.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": bool(scfg.SEGFORMER_TRT_FP16),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": scfg.SEGFORMER_TRT_CACHE_DIR,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": scfg.SEGFORMER_TRT_CACHE_DIR,
        }))
    if scfg.SEGFORMER_USE_CUDA and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class SegformerOnnx:
    def __init__(self):
        self.available = False
        self.error = None
        self.mode = "unavailable"        # "tensorrt" | "cuda" | "cpu" | "unavailable"
        self.session = None
        self.input_size = None           # (w, h) the model was exported at
        self.last_infer_ms = 0.0
        self.id2label = {}
        # Debug view only (want_class_ids=True): the latest per-pixel argmax
        # ADE20K class at the model's own output resolution (H/4, W/4), or
        # None. Models exported before the class_ids output existed simply
        # never fill it.
        self.last_class_ids = None
        self.has_class_ids = False

        self._class_weights = None
        self._frame_index = 0
        self._cached = None              # (out_w, out_h, prob, class_ids) from the last real inference

        if not scfg.SEGFORMER_ENABLED:
            self.error = "disabled in config_segformer.py (SEGFORMER_ENABLED=False)"
            return
        if not _ORT_AVAILABLE:
            self.error = "onnxruntime not installed — run: pip install onnxruntime-gpu"
            return
        if not os.path.exists(scfg.SEGFORMER_MODEL_PATH):
            self.error = (f"model not found: {scfg.SEGFORMER_MODEL_PATH} — create it once with: "
                          f"python tools/export_segformer_onnx.py")
            return

        try:
            self._create_session()
            self._load_metadata()
            self._warm_up()
        except Exception as e:
            self.error = f"failed to load '{scfg.SEGFORMER_MODEL_PATH}': {e}"
            self.session = None
            return

        self.available = True
        w, h = self.input_size
        print(f"[segformer] Running on {self.mode.upper()} — {w}x{h} input, "
              f"warm-up {self.last_infer_ms:.1f} ms/frame")

    # ------------------------------------------------------------ loading

    def _create_session(self):
        _preload_gpu_dlls()
        providers = _provider_list()

        if any(p[0] == "TensorrtExecutionProvider" for p in providers if isinstance(p, tuple)):
            print("[segformer] TensorRT enabled. The FIRST run builds an engine (can take a few")
            print("[segformer] minutes with no output) and caches it in "
                  f"{scfg.SEGFORMER_TRT_CACHE_DIR} — later runs load it in seconds.")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = scfg.SEGFORMER_CPU_THREADS
        options.inter_op_num_threads = 1
        options.log_severity_level = 3   # errors only — ORT's warnings are per-node noise here

        self.session = ort.InferenceSession(scfg.SEGFORMER_MODEL_PATH, sess_options=options,
                                            providers=providers)
        # What ORT actually got, not what was asked for: a provider whose
        # DLLs fail to load is silently dropped at session creation.
        active = self.session.get_providers()[0]
        self.mode = {"TensorrtExecutionProvider": "tensorrt",
                     "CUDAExecutionProvider": "cuda"}.get(active, "cpu")
        if self.mode == "cpu" and len(providers) > 1:
            print("[segformer] GPU providers requested but unavailable at runtime — on CPU. "
                  "Check onnxruntime-gpu matches your CUDA/cuDNN version.")

    def _load_metadata(self):
        meta = self.session.get_modelmeta().custom_metadata_map
        if "id2label" not in meta:
            raise ValueError("model has no id2label metadata — re-export it with "
                             "tools/export_segformer_onnx.py")
        id2label = {int(k): v for k, v in json.loads(meta["id2label"]).items()}
        self.id2label = id2label
        label2id = {v: k for k, v in id2label.items()}

        unknown = [name for name in scfg.DRIVEABLE_CLASSES if name not in label2id]
        if unknown:
            raise ValueError(f"DRIVEABLE_CLASSES not in this model's labels: {unknown}")
        unknown = [name for name in scfg.SKY_CLASSES if name not in label2id]
        if unknown:
            raise ValueError(f"SKY_CLASSES not in this model's labels: {unknown}")
        self._sky_ids = np.array([label2id[name] for name in scfg.SKY_CLASSES])

        weights = np.zeros(len(id2label), dtype=np.float32)
        for name in scfg.DRIVEABLE_CLASSES:
            weights[label2id[name]] = 1.0
        self._class_weights = weights

        _, h, w, _ = self.session.get_inputs()[0].shape
        self.input_size = (int(w), int(h))
        self.has_class_ids = any(o.name == "class_ids" for o in self.session.get_outputs())
        if scfg.SEMANTIC_SKY_MASK_ENABLED and not self.has_class_ids:
            print("[segformer] SEMANTIC_SKY_MASK_ENABLED needs the class_ids model output — "
                  "re-export with tools/export_segformer_onnx.py; sky depth is NOT being masked")

    def _warm_up(self):
        # First call on a fresh session pays allocator / kernel-selection
        # cost (and for TensorRT, the engine build) — pay it here, not on frame 1.
        w, h = self.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._run(dummy)
        self._run(dummy)

    # ------------------------------------------------------------ inference

    def _run(self, frame_bgr_resized, want_class_ids=False):
        outputs = ["driveable_prob"]
        if want_class_ids and self.has_class_ids:
            outputs.append("class_ids")
        t0 = time.perf_counter()
        result = self.session.run(outputs, {
            "image_bgr": frame_bgr_resized[None],
            "class_weights": self._class_weights,
        })
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        class_ids = result[1][0] if len(result) > 1 else None
        return result[0][0], class_ids

    def sky_mask(self, out_size):
        """
        (h, w) bool mask of pixels the model labels as sky
        (config_segformer.SKY_CLASSES) at out_size = (w, h), whose depth the
        caller should discard — see SEMANTIC_SKY_MASK_ENABLED for why. None
        when the mask is disabled, the model is unavailable, or this frame has
        no class map (inference failed): never guesses from a stale map.
        """
        if not (scfg.SEMANTIC_SKY_MASK_ENABLED and self.available) or self.last_class_ids is None:
            return None
        sky = np.isin(self.last_class_ids, self._sky_ids).astype(np.uint8)
        return cv2.resize(sky, tuple(out_size), interpolation=cv2.INTER_NEAREST) > 0

    def reset(self):
        """Drop the cached map — call on a playback seek/loop so the next
        frame is never served a probability map from a different scene."""
        self._frame_index = 0
        self._cached = None
        self.last_class_ids = None

    def driveable_probability(self, frame_bgr, out_size, want_class_ids=False):
        """
        frame_bgr : (H, W, 3) uint8, the camera color frame (any resolution)
        out_size  : (w, h) of the returned map — the processing resolution
                    the depth segmentation runs at.
        want_class_ids : also fetch the argmax class map into
                    self.last_class_ids (the debug view). Always fetched
                    when SEMANTIC_SKY_MASK_ENABLED (sky_mask needs it).

        Returns (h, w) float32 in [0, 1], or None if the model is
        unavailable or this frame's inference failed — the caller then runs
        pure-depth segmentation, exactly as before this model existed.
        """
        if not self.available:
            return None

        want_class_ids = want_class_ids or scfg.SEMANTIC_SKY_MASK_ENABLED
        run_every = max(1, int(scfg.SEGFORMER_RUN_EVERY_N_FRAMES))
        reuse = (self._cached is not None
                 and self._frame_index % run_every != 0
                 and self._cached[:2] == tuple(out_size))
        self._frame_index += 1
        if reuse:
            self.last_class_ids = self._cached[3]
            return self._cached[2]

        try:
            # Bilinear, not INTER_AREA: measured ~0.5 ms vs ~5 ms on a
            # 1280x720 frame, with no visible change in the output map.
            resized = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
            prob_small, class_ids = self._run(np.ascontiguousarray(resized), want_class_ids)
        except Exception as e:
            print(f"[segformer] inference error (pure-depth segmentation this frame): {e}")
            self.last_class_ids = None
            return None
        self.last_class_ids = class_ids

        # Bilinear: the model's output is 1/4 of its input — a smooth
        # upsample of a probability (not of class IDs) gives smooth edges.
        prob = cv2.resize(prob_small, tuple(out_size), interpolation=cv2.INTER_LINEAR)
        np.clip(prob, 0.0, 1.0, out=prob)
        self._cached = (out_size[0], out_size[1], prob, class_ids)
        return prob

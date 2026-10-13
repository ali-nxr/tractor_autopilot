"""
lane_assist/models/openpilot_supercombo.py — comma.ai openpilot's driving
model (supercombo) as a lane backend.

Everything model-specific is reused from the comma-operpilot project
(config_lane.LANE_MODELS["openpilot_supercombo"]["root"]): its
SupercomboRunner (ONNX session + recurrent state, as openpilot's modeld),
ModelFramePacker (openpilot's warp + 6-plane YUV packing) and openpilot's
own vendored transformations / output Parser. This file only adapts our
camera (RealSense intrinsics + ground-plane calibration) to it and turns
the parsed output into a LaneEstimate.

Two adaptations matter on a tractor:
  - Calibration: openpilot expects the camera's orientation relative to
    the road; LaneCamera supplies it from the fitted ground plane every
    frame (or from config), converted to openpilot's device_from_calib
    euler angles with openpilot's own rot2euler.
  - Scale: the network was trained on comma's mount (~1.2 m up). A camera
    at another height sees the same image of a road that is really
    (true height / model height) times bigger, so every metric output is
    rescaled by that ratio — exact for points on flat ground, which move
    along their own camera rays. The model's height is read from its own
    road-surface output (road-edge z), not assumed.
"""

import math
import os
import sys

import numpy as np
import cv2

from lane_assist.base import (EDGE_LEFT, EDGE_RIGHT, LANE_LEFT, LANE_RIGHT, Boundary, LaneEstimate,
                              LaneModel)


class _Calibration:
    """What ModelFramePacker needs: warp_matrix(wide) for the current camera."""

    def __init__(self, get_warp_matrix, euler, K):
        self._get_warp_matrix, self.euler, self.K = get_warp_matrix, euler, K

    def warp_matrix(self, wide):
        return self._get_warp_matrix(np.asarray(self.euler, dtype=np.float32), self.K, wide).astype(np.float32)


class OpenpilotSupercombo(LaneModel):
    name = "openpilot_supercombo"

    def __init__(self, settings, camera):
        super().__init__(settings, camera)
        root = settings["root"]
        if not os.path.isdir(root):
            self.error = f"comma-operpilot project not found at {root} (config_lane LANE_MODELS root)"
            return
        for p in (root, os.path.join(root, "vendor")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from op_video.runner import SupercomboRunner          # registers the CUDA DLLs first
            from op_video.video import ModelFramePacker
            from openpilot.common.transformations.camera import view_frame_from_device_frame
            from openpilot.common.transformations.model import get_warp_matrix
            from openpilot.common.transformations.orientation import rot2euler
            from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

            model_path = settings.get("model_path") or os.path.join(root, "models", "driving_supercombo.onnx")
            self._runner = SupercomboRunner(model_path, providers=list(settings["providers"]))
        except Exception as e:
            self.error = f"openpilot supercombo failed to load: {type(e).__name__}: {e}"
            return

        self._Packer = ModelFramePacker
        self._get_warp_matrix = get_warp_matrix
        self._rot2euler = rot2euler
        self._device_from_view = view_frame_from_device_frame.T
        self._X = np.asarray(ModelConstants.X_IDXS, dtype=float)
        self._Plan = Plan
        self._rate = float(settings["rate_hz"])
        self.mode = "cuda" if "CUDAExecutionProvider" in self._runner.providers else "cpu"
        self.available = True
        self.reset()
        print(f"[lane] openpilot supercombo on {self.mode.upper()} "
              f"(checkpoint {self._runner.checkpoint}, {self._rate:g} Hz ticks)", flush=True)

    def reset(self):
        if not self.available:
            return
        self._runner.reset()
        self._packer, self._packer_euler = None, None
        self._next_tick, self._last_t, self._frame_dt = None, None, None
        self._scale = None
        self._last = None

    # -------------------------------------------------------------- inference

    def _due(self, t):
        """Sub-sample frames to the network's trained tick rate, on the
        frame clock (a 30 fps stream -> ~20 Hz ticks; slower -> every frame)."""
        if self._last_t is not None:
            dt = t - self._last_t
            if dt <= 0 or dt > 1.0:                 # clock jumped: start over
                self.reset()
            else:
                self._frame_dt = dt if self._frame_dt is None else 0.8 * self._frame_dt + 0.2 * dt
        self._last_t = t
        period = 1.0 / self._rate
        if self._next_tick is None:
            self._next_tick = t + period
            return True
        if t >= self._next_tick - 0.5 * (self._frame_dt or 0.0):
            # keep an average of `rate` ticks/s; resync if we fell far behind
            self._next_tick = self._next_tick + period if t - self._next_tick < period else t + period
            return True
        return False

    def infer(self, frame_bgr, t):
        if not self.available or not self.camera.ready:
            return None
        if not self._due(t):
            return self._last

        V = self.camera.view_from_road()
        euler = self._rot2euler(self._device_from_view @ V)
        if self._packer is None or np.max(np.abs(euler - self._packer_euler)) > math.radians(0.2):
            self._packer = self._Packer(_Calibration(self._get_warp_matrix, euler, self.camera.K))
            self._packer_euler = euler

        h, w = frame_bgr.shape[:2]
        i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
        y = i420[:h]
        u = i420[h:h + h // 4].reshape(h // 2, w // 2)
        v = i420[h + h // 4:].reshape(h // 2, w // 2)
        out = self._runner.run(self._packer.pack(y, u, v, False), self._packer.pack(y, u, v, True),
                               is_rhd=bool(self.settings.get("rhd")))
        self._last = self._to_estimate(out, t)
        return self._last

    # ----------------------------------------------------------------- output

    def _to_estimate(self, out, t):
        s = self.settings
        X = self._X
        edges, edge_std = out["road_edges"][0], out["road_edges_stds"][0]      # (2,33,2) y,z
        lanes, lane_prob = out["lane_lines"][0], out["lane_lines_prob"][0, 1::2]  # (4,33,2), (4,)

        # metric scale: true camera height / the model's own road height
        near = (X >= 4.0) & (X <= 20.0)
        model_h = float(np.median(edges[:, near, 1]))
        if not (0.3 < model_h < 5.0):
            model_h = float(s["model_camera_height_m"])
        scale = 1.0
        if s.get("height_scaling", True) and self.camera.height_m:
            target = float(np.clip(self.camera.height_m / model_h, 0.25, 4.0))
            self._scale = target if self._scale is None else 0.8 * self._scale + 0.2 * target
            scale = self._scale

        def edge_conf(k):
            std = float(np.mean(edge_std[k, (X >= 3.0) & (X <= 25.0), 0])) * scale
            return float(np.clip(1.0 - std / s["edge_std_full_m"], 0.0, 1.0))

        boundaries = {
            LANE_LEFT: Boundary(lanes[1, :, 0] * scale, float(lane_prob[1])),
            LANE_RIGHT: Boundary(lanes[2, :, 0] * scale, float(lane_prob[2])),
            EDGE_LEFT: Boundary(edges[0, :, 0] * scale, edge_conf(0)),
            EDGE_RIGHT: Boundary(edges[1, :, 0] * scale, edge_conf(1)),
        }

        # planned path: positions at time steps -> lateral offset at our stations
        plan = out["plan"][0]
        pos = plan[:, self._Plan.POSITION] * scale
        plan_std = out["plan_stds"][0][:, self._Plan.POSITION][:, 1] * scale
        px, py = pos[:, 0], pos[:, 1]
        keep = np.concatenate([[True], np.diff(px) > 1e-3]) & (px >= 0)
        Xs = X * scale
        plan_y = np.full_like(Xs, np.nan)
        if keep.sum() >= 3:
            covered = Xs <= px[keep].max()
            plan_y[covered] = np.interp(Xs[covered], px[keep], py[keep])
        plan_conf = float(np.clip(1.0 - np.mean(plan_std[:17]) / s["plan_std_full_m"], 0.0, 1.0))

        v_ego = float(out["pose"][0][0]) * scale
        return LaneEstimate(
            t=t, x=Xs, boundaries=boundaries, plan_y=plan_y, plan_conf=plan_conf, model_speed_mps=v_ego,
            info={"scale": round(scale, 3), "model_road_height_m": round(model_h, 2),
                  "lane_probs": [round(float(p), 2) for p in lane_prob],
                  "edge_conf": [round(boundaries[EDGE_LEFT].conf, 2), round(boundaries[EDGE_RIGHT].conf, 2)]},
        )

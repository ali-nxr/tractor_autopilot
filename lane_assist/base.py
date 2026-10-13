"""
lane_assist/base.py — what every lane model produces, and how models are
plugged in.

A model backend turns one camera frame into a LaneEstimate: lateral
positions of lane lines / track edges / its planned path at stations ahead,
in the ROAD frame (see camera.py: x forward along the ground, y right,
z down; origin at the camera; metres), each with a 0..1 confidence. All
downstream logic (corridor choice, lock, steering, path, drawing) only sees
LaneEstimate, so models are interchangeable.

Adding a model:
    1. subclass LaneModel in lane_assist/models/<name>.py, implement infer()
    2. add an entry to config_lane.LANE_MODELS with "backend": "module:Class"
       plus whatever settings the backend reads
    3. set config_lane.LANE_MODEL to that entry's name
"""

import importlib
from dataclasses import dataclass, field

import numpy as np

# Boundary names a backend may fill (any subset).
LANE_LEFT, LANE_RIGHT = "lane_left", "lane_right"     # ego lane lines
EDGE_LEFT, EDGE_RIGHT = "edge_left", "edge_right"     # track / road edges


@dataclass
class Boundary:
    y: np.ndarray          # (N,) lateral position (m, + right) at LaneEstimate.x
    conf: float            # 0..1


@dataclass
class LaneEstimate:
    t: float                                   # frame time this came from
    x: np.ndarray                              # (N,) stations, metres ahead, increasing
    boundaries: dict = field(default_factory=dict)   # name -> Boundary
    plan_y: np.ndarray = None                  # (N,) model's planned path (NaN where not covered)
    plan_conf: float = 0.0
    model_speed_mps: float = None
    info: dict = field(default_factory=dict)   # backend diagnostics (shown in logs/debug)


class LaneModel:
    """Base class for lane model backends."""

    name = "base"

    def __init__(self, settings, camera):
        self.settings = settings
        self.camera = camera            # lane_assist.camera.LaneCamera
        self.available = False
        self.error = None
        self.mode = "unavailable"       # e.g. "cuda" / "cpu", for status display

    def infer(self, frame_bgr, t):
        """One frame -> LaneEstimate, or None (not calibrated yet / failed).
        Called for EVERY pipeline frame; a backend with its own tick rate
        may return its last estimate in between."""
        raise NotImplementedError

    def reset(self):
        """Forget temporal state (playback seek / loop, or assist toggled off)."""


def create_model(name, settings, camera):
    """Instantiate config_lane.LANE_MODELS[name]. Never raises — check .available."""
    spec = settings.get("backend")
    try:
        module_name, class_name = spec.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(settings, camera)
    except Exception as e:
        model = LaneModel(settings, camera)
        model.error = f"could not load lane model '{name}' ({spec}): {type(e).__name__}: {e}"
        return model

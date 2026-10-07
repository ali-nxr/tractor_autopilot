"""
shared_state.py — thread-safe holder for the latest processed frame +
decision data. The vision worker thread writes to it every frame, the UI
thread reads from it every tick. Nothing here beyond what the core system
actually needs: segmentation, IMU, obstacles, path, brake, steer.
"""

import threading
import time
from collections import deque

import config


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()

        # --- live video + core decision numbers ---
        self.annotated_frame = None      # BGR numpy array, ready to display
        self.speed_mps = 0.0
        self.brake_percent = 0.0
        self.steer_suggestion_deg = 0.0
        self.land_coverage_pct = 0.0
        self.fps = 0.0
        self.status_text = "starting..."

        # --- obstacles (see obstacle_decision.py / object_tracker.py) ---
        self.obstacles = []              # list of {bbox, distance_m, centroid_x_m, area_px, track_id, closing_speed_mps}
        self.nearest_obstacle_m = None
        self.closing_speed_mps = 0.0

        # --- drivable path + safety watchdog ---
        self.path_blocked_at_m = None    # float or None — None means clear ahead
        self.watchdog_degraded = False
        self.degraded_duration_s = 0.0

        # --- IMU (pitch/roll/yaw/accel/gyro) ---
        self.imu = {
            "available": False,
            "status": "not started",
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,          # relative heading only — drifts, no magnetometer
            "tilt_deg": 0.0,         # combined lean off vertical
            "accel_mps2": (0.0, 0.0, 0.0),
            "gyro_dps": (0.0, 0.0, 0.0),
            "rollover_risk": "ok",   # "ok" | "warning" | "danger"
        }

        # --- object detection (see yolo_detector.py) — informational
        # overlay only, never touches brake/steer/watchdog ---
        self.yolo_detections = []
        self.yolo_status = "not started"

        # --- rolling history for small live trend sparklines ---
        n = config.HISTORY_LENGTH
        self.history = {
            "t": deque(maxlen=n),
            "speed_kmh": deque(maxlen=n),
            "brake_pct": deque(maxlen=n),
            "nearest_m": deque(maxlen=n),
            "tilt_deg": deque(maxlen=n),
        }
        self._t0 = time.time()
        self._last_update = time.time()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            now = time.time()
            dt = now - self._last_update
            if dt > 0:
                self.fps = 1.0 / dt
            self._last_update = now

            t = now - self._t0
            if "speed_mps" in kwargs or "brake_percent" in kwargs:
                self.history["t"].append(t)
                self.history["speed_kmh"].append(self.speed_mps * 3.6)
                self.history["brake_pct"].append(self.brake_percent)
                self.history["nearest_m"].append(
                    self.nearest_obstacle_m if self.nearest_obstacle_m is not None else float("nan")
                )
                self.history["tilt_deg"].append(self.imu.get("tilt_deg", 0.0))

    def update_imu(self, imu_dict):
        with self._lock:
            self.imu = imu_dict

    def update_yolo(self, detections, status=None):
        with self._lock:
            self.yolo_detections = detections
            if status is not None:
                self.yolo_status = status

    def get_yolo_detections(self):
        """Thread-safe snapshot — called from the vision thread every
        frame to draw the LATEST available detections (which usually lag
        the current frame slightly, since detection runs asynchronously
        on its own thread; see main.py's YoloWorker)."""
        with self._lock:
            return list(self.yolo_detections)

    def snapshot(self):
        with self._lock:
            return {
                "annotated_frame": self.annotated_frame,
                "speed_mps": self.speed_mps,
                "brake_percent": self.brake_percent,
                "steer_suggestion_deg": self.steer_suggestion_deg,
                "land_coverage_pct": self.land_coverage_pct,
                "fps": self.fps,
                "status_text": self.status_text,
                "obstacles": list(self.obstacles),
                "nearest_obstacle_m": self.nearest_obstacle_m,
                "closing_speed_mps": self.closing_speed_mps,
                "path_blocked_at_m": self.path_blocked_at_m,
                "watchdog_degraded": self.watchdog_degraded,
                "degraded_duration_s": self.degraded_duration_s,
                "imu": dict(self.imu),
                "yolo_detections": list(self.yolo_detections),
                "yolo_status": self.yolo_status,
            }

    def snapshot_history(self):
        with self._lock:
            return {k: list(v) for k, v in self.history.items()}

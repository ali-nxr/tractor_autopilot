"""
speed_estimator.py — estimates the tractor's forward speed, fusing the
camera (optical flow + depth) with the IMU accelerometer when available.

Vision-only piece (always available, and the drift-free reference):
  Pick texture points on the segmented ground, track them frame-to-frame with
  Lucas-Kanade optical flow, and use the RealSense depth at each tracked
  point (before and after) to see how much closer that point got. For a
  static world point and a camera translating forward, the point's
  camera-frame Z decreases by exactly the distance the camera moved:

      forward_speed_m_s = median(z_prev - z_curr over tracked points) / dt

  Using ground points only keeps the estimate tied to points we trust are
  static and flat. Median (not mean) makes it robust to the odd mistracked
  point. This part alone is what V1 shipped with.

IMU fusion (new): raw accelerometer integration for velocity is not usable
on its own — small bias errors accumulate into the integral and the speed
estimate runs away within seconds, the same problem the pitch/roll
complementary filter in realsense_imu.py exists to solve, just one
derivative up. So instead of replacing vision with IMU, this fuses them the
same way: the IMU's forward-axis LINEAR acceleration (gravity-compensated
using the fused pitch/roll from the same IMU) is integrated frame-to-frame
for a fast, high-resolution delta-v between/around vision samples, and that
integrated estimate is continuously leaked back toward the vision speed
(config.SPEED_IMU_VISION_CORRECTION_RATE controls how fast) so accelerometer
bias can never run away — vision remains the ground truth, IMU just makes
the reading more responsive. If the IMU isn't available this transparently
degrades to vision-only, matching V1 behavior exactly.
"""

import math

import numpy as np
import cv2

import config

_LK_PARAMS = dict(
    winSize=config.LK_WIN_SIZE,
    maxLevel=config.LK_MAX_PYRAMID_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

_G_MPS2 = 9.81


class SpeedEstimator:
    def __init__(self):
        self.prev_gray = None
        self.prev_xyz = None
        self.prev_pts = None
        self.smoothed_speed_mps = 0.0
        self._imu_velocity_mps = 0.0

    def update(self, color_image, xyz, land_mask, dt, imu_state=None):
        """
        color_image : (H, W, 3) BGR, current frame
        xyz         : (H, W, 3) float32 meters, current frame's per-pixel 3D points
        land_mask   : (H, W) bool, current frame's segmented ground
        dt          : seconds since the previous call
        imu_state   : the dict from realsense_imu (get_state() /
                      unavailable_state()) for this same frame, or None.
        Returns the current smoothed forward speed estimate in m/s (>= 0).
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        vision_speed = 0.0

        if (
            self.prev_gray is not None
            and self.prev_pts is not None
            and len(self.prev_pts) >= config.SPEED_MIN_TRACK_POINTS
            and dt > 1e-3
        ):
            vision_speed = self._estimate_vision_speed(gray, xyz, dt)

        raw_speed = self._fuse_with_imu(vision_speed, imu_state, dt)

        alpha = config.SPEED_EMA_ALPHA
        self.smoothed_speed_mps = alpha * raw_speed + (1.0 - alpha) * self.smoothed_speed_mps

        self._reselect_features(gray, land_mask)
        self.prev_gray = gray
        self.prev_xyz = xyz

        return self.smoothed_speed_mps

    # ---------------- vision (optical flow + depth) ----------------

    def _estimate_vision_speed(self, gray, xyz, dt):
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **_LK_PARAMS
        )
        if curr_pts is None or status is None:
            return 0.0

        status = status.reshape(-1)
        prev_matched = self.prev_pts.reshape(-1, 2)[status == 1]
        curr_matched = curr_pts.reshape(-1, 2)[status == 1]
        if prev_matched.shape[0] == 0:
            return 0.0

        h, w = gray.shape

        # Vectorized in-bounds check + gather (replaces the old per-point
        # Python for-loop — meaningfully cheaper at 150 tracked points/frame,
        # and this runs every frame).
        pxi = np.round(prev_matched[:, 0]).astype(np.int32)
        pyi = np.round(prev_matched[:, 1]).astype(np.int32)
        cxi = np.round(curr_matched[:, 0]).astype(np.int32)
        cyi = np.round(curr_matched[:, 1]).astype(np.int32)

        in_bounds = (
            (pxi >= 0) & (pxi < w) & (pyi >= 0) & (pyi < h)
            & (cxi >= 0) & (cxi < w) & (cyi >= 0) & (cyi < h)
        )
        if not np.any(in_bounds):
            return 0.0

        pxi, pyi, cxi, cyi = pxi[in_bounds], pyi[in_bounds], cxi[in_bounds], cyi[in_bounds]

        z_prev = self.prev_xyz[pyi, pxi, 2]
        z_curr = xyz[cyi, cxi, 2]

        valid = (z_prev > 0.05) & (z_curr > 0.05)
        if np.count_nonzero(valid) < max(3, config.SPEED_MIN_TRACK_POINTS // 2):
            return 0.0

        deltas = z_prev[valid] - z_curr[valid]
        raw_speed = float(np.median(deltas)) / dt
        # Forward-only for V1 (reverse driving isn't handled yet); clamp outliers.
        return float(np.clip(raw_speed, 0.0, config.SPEED_MAX_REASONABLE_MPS))

    # ---------------- IMU fusion ----------------

    def _fuse_with_imu(self, vision_speed, imu_state, dt):
        if not imu_state or not imu_state.get("available") or dt <= 1e-4:
            # No IMU this run — behave exactly like V1 (vision-only), and
            # keep the integrator parked at the vision reading so it doesn't
            # jump if the IMU comes back later in the session.
            self._imu_velocity_mps = vision_speed
            return vision_speed

        forward_accel = self._forward_linear_accel(imu_state)

        self._imu_velocity_mps += forward_accel * dt
        # Leaky correction toward vision — this is what keeps accelerometer
        # bias from integrating into an ever-growing error. Framed the same
        # way as the pitch/roll complementary filter: fast/responsive from
        # the IMU, long-term-truth from vision.
        correction_rate = config.SPEED_IMU_VISION_CORRECTION_RATE
        self._imu_velocity_mps += (vision_speed - self._imu_velocity_mps) * min(1.0, correction_rate * dt)
        self._imu_velocity_mps = float(np.clip(self._imu_velocity_mps, 0.0, config.SPEED_MAX_REASONABLE_MPS))

        return self._imu_velocity_mps

    @staticmethod
    def _forward_linear_accel(imu_state):
        """
        Gravity-compensated forward-axis (camera Z) acceleration. Uses the
        already-fused pitch/roll from this same IMU reading to estimate
        gravity's projection onto the camera frame's axes (X: right, Y:
        down, Z: forward — see realsense_imu.py) and subtracts it from the
        raw accelerometer reading, leaving (approximately) the acceleration
        actually caused by the tractor moving. Small-angle-friendly, which
        covers normal tractor pitch/roll; extreme tilt would need a full
        rotation-matrix treatment, not worth it for this dashboard estimate.
        """
        pitch = math.radians(imu_state.get("pitch_deg", 0.0))
        roll = math.radians(imu_state.get("roll_deg", 0.0))
        ax, ay, az = imu_state.get("accel_mps2", (0.0, 0.0, 0.0))

        gravity_z = _G_MPS2 * math.sin(pitch)
        linear_z = az - gravity_z
        return linear_z

    # ---------------- feature reselection ----------------

    def _reselect_features(self, gray, land_mask):
        mask_u8 = (land_mask.astype(np.uint8)) * 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            mask=mask_u8,
            maxCorners=config.SPEED_FEATURE_MAX_CORNERS,
            qualityLevel=config.SPEED_FEATURE_QUALITY,
            minDistance=config.SPEED_FEATURE_MIN_DISTANCE,
        )
        self.prev_pts = pts

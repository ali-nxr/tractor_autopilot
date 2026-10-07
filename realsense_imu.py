"""
realsense_imu.py — fuses accelerometer + gyroscope motion data into a
pitch/roll/yaw estimate with a complementary filter.

This file no longer owns a pipeline. Motion frames are pulled from the SAME
rs.pipeline/rs.config as depth+color in realsense_capture.py — opening a
second independent pipeline against the same physical RealSense device was
the actual cause of "IMU not working": many USB controllers/driver stacks
only allow one pipeline to claim a device at a time, so the second
pipeline.start() either throws or silently never produces frames. Combining
every stream (video + motion) into one pipeline config, the way Intel's own
multi-stream examples do, avoids that failure mode entirely. ImuFusion here
just does the math on whatever motion frames realsense_capture.py hands it.

Why a complementary filter and not just the gyro or just the accel:
  - The gyro gives smooth, responsive angular velocity, but integrating it
    over time drifts (small bias errors accumulate).
  - The accelerometer, when the sensor isn't accelerating, measures gravity
    directly and can compute an absolute pitch/roll from it — but it's noisy
    frame-to-frame and gets corrupted by real acceleration (bumps, braking).
  - Blending them (mostly trust the gyro, slowly correct toward the accel's
    reading) gets a stable, responsive, drift-free-ish pitch/roll. Yaw has no
    equivalent correction available without a magnetometer, so yaw here is
    "relative heading since the reader started" — it WILL drift over time.
    Treat it as a rough turn-rate integral, not a true compass heading.

Axis convention (RealSense IMU frame, matches the color/depth optical frame):
  X: right, Y: down, Z: forward (out of the lens).
  With the camera mounted looking forward and roughly level:
    pitch = rotation about X (nose tipping up/down)
    roll  = rotation about Z (tipping side to side)
  Real mounting is never perfectly square to the chassis — use
  config.IMU_MOUNT_PITCH_OFFSET_DEG / IMU_MOUNT_ROLL_OFFSET_DEG (set from the
  IMU page's "zero on flat ground" reading) to correct for that instead of
  fighting the math here.
"""

import math
import time

import numpy as np

import config


class ImuFusion:
    """
    Stateless with respect to hardware — just takes motion samples (or
    rs.frame motion frames) and maintains the fused orientation estimate.
    Owned and fed by RealSenseCapture, one sample pair per video frame.
    """

    def __init__(self):
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._last_ts = None
        self._last_accel = np.zeros(3, dtype=np.float32)
        self._last_gyro = np.zeros(3, dtype=np.float32)
        self._zero_pitch_deg = 0.0
        self._zero_roll_deg = 0.0

    def zero_on_current_orientation(self):
        """Call this while the tractor is known to be on flat, level ground —
        makes the *current* reading the new pitch/roll zero-reference, on top
        of (not replacing) the static config mount offsets."""
        self._zero_pitch_deg = self._pitch_deg_raw()
        self._zero_roll_deg = self._roll_deg_raw()

    def _pitch_deg_raw(self):
        return math.degrees(self._pitch) + config.IMU_MOUNT_PITCH_OFFSET_DEG

    def _roll_deg_raw(self):
        return math.degrees(self._roll) + config.IMU_MOUNT_ROLL_OFFSET_DEG

    def update_from_rs_frames(self, accel_frame, gyro_frame):
        """
        accel_frame / gyro_frame: rs.frame (or falsy/None) pulled out of the
        same frameset as this iteration's color/depth via
        frameset.first_or_default(rs.stream.accel / rs.stream.gyro). Either
        can be missing on a given call (motion streams publish faster than
        video, so we just take whatever's most recent); returns the fused
        state dict either way.
        """
        got_new = False
        if accel_frame:
            d = accel_frame.as_motion_frame().get_motion_data()
            self._last_accel = np.array([d.x, d.y, d.z], dtype=np.float32)
            got_new = True
        if gyro_frame:
            d = gyro_frame.as_motion_frame().get_motion_data()
            self._last_gyro = np.array([d.x, d.y, d.z], dtype=np.float32)
            got_new = True
        return self._advance(got_new)

    def update_from_values(self, accel_xyz=None, gyro_xyz=None):
        """Same as update_from_rs_frames but for plain (x, y, z) tuples —
        useful for testing without real hardware/frames."""
        got_new = False
        if accel_xyz is not None:
            self._last_accel = np.array(accel_xyz, dtype=np.float32)
            got_new = True
        if gyro_xyz is not None:
            self._last_gyro = np.array(gyro_xyz, dtype=np.float32)
            got_new = True
        return self._advance(got_new)

    def _advance(self, got_new):
        now = time.time()
        dt = 0.0 if self._last_ts is None else max(0.0, now - self._last_ts)
        self._last_ts = now
        if got_new and dt > 0:
            self._update_orientation(dt)
        return self.get_state()

    def _update_orientation(self, dt):
        ax, ay, az = self._last_accel
        gx, gy, gz = self._last_gyro  # rad/s (RealSense gyro units)

        # Gyro-only integration (drifts over time, corrected below)
        gyro_pitch = self._pitch + gx * dt
        gyro_roll = self._roll + gz * dt
        self._yaw += gy * dt  # no correction source available for yaw

        # Accelerometer-derived absolute pitch/roll from the gravity vector.
        # With the camera level and stationary, gravity reads almost purely
        # on +Y (down): (ax, ay, az) ~= (0, g, 0).
        #   - Pitch is rotation about X (right) — nose up/down tilts gravity
        #     between Y and Z, so it shows up as az vs. ay.
        #   - Roll is rotation about Z (forward) — tipping side to side tilts
        #     gravity between Y and X, so it shows up as -ax vs. ay.
        # Each uses sqrt() of the OTHER two axes in the denominator (not just
        # ay) so a small amount of roll doesn't corrupt the pitch reading and
        # vice versa — standard 3-axis tilt-from-accelerometer formula.
        # Guard against divide-by-zero / degenerate readings.
        pitch_denom = math.sqrt(ax * ax + ay * ay)
        roll_denom = math.sqrt(ay * ay + az * az)
        accel_pitch = math.atan2(az, pitch_denom) if pitch_denom > 1e-6 else gyro_pitch
        accel_roll = math.atan2(-ax, roll_denom) if roll_denom > 1e-6 else gyro_roll

        alpha = config.IMU_COMPLEMENTARY_ALPHA
        self._pitch = alpha * gyro_pitch + (1.0 - alpha) * accel_pitch
        self._roll = alpha * gyro_roll + (1.0 - alpha) * accel_roll

    def get_state(self):
        pitch_deg = self._pitch_deg_raw() - self._zero_pitch_deg
        roll_deg = self._roll_deg_raw() - self._zero_roll_deg
        tilt_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, math.cos(math.radians(pitch_deg)) * math.cos(math.radians(roll_deg)))))
        )

        if tilt_deg >= config.IMU_TILT_DANGER_DEG:
            risk = "danger"
        elif tilt_deg >= config.IMU_TILT_WARNING_DEG:
            risk = "warning"
        else:
            risk = "ok"

        return {
            "available": True,
            "status": "running",
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": math.degrees(self._yaw) % 360.0,
            "tilt_deg": tilt_deg,
            "accel_mps2": tuple(float(v) for v in self._last_accel),
            "gyro_dps": tuple(float(math.degrees(v)) for v in self._last_gyro),
            "rollover_risk": risk,
        }


def unavailable_state(reason=None):
    return {
        "available": False,
        "status": reason or "IMU not available on this camera "
                             "(requires D435i / D455 motion module)",
        "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "tilt_deg": 0.0,
        "accel_mps2": (0.0, 0.0, 0.0), "gyro_dps": (0.0, 0.0, 0.0),
        "rollover_risk": "ok",
    }

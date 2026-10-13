"""
lane_assist/camera.py — the camera model every lane backend shares:
intrinsics, and the orientation + height of the camera relative to the
GROUND, which is what a lane model needs to (a) warp the image into its own
expected view and (b) turn its output into real metres.

Frames:
  view (camera) : x right, y down, z forward      (RealSense / OpenCV)
  road          : x forward along the ground, y right, z down (normal to
                  the ground), origin at the camera — the frame lane
                  estimates are expressed in. (openpilot calls this "calib".)

view_from_road(): 3x3 rotation whose columns are the road frame's forward,
right and down axes expressed in camera coordinates.

Calibration source (config_lane):
  - AUTO (default): from the ground plane ground_segmentation.py fits every
    frame — plane normal gives "down", camera height is the plane's distance.
    Follows the tractor pitching and rolling over terrain; smoothed.
  - MANUAL: CAMERA_PITCH_DOWN_DEG / CAMERA_ROLL_DEG / CAMERA_HEIGHT_M.
  Yaw (camera heading vs direction of travel) is never observable from the
  ground, so it always comes from CAMERA_YAW_DEG.
"""

import math

import numpy as np

import config_lane as lcfg

# Road axes for a level, square-mounted camera: forward = +z, right = +x, down = +y.
_LEVEL = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def view_from_road_manual(pitch_down_deg, roll_deg, yaw_deg):
    """Camera rotated (in its own axes) yaw right, then pitch down, then roll."""
    r = _ry(math.radians(yaw_deg)) @ _rx(-math.radians(pitch_down_deg)) @ _rz(math.radians(roll_deg))
    return r.T @ _LEVEL


def view_from_road_from_down(down, yaw_deg):
    """Road axes from the ground's 'down' direction in camera coordinates."""
    down = _unit(np.asarray(down, dtype=float))
    ez = np.array([0.0, 0.0, 1.0])
    fwd = _unit(ez - (ez @ down) * down)          # camera forward, projected onto the ground
    right = np.cross(down, fwd)
    psi = math.radians(yaw_deg)                   # camera yawed right -> road forward appears left
    fwd, right = math.cos(psi) * fwd - math.sin(psi) * right, math.sin(psi) * fwd + math.cos(psi) * right
    return np.column_stack([fwd, right, down])


def road_to_view(view_from_road, height_m, x, y):
    """Road-frame ground points -> camera frame (N,3), for a given calibration
    snapshot (each LaneEstimate carries the one it was computed with)."""
    x = np.asarray(x, dtype=float)
    return (np.asarray(view_from_road) @ np.stack([x, np.asarray(y, dtype=float),
                                                   np.full_like(x, float(height_m))])).T


def road_to_view(view_from_road, height_m, x, y):
    """Road-frame ground points -> camera frame (N,3), for a given calibration
    snapshot (each LaneEstimate carries the one it was computed with)."""
    x = np.asarray(x, dtype=float)
    return (np.asarray(view_from_road) @ np.stack([x, np.asarray(y, dtype=float),
                                                   np.full_like(x, float(height_m))])).T


class LaneCamera:
    def __init__(self, stream_intrinsics):
        """stream_intrinsics: anything with fx, fy, ppx, ppy, width, height."""
        override = lcfg.CAMERA_INTRINSICS
        if override:
            self.fx, self.fy = float(override["fx"]), float(override["fy"])
            self.cx, self.cy = float(override["cx"]), float(override["cy"])
        else:
            self.fx, self.fy = float(stream_intrinsics.fx), float(stream_intrinsics.fy)
            self.cx, self.cy = float(stream_intrinsics.ppx), float(stream_intrinsics.ppy)
        self.width, self.height = int(stream_intrinsics.width), int(stream_intrinsics.height)
        self._down = None
        self._height_m = None

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def manual_orientation(self):
        return lcfg.CAMERA_PITCH_DOWN_DEG is not None and lcfg.CAMERA_ROLL_DEG is not None

    def reset(self):
        self._down = None
        self._height_m = None

    def update_from_plane(self, plane):
        """plane: (a, b, c, d) with a*X + b*Y + c*Z + d = 0 in camera metres."""
        if plane is None:
            return
        n = np.asarray(plane[:3], dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return
        n, d = n / norm, float(plane[3]) / norm
        down = -math.copysign(1.0, d) * n            # from the camera toward the ground
        height = abs(d)
        a = lcfg.CALIBRATION_SMOOTHING_ALPHA
        if self._down is None:
            self._down, self._height_m = down, height
        else:
            self._down = _unit((1 - a) * self._down + a * down)
            self._height_m = (1 - a) * self._height_m + a * height

    @property
    def height_m(self):
        return lcfg.CAMERA_HEIGHT_M if lcfg.CAMERA_HEIGHT_M is not None else self._height_m

    @property
    def ready(self):
        if self.height_m is None:
            return False
        return self.manual_orientation or self._down is not None

    def view_from_road(self):
        if self.manual_orientation:
            return view_from_road_manual(lcfg.CAMERA_PITCH_DOWN_DEG, lcfg.CAMERA_ROLL_DEG, lcfg.CAMERA_YAW_DEG)
        return view_from_road_from_down(self._down, lcfg.CAMERA_YAW_DEG)

    def describe(self):
        if not self.ready:
            return "camera: waiting for ground plane"
        v = self.view_from_road()
        fwd, down = v[:, 0], v[:, 2]
        pitch_down = math.degrees(math.atan2(-fwd[1], fwd[2]))
        roll = math.degrees(math.atan2(-down[0], down[1]))
        src = "manual" if self.manual_orientation else "ground plane"
        return (f"camera ({src}): height {self.height_m:.2f} m, pitch down {pitch_down:+.1f} deg, "
                f"roll {roll:+.1f} deg, yaw {lcfg.CAMERA_YAW_DEG:+.1f} deg, "
                f"f=({self.fx:.0f},{self.fy:.0f}) c=({self.cx:.0f},{self.cy:.0f})")

    # ---------------------------------------------------------- projection

    def road_to_view(self, x, y, z=None):
        """Road-frame points (arrays) -> camera-frame (N,3). z defaults to the
        ground (camera height) — the corridor lies on the fitted ground plane."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.full_like(x, self.height_m) if z is None else np.asarray(z, dtype=float)
        return (self.view_from_road() @ np.stack([x, y, z])).T

    def project(self, pts_view):
        """Camera-frame (N,3) -> pixel (N,2), NaN behind the camera."""
        p = np.asarray(pts_view, dtype=float)
        uv = np.full((len(p), 2), np.nan)
        ok = p[:, 2] > 0.1
        uv[ok, 0] = self.fx * p[ok, 0] / p[ok, 2] + self.cx
        uv[ok, 1] = self.fy * p[ok, 1] / p[ok, 2] + self.cy
        return uv

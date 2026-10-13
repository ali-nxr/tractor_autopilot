"""
lane_assist/overlay.py — draws lane assist onto the HUD frame: the locked
corridor's edges + centreline (or, while searching, the current candidate
faintly), and a status badge. Inputs are camera-frame points, so this works
for any backend.
"""

import numpy as np
import cv2

import config_lane as lcfg
from lane_assist.camera import road_to_view


def to_view_polylines(lane_out):
    """LaneAssist output (road frame) -> camera-frame polylines for drawing,
    using the calibration the estimate was computed with. Done in the
    geometry process, so only small arrays travel to the render process."""
    src = lane_out.get("track") or lane_out.get("candidate")
    calib = lane_out.get("calibration")
    if src is None or calib is None:
        return None
    view_from_road, height = calib
    x = src["x"]
    keep = x <= lcfg.LANE_DRAW_MAX_M
    out = {"locked": lane_out.get("track") is not None}
    for name in ("left", "center", "right"):
        y = src[name][keep]
        ok = np.isfinite(y)
        out[name] = road_to_view(view_from_road, height, x[keep][ok], y[ok]) if ok.sum() >= 2 else None
    return out


def _project(pts, intr):
    z = pts[:, 2]
    ok = z > 0.3
    uv = np.stack([intr.fx * pts[ok, 0] / z[ok] + intr.ppx, intr.fy * pts[ok, 1] / z[ok] + intr.ppy], axis=1)
    return uv


def draw(frame, lane, intrinsics):
    """lane: geometry result["lane"] (see perception_core). In place."""
    if lane is None or not lane.get("enabled"):
        return frame
    h, w = frame.shape[:2]
    lines = lane.get("polylines")
    if lcfg.LANE_DRAW and lines is not None:
        locked = lines["locked"]
        color = lcfg.COLOR_LANE_LOCKED if locked else lcfg.COLOR_LANE_SEARCHING
        for name in ("left", "right", "center"):
            pts = lines.get(name)
            if pts is None:
                continue
            uv = _project(pts, intrinsics)
            uv = uv[(uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)]
            if len(uv) < 2:
                continue
            poly = np.round(uv).astype(np.int32)
            if name == "center":
                if locked:
                    cv2.polylines(frame, [poly], False, lcfg.COLOR_LANE_CENTER, 1, cv2.LINE_AA)
            else:
                cv2.polylines(frame, [poly], False, (0, 0, 0), 5 if locked else 3, cv2.LINE_AA)
                cv2.polylines(frame, [poly], False, color, 3 if locked else 1, cv2.LINE_AA)

    # status badge, top-right of the video
    state = lane.get("state", "OFF")
    if state in ("LOCKED", "COASTING"):
        steer = lane.get("steer_deg")
        steer_txt = "" if steer is None else (f"  steer {'R' if steer > 0 else 'L'} {abs(steer):.0f}deg"
                                               if abs(steer) >= 0.5 else "  steer straight")
        text = f"LANE {state}  {lane.get('kind', '')}  {lane.get('conf', 0):.2f}{steer_txt}"
        bg = (40, 140, 40) if state == "LOCKED" else (30, 140, 200)
    elif state == "SEARCHING":
        text = "LANE ASSIST: searching for a lane"
        bg = (90, 90, 90)
    else:
        text = f"LANE ASSIST: {state}"
        bg = (60, 60, 60)
    if lane.get("model_error"):
        text, bg = f"LANE ASSIST unavailable: {lane['model_error'][:60]}", (40, 40, 160)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x0, y0 = w - tw - 24, 12
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 14, y0 + th + 14), bg, -1)
    cv2.putText(frame, text, (x0 + 7, y0 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return frame

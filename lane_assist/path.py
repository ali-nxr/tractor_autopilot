"""
lane_assist/path.py — the drivable path ALONG the locked lane, as a drop-in
replacement for path_planner.plan_path() while lane assist is locked.

plan_path() searches every distance band for the free gap closest to the
previous heading ("always forward drivable space"). Here the lateral
position at each band is instead the locked corridor's centreline, so the
path follows the track. The safety rules are the same as plan_path's:
  - a band where a confirmed obstacle lies within the vehicle's corridor
    (half width + CORRIDOR_MARGIN_M around the lane centre) BLOCKS the path
    there (blocked_at_m) — the path never runs through an obstacle;
  - the near blind zone (no depth yet) is skipped, and the path ENDS where
    depth data runs out after ground was seen (range_end_m) — it is never
    drawn beyond what the depth sensor has verified;
  - heights come from real measured points (fitted plane as fallback).
The returned dict has exactly plan_path()'s keys, so the AR ribbon, the
overlay and the HUD use it unchanged.
"""

import numpy as np

import config
from path_planner import _project_to_pixel, _real_height_near


def plan_along_lane(center_xz, obstacle_mask, valid_depth, xyz, plane, intrinsics):
    """center_xz: (N, 2) camera-frame (X lateral, Z forward) samples of the
    locked centreline, Z increasing."""
    result = {"waypoints_m": [], "waypoints_px": [], "surface_left_px": [], "surface_right_px": [],
              "blocked_at_m": None, "range_end_m": None, "lane_locked": True}
    if plane is None or center_xz is None or len(center_xz) < 2:
        return result
    lane_x, lane_z = center_xz[:, 0], center_xz[:, 1]
    order = np.argsort(lane_z)
    lane_x, lane_z = lane_x[order], lane_z[order]

    n_steps = max(config.PATH_NUM_STEPS, 2)
    z_steps = np.linspace(config.MIN_RANGE_M, config.MAX_RANGE_M, n_steps)
    band_half = max((config.MAX_RANGE_M - config.MIN_RANGE_M) / (n_steps - 1) / 2.0, 0.15)
    half_w = config.TRACTOR_WIDTH_M / 2.0
    keep_out = half_w + config.CORRIDOR_MARGIN_M
    edge_window = max(config.PATH_LATERAL_BIN_M * 2.0, 0.15)

    x_all, y_all, z_all = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    search = keep_out + 2.0
    cand = (valid_depth & (np.abs(x_all) < search + float(np.abs(lane_x).max()))
            & (z_all >= config.MIN_RANGE_M - band_half) & (z_all <= config.MAX_RANGE_M + band_half))
    cx, cy, cz, cobs = x_all[cand], y_all[cand], z_all[cand], obstacle_mask[cand]

    started = False
    for z_target in z_steps:
        if z_target < lane_z[0] or z_target > lane_z[-1]:
            if started and z_target > lane_z[-1]:
                result["range_end_m"] = float(z_target)        # the lane estimate ends here
                break
            continue
        x_c = float(np.interp(z_target, lane_z, lane_x))
        in_band = np.abs(cz - z_target) < band_half
        near_lane = in_band & (np.abs(cx - x_c) < keep_out)
        if not near_lane.any():
            if not started:
                continue                                       # near blind zone
            result["range_end_m"] = float(z_target)            # depth ran out
            break
        started = True
        if cobs[near_lane].any():
            result["blocked_at_m"] = float(z_target)
            break

        bx, by = cx[in_band], cy[in_band]
        y_c = _real_height_near(x_c, bx, by, plane, float(z_target), max(half_w * 0.5, edge_window))
        result["waypoints_m"].append((x_c, float(z_target)))
        px = _project_to_pixel(x_c, y_c, float(z_target), intrinsics)
        if px is not None:
            result["waypoints_px"].append(px)
        yl = _real_height_near(x_c - half_w, bx, by, plane, float(z_target), edge_window)
        yr = _real_height_near(x_c + half_w, bx, by, plane, float(z_target), edge_window)
        lp = _project_to_pixel(x_c - half_w, yl, float(z_target), intrinsics)
        rp = _project_to_pixel(x_c + half_w, yr, float(z_target), intrinsics)
        if lp is not None and rp is not None:
            result["surface_left_px"].append(lp)
            result["surface_right_px"].append(rp)
    return result

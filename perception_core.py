"""
perception_core.py — the per-frame perception + decision step and the
render step, as plain functions/classes with no threading, UI or frame
source attached. The multi-process pipeline (mp_pipeline/) runs Perception
in its geometry process and render_frame() in its render process.

The stage order and every call are a stage-for-stage port of the vision
loop in main_recorded.py (which mirrors main.py): downsample -> ground
segmentation -> speed -> obstacles + tracking -> brake/steer -> path ->
AR ribbon -> (render) overlay. Nothing here changes what is decided; it
only lets each half run in its own process.
"""

import numpy as np
import cv2

import config
import config_segformer
import obstacle_decision
import overlay
import path_planner
import debug_view
from ar_ribbon import ArRibbon
from ground_segmentation import GroundSegmenter
from object_tracker import ObstacleTracker
from smoothing import SlewRateLimiter
from speed_estimator import SpeedEstimator
from yolo_detector import draw_detections

try:
    import config_lane
    from lane_assist import overlay as lane_overlay
    from lane_assist.assist import COASTING, LOCKED, OFF, LaneAssist
    from lane_assist.camera import road_to_view
    from lane_assist.path import plan_along_lane
    _LANE_IMPORTED = True
except Exception as _lane_import_error:          # plugin missing/broken: run without it
    print(f"[lane] lane assist plugin not loaded: {_lane_import_error}", flush=True)
    _LANE_IMPORTED = False

try:
    import config_lane
    from lane_assist import overlay as lane_overlay
    from lane_assist.assist import COASTING, LOCKED, OFF, LaneAssist
    from lane_assist.camera import road_to_view
    from lane_assist.path import plan_along_lane
    _LANE_IMPORTED = True
except Exception as _lane_import_error:          # plugin missing/broken: run without it
    print(f"[lane] lane assist plugin not loaded: {_lane_import_error}", flush=True)
    _LANE_IMPORTED = False


# ---------------------------------------------------------------- helpers
# Identical to main.py's — duplicated here (not imported) because importing
# main.py would pull in the Tk UI in every worker process.

def downsample_for_processing(depth_image_m, xyz, raw_valid_mask, ir_image, downscale):
    """NEAREST-sample downsampling — see main._downsample_for_processing for why
    averaging is wrong here (it invents phantom 3D points at depth holes).
    xyz may be None (the multi-process pipeline deprojects at processing
    resolution instead — see deproject_proc); xyz_proc is then None."""
    if downscale <= 1:
        return depth_image_m, xyz, raw_valid_mask, ir_image
    h, w = depth_image_m.shape[:2]
    proc_w, proc_h = max(1, w // downscale), max(1, h // downscale)
    nn = cv2.INTER_NEAREST
    depth_proc = cv2.resize(depth_image_m, (proc_w, proc_h), interpolation=nn)
    xyz_proc = None if xyz is None else cv2.resize(xyz, (proc_w, proc_h), interpolation=nn)
    valid_proc = cv2.resize(raw_valid_mask.astype(np.uint8), (proc_w, proc_h), interpolation=nn) > 0
    valid_proc &= depth_proc > 0
    ir_proc = None if ir_image is None else cv2.resize(ir_image, (proc_w, proc_h), interpolation=nn)
    return depth_proc, xyz_proc, valid_proc, ir_proc


def deproject_proc(depth_proc, intrinsics, downscale):
    """
    XYZ at PROCESSING resolution, straight from the (nearest-downsampled)
    depth — bit-identical to deprojecting at full resolution and then
    nearest-downsampling, which is what the capture sources + the single-
    process loops do, at ~1/downscale^2 of the cost. Identical because
    nearest-downsampling by an integer factor picks source pixel
    (downscale*u, downscale*v), and the same formula is evaluated with the
    source's own grid dtype (intrinsics.grid_dtype: float32 for playback,
    int64 for the live camera — they differ, so it matters).
    """
    h, w = depth_proc.shape[:2]
    dtype = np.dtype(getattr(intrinsics, "grid_dtype", "float32"))
    uu, vv = np.meshgrid(np.arange(w) * downscale, np.arange(h) * downscale)
    uu, vv = uu.astype(dtype), vv.astype(dtype)
    z = depth_proc
    x = (uu - intrinsics.ppx) * z / intrinsics.fx
    y = (vv - intrinsics.ppy) * z / intrinsics.fy
    return np.dstack((x, y, z)).astype(np.float32)


def z_as_xyz(depth):
    """A zero-copy (H, W, 3) view whose every channel is depth. For
    SpeedEstimator, which reads ONLY xyz[..., 2] (= depth) — so it needs no
    full-resolution deprojection at all."""
    s0, s1 = depth.strides
    return np.lib.stride_tricks.as_strided(depth, shape=depth.shape + (3,), strides=(s0, s1, 0),
                                           writeable=False)


def upsample_mask(mask, full_w, full_h, downscale):
    if downscale <= 1 or mask is None:
        return mask
    return cv2.resize(mask.astype(np.uint8) * 255, (full_w, full_h), interpolation=cv2.INTER_LINEAR) > 127


def upsample_mask_nearest(mask, full_w, full_h, downscale):
    if downscale <= 1 or mask is None:
        return mask
    return cv2.resize(mask.astype(np.uint8) * 255, (full_w, full_h), interpolation=cv2.INTER_NEAREST) > 127


def _scale_bboxes(obstacle_list, downscale):
    out = []
    for o in obstacle_list:
        o = dict(o)
        bx, by, bw, bh = o["bbox"]
        o["bbox"] = (bx * downscale, by * downscale, bw * downscale, bh * downscale)
        out.append(o)
    return out


# ------------------------------------------------------------- perception

class Perception:
    """
    Everything that carries state from one frame to the next, plus the
    per-frame step. reset() rebuilds all of it — used on a playback
    discontinuity (seek / loop / restart), exactly like main_recorded's
    _Stateful rebuild.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.ground_segmenter = GroundSegmenter()
        self.speed_estimator = SpeedEstimator()
        self.brake_smoother = SlewRateLimiter(config.MAX_BRAKE_CHANGE_PCT_PER_S)
        self.steer_smoother = SlewRateLimiter(config.MAX_STEER_CHANGE_DEG_PER_S)
        self.obstacle_tracker = ObstacleTracker()
        self.ar_ribbon = ArRibbon()
        self.degraded_since = None
        self.last_frame_time = None
        self.lane_assist = (LaneAssist() if _LANE_IMPORTED and config_lane.LANE_ASSIST_AVAILABLE else None)
        self.lane_assist = (LaneAssist() if _LANE_IMPORTED and config_lane.LANE_ASSIST_AVAILABLE else None)

    def step(self, frame, now, nominal_dt, intrinsics, semantic, timing, speed_mps=None,
             lane_estimate=None, lane_enabled=False, lane_model_error=None):
        """
        frame    : dict color/depth/xyz/raw_valid/ir (full resolution) + imu_state
        now      : this frame's time (recording clock in playback, wall clock live)
        semantic : dict from the semantic process — prob (proc res) and
                   sky (proc res bool), each possibly None
        timing   : dict to add per-stage seconds into
        lane_estimate / lane_enabled: the lane model's output for this frame
                   and whether lane assist is switched on (lane_assist/).
                   While LOCKED the path follows the lane and the steer
                   suggestion comes from it — unless an obstacle is in the
                   corridor, which keeps the obstacle-avoidance steer.
        lane_estimate / lane_enabled: the lane model's output for this frame
                   and whether lane assist is switched on (lane_assist/).
                   While LOCKED the path follows the lane and the steer
                   suggestion comes from it — unless an obstacle is in the
                   corridor, which keeps the obstacle-avoidance steer.
        speed_mps: this frame's speed if it was estimated elsewhere (the
                   multi-process pipeline runs SpeedEstimator in its own
                   process — see mp_pipeline/workers.speed_main); None =
                   estimate it here, as the single-process loops do.

        Returns (result_for_render, hud_numbers).
        """
        import time
        t0 = time.perf_counter()
        color = frame["color"]
        pd = config.PROCESSING_DOWNSCALE
        full_h, full_w = frame["depth"].shape[:2]

        dt = nominal_dt if self.last_frame_time is None else max(1e-4, now - self.last_frame_time)
        self.last_frame_time = now

        depth_proc, xyz_proc, valid_proc, ir_proc = downsample_for_processing(
            frame["depth"], frame.get("xyz"), frame["raw_valid"], frame["ir"], pd)
        if xyz_proc is None:
            xyz_proc = deproject_proc(depth_proc, intrinsics, pd)
        sky_proc = semantic.get("sky")
        if sky_proc is not None:
            valid_proc = valid_proc & ~sky_proc

        seg = self.ground_segmenter.segment(depth_proc, xyz_proc, ir_image=ir_proc,
                                            raw_valid_mask=valid_proc,
                                            semantic_prob=semantic.get("prob"))
        t1 = time.perf_counter(); timing["segment"] = timing.get("segment", 0.0) + (t1 - t0)
        land_mask_proc = seg["land_mask"]
        valid_depth_proc = seg["valid_depth"]
        dist_to_plane_proc = seg["dist_to_plane"]

        total_px = land_mask_proc.size
        land_coverage_pct = 100.0 * float(np.sum(land_mask_proc)) / total_px if total_px else 0.0
        plane_found = seg["plane"] is not None
        ground_ok = plane_found and (land_coverage_pct >= config.MIN_LAND_COVERAGE_PCT)
        if ground_ok:
            self.degraded_since = None
            degraded_duration = 0.0
        else:
            if self.degraded_since is None:
                self.degraded_since = now
            degraded_duration = now - self.degraded_since
        watchdog_triggered = degraded_duration >= config.WATCHDOG_MAX_DEGRADED_S

        if speed_mps is None:
            land_mask_full = upsample_mask(land_mask_proc, full_w, full_h, pd)
            speed_mps = self.speed_estimator.update(color, frame["xyz"], land_mask_full, dt,
                                                    imu_state=frame["imu_state"])
        t2 = time.perf_counter(); timing["speed"] = timing.get("speed", 0.0) + (t2 - t1)

        corridor_mask_proc = obstacle_decision.build_corridor_mask(xyz_proc, valid_depth_proc)
        raised_mask_proc = obstacle_decision.build_raised_mask(
            dist_to_plane_proc, valid_depth_proc, land_mask_proc, xyz=xyz_proc)
        untrusted_proc = (obstacle_decision.glare_mask(color, land_mask_proc.shape)
                          if config.OBSTACLE_REJECT_GLARE else None)
        blobs, blob_labels = obstacle_decision.detect_obstacle_blobs(
            raised_mask_proc, xyz_proc, dist_to_plane_proc, untrusted_proc)
        tracked = self.obstacle_tracker.update(blobs, now)
        confirmed = [b for b in tracked if b["confirmed"]]
        obstacle_mask_proc = obstacle_decision.blobs_mask(blob_labels, confirmed)
        obstacles = obstacle_decision.corridor_obstacles(confirmed, blob_labels, corridor_mask_proc, xyz_proc)
        display_obstacles = _scale_bboxes(confirmed, pd)
        obstacles = _scale_bboxes(obstacles, pd)

        nearest_distance = obstacles[0]["distance_m"] if obstacles else None
        closing_speed_mps = obstacles[0]["closing_speed_mps"] if obstacles else 0.0
        brake_raw = obstacle_decision.compute_brake_percent(nearest_distance, speed_mps,
                                                            closing_speed_mps=closing_speed_mps)
        steer_raw = obstacle_decision.compute_steer_suggestion(obstacles[0] if obstacles else None)

        lane_out = None
        if self.lane_assist is not None:
            if lane_enabled:
                if self.lane_assist.state == OFF:
                    self.lane_assist.reset()
                lane_out = self.lane_assist.update(lane_estimate, now, speed_mps)
                if lane_out["steer_deg"] is not None and not obstacles:
                    steer_raw = lane_out["steer_deg"]
            elif self.lane_assist.state != OFF:
                self.lane_assist.reset(OFF)

        lane_out = None
        if self.lane_assist is not None:
            if lane_enabled:
                if self.lane_assist.state == OFF:
                    self.lane_assist.reset()
                lane_out = self.lane_assist.update(lane_estimate, now, speed_mps)
                if lane_out["steer_deg"] is not None and not obstacles:
                    steer_raw = lane_out["steer_deg"]
            elif self.lane_assist.state != OFF:
                self.lane_assist.reset(OFF)
        if watchdog_triggered:
            brake_percent = self.brake_smoother.force(100.0)
            steer_deg = self.steer_smoother.force(0.0)
        else:
            brake_percent = self.brake_smoother.update(brake_raw, dt)
            steer_deg = self.steer_smoother.update(steer_raw, dt)
        t3 = time.perf_counter(); timing["obstacles"] = timing.get("obstacles", 0.0) + (t3 - t2)

        lane_path = None
        if lane_out is not None and lane_out["state"] in (LOCKED, COASTING) and lane_out["calibration"]:
            track = lane_out["track"]
            ok = np.isfinite(track["center"])
            if ok.sum() >= 2:
                v = road_to_view(*lane_out["calibration"], track["x"][ok], track["center"][ok])
                lane_path = plan_along_lane(v[:, [0, 2]], obstacle_mask_proc, valid_depth_proc, xyz_proc,
                                            seg["plane"], intrinsics)
        path_result = lane_path if lane_path is not None else path_planner.plan_path(
            land_mask_proc, obstacle_mask_proc, valid_depth_proc, xyz_proc, seg["plane"], intrinsics)
        ribbon = None
        if config.AR_ENABLED:
            ribbon = self.ar_ribbon.update(path_result, xyz_proc, valid_depth_proc, obstacle_mask_proc,
                                           seg["plane"], intrinsics)
        t4 = time.perf_counter(); timing["path"] = timing.get("path", 0.0) + (t4 - t3)

        if watchdog_triggered:
            status_text = f"PERCEPTION DEGRADED {degraded_duration:.1f}s — safety brake engaged"
        elif ground_ok:
            status_text = "running"
        elif plane_found:
            status_text = "low ground visibility (coasting)"
        else:
            status_text = "no ground plane found (coasting)"

        result = {
            # proc-resolution masks; the render process upsamples them
            "land_mask": land_mask_proc,
            "boundary_contour": seg["boundary_contour"],
            "corridor_mask": corridor_mask_proc,
            "confidence_mask": seg["confidence_mask"],
            "sky": sky_proc,
            "semantic_ground": seg.get("semantic_ground"),
            "display_obstacles": display_obstacles,
            "brake_percent": brake_percent,
            "steer_deg": steer_deg,
            "path_result": path_result,
            "speed_mps": speed_mps,
            "ribbon": ribbon,
            "watchdog_triggered": watchdog_triggered,
            "degraded_duration": degraded_duration,
            "plane": seg["plane"],
            "lane": None,
        }
        lane_hud = None
        if self.lane_assist is not None:
            lane_hud = {"enabled": bool(lane_enabled), "state": self.lane_assist.state if lane_out is None
                        else lane_out["state"], "kind": None, "conf": 0.0, "steer_deg": None, "offset_m": None,
                        "model_error": lane_model_error}
            if lane_out is not None:
                lane_hud.update({k: lane_out[k] for k in ("kind", "conf", "steer_deg", "offset_m")})
            result["lane"] = dict(lane_hud, polylines=None if lane_out is None
                                  else lane_overlay.to_view_polylines(lane_out))
        if config.DEBUG:
            # What debug_view.build() reads from the segmentation result.
            result["seg_debug"] = {k: seg.get(k) for k in (
                "land_mask", "valid_depth", "dist_to_plane", "plane", "semantic_seeded",
                "semantic_prob", "geo_land_raw", "semantic_ground")}
        hud = {
            "speed_mps": speed_mps,
            "obstacles": obstacles,
            "nearest_obstacle_m": nearest_distance,
            "closing_speed_mps": closing_speed_mps,
            "brake_percent": brake_percent,
            "steer_suggestion_deg": steer_deg,
            "land_coverage_pct": land_coverage_pct,
            "path_blocked_at_m": path_result.get("blocked_at_m"),
            "watchdog_degraded": watchdog_triggered,
            "degraded_duration_s": degraded_duration,
            "status_text": status_text,
            "lane": lane_hud,
        }
        return result, hud


# ----------------------------------------------------------------- render

def render_frame(frame, result, yolo_detections, intrinsics=None):
    """The annotated HUD frame for one perception result (main_recorded's
    overlay block, unchanged)."""
    color = frame["color"]
    raw_valid = frame["raw_valid"]
    pd = config.PROCESSING_DOWNSCALE
    full_h, full_w = color.shape[:2]

    land_mask_full = upsample_mask(result["land_mask"], full_w, full_h, pd)
    corridor_mask_full = upsample_mask(result["corridor_mask"], full_w, full_h, pd)
    confidence_mask_full = upsample_mask_nearest(result["confidence_mask"], full_w, full_h, pd)
    contour = result["boundary_contour"]
    boundary_contour_full = contour * pd if (pd > 1 and contour is not None) else contour

    # Masked sky is a deliberate rejection, not untrusted sensor depth —
    # keep it out of the low-confidence overlay.
    raw_valid_display = raw_valid
    if result["sky"] is not None and raw_valid is not None:
        raw_valid_display = raw_valid & ~upsample_mask_nearest(result["sky"], full_w, full_h, pd)

    semantic_mask_full = None
    if config_segformer.SEMANTIC_SHOW_OVERLAY and result["semantic_ground"] is not None:
        semantic_mask_full = upsample_mask(result["semantic_ground"], full_w, full_h, pd)

    annotated = overlay.draw_annotations(
        color, land_mask_full, boundary_contour_full, corridor_mask_full,
        result["display_obstacles"], result["brake_percent"], result["steer_deg"],
        result["path_result"], speed_mps=result["speed_mps"],
        confidence_mask=confidence_mask_full, raw_valid_mask=raw_valid_display,
        ribbon=result["ribbon"], semantic_mask=semantic_mask_full,
    )
    if result["watchdog_triggered"]:
        overlay.draw_watchdog_banner(annotated, result["degraded_duration"])
    if result.get("lane") and intrinsics is not None:
        lane_overlay.draw(annotated, result["lane"], intrinsics)
    if yolo_detections:
        draw_detections(annotated, yolo_detections)
    return annotated


def render_debug(frame, result, semantic, semantic_info, intrinsics=None):
    """debug_view mosaic for one frame (config.DEBUG)."""
    pd = config.PROCESSING_DOWNSCALE
    depth_proc, xyz_proc, _, _ = downsample_for_processing(frame["depth"], frame.get("xyz"),
                                                           frame["raw_valid"], None, pd)
    if xyz_proc is None:
        xyz_proc = deproject_proc(depth_proc, intrinsics, pd)
    return debug_view.build(frame["color"], frame["depth"], frame["raw_valid"], xyz_proc,
                            result["seg_debug"], semantic.get("prob"), semantic_info)

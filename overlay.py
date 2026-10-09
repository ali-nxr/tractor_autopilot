"""
overlay.py — draws all the visual annotations onto the color frame:
land fill, land boundary, corridor outline, obstacle boxes + distances,
the AR-style tractor-width drivable surface, and the real drivable path
center line (see path_planner.py).
"""

import numpy as np
import cv2

import config
import config_segformer
from obstacle_decision import brake_color


def _blend_masked(frame, mask_bool, color, alpha):
    """
    Same visual result as `overlay = frame.copy(); overlay[mask_bool] = color;
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)` — semi-transparent
    color wash over just the masked pixels, everything else unchanged — but
    roughly an order of magnitude faster. Boolean fancy-index assignment
    (`array[bool_mask] = color`) turned out to be the dominant cost in this
    whole module (confirmed by profiling, not assumed): computing the blend
    over the WHOLE frame with cv2.addWeighted (a highly optimized, SIMD
    vectorized C++ call) and then using cv2.copyTo with a uint8 mask to keep
    only the masked pixels is faster than assigning into only the ~1M
    relevant pixel positions directly. Verified byte-for-byte identical
    output against the naive approach, including empty/full mask edge cases.
    """
    solid = np.empty_like(frame)
    solid[:] = color
    fully_blended = cv2.addWeighted(solid, alpha, frame, 1.0 - alpha, 0)
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    result = frame.copy()
    cv2.copyTo(fully_blended, mask_u8, result)
    return result


def draw_annotations(color_image, land_mask, boundary_contour, corridor_mask,
                      obstacles, brake_percent, steer_suggestion_deg, path_result,
                      speed_mps=0.0, confidence_mask=None, raw_valid_mask=None,
                      ribbon=None, semantic_mask=None):
    """
    ribbon=None -> legacy look (flat surface polygon + center line + text).
    ribbon=<ar_ribbon.ArRibbon.update() result> -> the AR look: smooth
    ground-hugging ribbon, wheel-line edges, ground distance stripes, STOP
    line, occlusion, and the declutter switches in config (AR_SHOW_*).
    semantic_mask=<bool mask> -> thin outline of what SegFormer alone calls
    driveable (config_segformer.SEMANTIC_SHOW_OVERLAY), for tuning.
    """
    frame = color_image.copy()
    h, w = frame.shape[:2]
    ar = ribbon is not None

    # --- low-confidence regions (sensor couldn't trust the depth here, e.g.
    #     black/IR-absorbing material) — shown honestly instead of silently
    #     folding into land or obstacle classification. Kept in AR mode too:
    #     it's safety information, not decoration. ---
    if config.SHOW_LOW_CONFIDENCE_OVERLAY and confidence_mask is not None and raw_valid_mask is not None:
        untrusted = raw_valid_mask & (~confidence_mask)
        if untrusted.any():
            frame = _blend_masked(frame, untrusted, config.COLOR_LOW_CONFIDENCE, 0.5)

    # --- land fill (semi-transparent green) ---
    if (not ar or config.AR_SHOW_LAND_FILL) and land_mask.any():
        frame = _blend_masked(frame, land_mask, config.COLOR_LAND_FILL, 0.35)

    # --- land boundary ---
    if boundary_contour is not None and (not ar or config.AR_SHOW_LAND_BOUNDARY):
        cv2.drawContours(frame, [boundary_contour], -1, config.COLOR_LAND_BOUNDARY, 1 if ar else 2,
                         cv2.LINE_AA)

    # --- SegFormer driveable outline (tuning aid, off by default) ---
    if semantic_mask is not None and semantic_mask.any():
        sem_contours, _ = cv2.findContours((semantic_mask.astype(np.uint8)) * 255,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, sem_contours, -1, config_segformer.COLOR_SEMANTIC_OUTLINE, 1,
                         cv2.LINE_AA)

    # --- corridor outline ---
    if not ar or config.AR_SHOW_CORRIDOR_OUTLINE:
        corridor_u8 = (corridor_mask.astype(np.uint8)) * 255
        corridor_contours, _ = cv2.findContours(corridor_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if corridor_contours:
            largest = max(corridor_contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest], -1, config.COLOR_CORRIDOR, 1)

    # --- drivable surface. Drawn before obstacle markers so a marker for
    #     something standing IN the strip still renders on top of it. ---
    if ar:
        frame = _draw_ar_ribbon(frame, ribbon)
    else:
        frame = _draw_drivable_surface(frame, path_result, config.COLOR_DRIVABLE_SURFACE)

    # --- obstacles ---
    for obs in obstacles:
        if ar:
            _draw_obstacle_brackets(frame, obs)
        else:
            bx, by, bw, bh = obs["bbox"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), config.COLOR_OBSTACLE_BOX, 2)
            label = f'{obs["distance_m"]:.1f} m'
            cv2.putText(frame, label, (bx, max(by - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_OBSTACLE_BOX, 2)

    # --- center line (legacy always; AR only if enabled) ---
    if not ar or config.AR_SHOW_CENTER_LINE:
        path_color = brake_color(brake_percent)
        _draw_real_path(frame, path_result, path_color, show_label=not ar)

    # --- status text block (the HUD panels already show all of this) ---
    if not ar or config.AR_SHOW_STATUS_TEXT:
        nearest = obstacles[0]["distance_m"] if obstacles else None
        _draw_status_text(frame, nearest, brake_percent, steer_suggestion_deg, speed_mps)

    return frame


def _put_label(frame, text, org, color, scale=0.6, thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_obstacle_brackets(frame, obs):
    x, y, w, h = obs["bbox"]
    c = config.COLOR_OBSTACLE_BOX
    t = max(10, min(w, h) // 4)
    for (px, py, dx, dy) in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * t, py), c, 3, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * t), c, 3, cv2.LINE_AA)
    label = f'{obs["distance_m"]:.1f} m'
    ly = max(y - 6, 22)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, ly - th - 8), (x + tw + 12, ly + 2), (20, 20, 20), -1)
    cv2.putText(frame, label, (x + 6, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)


def _draw_no_path_banner(frame, stop_z):
    h, w = frame.shape[:2]
    msg = "NO DRIVABLE PATH" + (f" - blocked at {stop_z:.1f} m" if stop_z is not None else "")
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x0 = (w - tw) // 2 - 16; y1 = h - 24; y0 = y1 - th - 22
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 32, y1), (25, 25, 25), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 32, y1), config.COLOR_AR_STOP, 2)
    cv2.putText(frame, msg, (x0 + 16, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_AR_STOP, 2,
                cv2.LINE_AA)


def _draw_ar_ribbon(frame, rb):
    """
    AR drivable surface. Performance-conscious: everything is computed
    inside the ribbon's bounding box only, the fade/color ramp is applied
    per image ROW (distance maps monotonically to row for ground ahead) —
    no per-segment full-frame masks.
    """
    if not rb.get("valid"):
        if rb.get("no_path"):
            _draw_no_path_banner(frame, rb.get("stop_z"))
        return frame
    h, w = frame.shape[:2]
    L, R, z = rb["left"], rb["right"], rb["z"]
    if not (np.isfinite(L).all() and np.isfinite(R).all()):
        return frame
    poly = np.vstack([L, R[::-1]]).round().astype(np.int32)
    pad = 24
    x0 = max(int(poly[:, 0].min()) - pad, 0); x1 = min(int(poly[:, 0].max()) + pad, w)
    y0 = max(int(poly[:, 1].min()) - pad, 0); y1 = min(int(poly[:, 1].max()) + pad, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    off = np.array([x0, y0], np.int32)

    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [poly - off], 1)

    # distance fade + colour ramp toward the stop point, per image row
    frac = (z - z[0]) / max(z[-1] - z[0], 1e-6)
    a = config.AR_ALPHA_NEAR + (config.AR_ALPHA_FAR - config.AR_ALPHA_NEAR) * frac
    col = np.tile(np.array(config.COLOR_DRIVABLE_SURFACE, np.float32), (len(z), 1))
    stop_z = rb.get("stop_z")
    if stop_z is not None:
        k = np.clip(1.0 - (stop_z - z) / config.AR_WARN_RAMP_M, 0, 1)[:, None]
        col = col * (1 - k) + np.array(config.COLOR_AR_WARN, np.float32) * k
    vc = (L[:, 1] + R[:, 1]) / 2.0
    order = np.argsort(vc)
    rows = np.arange(y0, y1, dtype=np.float32)
    a_row = np.interp(rows, vc[order], a[order])
    col_row = np.stack([np.interp(rows, vc[order], col[order, c]) for c in range(3)], axis=-1)

    # occlusion: anything standing above the ground stays in FRONT
    # occlusion: anything standing above the ground stays in FRONT
    rh, rw = mask.shape
    occ = rb.get("occlusion")
    if occ is not None:
        occ_full = cv2.resize(occ.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        occ_roi = occ_full[y0:y1, x0:x1]
    else:
        occ_roi = np.zeros((rh, rw), np.uint8)
    visible = (1 - occ_roi).astype(np.uint8)            # 1 where NOT occluded
    fill_mask = mask & visible  # (cv2.bitwise_and measured ~16 ms here; numpy is ~0.1 ms)

    # fill: alpha and colour depend only on the ROW, so the blend is a
    # per-row affine map (roi*(1-a) + col*a); computed once, then copied
    # through the (mask AND not-occluded) stencil.
    roi = frame[y0:y1, x0:x1]
    blended = (roi.astype(np.float32) * (1.0 - a_row)[:, None, None]
               + (col_row * a_row[:, None])[:, None, :])
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    roi = roi.copy()
    cv2.copyTo(blended, fill_mask, roi)

    # glowing wheel-line edges: drawn + blurred at 1/4 resolution (same
    # soft look, ~10x cheaper than a full-res wide Gaussian), occluded too
    q = 4
    small = np.zeros((max(rh // q, 1), max(rw // q, 1), 3), np.uint8)
    for E in (L, R):
        pts = ((E - off) / q).round().astype(np.int32)
        cv2.polylines(small, [pts], False, config.COLOR_AR_GLOW, 2, cv2.LINE_AA)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    glow = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    np.multiply(glow, visible[..., None], out=glow)
    roi = cv2.addWeighted(roi, 1.0, glow, 0.8, 0)

    # crisp edges + 1 m ground stripes, drawn on a copy so occlusion applies
    lines = roi.copy()
    for E in (L, R):
        cv2.polylines(lines, [E.round().astype(np.int32) - off], False, config.COLOR_AR_EDGE, 2, cv2.LINE_AA)
    labels = []
    for zi in range(int(np.ceil(z[0])), int(np.floor(z[-1])) + 1):
        j = int(np.argmin(np.abs(z - zi)))
        major = zi % 5 == 0
        p1 = tuple((L[j].round().astype(np.int32) - off).tolist())
        p2 = tuple((R[j].round().astype(np.int32) - off).tolist())
        cv2.line(lines, p1, p2, config.COLOR_AR_EDGE if major else config.COLOR_AR_STRIPE,
                 2 if major else 1, cv2.LINE_AA)
        if major:
            labels.append((f"{zi} m", (int(R[j][0]) + 8, int(R[j][1]) + 5)))
    cv2.copyTo(lines, visible, roi)
    frame[y0:y1, x0:x1] = roi

    for text, org in labels:
        _put_label(frame, text, org, config.COLOR_AR_EDGE)

    sl = rb.get("stop_line")
    if sl is not None and np.isfinite(np.array(sl)).all():
        pl, pr = (int(sl[0][0]), int(sl[0][1])), (int(sl[1][0]), int(sl[1][1]))
        cv2.line(frame, pl, pr, config.COLOR_AR_STOP, 6, cv2.LINE_AA)
        mid = ((pl[0] + pr[0]) // 2 - 36, (pl[1] + pr[1]) // 2 - 12)
        _put_label(frame, "STOP", mid, config.COLOR_AR_STOP, scale=0.9, thick=2)
    return frame


def _draw_drivable_surface(frame, path_result, color):
    """
    AR-style: fills the ACTUAL tractor-width strip of ground ahead as a
    semi-transparent surface — not a flat rectangle pasted over the
    video. Each point on both edges of the polygon is a REAL (x, y, z)
    point (the tractor's own width, offset from the verified-clear path
    center) with its OWN independently-sampled real ground height,
    projected through the real camera intrinsics (see path_planner.py) —
    that combination is what makes it follow genuine bumps and slopes in
    the image, including tracking a side-slope correctly, rather than
    floating above or cutting through the real ground, or assuming the
    ground is flat across the tractor's whole width.

    Stops exactly where the path itself stops (surface_left_px /
    surface_right_px are built by path_planner.py from the SAME loop
    that stops at blocked_at_m) — never drawn through or past an
    obstacle, same guarantee the center-line path already has.
    """
    left = path_result.get("surface_left_px") or []
    right = path_result.get("surface_right_px") or []
    if len(left) < 2 or len(right) < 2:
        return frame

    polygon = np.array(left + list(reversed(right)), dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return _blend_masked(frame, mask > 0, color, config.DRIVABLE_SURFACE_ALPHA)


def _draw_real_path(frame, path_result, color, show_label=True):
    """
    Draws the actual planned path from path_planner.plan_path(): a polyline
    through waypoints_px, starting at the bottom-center of the frame. If the
    plan reports a blocked distance, the path is drawn in the danger color
    and a "path blocked" label is shown instead of pretending the path
    continues past an obstacle it can't actually get around.
    """
    h, w = frame.shape[:2]
    waypoints_px = path_result.get("waypoints_px") or []
    blocked_at_m = path_result.get("blocked_at_m")

    draw_color = config.COLOR_PATH_DANGER if blocked_at_m is not None else color
    points = [(w // 2, h - 1)] + waypoints_px

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], draw_color, 3, cv2.LINE_AA)

    if len(points) == 1:
        # No usable waypoint at all (e.g. blocked right from MIN_RANGE_M) —
        # still mark the start point so it's clear the path is empty, not
        # just not drawn.
        cv2.circle(frame, points[0], 5, draw_color, -1)

    if blocked_at_m is not None and show_label:
        label = f"Path blocked at {blocked_at_m:.1f} m"
        cv2.putText(frame, label, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    config.COLOR_PATH_DANGER, 1, cv2.LINE_AA)


def draw_watchdog_banner(frame, degraded_duration_s):
    """
    Full-width warning strip across the top of the frame when the
    perception watchdog has tripped (see config.WATCHDOG_MAX_DEGRADED_S /
    main.py) — this is a distinct, louder signal than the normal status
    text line, since it means the safety brake is being forced on, not
    just that braking is at some computed percentage.
    """
    h, w = frame.shape[:2]
    bar_h = 40
    cv2.rectangle(frame, (0, 0), (w, bar_h), config.COLOR_PATH_DANGER, -1)
    label = f"PERCEPTION DEGRADED {degraded_duration_s:.1f}s — SAFETY BRAKE ENGAGED"
    cv2.putText(frame, label, (10, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def _draw_status_text(frame, nearest_distance_m, brake_percent, steer_suggestion_deg, speed_mps):
    lines = [
        f"Speed: {speed_mps * 3.6:.1f} km/h",
        f"Nearest obstacle: {'--' if nearest_distance_m is None else f'{nearest_distance_m:.2f} m'}",
        f"Brake: {brake_percent:.0f}%",
        f"Steer suggestion: {steer_suggestion_deg:+.1f} deg",
    ]
    y0 = 24
    for i, line in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)

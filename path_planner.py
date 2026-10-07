"""
path_planner.py — computes an actual drivable path ahead of the tractor,
plus a tractor-width DRIVABLE SURFACE (a ribbon the full width of the
tractor, hugging the real measured ground) for an AR-style overlay — see
overlay.py's _draw_drivable_surface.

This replaces the old behavior (see overlay.py's previous _draw_path_curve),
which was purely cosmetic: a curve bent by steer_suggestion_deg with no
relationship to what the ground segmentation or obstacle detection actually
saw. README (before this change) was explicit about this:
"Steering 'path' is an illustrative curve for the dashboard, not a real
planned trajectory."

Method, step by step with increasing forward distance (real-world Z, meters):
  1. Slice the point cloud to a thin band around that distance.
  2. Bin that slice sideways (real-world X, meters) into small columns.
  3. A column is "clear" if it has valid depth data and none of it is flagged
     as an obstacle (obstacle_decision.build_raised_mask) — no fixed corridor
     assumption at this stage, so the search can look wider than the tight
     braking corridor for a gap to steer into.
  4. Find every contiguous run of clear columns at least TRACTOR_WIDTH_M
     wide (the tractor must actually physically fit), and take the one whose
     center is closest to the previous waypoint (keeps the path continuous
     instead of jumping between two equally-good gaps frame to frame).
  5. The path point at that distance is the center of that run. If NO run is
     wide enough, or there's no depth data at all at that distance, the path
     stops there — that is a genuinely blocked distance, not a rendering
     artifact. The drivable surface stops at exactly the same point, for the
     same reason: it must never be drawn through or past an obstacle.
  6. Each waypoint's HEIGHT (both the center line AND the tractor's left/
     right edges) comes from the REAL measured ground points nearest that
     exact lateral position — not from the fitted plane equation. The plane
     is only used as a fallback where no real points exist. This matters on
     bumpy/rutted/sloped real fields: a single flat plane is an average
     surface, and drawing a fixed-height surface makes it visibly float
     above dips or cut through rises on the real ground as seen on camera.
     Sampling each edge's own real height independently (not just copying
     the center height across the whole width) is what lets the surface
     follow a side-slope instead of assuming the ground is locally flat
     across the tractor's full width.

Every (x, y, z) point — center line AND both edges — is projected to a
pixel coordinate using the same camera intrinsics realsense_capture.py
uses to deproject in the first place, which is what makes the edges land
on the actual floor in the image rather than a flat rectangle pasted over
it: a real point at real (x, y, z) projects to the correct screen position
for whatever height the real ground actually has there.
"""

import numpy as np

import config


def _ground_height_at(plane, x, z):
    """
    Y (camera-frame, down-positive) of the fitted ground plane at (x, z).
    a*X + b*Y + c*Z + d = 0  =>  Y = -(a*X + c*Z + d) / b
    Used ONLY as a fallback when a waypoint (or one of its edges) has no
    real measured points nearby at all — see the notes in plan_path() below.
    """
    a, b, c, d = plane
    if abs(b) < 1e-6:
        return 0.0
    return -(a * x + c * z + d) / b


def _project_to_pixel(x, y, z, intrinsics):
    if z <= 1e-6:
        return None
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (int(round(u)), int(round(v)))


def _real_height_near(x_query, cx_in_band, cy_in_band, plane, z_target, window_m):
    """
    Median real measured height (Y) of points within window_m of x_query,
    in the current z-band slice — falls back to the fitted plane if no
    real points fall in that window. Used independently for the center
    line AND for each of the tractor's two edges, so a side-slope (ground
    higher on one side than the other at the same forward distance) is
    reflected correctly instead of the whole tractor-width surface being
    drawn at one flat height copied from the center.
    """
    mask = np.abs(cx_in_band - x_query) < window_m
    if mask.any():
        return float(np.median(cy_in_band[mask]))
    return _ground_height_at(plane, x_query, z_target)


def _close_small_gaps(present_bin, max_gap):
    """
    Fills INTERIOR runs of False in present_bin that are <= max_gap long and
    have True on both sides — i.e. a small "no sample landed here" gap
    surrounded by confirmed data, not a real absence of ground. Gaps at
    either end of the array (no data on one side) are left alone, since
    those aren't bracketed by confirmed-clear data.
    """
    n = present_bin.shape[0]
    if n == 0:
        return present_bin
    result = present_bin.copy()
    i = 0
    while i < n:
        if result[i]:
            i += 1
            continue
        j = i
        while j < n and not result[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and gap_len <= max_gap:
            result[i:j] = True
        i = j
    return result


def _find_clear_runs(free_bin, min_run_len):
    """Contiguous runs of True in free_bin, at least min_run_len long."""
    runs = []
    run_start = None
    for i, val in enumerate(free_bin):
        if val and run_start is None:
            run_start = i
        elif not val and run_start is not None:
            runs.append((run_start, i))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(free_bin)))
    return [r for r in runs if (r[1] - r[0]) >= min_run_len]


def plan_path(land_mask, raised_mask, valid_depth, xyz, plane, intrinsics):
    """
    Returns dict:
      waypoints_m       : list of (x_m, z_m) real-world center-line points, near-to-far
      waypoints_px       : list of (u, v) pixel coords for the same points
      surface_left_px     : list of (u, v) — the tractor's LEFT edge at each
                            reached waypoint, real height, real projection
      surface_right_px    : list of (u, v) — same, right edge
      blocked_at_m        : distance (m) of the first step where ground IS
                            visible but no tractor-width gap exists (a real
                            obstacle / too narrow), or None.
      range_end_m         : distance (m) where usable depth data ran out
                            after real ground was seen (sensor range limit),
                            or None — NOT an obstacle, path just ends. surface_left_px/surface_right_px
                            stop at exactly this point too, same as
                            waypoints_px — the surface is never drawn
                            through or past an obstacle.
    """
    empty_result = {
        "waypoints_m": [], "waypoints_px": [],
        "surface_left_px": [], "surface_right_px": [],
        "blocked_at_m": None, "range_end_m": None,
    }

    if plane is None:
        return empty_result

    x_all = xyz[..., 0]
    y_all = xyz[..., 1]
    z_all = xyz[..., 2]

    n_steps = max(config.PATH_NUM_STEPS, 2)
    z_steps = np.linspace(config.MIN_RANGE_M, config.MAX_RANGE_M, n_steps)
    band_half = (config.MAX_RANGE_M - config.MIN_RANGE_M) / (n_steps - 1) / 2.0
    band_half = max(band_half, 0.15)  # never slice thinner than this

    # Snapshot ONCE per call: the vehicle width can be changed at runtime
    # from the UI (vehicle_profile.py) on another thread — reading it once
    # guarantees gap-fitting, search window and AR surface width all use
    # the same value within a single frame.
    vehicle_width_m = config.TRACTOR_WIDTH_M
    bin_width = max(config.PATH_LATERAL_BIN_M, 0.02)
    search_half = (vehicle_width_m / 2.0) + config.CORRIDOR_MARGIN_M + config.PATH_LATERAL_SEARCH_MARGIN_M
    n_bins = max(int(np.ceil((2.0 * search_half) / bin_width)), 1)
    bin_edges = np.linspace(-search_half, search_half, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    min_bins_for_width = max(int(np.ceil(vehicle_width_m / bin_width)), 1)
    half_tractor_width_m = vehicle_width_m / 2.0
    # Small window for sampling each edge's own real ground height — wide
    # enough to usually catch a few real points, narrow enough to stay a
    # genuinely LOCAL sample of that edge rather than blending in height
    # from elsewhere across the run.
    edge_height_window_m = max(bin_width * 2.0, 0.15)

    # Pre-restrict to the candidate region ONCE, instead of re-scanning the
    # full ~921k-pixel frame on every one of the ~PATH_NUM_STEPS per-distance
    # steps below. This does not change which points are ever considered at
    # any step — z_lo/z_hi are exactly the tightest bounds that could ever
    # be reached by ANY z_target's in-band window across the whole
    # z_steps range (MIN_RANGE_M-band_half to MAX_RANGE_M+band_half), and
    # the |x|<search_half bound is identical to what each step already
    # required — it just computes each step's boolean masks against this
    # much smaller candidate array instead of the full frame every time.
    z_lo = config.MIN_RANGE_M - band_half
    z_hi = config.MAX_RANGE_M + band_half
    candidate = valid_depth & (np.abs(x_all) < search_half) & (z_all >= z_lo) & (z_all <= z_hi)
    cx = x_all[candidate]
    cy = y_all[candidate]
    cz = z_all[candidate]
    craised = raised_mask[candidate]

    waypoints_m = []
    waypoints_px = []
    surface_left_px = []
    surface_right_px = []
    blocked_at_m = None
    range_end_m = None
    started = False  # True once the first distance with real ground data is reached
    started_with_path = False  # True once the first real waypoint has been placed
    prev_x = 0.0  # start assuming straight ahead from the tractor's centerline

    for z_target in z_steps:
        in_band = np.abs(cz - z_target) < band_half
        if not in_band.any():
            if not started:
                # Near BLIND ZONE, not an obstacle: a camera mounted on a
                # tractor physically cannot see the ground right in front
                # of the bumper (confirmed: ~3.7 m for a 2.3 m-high camera
                # pitched 14 deg down). The old code treated this as
                # "blocked at 0.5 m" on EVERY real frame, so the path and
                # drivable surface never drew at all on real hardware.
                continue
            # Data ran out AFTER real ground was already seen — the end of
            # usable sensor range (or ground dropping out of view), not a
            # detected obstacle. The path simply ends here; it is NOT
            # reported as blocked, since nothing was actually detected in
            # the way. Obstacles are still caught separately, and braking
            # never depended on this value.
            range_end_m = float(z_target)
            break
        started = True

        cx_in_band = cx[in_band]
        cy_in_band = cy[in_band]
        x_present = cx_in_band
        x_blocked = cx_in_band[craised[in_band]]

        present_bin = np.zeros(n_bins, dtype=bool)
        idx = np.clip(np.digitize(x_present, bin_edges) - 1, 0, n_bins - 1)
        present_bin[np.unique(idx)] = True
        present_bin = _close_small_gaps(present_bin, config.PATH_LATERAL_GAP_CLOSE_BINS)

        blocked_bin = np.zeros(n_bins, dtype=bool)
        if x_blocked.size:
            bidx = np.clip(np.digitize(x_blocked, bin_edges) - 1, 0, n_bins - 1)
            blocked_bin[np.unique(bidx)] = True

        free_bin = present_bin & (~blocked_bin)

        runs = _find_clear_runs(free_bin, min_bins_for_width)
        if not runs:
            if blocked_bin.any():
                # Real raised/obstacle points are in this band and no
                # tractor-width gap exists around them — genuinely blocked.
                blocked_at_m = float(z_target)
                break
            # No obstacle points at all — the gap is too narrow only
            # because DATA is missing (the band where ground first comes
            # into view is only partly filled, or depth is dropping out at
            # the far end of sensor range). Missing data is not an
            # obstacle. Confirmed as a real failure on realistic synthetic
            # depth: the partly-filled first visible band was being
            # reported as "blocked", so no path or surface drew at all.
            if not started_with_path:
                continue
            range_end_m = float(z_target)
            break
        started_with_path = True

        # HOLD THE LINE: inside each clear run, the tractor's center may sit
        # anywhere in [run_lo + half, run_hi - half] (full vehicle width
        # still inside verified-clear ground). Pick the run and the position
        # CLOSEST to the previous heading, so on open ground the path stays
        # straight and only moves over when an obstacle or edge forces it.
        # (The old rule took the run's CENTER, which zig-zagged on open
        # ground as the visible slice width changed — confirmed on realistic
        # synthetic field depth.) Safety is unchanged: the whole vehicle
        # width is still inside a verified-clear run at every waypoint.
        def fit_interval(r):
            lo = bin_edges[r[0]] + half_tractor_width_m
            hi = bin_edges[r[1]] - half_tractor_width_m
            return lo, max(hi, lo)

        def best_x_in(r):
            lo, hi = fit_interval(r)
            return float(np.clip(prev_x, lo, hi))

        best_run = min(runs, key=lambda r: abs(best_x_in(r) - prev_x))
        x_target = best_x_in(best_run)
        prev_x = x_target

        # Ground height for this waypoint: real measured points right around
        # it (median — robust to the odd noisy point); fitted plane only as
        # a fallback where no measurements exist.
        y_target = _real_height_near(x_target, cx_in_band, cy_in_band, plane, float(z_target),
                                     max(half_tractor_width_m * 0.5, edge_height_window_m))

        waypoints_m.append((float(x_target), float(z_target)))

        px = _project_to_pixel(x_target, y_target, float(z_target), intrinsics)
        if px is not None:
            waypoints_px.append(px)

        # Tractor-width drivable surface: the run is guaranteed (by
        # min_bins_for_width above) at least TRACTOR_WIDTH_M wide, so
        # these two edges — at exactly half the tractor's width either
        # side of the chosen run's center — are themselves guaranteed to
        # fall inside the verified-clear run, not past its edges.
        x_left = x_target - half_tractor_width_m
        x_right = x_target + half_tractor_width_m
        y_left = _real_height_near(x_left, cx_in_band, cy_in_band, plane, float(z_target), edge_height_window_m)
        y_right = _real_height_near(x_right, cx_in_band, cy_in_band, plane, float(z_target), edge_height_window_m)
        left_px = _project_to_pixel(x_left, y_left, float(z_target), intrinsics)
        right_px = _project_to_pixel(x_right, y_right, float(z_target), intrinsics)
        # Both edges of the SAME waypoint are kept together or not at all —
        # a polygon built from a left-only or right-only point at some
        # step would pinch the ribbon in toward the center line instead of
        # correctly showing "no valid surface data at this exact step".
        if left_px is not None and right_px is not None:
            surface_left_px.append(left_px)
            surface_right_px.append(right_px)

    return {
        "waypoints_m": waypoints_m,
        "waypoints_px": waypoints_px,
        "surface_left_px": surface_left_px,
        "surface_right_px": surface_right_px,
        "blocked_at_m": blocked_at_m,
        "range_end_m": range_end_m,
    }

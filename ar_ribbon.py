"""
ar_ribbon.py — geometry for the AR drivable-surface ribbon (drawn by
overlay._draw_ar_ribbon).

Takes the planner's verified-clear waypoints (path_planner.plan_path) and
turns them into a smooth, tractor-width ribbon lying on the REAL measured
ground:

  1. SMOOTH but FAITHFUL centerline — monotone cubic (PCHIP / Fritsch-
     Carlson) interpolation through the waypoints. Unlike a least-squares
     curve fit, it passes EXACTLY through every verified-clear waypoint and
     never overshoots between two of them, so the smoothed ribbon can't be
     bent out over an obstacle the real path swerved around.
  2. REAL ground height at each edge — measured points binned into a
     0.25 m ground grid (obstacle points excluded), 3x3-neighbourhood mean;
     the fitted plane is only a fallback where no measurements exist.
     This is what makes the ribbon hug bumps and side-slopes.
  3. ANTI-JITTER (temporal smoothing) on a fixed distance grid, with
     safety rules: a large lateral change (a genuine re-route around a new
     obstacle) SNAPS instantly instead of lagging; the far end SHRINKS
     instantly when something new blocks the way and only GROWS gradually.
     Display only — braking/steering never read any of this.
"""

import numpy as np

import config


def _pchip_slopes(x, y):
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
        return m
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def end_slope(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3 * d0):
            return 3 * d0
        return s

    m[0] = end_slope(h[0], h[1], d[0], d[1])
    m[-1] = end_slope(h[-1], h[-2], d[-1], d[-2])
    return m


def pchip(x, y, xq):
    """Monotone cubic Hermite interpolation. Outside [x0, xn] the value is
    held constant (no extrapolation)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xq = np.clip(np.asarray(xq, float), x[0], x[-1])
    m = _pchip_slopes(x, y)
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (xq - x[i]) / h
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


class _GroundGrid:
    """Measured ground height (camera-frame Y) on a coarse (x, z) grid."""

    def __init__(self, xyz, valid, raised, plane, x_lim, z_max, cell):
        self.cell, self.x_lim, self.plane = cell, x_lim, plane
        self.nx = int(np.ceil(2 * x_lim / cell)) + 1
        self.nz = int(np.ceil(z_max / cell)) + 1
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        m = valid & ~raised & (np.abs(x) < x_lim) & (z > 0) & (z < z_max)
        ix = ((x[m] + x_lim) / cell).astype(np.int64)
        iz = (z[m] / cell).astype(np.int64)
        key = iz * self.nx + ix
        size = self.nx * self.nz
        sums = np.bincount(key, weights=y[m], minlength=size)[:size].reshape(self.nz, self.nx)
        cnts = np.bincount(key, minlength=size)[:size].reshape(self.nz, self.nx).astype(float)
        # 3x3 neighbourhood sums (fills small gaps, lightly smooths noise)
        pad_s = np.pad(sums, 1); pad_c = np.pad(cnts, 1)
        self.s3 = sum(pad_s[dz:dz + self.nz, dx:dx + self.nx] for dz in range(3) for dx in range(3))
        self.c3 = sum(pad_c[dz:dz + self.nz, dx:dx + self.nx] for dz in range(3) for dx in range(3))

    def height(self, xq, zq):
        ix = np.clip(((xq + self.x_lim) / self.cell).astype(np.int64), 0, self.nx - 1)
        iz = np.clip((zq / self.cell).astype(np.int64), 0, self.nz - 1)
        c = self.c3[iz, ix]
        a, b, cc, d = self.plane
        plane_y = -(a * xq + cc * zq + d) / b if abs(b) > 1e-6 else np.zeros_like(xq)
        return np.where(c > 0, self.s3[iz, ix] / np.maximum(c, 1), plane_y)


def _project(x, y, z, intr):
    z = np.maximum(z, 1e-3)
    return np.stack([intr.fx * x / z + intr.ppx, intr.fy * y / z + intr.ppy], axis=-1)


class ArRibbon:
    def __init__(self):
        step = config.AR_RIBBON_SAMPLE_M
        self._grid = np.arange(0.0, config.MAX_RANGE_M + 2.0 + 1e-9, step)
        self.reset()

    def reset(self):
        self._x_prev = None     # smoothed centerline x on self._grid (NaN = unset)
        self._end_prev = None   # smoothed far-end distance

    def update(self, path_result, xyz, valid, raised, plane, intrinsics):
        stop_z = path_result.get("blocked_at_m")
        wm = path_result.get("waypoints_m") or []
        base = {"valid": False, "no_path": stop_z is not None and len(wm) < 2,
                "stop_z": stop_z, "occlusion": raised}
        if plane is None or len(wm) < 2:
            self.reset()
            return base

        zw = np.array([p[1] for p in wm]); xw = np.array([p[0] for p in wm])
        z0 = zw[0]
        target_end = (stop_z - config.AR_STOP_GAP_M) if stop_z is not None else zw[-1]
        target_end = max(target_end, z0 + config.AR_RIBBON_SAMPLE_M)
        # far end: shrink instantly (new obstacle), grow gradually
        if self._end_prev is None or target_end < self._end_prev:
            end = target_end
        else:
            end = min(target_end, self._end_prev + config.AR_MAX_GROW_M_PER_FRAME)
        self._end_prev = end

        idx = np.nonzero((self._grid >= z0) & (self._grid <= end))[0]
        if len(idx) < 2:
            return base
        zs = self._grid[idx]
        x_new = pchip(zw, xw, zs)

        # anti-jitter, with instant snap on a genuine re-route
        if self._x_prev is not None:
            prev = self._x_prev[idx]
            ok = ~np.isnan(prev)
            if ok.any() and np.max(np.abs(x_new[ok] - prev[ok])) < config.AR_SNAP_SHIFT_M:
                a = config.AR_TEMPORAL_ALPHA
                x_new = np.where(ok, a * x_new + (1 - a) * prev, x_new)
        self._x_prev = np.full(self._grid.shape, np.nan)
        self._x_prev[idx] = x_new

        half = config.TRACTOR_WIDTH_M / 2.0
        x_lim = half + config.CORRIDOR_MARGIN_M + config.PATH_LATERAL_SEARCH_MARGIN_M + 1.0
        grid = _GroundGrid(xyz, valid, raised, plane, x_lim, config.MAX_RANGE_M + 2.0,
                           config.AR_GROUND_CELL_M)
        xl, xr = x_new - half, x_new + half
        left = _project(xl, grid.height(xl, zs), zs, intrinsics)
        right = _project(xr, grid.height(xr, zs), zs, intrinsics)

        stop_line = None
        if stop_z is not None:
            zst = np.array([stop_z - 0.75 * config.AR_STOP_GAP_M])
            xc = pchip(zw, xw, zst)
            pl = _project(xc - half, grid.height(xc - half, zst), zst, intrinsics)[0]
            pr = _project(xc + half, grid.height(xc + half, zst), zst, intrinsics)[0]
            stop_line = (pl, pr)

        base.update(valid=True, left=left, right=right, z=zs, stop_line=stop_line, no_path=False)
        return base

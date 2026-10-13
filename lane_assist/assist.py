"""
lane_assist/assist.py — model-agnostic lane assist: pick a corridor from a
LaneEstimate, LOCK onto it, and turn the locked centreline into a steer
suggestion. Runs in the geometry process (cheap; no model here).

States:
  OFF        lane assist disabled
  SEARCHING  no lock; a valid corridor must be seen consistently for
             LOCK_ACQUIRE_S before it locks (one good frame is not enough —
             openpilot's output "thrashes" on open ground)
  LOCKED     following the corridor; every estimate consistent with it
             (within LOCK_MAX_JUMP_M at LOCK_REF_DISTANCE_M) updates it
  COASTING   no consistent estimate right now; the last corridor is kept
             for up to LOCK_COAST_S, then the lock is dropped

Steering: pure pursuit on the locked centreline, from the rear axle:
  lookahead L = clip(LOOKAHEAD_MIN_M + speed * LOOKAHEAD_TIME_S, MIN, MAX)
  y_L = lateral offset of the centreline at distance L (vehicle frame)
  curvature k = 2 * y_L / L^2,  steer = atan(WHEELBASE_M * k)
Sign convention matches the HUD: positive = steer right.
"""

import math

import numpy as np

import config
import config_lane as lcfg
from lane_assist.base import EDGE_LEFT, EDGE_RIGHT, LANE_LEFT, LANE_RIGHT

OFF, SEARCHING, LOCKED, COASTING = "OFF", "SEARCHING", "LOCKED", "COASTING"


class _Corridor:
    def __init__(self, kind, center, left, right, conf):
        self.kind, self.center, self.left, self.right, self.conf = kind, center, left, right, conf


class LaneAssist:
    def __init__(self):
        self.stations = np.arange(0.0, lcfg.LANE_RANGE_M + 1e-9, lcfg.LANE_STATION_STEP_M)
        self.reset()

    def reset(self, state=SEARCHING):
        self.state = state
        self.track = None           # locked _Corridor
        self._candidate_since = None
        self._last_candidate = None
        self._last_good_t = None
        self._last_estimate_t = None
        self._calibration = None        # (view_from_road, camera_height_m) of the estimate being used
        self._calibration = None        # (view_from_road, camera_height_m) of the estimate being used

    # ---------------------------------------------------------------- corridor

    def _ref(self, arr):
        return float(np.interp(lcfg.LOCK_REF_DISTANCE_M, self.stations, arr))

    def _resample(self, est, y):
        """Model stations -> ours. NaN beyond what the estimate covers."""
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        out = np.full(self.stations.shape, np.nan)
        if ok.sum() < 2:
            return out
        x, y = est.x[ok], y[ok]
        inside = (self.stations >= x[0]) & (self.stations <= x[-1])
        out[inside] = np.interp(self.stations[inside], x, y)
        return out

    def _corridor(self, est, kind):
        b = est.boundaries
        if kind == "plan":
            if est.plan_y is None or est.plan_conf < lcfg.PLAN_MIN_CONF:
                return None
            center = self._resample(est, est.plan_y)
            half = config.TRACTOR_WIDTH_M / 2.0
            left, right, conf = center - half, center + half, est.plan_conf
        else:
            names = (LANE_LEFT, LANE_RIGHT) if kind == "lane_lines" else (EDGE_LEFT, EDGE_RIGHT)
            min_conf = lcfg.LANE_LINE_MIN_PROB if kind == "lane_lines" else lcfg.EDGE_MIN_CONF
            if not all(n in b for n in names):
                return None
            lb, rb = b[names[0]], b[names[1]]
            conf = min(lb.conf, rb.conf)
            if conf < min_conf:
                return None
            left, right = self._resample(est, lb.y), self._resample(est, rb.y)
            width = right - left
            near = (self.stations >= 3.0) & (self.stations <= lcfg.LOCK_REF_DISTANCE_M)
            w_near = width[near]
            if (not np.isfinite(w_near).all() or (w_near < lcfg.MIN_CORRIDOR_WIDTH_M).any()
                    or np.median(w_near) > lcfg.MAX_CORRIDOR_WIDTH_M):
                return None                            # crossing / implausible edges
            center = 0.5 * (left + right)
        if not np.isfinite(self._ref(np.nan_to_num(center, nan=np.inf))):
            return None                                # does not reach the reference distance
        return _Corridor(kind, center, left, right, conf)

    def _pick(self, est):
        """The highest-priority valid corridor — and while locked, the
        highest-priority one CONSISTENT with the lock (so a better source
        takes over as soon as it agrees with what is being followed, and a
        valid-but-different corridor elsewhere cannot hijack the lock)."""
        valid = [c for c in (self._corridor(est, k) for k in lcfg.LANE_SOURCE_PRIORITY) if c is not None]
        if self.track is not None:
            ref = self._ref(self.track.center)
            for c in valid:
                if abs(self._ref(c.center) - ref) < lcfg.LOCK_MAX_JUMP_M:
                    return c
        return valid[0] if valid else None

    # ------------------------------------------------------------------- update

    def update(self, est, now, speed_mps):
        """est: LaneEstimate or None. Returns the dict the HUD/path/overlay use."""
        if est is not None and est.t == self._last_estimate_t:
            est_new = None                   # backend returned its previous tick again
        else:
            est_new = est
            if est is not None:
                self._last_estimate_t = est.t
                if "view_from_road" in est.info:
                    self._calibration = (est.info["view_from_road"], est.info["camera_height_m"])
                if "view_from_road" in est.info:
                    self._calibration = (est.info["view_from_road"], est.info["camera_height_m"])
        cand = self._pick(est_new) if est_new is not None else None

        if self.state in (LOCKED, COASTING):
            if cand is not None and abs(self._ref(cand.center) - self._ref(self.track.center)) < lcfg.LOCK_MAX_JUMP_M:
                a = lcfg.LOCK_SMOOTHING_ALPHA
                t = self.track
                blend = lambda old, new: np.where(np.isfinite(new) & np.isfinite(old), a * new + (1 - a) * old,
                                                  np.where(np.isfinite(new), new, old))
                self.track = _Corridor(cand.kind, blend(t.center, cand.center), blend(t.left, cand.left),
                                       blend(t.right, cand.right), cand.conf)
                self.state, self._last_good_t = LOCKED, now
            elif est_new is not None or self.state == COASTING:
                if now - self._last_good_t <= lcfg.LOCK_COAST_S:
                    self.state = COASTING
                else:
                    self.reset(SEARCHING)
        else:   # SEARCHING
            if cand is None:
                if est_new is not None:
                    self._candidate_since, self._last_candidate = None, None
            else:
                consistent = (self._last_candidate is not None and
                              abs(self._ref(cand.center) - self._ref(self._last_candidate.center)) < lcfg.LOCK_CONSISTENCY_M)
                if not consistent:
                    self._candidate_since = now
                self._last_candidate = cand
                if now - self._candidate_since >= lcfg.LOCK_ACQUIRE_S:
                    self.track, self.state, self._last_good_t = cand, LOCKED, now

        return self._output(speed_mps, cand)

    # ------------------------------------------------------------------ outputs

    def _steer(self, speed_mps):
        t = self.track
        x_v = self.stations + lcfg.CAMERA_X_FROM_REAR_AXLE_M
        y_v = t.center + lcfg.CAMERA_Y_OFFSET_M
        ok = np.isfinite(y_v)
        if ok.sum() < 2:
            return None, None
        x_v, y_v = x_v[ok], y_v[ok]
        L = float(np.clip(lcfg.LOOKAHEAD_MIN_M + max(speed_mps or 0.0, 0.0) * lcfg.LOOKAHEAD_TIME_S,
                          lcfg.LOOKAHEAD_MIN_M, lcfg.LOOKAHEAD_MAX_M))
        L = min(L, float(x_v[-1]))
        y_l = float(np.interp(L, x_v, y_v))
        curvature = 2.0 * y_l / (L * L)
        steer = math.degrees(math.atan(lcfg.WHEELBASE_M * curvature))
        return float(np.clip(steer, -config.MAX_STEER_SUGGESTION_DEG, config.MAX_STEER_SUGGESTION_DEG)), L

    def _output(self, speed_mps, cand):
        out = {"state": self.state, "kind": None, "conf": 0.0, "steer_deg": None, "lookahead_m": None,
               "offset_m": None, "track": None, "candidate": None, "calibration": self._calibration}
        if self.track is not None:
            t = self.track
            out.update(kind=t.kind, conf=round(float(t.conf), 2))
            out["steer_deg"], out["lookahead_m"] = self._steer(speed_mps)
            c0 = t.center[np.isfinite(t.center)]
            # vehicle position relative to the corridor centre at the camera (+ = vehicle right of centre)
            out["offset_m"] = None if c0.size == 0 else round(float(-c0[0] + lcfg.CAMERA_Y_OFFSET_M), 2)
            out["track"] = {"x": self.stations, "center": t.center, "left": t.left, "right": t.right}
        elif cand is not None:
            out["candidate"] = {"x": self.stations, "center": cand.center, "left": cand.left, "right": cand.right,
                                "kind": cand.kind}
        return out

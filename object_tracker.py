"""
object_tracker.py — assigns persistent IDs to obstacles across frames, so
closing-speed and distance history are computed per REAL tracked object,
not just "whatever's nearest this frame" (which can silently jump between
different objects as they move relative to each other).

This directly replaces the limitation obstacle_decision.ClosingSpeedEstimator
documents about itself: "Deliberately resets whenever the nearest obstacle
disappears... this is intentionally a simple 'nearest distance right now'
differentiator, not real multi-object tracking." That class is left in
obstacle_decision.py for backward compatibility (existing tests reference
it directly), but main.py's actual wiring uses ObstacleTracker instead —
a strict upgrade: real per-object tracking instead of a single global
nearest-distance heuristic.

Deliberately simple — nearest-centroid greedy matching, no Kalman filter,
no re-identification after a long occlusion, no appearance/visual
matching. Good enough for "this is probably the same blob as last frame,
while it stays visible, nearby, and doesn't jump around" — appropriate for
a handful of obstacles per frame at video frame rate, not a general
multi-object-tracking problem.
"""

import math

import config


class ObstacleTracker:
    def __init__(self, max_match_distance_m=None, max_missed_frames=None):
        self.max_match_distance_m = (
            max_match_distance_m if max_match_distance_m is not None
            else config.OBSTACLE_TRACK_MAX_MATCH_DISTANCE_M
        )
        self.max_missed_frames = (
            max_missed_frames if max_missed_frames is not None
            else config.OBSTACLE_TRACK_MAX_MISSED_FRAMES
        )
        self._tracks = {}   # track_id -> {last_x_m, last_distance_m, last_time, closing_speed_mps, missed_frames}
        self._next_id = 1

    def update(self, obstacles, now):
        """
        obstacles: list of dicts from obstacle_decision.find_obstacles()
        (bbox, distance_m, centroid_x_m, area_px) — NOT mutated.

        Returns a NEW list of dicts, each the same as the input plus:
          track_id          : int, stable across frames for the same
                               physical object (best-effort)
          closing_speed_mps : float, THIS object's own smoothed closing
                               rate (see obstacle_decision.py's
                               ClosingSpeedEstimator for the same math,
                               applied per-track here instead of globally)

        Still sorted nearest-first, same as the input convention.
        """
        unmatched_track_ids = set(self._tracks.keys())
        results = []

        # Match nearest-first: the closest obstacle is both the safety-
        # critical one to get right AND, being closest, usually has the
        # least position ambiguity — greedy matching starting there is a
        # reasonable, simple heuristic for the small obstacle counts this
        # sees per frame.
        for obs in sorted(obstacles, key=lambda o: o["distance_m"]):
            best_id = None
            best_dist = None
            for tid in unmatched_track_ids:
                track = self._tracks[tid]
                d = math.hypot(
                    obs["centroid_x_m"] - track["last_x_m"],
                    obs["distance_m"] - track["last_distance_m"],
                )
                if d <= self.max_match_distance_m and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                unmatched_track_ids.discard(best_id)
                track = self._tracks[best_id]
                dt = now - track["last_time"]
                closing_speed = track["closing_speed_mps"]
                if dt > 1e-3:
                    raw_rate = (track["last_distance_m"] - obs["distance_m"]) / dt
                    alpha = config.CLOSING_SPEED_EMA_ALPHA
                    closing_speed = alpha * raw_rate + (1.0 - alpha) * closing_speed
                track["last_x_m"] = obs["centroid_x_m"]
                track["last_distance_m"] = obs["distance_m"]
                track["last_time"] = now
                track["closing_speed_mps"] = closing_speed
                track["missed_frames"] = 0
                track["hits"] = track.get("hits", 0) + 1
                track_id = best_id
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {
                    "last_x_m": obs["centroid_x_m"],
                    "last_distance_m": obs["distance_m"],
                    "last_time": now,
                    "closing_speed_mps": 0.0,
                    "missed_frames": 0,
                    "hits": 1,
                    "confirmed": False,
                }

            # PERSISTENCE: an object only counts (confirmed=True) once it has
            # been seen in OBSTACLE_CONFIRM_FRAMES frames — one noisy frame
            # can't create an obstacle. EXCEPTION for safety: a big object
            # that is already close is confirmed on the FIRST frame, so a
            # person stepping out in front never waits for confirmation.
            # Once confirmed, a track stays confirmed for as long as it lives.
            track = self._tracks[track_id]
            if not track.get("confirmed", False):
                close_and_big = (obs["distance_m"] <= config.OBSTACLE_IMMEDIATE_CONFIRM_M
                                 and obs.get("height_m", 0.0) >= config.OBSTACLE_IMMEDIATE_MIN_HEIGHT_M)
                track["confirmed"] = close_and_big or track.get("hits", 1) >= config.OBSTACLE_CONFIRM_FRAMES
            obs_out = dict(obs)
            obs_out["track_id"] = track_id
            obs_out["closing_speed_mps"] = track["closing_speed_mps"]
            obs_out["confirmed"] = track["confirmed"]
            obs_out["track_hits"] = track.get("hits", 1)
            results.append(obs_out)

        # Age out tracks that didn't match anything this frame — a real
        # object that's briefly occluded gets a few frames of grace
        # (max_missed_frames) before its ID is dropped, so a short flicker
        # in detection doesn't immediately fragment into a "new" object.
        for tid in list(unmatched_track_ids):
            self._tracks[tid]["missed_frames"] += 1
            if self._tracks[tid]["missed_frames"] > self.max_missed_frames:
                del self._tracks[tid]

        results.sort(key=lambda o: o["distance_m"])
        return results

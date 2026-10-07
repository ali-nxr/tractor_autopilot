"""
vehicle_profile.py — runtime vehicle size (width + length).

set_vehicle_size() validates the values, applies them LIVE to config (the
vision thread reads config.TRACTOR_WIDTH_M fresh every frame, so the
corridor, path gap-fitting and AR drivable surface all update on the next
frame), and saves them to config.VEHICLE_PROFILE_PATH so they survive a
restart. load_profile() is called once at startup.

Thread safety: the UI thread writes, the vision thread reads. Each write
is a single float attribute assignment (atomic in CPython), and
path_planner snapshots the width once per call, so a change can never be
applied half-way through one frame's path computation.
"""

import json
import os

import config

FT_PER_M = 1.0 / 0.3048


def to_meters(value, unit):
    return float(value) * 0.3048 if unit == "ft" else float(value)


def from_meters(value_m, unit):
    return value_m * FT_PER_M if unit == "ft" else value_m


def validate(width_m, length_m):
    """Returns an error message string, or None if both values are OK."""
    wlo, whi = config.VEHICLE_WIDTH_RANGE_M
    llo, lhi = config.VEHICLE_LENGTH_RANGE_M
    for name, v in (("Width", width_m), ("Length", length_m)):
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
            return f"{name} is not a valid number"
    if not (wlo <= width_m <= whi):
        return f"Width must be {wlo:.1f}–{whi:.1f} m ({wlo * FT_PER_M:.1f}–{whi * FT_PER_M:.1f} ft)"
    if not (llo <= length_m <= lhi):
        return f"Length must be {llo:.1f}–{lhi:.1f} m ({llo * FT_PER_M:.1f}–{lhi * FT_PER_M:.1f} ft)"
    return None


def set_vehicle_size(width_m, length_m, save=True):
    """Validates, applies live, optionally persists. Raises ValueError on
    invalid input — nothing is changed in that case."""
    err = validate(width_m, length_m)
    if err:
        raise ValueError(err)
    config.TRACTOR_WIDTH_M = float(width_m)
    config.TRACTOR_LENGTH_M = float(length_m)
    print(f"[vehicle] Size set: width={width_m:.3f} m ({width_m * FT_PER_M:.1f} ft), "
          f"length={length_m:.3f} m ({length_m * FT_PER_M:.1f} ft)")
    if save:
        save_profile()


def save_profile(path=None):
    path = path or config.VEHICLE_PROFILE_PATH
    data = {"width_m": config.TRACTOR_WIDTH_M, "length_m": config.TRACTOR_LENGTH_M}
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the saved profile
    except OSError as e:
        print(f"[vehicle] Could not save profile (size still applied for this session): {e}")


def load_profile(path=None):
    """Loads a saved profile if one exists. Never raises: a missing,
    corrupt or out-of-range file just keeps the config.py defaults."""
    path = path or config.VEHICLE_PROFILE_PATH
    if not os.path.exists(path):
        print(f"[vehicle] No saved profile — using defaults: width={config.TRACTOR_WIDTH_M:.3f} m, "
              f"length={config.TRACTOR_LENGTH_M:.3f} m")
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        set_vehicle_size(float(data["width_m"]), float(data["length_m"]), save=False)
        print(f"[vehicle] Loaded saved profile from {path}")
        return True
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[vehicle] Ignoring invalid profile file {path} ({e}) — using defaults")
        return False

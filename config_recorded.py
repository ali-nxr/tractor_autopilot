"""
config_recorded.py — the ONLY configuration difference between running on
a live camera (main.py) and running on a recorded sequence
(main_recorded.py).

Two kinds of settings live here and they are deliberately kept apart:

  PLAYBACK_* — settings that only exist for playback (which folder, which
  sequence, loop, speed). Nothing in the perception code has ever heard of
  these; they are read by main_recorded.py and handed to the player.

  OVERRIDES — the handful of values in config.py that must CHANGE for
  recorded data to be processed sensibly. apply() writes them into the
  live config module at startup, before any perception module reads them.

Why apply() writes into config.py rather than this file being imported
instead of it: every perception module (ground_segmentation, overlay,
obstacle_decision, path_planner, ...) does `import config` and reads
attributes at call time. Pushing the overrides into that one module means
all of them see recorded-mode values with ZERO changes to any of them —
and, just as importantly, there is exactly one definition of each tunable
instead of two copies that can drift apart. The list below is short on
purpose: anything NOT in it is identical to the live system, so what you
see in playback is what the live system would have decided from the same
frames.
"""

import os

import config

# ---------------- Where the recordings are ----------------
RECORDING_DIR = r"D:\tractor_realsense_data"

# Which sequence to play. None = pick automatically (see PREFER below).
# Otherwise: a full path, a folder index ("0", "1"), or any part of the
# file name ("145904"). The --file command-line argument overrides this.
SEQUENCE = None
# Used only when SEQUENCE is None: "longest" (most footage to look at),
# "first", or "last".
PREFER = "longest"

# ---------------- Transport ----------------
PLAYBACK_LOOP = True
PLAYBACK_START_PAUSED = False
# 0.0 = run as fast as the perception pipeline can accept frames, which
# is the right default here: perception costs ~75-100 ms/frame (see
# main.py's [timing] log) against a ~30 fps recording, so anything that
# paced to the wall clock would have to SKIP most of the recording. At 0
# every recorded frame is processed and the run is reproducible — it does
# not depend on how fast this machine happens to be. Set 1.0 to watch it
# at true recorded speed instead.
PLAYBACK_SPEED = 0.0
# False = never skip a recorded frame (the producer waits for the
# consumer). Only set True if holding a chosen speed matters more than
# seeing every frame.
PLAYBACK_DROP_FRAMES = False
PLAYBACK_SKIP_SECONDS = 5.0           # the -5s / +5s buttons

# ---------------- What to pull out of the recording ----------------
PLAYBACK_WANT_MOTION = True           # use recorded accel/gyro if present
PLAYBACK_WANT_IR = False              # see OVERRIDES note on IR below
# Integrate the IMU on the RECORDING's clock instead of the wall clock.
# Keep this on: a complementary filter integrates gyro over dt, so
# replaying a 30 fps recording through a ~10 fps pipeline would otherwise
# feed roughly 3x the true dt into every step and the attitude estimate
# would drift by exactly that ratio. See playback/imu_adapter.py.
PLAYBACK_RECORDED_TIMEBASE = True

# ---------------- UI ----------------
PLAYBACK_SHOW_CONTROLS = True         # transport bar under the HUD
# The live app polls a Windows performance counter every few seconds to
# diagnose CPU throttling. That is a question about live hardware
# behaviour, not about recorded data, and it spawns a PowerShell process
# each time — off here by default.
PLAYBACK_CPU_MONITOR = False


# ---------------- Overrides applied to config.py ----------------
def _overrides():
    """
    {config attribute: recorded-mode value}. Each entry has a reason, and
    anything without one does not belong here — silently diverging from
    the live configuration would make playback results untrustworthy as
    evidence about the live system.
    """
    return {
        # These recordings contain depth + color (+ IMU) only; no infrared
        # stream was captured. Leaving the IR confidence check enabled
        # would have ground_segmentation.py expect an ir_image that can
        # never arrive. The player reports this at startup too.
        "ENABLE_IR_CONFIDENCE": False,

        "WINDOW_TITLE": "Tractor Vision — Recorded Playback",

        # Deliberately NOT overridden: YOLO_MIN_INTERVAL_S. Unthrottling
        # detection here looked right on paper — playback has no real-time
        # deadline to protect — and measured badly: it cost ~40% of
        # playback throughput (segment time 118 -> 160-190 ms/frame) for
        # effectively nothing. Perception runs slower than real time here
        # (~4 FPS, ~250 ms/frame on this machine), so the live 0.3 s
        # throttle already fires about once per frame on its own. Keeping
        # the live value also keeps one fewer behavioural difference
        # between what playback shows and what the live system would do.
    }


def apply(verbose=True):
    """
    Write the overrides into the live config module. Call this FIRST in
    main_recorded.py — before importing any perception module, so nothing
    can have cached a live-mode value.

    Returns the list of (name, old, new) actually changed.
    """
    changed = []
    for name, new in _overrides().items():
        old = getattr(config, name, None)
        if old == new:
            continue
        setattr(config, name, new)
        changed.append((name, old, new))

    if verbose and changed:
        print("[config_recorded] config.py overrides for recorded playback:")
        for name, old, new in changed:
            print(f"[config_recorded]   {name}: {old!r} -> {new!r}")
    return changed


def resolve_recording_dir():
    """RECORDING_DIR, or an explicit error naming what was looked for —
    an empty video window is a much worse way to learn the path is wrong."""
    folder = os.path.abspath(os.path.expanduser(RECORDING_DIR))
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Recording folder not found: {folder}\n"
            f"Set RECORDING_DIR in config_recorded.py, or pass --dir on the command line."
        )
    return folder

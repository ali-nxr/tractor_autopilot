"""
playback/options.py — every tunable the player itself needs, in ONE plain
dataclass with working defaults.

This is the whole reason the package is reusable: nothing under playback/
ever reads a host project's config module. The host builds a
PlaybackOptions, hands it over, and that is the entire contract. Dropping
this package into another project therefore needs no edits to anything in
here — see playback/README.md.
"""

from dataclasses import dataclass


@dataclass
class PlaybackOptions:
    """Options for RealSenseBagSource + PlaybackController."""

    # ---------------- what to play ----------------
    path: str = ""

    # ---------------- stream selection ----------------
    # The player always asks for depth + color. These two say whether it
    # should ALSO try for the optional streams; each is only attempted if
    # the recording actually advertises it, and is dropped automatically
    # if the recording advertises it but holds no messages for it (a real
    # case — see RealSenseBagSource's stream ladder).
    want_motion: bool = True          # accel + gyro (IMU)
    want_ir: bool = False             # infrared, for depth-confidence checks
    ir_stream_index: int = 1

    # ---------------- frame preparation ----------------
    align_to_color: bool = True       # rs.align(color): depth in color's frame
    color_to_bgr: bool = True         # recordings are usually RGB8; OpenCV wants BGR
    copy_frames: bool = True          # see RealSenseBagSource._to_array

    enable_depth_filters: bool = True
    spatial_filter_alpha: float = 0.5
    spatial_filter_delta: int = 20
    temporal_filter_alpha: float = 0.4
    temporal_filter_delta: int = 20
    spatial_holes_fill: int = 0       # 0 = off; hole-filling invents ground depth

    # ---------------- transport policy (PlaybackController) ----------------
    loop: bool = True                 # False = hold on the last frame at the end
    start_paused: bool = False
    # 0.0 = unthrottled: deliver frames as fast as the consumer accepts
    # them. This is the right default for perception testing — every
    # recorded frame gets processed, and the result does not depend on how
    # fast this particular machine happens to be. >0 paces to the
    # recording's own timestamps (1.0 = original speed, 0.5 = half, ...).
    speed: float = 0.0
    # False = never skip a recorded frame (deterministic; the producer
    # waits for the consumer). True = keep-newest/drop-old like a live
    # camera, which is what you want when pacing at a fixed speed matters
    # more than seeing every frame.
    drop_frames: bool = False

    # ---------------- timing ----------------
    # Drive the host's per-frame dt from the RECORDING's timestamps rather
    # than the wall clock. Keep this on: it is what makes a slow (or
    # paused, or stepped) playback produce the same speed estimates,
    # slew-limited brake/steer values and watchdog timings the live system
    # would have produced from the same frames.
    use_recorded_timebase: bool = True

    # ---------------- robustness ----------------
    probe_timeout_ms: int = 1500      # per stream-combination attempt
    frame_timeout_ms: int = 5000      # normal per-frame read

    def resolved_speed(self) -> float:
        return max(0.0, float(self.speed))

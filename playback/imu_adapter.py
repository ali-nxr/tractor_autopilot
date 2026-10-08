"""
playback/imu_adapter.py — run a host project's existing IMU fusion on the
RECORDING's clock instead of the wall clock.

Why this exists. A complementary filter integrates the gyro over dt. A
live fusion gets dt from time.time(), which is correct live because
frames arrive in real time. During playback that assumption breaks in
both directions: processing a 30 fps recording at ~10 fps feeds roughly
3x the true dt into every integration step (so pitch/roll/yaw drift ~3x
too fast), and a paused player feeds one enormous dt on resume. Stepping
frame by frame is worse still — the dt becomes however long the operator
took to press the button.

The fix is to integrate by recorded time. Rather than fork the host's
fusion math, this wraps whatever class the host already uses and swaps
only the time source, so the filter itself — the same constants, the same
blend, the same axis convention — stays byte-identical to the live one.

Usage:

    from realsense_imu import ImuFusion              # host's own class
    from playback.imu_adapter import playback_fusion

    fusion = playback_fusion(ImuFusion)()            # subclass, then build

The wrapper needs three methods from the base class: `_advance(got_new)`
(the step it replaces), `_update_orientation(dt)` and `get_state()`, plus
a `_last_ts` instance attribute it may reset. If any method is missing,
`playback_fusion` returns the base class UNCHANGED (with a warning)
rather than producing something subtly broken — playback then simply uses
wall-clock dt, which is still usable, just less faithful.
"""

_REQUIRED_METHODS = ("_advance", "_update_orientation", "get_state")


def supports_recorded_timebase(fusion_cls):
    """Whether `fusion_cls` exposes the internals the wrapper drives."""
    return all(callable(getattr(fusion_cls, name, None)) for name in _REQUIRED_METHODS)


def playback_fusion(fusion_cls, verbose=True):
    """
    Return a subclass of `fusion_cls` that integrates on recorded time.

    The subclass adds:
      set_sample_time(seconds)  — the current frame's recording timestamp
      reset_timebase()          — forget the last timestamp (after a seek
                                  or a loop wrap, where the delta is
                                  meaningless)

    RealSenseBagSource calls both automatically when they exist, so the
    host only has to pass the instance in.
    """
    if not supports_recorded_timebase(fusion_cls):
        if verbose:
            print(f"[playback] {fusion_cls.__name__} does not expose the internals needed "
                  f"to drive it from recorded time — using it unchanged (IMU integration "
                  f"will follow the wall clock, so attitude drifts with playback speed).")
        return fusion_cls

    class PlaybackImuFusion(fusion_cls):
        """`fusion_cls`, integrating by recording time rather than wall time."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sample_time_s = None

        def set_sample_time(self, seconds):
            self._sample_time_s = float(seconds)

        def reset_timebase(self):
            """Next sample starts a new timeline: dt is 0, so no spurious
            integration across a seek or a loop boundary."""
            self._sample_time_s = None
            self._last_ts = None

        def _advance(self, got_new):
            sample_time = self._sample_time_s
            if sample_time is None:
                # No recorded timestamp supplied — behave exactly like the
                # base class rather than inventing a dt.
                return super()._advance(got_new)
            previous = self._last_ts
            self._last_ts = sample_time
            dt = 0.0 if previous is None else max(0.0, sample_time - previous)
            if got_new and dt > 0:
                self._update_orientation(dt)
            return self.get_state()

    PlaybackImuFusion.__name__ = f"Playback{fusion_cls.__name__}"
    PlaybackImuFusion.__qualname__ = PlaybackImuFusion.__name__
    return PlaybackImuFusion

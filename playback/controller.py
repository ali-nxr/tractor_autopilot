"""
playback/controller.py — transport policy for a recorded sequence:
play / pause / step / seek / restart / speed / loop.

Split deliberately from RealSenseBagSource. The source answers exactly
one question ("give me the next recorded frameset"); everything about
WHEN that happens lives here. That separation is what lets the same
source be driven by a Tk app, a headless batch script, or a test that
steps one frame at a time, with no changes to either side.

Thread model: next_frame() is called from exactly one reader thread
(PlaybackCapturePipeline's). Every control method is safe to call from
any other thread — a UI thread, typically — and takes effect on the
reader's next iteration. State is guarded by one lock and the wait is on
a Condition, so pause costs nothing and resume is immediate rather than
polled.

One rule makes that safe without trusting anything about librealsense's
own locking: the SDK playback object is touched ONLY from the reader
thread. Controls just record an intent (a pending seek, a step credit)
for the reader to apply, and everything a UI needs to display — position,
frame index, pass count — is served from the last frame's metadata rather
than queried live off the device.

Discontinuities. A seek, a restart, or a loop wrap means the next frame
does NOT continue from the previous one. Anything holding frame-to-frame
state — optical-flow speed estimation, obstacle tracks, temporal depth
and plane smoothing, slew-rate limiters — would otherwise carry nonsense
across the cut (a seek backwards looks like the world lurching). The
controller detects all three cases and invokes on_discontinuity(reason)
so the host can reset that state in one place.
"""

import threading
import time

from .bag_source import PlaybackEnded


class PlaybackController:
    """Transport control around a RealSenseBagSource."""

    def __init__(self, source, options=None, on_discontinuity=None):
        self.source = source
        self.options = options if options is not None else source.options
        self._on_discontinuity = on_discontinuity

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)

        self._paused = bool(self.options.start_paused)
        self._steps_pending = 0
        self._speed = self.options.resolved_speed()
        self._loop = bool(self.options.loop)
        self._stopped = False
        self._at_end = False
        self._pending_seek = None
        self._pending_reason = None

        # Pacing state: wall-clock and recording-clock anchors for the
        # current playing stretch. Both are re-anchored on every
        # pause/resume/seek/speed change, so pacing never tries to "catch
        # up" for time spent paused.
        self._anchor_wall = None
        self._anchor_recorded = None

        # Read once, from this thread, before the reader starts — so no
        # other thread ever has to query the SDK for it.
        self._duration_s = self.source.duration_s
        self._position_s = 0.0
        self._last_recorded_time_s = None

        self.last_meta = {}
        self.frames_delivered = 0

    # ------------------------------------------------------------ controls

    def play(self):
        with self._wake:
            if self._at_end and not self._loop:
                # Pressing play at the end of a non-looping sequence means
                # "play it again", which is what a user expects.
                self._pending_seek = 0.0
                self._pending_reason = "restart"
                self._at_end = False
            self._paused = False
            self._reanchor_locked()
            self._wake.notify_all()

    def pause(self):
        with self._wake:
            self._paused = True
            self._wake.notify_all()

    def toggle_pause(self):
        if self.paused:
            self.play()
        else:
            self.pause()

    def step(self, count=1):
        """Advance `count` frames while staying paused — the useful mode
        for inspecting one specific frame's perception output."""
        with self._wake:
            self._paused = True
            self._steps_pending += max(1, int(count))
            self._wake.notify_all()

    def seek(self, seconds, reason="seek"):
        """Request a jump. Applied by the reader thread before its next
        read, so the SDK is only ever touched from one thread."""
        with self._wake:
            self._pending_seek = max(0.0, float(seconds))
            self._pending_reason = reason
            self._at_end = False
            self._wake.notify_all()

    def seek_relative(self, delta_seconds):
        self.seek(self.position_s + float(delta_seconds), reason="seek")

    def seek_fraction(self, fraction):
        self.seek(max(0.0, min(1.0, float(fraction))) * self.duration_s)

    def restart(self):
        self.seek(0.0, reason="restart")

    def set_speed(self, speed):
        """0 = unthrottled (process every frame as fast as possible);
        >0 paces to the recording's own timestamps (1.0 = real time)."""
        with self._wake:
            self._speed = max(0.0, float(speed))
            self._reanchor_locked()
            self._wake.notify_all()

    def set_loop(self, loop):
        with self._wake:
            self._loop = bool(loop)
            self.options.loop = self._loop
            self._wake.notify_all()

    def stop(self):
        with self._wake:
            self._stopped = True
            self._wake.notify_all()

    # --------------------------------------------------------------- state

    @property
    def paused(self):
        with self._lock:
            return self._paused

    @property
    def speed(self):
        with self._lock:
            return self._speed

    @property
    def loop(self):
        with self._lock:
            return self._loop

    @property
    def at_end(self):
        with self._lock:
            return self._at_end

    @property
    def stopped(self):
        with self._lock:
            return self._stopped

    @property
    def duration_s(self):
        """Length of the recording. Fixed, so it is safe to read anywhere."""
        return self._duration_s

    @property
    def position_s(self):
        """Where playback is, as of the last delivered frame — served from
        that frame's metadata, never queried off the SDK from here."""
        with self._lock:
            return self._position_s

    @property
    def progress(self):
        duration = self._duration_s
        return 0.0 if duration <= 0 else max(0.0, min(1.0, self.position_s / duration))

    def status(self):
        """One snapshot for a UI to render — never raises, never blocks on
        the reader, never touches the SDK."""
        with self._lock:
            paused, speed, loop, at_end = self._paused, self._speed, self._loop, self._at_end
            position = self._position_s
            meta = dict(self.last_meta)
            delivered = self.frames_delivered
        duration = self._duration_s
        return {
            "name": self.source.info.name,
            "paused": paused,
            "at_end": at_end,
            "loop": loop,
            "speed": speed,
            "position_s": position,
            "duration_s": duration,
            "progress": 0.0 if duration <= 0 else max(0.0, min(1.0, position / duration)),
            "frame_index": meta.get("frame_index", 0),
            "pass_index": meta.get("pass_index", 0),
            "frames_per_pass": self.source.frames_per_pass,
            "frames_delivered": delivered,
            "recorded_time_s": meta.get("recorded_time_s", 0.0),
        }

    # ------------------------------------------------------- reader thread

    def next_frame(self):
        """
        Block until the next frame should be delivered, then read it.

        Returns (frames_tuple, meta) or None when the controller was
        stopped or the sequence ended with looping off. meta gains a
        "discontinuity" key naming the reason whenever this frame does not
        continue from the previous one.
        """
        reason = self._await_turn()
        if reason == "stopped":
            return None

        if reason is None:
            self._pace()

        try:
            frames, meta = self.source.read_frames()
        except PlaybackEnded:
            with self._wake:
                self._at_end = True
                self._paused = True
            return None

        if meta.get("wrapped"):
            reason = "loop"

        meta["discontinuity"] = reason
        if reason is not None:
            self._notify_discontinuity(reason)
            with self._wake:
                self._reanchor_locked()

        with self._lock:
            self.last_meta = meta
            self._position_s = meta.get("position_s", self._position_s)
            self._last_recorded_time_s = meta.get("frame_time_s")
            self.frames_delivered += 1
        return frames, meta

    def _await_turn(self):
        """
        Wait until a frame is due. Returns the discontinuity reason for
        the frame about to be read: None (continues normally), "seek",
        "restart", or "stopped".
        """
        while True:
            with self._wake:
                if self._stopped:
                    return "stopped"

                if self._pending_seek is not None:
                    target = self._pending_seek
                    reason = self._pending_reason or "seek"
                    self._pending_seek = None
                    self._pending_reason = None
                else:
                    if self._steps_pending > 0:
                        self._steps_pending -= 1
                        self._reanchor_locked()
                        return None
                    if not self._paused:
                        return None
                    self._wake.wait(0.2)
                    continue

            # Lock deliberately RELEASED for the seek itself: it is an SDK
            # call that re-indexes the file and takes long enough to be
            # felt (~0.1 s on a 2.3 GB recording). Holding the lock across
            # it would stall every status() read, which is what the UI
            # polls — so dragging the timeline would visibly freeze the
            # window at the exact moment it should feel responsive.
            self.source.seek(target)
            with self._wake:
                self._reanchor_locked()
            # A seek always yields its frame, even while paused —
            # otherwise scrubbing the timeline would show nothing.
            return reason

    def _reanchor_locked(self):
        """Restart the pacing clock from wherever the recording is now.
        Must hold the lock."""
        self._anchor_wall = None
        self._anchor_recorded = None

    def _pace(self):
        """
        Sleep so the recording plays at `speed` x its original rate.

        Driven by the frames' own timestamps rather than an assumed frame
        rate, so a recording with uneven frame intervals (these are: ~21
        effective fps from a 30 fps request) plays back with its real
        timing, not a smoothed average.

        At speed 0 this does nothing at all: frames are delivered as fast
        as the consumer takes them, which is what makes a slow perception
        pipeline still see every recorded frame.
        """
        with self._lock:
            speed = self._speed
            anchor_wall = self._anchor_wall
            anchor_recorded = self._anchor_recorded
            recorded_now = self._last_recorded_time_s
        if speed <= 0:
            return

        if recorded_now is None:
            return
        if anchor_wall is None or anchor_recorded is None:
            with self._lock:
                self._anchor_wall = time.monotonic()
                self._anchor_recorded = recorded_now
            return

        # Where the NEXT frame should land is unknown until it is read, so
        # pace against the last frame's own position in the recording:
        # sleep off however much wall time is still owed for the recorded
        # time already consumed.
        recorded_elapsed = recorded_now - anchor_recorded
        wall_elapsed = time.monotonic() - anchor_wall
        owed = (recorded_elapsed / speed) - wall_elapsed
        if owed > 0:
            time.sleep(min(owed, 1.0))

    def _notify_discontinuity(self, reason):
        if self._on_discontinuity is None:
            return
        try:
            self._on_discontinuity(reason)
        except Exception as e:
            print(f"[playback] on_discontinuity({reason!r}) raised, continuing: {e}")

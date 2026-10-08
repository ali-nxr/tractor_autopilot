"""
playback/pipeline.py — reads a recorded sequence on its own thread and
hands frames to a processing loop, with the SAME get_latest() contract a
live camera-capture pipeline has: returns (frames_tuple, capture_time),
both None before the first frame.

The important difference from a live pipeline is what happens when
processing is slower than capture, and it is the opposite choice:

  A LIVE pipeline must drop. Frames keep arriving whether or not anyone
  is ready, and a stale frame is worthless to a safety loop, so
  keep-newest/drop-old is correct.

  A PLAYER must NOT drop by default. The recording is not going
  anywhere, and silently skipping most of it would mean the perception
  pipeline never sees the frames that matter — and that which frames got
  skipped depends on how fast this particular machine is, which makes
  results unreproducible. So the default is backpressure: the reader
  prepares exactly one frame ahead and then waits for the consumer to
  take it.

One frame ahead, not zero, is the point: reading + aligning + filtering
the next frame overlaps with processing the current one, exactly the
overlap the live pipeline gets, so nothing is given up for the
determinism. Set options.drop_frames=True for live-like behaviour when
pacing at a fixed speed matters more than seeing every frame.
"""

import threading
import time


class PlaybackCapturePipeline:
    """
    Drop-in replacement for a live capture pipeline, driven by a
    PlaybackController.

    get_latest() is the only method the processing loop needs, and it
    matches the live signature exactly, so the loop itself needs no
    playback-specific branches.
    """

    def __init__(self, controller, drop_frames=None):
        self.controller = controller
        self.source = controller.source
        self.drop_frames = (controller.options.drop_frames
                            if drop_frames is None else bool(drop_frames))

        self._lock = threading.Lock()
        self._latest_frames = None
        self._latest_capture_time = None
        self._latest_meta = None
        self._handed_out_capture_time = None
        self._slot_free = threading.Event()
        self._slot_free.set()          # nothing produced yet
        self._stop_event = threading.Event()

        self.last_error = None
        self.ended = False             # sequence finished, looping off
        self.frames_read = 0

        self._thread = threading.Thread(target=self._run, name="playback-reader",
                                        daemon=True)

    # ------------------------------------------------------------ lifecycle

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.controller.stop()
        # Release a reader parked on backpressure so it can see the stop.
        self._slot_free.set()

    def join(self, timeout=2.0):
        self._thread.join(timeout)

    # -------------------------------------------------------------- consumer

    def get_latest(self):
        """
        Non-blocking. Returns (frames_tuple, capture_time) — both None if
        nothing has been captured yet, and the SAME pair as last time if
        no new frame has arrived since.

        Taking a NEW frame frees the reader's slot, so the next frame is
        being prepared while the caller processes this one.

        Only a frame handed out for the FIRST time frees that slot. That
        distinction is the whole backpressure mechanism: a processing loop
        polls repeatedly while waiting, and if every poll freed the slot
        the reader would run a frame ahead of itself and overwrite frames
        the consumer never saw — silently dropping exactly what the
        default no-drop policy exists to preserve.

        It does assume the caller processes a frame once it has been
        handed one (which is what a capture_time change means to a
        processing loop), rather than fetching more frames first.
        """
        with self._lock:
            frames = self._latest_frames
            capture_time = self._latest_capture_time
            is_new = capture_time is not None and capture_time != self._handed_out_capture_time
            if is_new:
                self._handed_out_capture_time = capture_time
        if is_new:
            self._slot_free.set()
        return frames, capture_time

    def get_latest_meta(self):
        """Playback metadata for the frame get_latest() last returned:
        position_s, frame_index, pass_index, discontinuity. Separate from
        get_latest() so the live-compatible signature stays untouched."""
        with self._lock:
            return dict(self._latest_meta) if self._latest_meta else None

    # ---------------------------------------------------------------- reader

    def _run(self):
        while not self._stop_event.is_set():
            if not self.drop_frames:
                # Backpressure: wait until the consumer has taken the
                # previous frame. Timed wait so a stop is still noticed
                # promptly if the consumer goes away entirely.
                if not self._slot_free.wait(0.2):
                    continue
                if self._stop_event.is_set():
                    break
                self._slot_free.clear()

            try:
                result = self.controller.next_frame()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self._slot_free.set()
                time.sleep(0.05)
                continue

            if result is None:
                # Stopped, or the sequence ended with looping off. Keep the
                # last frame visible and idle — the controller has paused
                # itself, so play()/seek() can still revive this.
                self.ended = self.controller.at_end
                self._slot_free.set()
                if self._stop_event.is_set() or self.controller.stopped:
                    break
                time.sleep(0.05)
                continue

            frames, meta = result
            self.ended = False
            self.frames_read += 1
            with self._lock:
                self._latest_frames = frames
                self._latest_capture_time = time.time()
                self._latest_meta = meta

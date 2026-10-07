"""
camera_pipeline.py — runs RealSenseCapture.get_frames() on its OWN thread,
decoupled from frame PROCESSING (segmentation, obstacles, path, overlay,
etc.) on the main vision thread — see main.py. Always exposes only the
LATEST captured frame (drop-old-keep-newest), so capture and processing
genuinely run CONCURRENTLY instead of strictly sequentially: with capture
now confirmed at ~25ms/frame and processing at ~75ms/frame (real,
measured numbers — see main.py's [timing] log), sequential execution costs
~100ms/frame; pipelined, the two overlap and total throughput is bounded
by whichever is slower (~75ms here), not their sum.

This is real parallelism, not just concurrency: pyrealsense2's blocking
wait-for-frames call is a C++-backed extension call that releases
Python's GIL while blocked on hardware I/O, and the numpy/OpenCV-heavy
processing on the other thread releases the GIL for its own C-level
calls too — confirmed on real hardware (24 physical cores, ~4.8GHz
turbo, only 18-27% total CPU utilization) to have real idle capacity to
actually use, not just a theoretical benefit.
"""

import threading
import time


class CameraCapturePipeline:
    def __init__(self, camera):
        self.camera = camera
        self._latest_frames = None
        self._latest_capture_time = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        """Non-blocking. Returns (frames_tuple, capture_time) — both None
        if nothing has been captured yet."""
        with self._lock:
            return self._latest_frames, self._latest_capture_time

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frames = self.camera.get_frames()
            except Exception as e:
                self.last_error = str(e)
                continue
            with self._lock:
                self._latest_frames = frames
                self._latest_capture_time = time.time()

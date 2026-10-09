"""
ui/screen_recorder.py — records the app's own windows (the HUD and, when
config.DEBUG is on, the debug window) to video files in
config.RECORD_VIDEO_DIR, when config.RECORD_VIDEO is True.

Capture: on Windows, a BitBlt from the window's own device context. Under
the desktop compositor (DWM, always on since Windows 8) that reads the
window's own off-screen surface — so a window partly covered by another
one (e.g. the debug window over the HUD) still records its own content,
unaffected by display scaling. Measured ~5 ms per 1500x900 capture;
PrintWindow gave identical images (also under a covering window) but
took 9-24 ms. Elsewhere, falls back to a PIL screen grab of the window's
area (which does record anything covering it).

Timing: captures run on the Tk thread via root.after() at
RECORD_VIDEO_FPS, but frames are written according to WALL-CLOCK time — if
the UI stalls for half a second, the last frame is repeated to fill it — so
the video plays back at real speed. Encoding (cv2.VideoWriter) runs on a
separate thread per file, behind a small bounded queue, so the UI thread
never waits on the encoder; if the encoder falls behind, frames are dropped
and counted rather than blocking the app.
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import cv2

import config

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _gdi32 = ctypes.windll.gdi32

    # Explicit signatures: handles are 64-bit on 64-bit Python, and ctypes'
    # default int return type would silently truncate them.
    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.GetDC.restype = wintypes.HDC
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    _gdi32.CreateCompatibleDC.restype = wintypes.HDC
    _gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    _gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    _gdi32.SelectObject.restype = wintypes.HGDIOBJ
    _gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    _gdi32.DeleteDC.argtypes = [wintypes.HDC]
    _gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    _gdi32.BitBlt.restype = wintypes.BOOL
    _gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                 ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]

    class _BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    _SRCCOPY = 0x00CC0020

    def _capture_hwnd(hwnd):
        rect = wintypes.RECT()
        _user32.GetClientRect(hwnd, ctypes.byref(rect))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return None
        window_dc = _user32.GetDC(hwnd)
        mem_dc = _gdi32.CreateCompatibleDC(window_dc)
        bitmap = _gdi32.CreateCompatibleBitmap(window_dc, w, h)
        old = _gdi32.SelectObject(mem_dc, bitmap)
        try:
            if not _gdi32.BitBlt(mem_dc, 0, 0, w, h, window_dc, 0, 0, _SRCCOPY):
                return None
            header = _BITMAPINFOHEADER(biSize=ctypes.sizeof(_BITMAPINFOHEADER), biWidth=w,
                                       biHeight=-h, biPlanes=1, biBitCount=32, biCompression=0)
            buffer = np.empty((h, w, 4), np.uint8)
            if not _gdi32.GetDIBits(mem_dc, bitmap, 0, h, buffer.ctypes.data,
                                    ctypes.byref(header), 0):
                return None
            return np.ascontiguousarray(buffer[..., :3])   # BGRA -> BGR
        finally:
            _gdi32.SelectObject(mem_dc, old)
            _gdi32.DeleteObject(bitmap)
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(hwnd, window_dc)


def capture_window(toplevel):
    """The window's client area as a BGR image, or None (closed/minimized)."""
    try:
        if not toplevel.winfo_exists() or not toplevel.winfo_viewable():
            return None
        if sys.platform == "win32":
            # wm_frame() is the decorated top-level window that owns the
            # client area Tk draws into.
            return _capture_hwnd(int(toplevel.wm_frame(), 16))
        from PIL import ImageGrab
        x, y = toplevel.winfo_rootx(), toplevel.winfo_rooty()
        w, h = toplevel.winfo_width(), toplevel.winfo_height()
        grab = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        return cv2.cvtColor(np.asarray(grab), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


class _VideoTrack:
    """One output file, encoded on its own thread."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.size = None              # (w, h), fixed by the first frame
        self.frames_written = 0
        self.frames_dropped = 0
        self._writer = None
        self._queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, name=f"rec-{os.path.basename(path)}",
                                        daemon=True)
        self._thread.start()

    def submit(self, frame, copies):
        if self.size is None:
            h, w = frame.shape[:2]
            self.size = (w - w % 2, h - h % 2)   # even dimensions: some codecs require them
        if frame.shape[1::-1] != self.size:
            # The window was resized after recording started: keep the
            # file's size (a video stream cannot change resolution).
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        try:
            self._queue.put_nowait((frame, copies))
        except queue.Full:
            self.frames_dropped += copies

    def close(self):
        self._queue.put((None, 0))
        self._thread.join(timeout=10)

    def _run(self):
        while True:
            frame, copies = self._queue.get()
            if frame is None:
                break
            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*config.RECORD_VIDEO_CODEC)
                self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
                if not self._writer.isOpened():
                    print(f"[record] could not open a video writer for {self.path} "
                          f"(codec {config.RECORD_VIDEO_CODEC!r}) — this window is not being recorded")
                    self._writer = None
                    self._drain()
                    return
            for _ in range(copies):
                self._writer.write(frame)
            self.frames_written += copies
        if self._writer is not None:
            self._writer.release()

    def _drain(self):
        while True:
            frame, _ = self._queue.get()
            if frame is None:
                return


class AppRecorder:
    """
    sources: [(name, get_toplevel)] — get_toplevel() returns the Tk window
    to capture this tick, or None once that window is gone (its file is
    then finalized).
    """

    def __init__(self, root, sources):
        self.root = root
        self.sources = sources
        self.fps = float(config.RECORD_VIDEO_FPS)
        self._tracks = {}
        self._started_at = None
        self._after_id = None
        self._stopped = False
        # UI-thread health: captures run on the Tk thread, so a long gap
        # between ticks means the WHOLE UI was frozen for that long — the
        # video then repeats one frame through the gap. Reported at stop().
        self._last_tick = None
        self._max_gap_s = 0.0
        self._frozen_s = 0.0

        os.makedirs(config.RECORD_VIDEO_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._paths = {name: os.path.join(config.RECORD_VIDEO_DIR, f"{stamp}_{name}.{config.RECORD_VIDEO_EXT}")
                       for name, _ in sources}

    @property
    def started(self):
        return self._started_at is not None

    def start(self):
        if self.started:
            return
        self._started_at = time.perf_counter()
        for name, path in self._paths.items():
            print(f"[record] recording {name} window -> {os.path.abspath(path)}")
        self._tick()

    def _tick(self):
        if self._stopped:
            return
        now = time.perf_counter()
        if self._last_tick is not None:
            gap = now - self._last_tick
            self._max_gap_s = max(self._max_gap_s, gap)
            if gap > 0.5:
                self._frozen_s += gap
        self._last_tick = now
        # Frames owed so far at the target rate, from the wall clock.
        due = int((now - self._started_at) * self.fps) + 1
        for name, get_toplevel in self.sources:
            track = self._tracks.get(name)
            if track is False:
                continue                      # this window was closed; its file is done
            toplevel = get_toplevel()
            if toplevel is None:
                if track is not None:
                    track.close()
                    self._report(name, track)
                self._tracks[name] = False
                continue
            frame = capture_window(toplevel)
            if frame is None:
                continue                      # minimized for now — keep the track open
            if track is None:
                track = self._tracks[name] = _VideoTrack(self._paths[name], self.fps)
                track.next_index = due - 1    # a window that appears late starts at "now"
            copies = due - track.next_index
            if copies > 0:
                track.submit(frame, copies)
                track.next_index = due
        self._after_id = self.root.after(max(1, int(1000 / self.fps)), self._tick)

    def stop(self):
        """Finalize every file. Call before the Tk root is destroyed."""
        if self._stopped:
            return
        self._stopped = True
        if not self.started:
            print("[record] app closed before the first frame was shown — nothing recorded")
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        for name, track in self._tracks.items():
            if track:
                track.close()
                self._report(name, track)
        if self.started and self._max_gap_s > 0.5:
            total = time.perf_counter() - self._started_at
            print(f"[record] note: the UI thread was blocked for {self._frozen_s:.1f} s of {total:.1f} s "
                  f"(longest {self._max_gap_s:.1f} s) — the video repeats the last frame through those "
                  f"stretches, so it shows FEWER updates than the pipeline actually produced. The UI "
                  f"shares Python's GIL with the vision, YOLO and encoder threads.")

    def _report(self, name, track):
        seconds = track.frames_written / self.fps
        line = f"[record] {name}: {track.frames_written} frames ({seconds:.1f} s) saved to {track.path}"
        if track.frames_dropped:
            line += f"  ({track.frames_dropped} frames dropped — encoder could not keep up)"
        print(line)

"""
mp_pipeline/app.py — starts the multi-process pipeline (see workers.py)
and connects it to the existing UI.

The UI process keeps exactly the interfaces the UI already uses:
  - SharedState is filled by a bridge thread from the render process's
    messages + shared-memory display buffers;
  - PlaybackBar talks to ControllerProxy, which has the PlaybackController
    methods the bar calls, forwards them to the capture process, and
    answers status() from the latest status the capture process pushed.
So ui/app.py, the playback bar, the debug window and the recorder run
unchanged — they just receive frames from other processes.
"""

import multiprocessing as mp
import queue
import threading
import time
import types

import config
import config_lane
import config_segformer
import vehicle_profile
from mp_pipeline import workers


class _NoMainReimport:
    """
    On Windows, multiprocessing's spawn re-runs the parent's __main__ script
    in every child before calling the target. Here that script is main.py /
    main_recorded.py, whose imports pull in Tk, ultralytics and torch — seconds
    of startup and hundreds of MB per worker, for code the workers never use
    (their targets all live in mp_pipeline.workers). spawn only re-runs the
    main script if __main__ has a __file__, so hide it while starting
    processes and restore it straight after.
    """

    def __enter__(self):
        import sys
        self._main = sys.modules["__main__"]
        self._file = getattr(self._main, "__file__", None)
        if self._file is not None:
            del self._main.__file__
        return self

    def __exit__(self, *exc):
        if self._file is not None:
            self._main.__file__ = self._file
        return False


def _config_snapshot():
    """Every UPPER_CASE setting of both config modules, as the parent sees
    them now — re-applied in each spawned process (see workers._bootstrap)."""
    snap = {}
    for name, module in (("config", config), ("config_segformer", config_segformer), ("config_lane", config_lane)):
        snap[name] = {k: v for k, v in vars(module).items()
                      if k.isupper() and not callable(v) and not isinstance(v, types.ModuleType)}
    return snap


class ControllerProxy:
    """What playback.ui_controls.PlaybackBar needs from a PlaybackController."""

    def __init__(self, cmd_q, info, options):
        self._cmd_q = cmd_q
        self._status = {"paused": bool(options.start_paused), "at_end": False,
                        "loop": bool(options.loop), "speed": options.resolved_speed(),
                        "position_s": 0.0, "duration_s": info["duration_s"], "progress": 0.0,
                        "frame_index": 0, "pass_index": 0, "frames_per_pass": 0,
                        "frames_delivered": 0, "recorded_time_s": 0.0, "name": info["name"]}
        summary = info["summary"]
        self.source = types.SimpleNamespace(info=types.SimpleNamespace(summary=lambda: summary,
                                                                       name=info["name"]))

    def update_status(self, status):
        self._status = status

    def _send(self, name, *args):
        self._cmd_q.put((name, args))

    # commands
    def play(self): self._send("play")
    def pause(self): self._send("pause")
    def toggle_pause(self): self._send("toggle_pause")
    def step(self, count=1): self._send("step", count)
    def seek(self, seconds, reason="seek"): self._send("seek", seconds, reason)
    def seek_relative(self, delta): self._send("seek_relative", delta)
    def seek_fraction(self, fraction): self._send("seek_fraction", fraction)
    def restart(self): self._send("restart")
    def set_speed(self, speed): self._send("set_speed", speed); self._status["speed"] = speed
    def set_loop(self, loop): self._send("set_loop", loop); self._status["loop"] = bool(loop)
    def stop(self): pass                       # the pipeline shutdown handles this

    # state
    def status(self): return dict(self._status)
    paused = property(lambda self: self._status["paused"])
    speed = property(lambda self: self._status["speed"])
    loop = property(lambda self: self._status["loop"])
    at_end = property(lambda self: self._status["at_end"])
    duration_s = property(lambda self: self._status["duration_s"])
    position_s = property(lambda self: self._status["position_s"])


class Pipeline:
    """Owns the worker processes, queues and shared memory."""

    def __init__(self, mode, source_args=None, yolo=True, ring_count=None):
        self.mode = mode
        self.ctx = mp.get_context("spawn")
        ctx = self.ctx
        self.ring_count = ring_count or config.MP_RING_SLOTS
        self.snapshot = _config_snapshot()
        self.q_sem, self.q_geo, self.q_render = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.q_speed_in, self.q_speed_out, self.q_mask = ctx.Queue(), ctx.Queue(), ctx.Queue()
        # Lane assist plugin (config_lane): its own process, toggled at runtime
        # through a shared flag both the lane and geometry processes read.
        self.lane = bool(config.MULTIPROCESS_PIPELINE and config_lane.LANE_ASSIST_AVAILABLE)
        self.q_lane_in = ctx.Queue() if self.lane else None
        self.q_lane_out = ctx.Queue() if self.lane else None
        self.q_plane = ctx.Queue() if self.lane else None
        self.lane_enabled = ctx.Value("b", int(config_lane.LANE_ASSIST_ENABLED)) if self.lane else None
        self.free_q, self.ui_q = ctx.Queue(), ctx.Queue()
        self.capture_cmd, self.geometry_cmd = ctx.Queue(), ctx.Queue()
        self.info_q = ctx.Queue()
        self.yolo_in = ctx.Queue(maxsize=1) if yolo else None
        self.yolo_out = ctx.Queue(maxsize=2) if yolo else None
        self.source_args = source_args
        self.procs = []
        self.display = None
        self.info = None

    def start(self, timeout_s=180):
        ctx = self.ctx
        cap = ctx.Process(target=workers.capture_main, name="tv-capture", daemon=True, args=(
            self.snapshot, self.mode, self.source_args, self.ring_count,
            tuple(q for q in (self.q_sem, self.q_speed_in, self.q_lane_in) if q is not None),
            self.free_q,
            self.capture_cmd, self.info_q, self.ui_q))
        with _NoMainReimport():
            cap.start()
        self.procs.append(cap)
        # The ring's frame size is only known once the source is open.
        deadline = time.time() + timeout_s
        while self.info is None:
            try:
                self.info = self.info_q.get(timeout=0.5)
            except queue.Empty:
                if not cap.is_alive() or time.time() > deadline:
                    raise RuntimeError("capture process failed to start (see its error above)")

        from mp_pipeline.shm import DisplayRing
        import debug_view
        h, w = self.info["height"], self.info["width"]
        debug_shape = None
        if config.DEBUG:
            pw, ph = debug_view.panel_size((h, w))
            debug_shape = (2 * ph, 3 * pw, 3)
        self.display = DisplayRing.create(h, w, debug_shape, count=4)
        ring, disp = self.info["ring"], self.display.handle()

        specs = [
            ("tv-semantic", workers.semantic_main, (self.snapshot, ring, self.q_sem, self.q_geo, self.ui_q)),
            ("tv-speed", workers.speed_main, (self.snapshot, ring, self.info["nominal_dt"], self.q_speed_in,
                                              self.q_speed_out, self.q_mask, self.ui_q)),
            ("tv-geometry", workers.geometry_main, (self.snapshot, ring, self.info["intrinsics"],
                                                    self.info["nominal_dt"], self.q_geo, self.q_render,
                                                    self.geometry_cmd, self.ui_q, self.q_speed_out,
                                                    self.q_mask, self.q_lane_out, self.q_plane,
                                                    self.lane_enabled)),
            ("tv-render", workers.render_main, (self.snapshot, ring, disp, self.info["intrinsics"],
                                                self.q_render, self.free_q,
                                                self.ui_q, self.yolo_in, self.yolo_out)),
        ]
        if self.yolo_in is not None:
            specs.append(("tv-yolo", workers.yolo_main, (self.snapshot, self.yolo_in, self.yolo_out)))
        if self.lane:
            specs.append(("tv-lane", workers.lane_main, (self.snapshot, ring, self.info["intrinsics"],
                                                         self.q_lane_in, self.q_lane_out, self.q_plane,
                                                         self.lane_enabled, self.ui_q)))
        for name, target, args in specs:
            p = ctx.Process(target=target, name=name, args=args, daemon=True)
            with _NoMainReimport():
                p.start()
            self.procs.append(p)
        vehicle_profile.add_listener(lambda w_m, l_m: self.geometry_cmd.put((w_m, l_m)))
        return self.info

    def set_lane_enabled(self, on):
        if self.lane_enabled is not None:
            self.lane_enabled.value = int(bool(on))
            print(f"[lane] lane assist {'ON' if on else 'OFF'}", flush=True)

    def stop(self):
        self.capture_cmd.put(None)             # capture sends None down the chain
        deadline = time.time() + 6.0
        for p in self.procs:
            # Keep reading the UI queue while waiting: render cannot finish
            # flushing its last messages if nobody reads them.
            while p.is_alive() and time.time() < deadline:
                try:
                    self.ui_q.get(timeout=0.05)
                except queue.Empty:
                    pass
            p.join(timeout=0.1)
        for p in self.procs:
            if p.is_alive():
                print(f"[mp] {p.name} did not exit in time — terminating", flush=True)
                p.terminate()
        if self.display is not None:
            self.display.close()


def _bridge(pipeline, state, proxy, stop_event):
    """UI-process thread: render messages -> SharedState (latest frame wins)."""
    display = pipeline.display
    while not stop_event.is_set():
        try:
            msg = pipeline.ui_q.get(timeout=0.2)
        except queue.Empty:
            continue
        # Drain to the newest frame: if the UI fell behind, showing every
        # stale frame would only add latency. Non-frame messages are applied.
        latest = None
        while True:
            kind = msg.get("type")
            if kind == "frame":
                latest = msg
            elif kind == "playback_status" and proxy is not None:
                proxy.update_status(msg["status"])
            elif kind == "error":
                state.update(status_text=f"{msg['stage']} process error: {msg['text']}")
            try:
                msg = pipeline.ui_q.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            continue
        images = display.read(latest["buf"], latest["seq"])
        if images is None:
            continue                            # overwritten while reading — a newer one is coming
        annotated, debug_img = images
        hud = latest["hud"]
        if proxy is not None:
            hud = dict(hud, status_text=f"{hud['status_text']}  [recorded: {proxy.source.info.name}]")
        state.update(annotated_frame=annotated, **hud)
        if latest["imu_state"] is not None:
            state.update_imu(latest["imu_state"])
        dets, yolo_status = latest["yolo"]
        state.update_yolo(dets, status=yolo_status)
        if debug_img is not None:
            state.update_debug(debug_img, latest["probe"])


def run(mode, source_args=None, yolo=True, show_playback_bar=True):
    """Start the pipeline and the (unchanged) Tk UI; returns when the window closes."""
    from shared_state import SharedState
    from ui.app import TractorVisionApp

    pipeline = Pipeline(mode, source_args, yolo=yolo)
    print(f"[mp] starting multi-process pipeline ({mode}) — capture / semantic / speed / geometry / render"
          f"{' / yolo' if yolo else ''} each in its own process", flush=True)
    info = pipeline.start()
    state = SharedState()
    proxy = ControllerProxy(pipeline.capture_cmd, info, source_args[0]) if mode == "recorded" else None
    stop_event = threading.Event()
    threading.Thread(target=_bridge, args=(pipeline, state, proxy, stop_event),
                     daemon=True, name="mp-bridge").start()

    def on_close():
        stop_event.set()
        pipeline.stop()

    app = TractorVisionApp(state, on_close=on_close,
                           on_imu_zero=lambda: pipeline.capture_cmd.put(("zero_imu", ())),
                           on_lane_toggle=pipeline.set_lane_enabled if pipeline.lane else None,
                           lane_initially_on=bool(config_lane.LANE_ASSIST_ENABLED))
    if proxy is not None and show_playback_bar:
        import tkinter as tk
        import config_recorded
        from ui import theme
        from playback.ui_controls import PlaybackBar
        bar = PlaybackBar(app.root, proxy, palette=PlaybackBar.palette_from(theme),
                          skip_seconds=config_recorded.PLAYBACK_SKIP_SECONDS)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.reserve_space(app.root)
        bar.start_polling()
        bar.bind_keys(app.root)
        app.root.title(f"{config.WINDOW_TITLE} — {bar.describe_sequence()}")
    app.run()
    return 0

"""
main_recorded.py — Tractor Vision entry point for RECORDED RealSense
sequences. Same perception and decision logic as main.py; the only
difference is where the frames come from.

Run:
    python main_recorded.py                     # auto-pick a sequence
    python main_recorded.py --list              # show what is available
    python main_recorded.py --file 145904       # name fragment, index, or path
    python main_recorded.py --speed 1 --paused  # watch at recorded speed

"Same logic as main" is enforced structurally, not by copy-paste. The
per-frame stage sequence here is the same code path main.py runs, and the
shared helpers (downsample/upsample, the YOLO worker thread, the CPU
diagnostics) are IMPORTED from main.py rather than duplicated — so they
cannot drift apart as main.py changes. config.py is shared too: only the
few values in config_recorded.OVERRIDES differ, each with a stated reason.

Three things must be different for a recording, and each is a correctness
issue rather than a convenience:

  THE CLOCK. Every per-frame number that involves time — the optical-flow
  speed estimate (metres moved / dt), the brake and steer slew limiters
  (max change per second), obstacle track ages, the perception watchdog's
  degraded-seconds — is computed from dt. Live, dt is wall-clock time
  because frames arrive in real time. Replaying a ~30 fps recording
  through a ~10 fps pipeline would make wall-clock dt roughly 3x the
  truth, so the recording would appear to be moving 3x faster and every
  rate limit would effectively triple. So dt here comes from the FRAMES'
  OWN timestamps. The displayed FPS stays wall-clock, because that one
  really is about this machine.

  NO DROPPED FRAMES. A live loop must drop stale frames; a player must
  not, or perception never sees most of the recording and which frames it
  skipped depends on machine speed. See playback/pipeline.py.

  DISCONTINUITIES. Pausing, stepping, scrubbing and looping are the whole
  point of a player, and each one means the next frame does not follow
  the previous one. Everything holding frame-to-frame state — optical
  flow, obstacle tracks, plane and mask temporal smoothing, the AR
  ribbon, the slew limiters — is rebuilt on those cuts, or it would carry
  nonsense across the seam (a wrap looks like the world teleporting).
"""

import argparse
import sys
import threading
import time

import numpy as np

# Overrides must land in config.py BEFORE any perception module is
# imported, so nothing can capture a live-mode value at import time.
# This is why the import block below is split in two.
import config
import config_recorded

config_recorded.apply()

import main as live                                   # noqa: E402
import obstacle_decision                              # noqa: E402
import overlay                                        # noqa: E402
import path_planner                                   # noqa: E402
import vehicle_profile                                # noqa: E402
from ar_ribbon import ArRibbon                        # noqa: E402
from ground_segmentation import GroundSegmenter       # noqa: E402
from object_tracker import ObstacleTracker            # noqa: E402
from realsense_imu import ImuFusion, unavailable_state  # noqa: E402
from shared_state import SharedState                  # noqa: E402
from smoothing import SlewRateLimiter                 # noqa: E402
from speed_estimator import SpeedEstimator            # noqa: E402
from ui.app import TractorVisionApp                   # noqa: E402

from playback import (                                # noqa: E402
    PlaybackCapturePipeline,
    PlaybackController,
    PlaybackOptions,
    RealSenseBagSource,
    list_sequences,
    playback_fusion,
    select_sequence,
)

# Reused from main.py verbatim — same behaviour, one definition.
_downsample_for_processing = live._downsample_for_processing
_upsample_mask = live._upsample_mask
_upsample_mask_nearest = live._upsample_mask_nearest
YoloWorker = live.YoloWorker


class RecordedVisionWorker:
    """
    main.py's VisionWorker, reading a recorded sequence.

    The per-frame body below is deliberately a stage-for-stage match of
    VisionWorker._run: capture -> downsample -> ground segmentation ->
    speed -> obstacles + tracking -> brake/steer -> path -> AR ribbon ->
    overlay -> shared_state. Only the frame source, the clock and the
    discontinuity handling differ.
    """

    def __init__(self, shared_state: SharedState, playback_options: PlaybackOptions,
                 sequence_info, yolo_worker=None):
        self.state = shared_state
        self.playback_options = playback_options
        self.sequence_info = sequence_info
        self.yolo_worker = yolo_worker

        self.source = None
        self.controller = None
        self.capture_pipeline = None

        # Set by the reader thread, consumed by the processing loop — the
        # reset itself must happen on the thread that OWNS these objects,
        # never from the reader's callback.
        self._reset_reason = None
        self._reset_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="recorded-vision",
                                        daemon=True)

    # ------------------------------------------------------------ lifecycle

    def open(self):
        """
        Build the playback stack on the CALLING thread so a bad path or an
        unreadable file fails immediately and visibly, instead of
        disappearing into a worker thread and showing up as an app that
        just never displays anything.
        """
        fusion = None
        if self.playback_options.want_motion:
            # The project's own ImuFusion, re-timed onto the recording's
            # clock — same filter, same constants. See imu_adapter.
            fusion_cls = playback_fusion(ImuFusion)
            fusion = fusion_cls()

        self.source = RealSenseBagSource(
            self.playback_options,
            imu_fusion=fusion,
            unavailable_imu_state=unavailable_state,
            info=self.sequence_info,
        )
        self.controller = PlaybackController(
            self.source, on_discontinuity=self._request_reset
        )
        self.capture_pipeline = PlaybackCapturePipeline(self.controller)

        print("=" * 72)
        for line in self.source.status_lines():
            print(f"[playback] {line}")
        print("=" * 72)
        return self.controller

    def start(self):
        self.capture_pipeline.start()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self.capture_pipeline is not None:
            self.capture_pipeline.stop()

    def zero_imu(self):
        if self.source is not None:
            self.source.zero_imu()

    def _request_reset(self, reason):
        with self._reset_lock:
            self._reset_reason = reason

    def _take_reset(self):
        with self._reset_lock:
            reason, self._reset_reason = self._reset_reason, None
        return reason

    # -------------------------------------------------------- frame-to-frame

    class _Stateful:
        """
        Every component that carries state from one frame to the next,
        grouped so a discontinuity can rebuild all of them at once and
        none can be forgotten. Rebuilding (rather than clearing) is used
        because most of these expose no reset, and a fresh instance is
        unambiguously clean.
        """

        def __init__(self):
            self.ground_segmenter = GroundSegmenter()
            self.speed_estimator = SpeedEstimator()
            self.brake_smoother = SlewRateLimiter(config.MAX_BRAKE_CHANGE_PCT_PER_S)
            self.steer_smoother = SlewRateLimiter(config.MAX_STEER_CHANGE_DEG_PER_S)
            self.obstacle_tracker = ObstacleTracker()
            self.ar_ribbon = ArRibbon()
            self.degraded_since = None
            self.last_frame_time = None

    def _run(self):
        state = self.state
        stateful = self._Stateful()

        state.update(status_text="playback starting...")
        live._print_cpu_startup_info()

        # Nominal frame interval from the recording itself — used for the
        # single frame right after a cut, where there is no previous
        # timestamp to difference against and a dt of 0 would stall the
        # rate limiters.
        color_stream = self.sequence_info.stream("color")
        nominal_dt = 1.0 / (color_stream.fps if color_stream and color_stream.fps else config.FPS)

        timing = _StageTiming()
        last_seen_capture_time = None

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frames, capture_time = self.capture_pipeline.get_latest()
            if frames is None:
                if self.capture_pipeline.last_error:
                    state.update(status_text=f"Frame error: {self.capture_pipeline.last_error}")
                time.sleep(0.002)
                continue
            if capture_time == last_seen_capture_time:
                # No NEW frame since the last iteration: either the reader
                # is still preparing one, or playback is paused. Either
                # way the last result stays on screen.
                time.sleep(0.002)
                continue
            last_seen_capture_time = capture_time

            color_image, depth_image_m, xyz, ir_image, raw_valid_mask, imu_state = frames
            t1 = time.perf_counter()
            timing.capture += (t1 - t0)

            # A seek, a restart or a loop wrap means this frame does not
            # continue from the previous one — rebuild everything that
            # assumed it did, before any of it is used.
            reset_reason = self._take_reset()
            if reset_reason is not None:
                stateful = self._Stateful()
                print(f"[playback] {reset_reason}: per-frame state reset "
                      f"(speed estimate, obstacle tracks, plane/mask smoothing, "
                      f"AR ribbon, brake/steer limiters)")

            if imu_state is not None:
                state.update_imu(imu_state)

            if color_image is None:
                continue

            # THE RECORDING'S OWN CLOCK (see this module's docstring).
            meta = self.capture_pipeline.get_latest_meta() or {}
            now = meta.get("frame_time_s")
            if now is None:
                now = time.time()
            if stateful.last_frame_time is None:
                dt = nominal_dt
            else:
                dt = max(1e-4, now - stateful.last_frame_time)
            stateful.last_frame_time = now

            pd = config.PROCESSING_DOWNSCALE
            full_h, full_w = depth_image_m.shape[:2]
            depth_proc, xyz_proc, valid_proc, ir_proc = _downsample_for_processing(
                depth_image_m, xyz, raw_valid_mask, ir_image, pd
            )

            seg = stateful.ground_segmenter.segment(
                depth_proc, xyz_proc, ir_image=ir_proc, raw_valid_mask=valid_proc
            )
            t2 = time.perf_counter()
            timing.segment += (t2 - t1)
            land_mask_proc = seg["land_mask"]
            boundary_contour_proc = seg["boundary_contour"]
            dist_to_plane_proc = seg["dist_to_plane"]
            valid_depth_proc = seg["valid_depth"]
            confidence_mask_proc = seg["confidence_mask"]

            total_px = land_mask_proc.size
            land_coverage_pct = 100.0 * float(np.sum(land_mask_proc)) / total_px if total_px else 0.0

            plane_found = seg["plane"] is not None
            ground_ok = plane_found and (land_coverage_pct >= config.MIN_LAND_COVERAGE_PCT)
            if ground_ok:
                stateful.degraded_since = None
                degraded_duration = 0.0
            else:
                if stateful.degraded_since is None:
                    stateful.degraded_since = now
                degraded_duration = now - stateful.degraded_since
            watchdog_triggered = degraded_duration >= config.WATCHDOG_MAX_DEGRADED_S

            land_mask_full = _upsample_mask(land_mask_proc, full_w, full_h, pd)
            speed_mps = stateful.speed_estimator.update(
                color_image, xyz, land_mask_full, dt, imu_state=imu_state)
            t3 = time.perf_counter()
            timing.speed += (t3 - t2)

            corridor_mask_proc = obstacle_decision.build_corridor_mask(xyz_proc, valid_depth_proc)
            raised_mask_proc = obstacle_decision.build_raised_mask(
                dist_to_plane_proc, valid_depth_proc, land_mask_proc, xyz=xyz_proc)
            untrusted_proc = (obstacle_decision.glare_mask(color_image, land_mask_proc.shape)
                              if config.OBSTACLE_REJECT_GLARE else None)
            blobs, blob_labels = obstacle_decision.detect_obstacle_blobs(
                raised_mask_proc, xyz_proc, dist_to_plane_proc, untrusted_proc)
            tracked = stateful.obstacle_tracker.update(blobs, now)
            confirmed = [b for b in tracked if b["confirmed"]]
            obstacle_mask_proc = obstacle_decision.blobs_mask(blob_labels, confirmed)
            obstacles = obstacle_decision.corridor_obstacles(
                confirmed, blob_labels, corridor_mask_proc, xyz_proc)

            display_obstacles = _scale_bboxes(confirmed, pd)
            obstacles = _scale_bboxes(obstacles, pd)

            nearest_distance = obstacles[0]["distance_m"] if obstacles else None
            closing_speed_mps = obstacles[0]["closing_speed_mps"] if obstacles else 0.0

            brake_raw = obstacle_decision.compute_brake_percent(
                nearest_distance, speed_mps, closing_speed_mps=closing_speed_mps
            )
            steer_raw = obstacle_decision.compute_steer_suggestion(
                obstacles[0] if obstacles else None
            )

            if watchdog_triggered:
                brake_percent = stateful.brake_smoother.force(100.0)
                steer_deg = stateful.steer_smoother.force(0.0)
            else:
                brake_percent = stateful.brake_smoother.update(brake_raw, dt)
                steer_deg = stateful.steer_smoother.update(steer_raw, dt)
            t4 = time.perf_counter()
            timing.obstacles += (t4 - t3)

            path_result = path_planner.plan_path(
                land_mask_proc, obstacle_mask_proc, valid_depth_proc, xyz_proc,
                seg["plane"], self.source.intrinsics
            )
            t5 = time.perf_counter()
            timing.path += (t5 - t4)

            corridor_mask_full = _upsample_mask(corridor_mask_proc, full_w, full_h, pd)
            confidence_mask_full = _upsample_mask_nearest(confidence_mask_proc, full_w, full_h, pd)
            boundary_contour_full = (
                boundary_contour_proc * pd
                if (pd > 1 and boundary_contour_proc is not None)
                else boundary_contour_proc
            )

            ribbon = None
            if config.AR_ENABLED:
                ribbon = stateful.ar_ribbon.update(
                    path_result, xyz_proc, valid_depth_proc, obstacle_mask_proc,
                    seg["plane"], self.source.intrinsics)

            annotated = overlay.draw_annotations(
                color_image, land_mask_full, boundary_contour_full, corridor_mask_full,
                display_obstacles, brake_percent, steer_deg, path_result, speed_mps=speed_mps,
                confidence_mask=confidence_mask_full, raw_valid_mask=raw_valid_mask,
                ribbon=ribbon,
            )
            if watchdog_triggered:
                overlay.draw_watchdog_banner(annotated, degraded_duration)

            if self.yolo_worker is not None:
                self.yolo_worker.submit_frame(color_image)
                live.draw_detections(annotated, state.get_yolo_detections())

            t6 = time.perf_counter()
            timing.overlay += (t6 - t5)

            if watchdog_triggered:
                status_text = f"PERCEPTION DEGRADED {degraded_duration:.1f}s — safety brake engaged"
            elif ground_ok:
                status_text = "running"
            elif plane_found:
                status_text = "low ground visibility (coasting)"
            else:
                status_text = "no ground plane found (coasting)"
            status_text = f"{status_text}  [recorded: {self.sequence_info.name}]"

            state.update(
                annotated_frame=annotated,
                speed_mps=speed_mps,
                obstacles=obstacles,
                nearest_obstacle_m=nearest_distance,
                closing_speed_mps=closing_speed_mps,
                brake_percent=brake_percent,
                steer_suggestion_deg=steer_deg,
                land_coverage_pct=land_coverage_pct,
                path_blocked_at_m=path_result.get("blocked_at_m"),
                watchdog_degraded=watchdog_triggered,
                degraded_duration_s=degraded_duration,
                status_text=status_text,
            )
            t7 = time.perf_counter()
            timing.state_update += (t7 - t6)

            timing.frames += 1
            if timing.frames >= 10:
                print(timing.report(extra=self._progress_note()))
                timing.reset()

        if self.capture_pipeline is not None:
            self.capture_pipeline.stop()
        if self.source is not None:
            self.source.stop()

    def _progress_note(self):
        status = self.controller.status()
        total = status["frames_per_pass"]
        frame = f"{status['frame_index']}" + (f"/{total}" if total else "")
        return (f"frame={frame}  t={status['position_s']:.1f}/{status['duration_s']:.1f}s"
                + (f"  pass={status['pass_index'] + 1}" if status["pass_index"] else "")
                + ("  PAUSED" if status["paused"] else ""))


def _scale_bboxes(obstacle_list, downscale):
    """
    bbox is in PROCESSING-resolution pixels -> scale for the full-res
    display. distance/centroid/height/width are real metres and need no
    scaling. (Same as main.py's inner _bbox_to_full.)
    """
    out = []
    for o in obstacle_list:
        o = dict(o)
        bx, by, bw, bh = o["bbox"]
        o["bbox"] = (bx * downscale, by * downscale, bw * downscale, bh * downscale)
        out.append(o)
    return out


class _StageTiming:
    """main.py's inline per-stage timing accumulators, as one object so
    the loop body above stays readable."""

    _STAGES = ("capture", "segment", "speed", "obstacles", "path", "overlay", "state_update")

    def __init__(self):
        self.reset()

    def reset(self):
        for stage in self._STAGES:
            setattr(self, stage, 0.0)
        self.frames = 0

    def report(self, extra=""):
        n = max(1, self.frames)
        ms = {stage: getattr(self, stage) / n * 1000 for stage in self._STAGES}
        total = sum(ms.values())
        line = ("[timing] " + "  ".join(f"{stage}={ms[stage]:.1f}" for stage in self._STAGES)
                + f"  (all ms/frame)  TOTAL={total:.1f}ms/frame (~{1000 / max(total, 1e-6):.1f} FPS)")
        if extra:
            line += f"  |  {extra}"
        return line


def _build_playback_options(args, info):
    """config_recorded defaults, with any explicit command-line override
    on top. Motion/IR are additionally gated on what the sequence holds,
    so a recording without them starts clean instead of starting with a
    failed attempt."""
    return PlaybackOptions(
        path=info.path,
        want_motion=config_recorded.PLAYBACK_WANT_MOTION and info.has_motion,
        want_ir=config_recorded.PLAYBACK_WANT_IR and info.has_ir,
        enable_depth_filters=config.ENABLE_DEPTH_FILTERS,
        spatial_filter_alpha=config.SPATIAL_FILTER_ALPHA,
        spatial_filter_delta=config.SPATIAL_FILTER_DELTA,
        temporal_filter_alpha=config.TEMPORAL_FILTER_ALPHA,
        temporal_filter_delta=config.TEMPORAL_FILTER_DELTA,
        ir_stream_index=config.IR_STREAM_INDEX,
        loop=(config_recorded.PLAYBACK_LOOP if args.loop is None else args.loop),
        start_paused=(config_recorded.PLAYBACK_START_PAUSED or args.paused),
        speed=(config_recorded.PLAYBACK_SPEED if args.speed is None else args.speed),
        drop_frames=config_recorded.PLAYBACK_DROP_FRAMES,
        use_recorded_timebase=config_recorded.PLAYBACK_RECORDED_TIMEBASE,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Tractor Vision perception pipeline on a recorded "
                    "RealSense sequence.")
    parser.add_argument("--dir", default=None,
                        help="folder of recordings (default: config_recorded.RECORDING_DIR)")
    parser.add_argument("--file", default=None,
                        help="which sequence: a path, a folder index (0, 1, ...), "
                             "or part of the file name")
    parser.add_argument("--list", action="store_true",
                        help="list the sequences in the folder and exit")
    parser.add_argument("--speed", type=float, default=None,
                        help="0 = as fast as perception allows (default), 1 = recorded speed")
    parser.add_argument("--paused", action="store_true", help="start paused on the first frame")
    loop_group = parser.add_mutually_exclusive_group()
    loop_group.add_argument("--loop", dest="loop", action="store_const", const=True, default=None,
                            help="replay from the start at the end (default)")
    loop_group.add_argument("--no-loop", dest="loop", action="store_const", const=False,
                            help="hold on the last frame at the end")
    parser.add_argument("--no-yolo", action="store_true", help="skip object detection")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    folder = args.dir or config_recorded.resolve_recording_dir()

    if args.list:
        sequences = list_sequences(folder)
        if not sequences:
            print(f"No recorded sequences found in {folder}")
            return 1
        print(f"Recorded sequences in {folder}:")
        for index, info in enumerate(sequences):
            print(f"  [{index}] {info.summary()}")
            for stream in info.streams:
                print(f"        {stream.describe()}")
        return 0

    try:
        info = select_sequence(folder, args.file or config_recorded.SEQUENCE,
                               prefer=config_recorded.PREFER)
    except (FileNotFoundError, ValueError) as e:
        print(f"[main_recorded] {e}")
        return 1

    # Same startup order as main.py: cap OpenCV's internal threading
    # before anything else (including CUDA setup), then restore the saved
    # vehicle size so the very first frame uses the right corridor width.
    live._limit_cv2_threads()
    vehicle_profile.load_profile()

    shared_state = SharedState()

    yolo_worker = None
    if config.YOLO_ENABLED and not args.no_yolo:
        yolo_worker = YoloWorker(shared_state)
        yolo_worker.start()

    worker = RecordedVisionWorker(
        shared_state, _build_playback_options(args, info), info, yolo_worker=yolo_worker)
    try:
        controller = worker.open()
    except Exception as e:
        print(f"[main_recorded] Could not open {info.name}: {type(e).__name__}: {e}")
        if yolo_worker is not None:
            yolo_worker.stop()
        return 1
    worker.start()

    cpu_monitor = None
    if config_recorded.PLAYBACK_CPU_MONITOR:
        cpu_monitor = live.CpuPerfMonitorThread(interval_s=3.0)
        cpu_monitor.start()

    def on_close():
        worker.stop()
        if yolo_worker is not None:
            yolo_worker.stop()
        if cpu_monitor is not None:
            cpu_monitor.stop()
        controller.stop()

    app = TractorVisionApp(shared_state, on_close=on_close, on_imu_zero=worker.zero_imu)

    if config_recorded.PLAYBACK_SHOW_CONTROLS:
        # Imported here, not at module scope, so the player's Tk
        # dependency stays optional for headless use of this file.
        import tkinter as tk
        from ui import theme
        from playback.ui_controls import PlaybackBar

        bar = PlaybackBar(app.root, controller,
                          palette=PlaybackBar.palette_from(theme),
                          skip_seconds=config_recorded.PLAYBACK_SKIP_SECONDS)
        # side=BOTTOM packs it ABOVE the already-packed status bar, so the
        # transport sits between the video and the status strip.
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        # ...and give the window the height back, or the bar takes it from
        # the fixed-size side panels instead and clips their lower rows.
        bar.reserve_space(app.root)
        bar.start_polling()
        bar.bind_keys(app.root)
        # The bar has no row to spare for "what am I looking at", and the
        # title bar is free real estate.
        app.root.title(f"{config.WINDOW_TITLE} — {bar.describe_sequence()}")

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

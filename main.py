"""
main.py — Tractor Vision entry point. Vision + decision-logic only, no
actuator control.

One background thread (VisionWorker) does: camera capture -> ground
segmentation -> obstacle detection + tracking -> drivable path planning ->
brake + steer -> overlay drawing -> shared_state update, every frame. The
Tkinter UI runs on the main thread and reads from shared_state each tick.

Deliberately lean: no object detection, no manual calibration, no session
recording, no analytics — just the core perception/decision loop, kept as
fast and correct as possible.

Run:
    python main.py
"""

import threading
import time
import subprocess
import numpy as np
import cv2

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import ctypes
    _KERNEL32 = ctypes.windll.kernel32
    _HAS_WIN32_CORE_API = True
except Exception:
    _HAS_WIN32_CORE_API = False


def _current_core_index():
    """
    Best-effort: which logical CPU core THIS THREAD is actually running on
    right now. Tries the raw Windows API first (GetCurrentProcessorNumber
    — the same low-level call Task Manager's own per-core view is built
    on) since psutil.Process().cpu_num() turned out to fail silently on
    Windows in practice (confirmed: the field never appeared in a real
    log, with no error shown, because the exception around it was being
    swallowed) — not just a theoretical concern. Falls back to psutil,
    then returns a clearly-labeled failure reason rather than just
    vanishing from the log the way it did before.
    """
    if _HAS_WIN32_CORE_API:
        try:
            return str(_KERNEL32.GetCurrentProcessorNumber())
        except Exception as e:
            return f"win32-error({e})"
    if _PSUTIL_AVAILABLE:
        try:
            return str(psutil.Process().cpu_num())
        except Exception as e:
            return f"psutil-error({e})"
    return "unavailable(neither win32 nor psutil worked)"


def _downsample_for_processing(depth_image_m, xyz, raw_valid_mask, ir_image, downscale):
    """
    Reduces the WORKING resolution for the expensive geometric math
    (segmentation, obstacles, path search) while capture and the final
    displayed video both stay at full resolution — capture has to stay
    untouched since color/depth's specific resolution pairing is what
    makes the IMU motion streams resolve at all (see realsense_capture.py).
    Real-world X/Y/Z values in xyz are unaffected by this — downscaling
    only reduces SAMPLE DENSITY, not correctness of the samples that
    remain, which is why path_planner's meters-based search and the
    obstacle distances it reports need no compensation for this at all.
    NEAREST-sample downsampling, deliberately NOT INTER_AREA averaging.
    An earlier version averaged (INTER_AREA) — confirmed on realistic
    synthetic depth to create PHANTOM 3D points: invalid depth pixels are
    stored as 0, so any 2x2 block that was half valid / half invalid got
    averaged into a fake point at roughly HALF its true distance, floating
    in mid-air (measured +1.7 m off the ground plane). Averaging also
    blends foreground and background across object edges into points that
    exist on neither. Real RealSense depth is full of small holes, so this
    scattered phantom obstacles across real frames. Nearest-sampling only
    ever keeps a point the sensor actually measured.
    """
    if downscale <= 1:
        return depth_image_m, xyz, raw_valid_mask, ir_image

    h, w = depth_image_m.shape[:2]
    proc_w, proc_h = max(1, w // downscale), max(1, h // downscale)
    nn = cv2.INTER_NEAREST

    depth_proc = cv2.resize(depth_image_m, (proc_w, proc_h), interpolation=nn)
    xyz_proc = cv2.resize(xyz, (proc_w, proc_h), interpolation=nn)
    valid_proc = cv2.resize(raw_valid_mask.astype(np.uint8), (proc_w, proc_h), interpolation=nn) > 0
    # Belt-and-braces: a sample is only valid if its own depth is real.
    valid_proc &= depth_proc > 0

    ir_proc = None
    if ir_image is not None:
        ir_proc = cv2.resize(ir_image, (proc_w, proc_h), interpolation=nn)

    return depth_proc, xyz_proc, valid_proc, ir_proc


def _upsample_mask(mask, full_w, full_h, downscale):
    """Boolean mask back to full resolution for display — bilinear +
    rethreshold gives a smoother boundary than nearest-neighbor would,
    matching the same technique ground_segmentation.py's own internal
    downscale step already uses for exactly this reason."""
    if downscale <= 1:
        return mask
    mask_u8 = (mask.astype(np.uint8)) * 255
    return cv2.resize(mask_u8, (full_w, full_h), interpolation=cv2.INTER_LINEAR) > 127


def _upsample_mask_nearest(mask, full_w, full_h, downscale):
    """Same as _upsample_mask but nearest-neighbor — used only for the
    confidence mask, which feeds a coarse "was this pixel trustworthy"
    overlay highlight, not a boundary that benefits from smoothing."""
    if downscale <= 1 or mask is None:
        return mask
    mask_u8 = (mask.astype(np.uint8)) * 255
    return cv2.resize(mask_u8, (full_w, full_h), interpolation=cv2.INTER_NEAREST) > 127


import config
from shared_state import SharedState
from realsense_capture import RealSenseCapture
from ground_segmentation import GroundSegmenter
import obstacle_decision
import path_planner
import overlay
from speed_estimator import SpeedEstimator
from smoothing import SlewRateLimiter
from object_tracker import ObstacleTracker
from camera_pipeline import CameraCapturePipeline
from yolo_detector import AgriYoloDetector, draw_detections
import vehicle_profile
from ar_ribbon import ArRibbon
from ui.app import TractorVisionApp


def _limit_cv2_threads():
    """
    Caps OpenCV's own internal per-call threading (config.CV2_NUM_THREADS)
    — this was proven necessary earlier in this project's history (real,
    measured effect, not theoretical) but was lost when the codebase was
    stripped down and rebuilt: only the matching torch-side cap got
    restored (scoped inside yolo_detector.py, affecting YOLO's own CPU-side
    work only), not this one — which matters for the MAIN vision loop's
    own heavy cv2 calls (connectedComponentsWithStats, morphologyEx — the
    core cost of the segment/obstacles stages). OpenCV defaults to
    auto-threading across every available core for each call; with YOLO's
    own thread now also active, that's real oversubscription again. Called
    once, at startup, before anything else runs.
    """
    try:
        cv2.setNumThreads(config.CV2_NUM_THREADS)
        print(f"[main] cv2.setNumThreads({config.CV2_NUM_THREADS}) "
              f"(was going to auto-use all {cv2.getNumberOfCPUs()} logical CPUs by default)")
    except Exception as e:
        print(f"[main] Could not limit cv2 threads (non-fatal): {e}")


def _print_cpu_startup_info():
    """
    Prints what the OS itself reports for CPU cores + clock speed range,
    once at startup — the same numbers Task Manager's Performance tab
    shows, but in the log directly instead of something that has to be
    read off screen and typed back. If this alone shows a max frequency
    far below the CPU's actual rated boost clock, that's diagnostic on
    its own, before a single frame is even processed.
    """
    if not _PSUTIL_AVAILABLE:
        print("[cpu] psutil not installed — run: pip install psutil")
        print("[cpu] (needed for the CPU clock-speed diagnostics below)")
        return
    try:
        freq = psutil.cpu_freq()
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        print("=" * 72)
        print(f"[cpu] Logical cores: {logical}  Physical cores: {physical}")
        if freq is not None:
            print(f"[cpu] Reported clock — current: {freq.current:.0f} MHz  "
                  f"min: {freq.min:.0f} MHz  max: {freq.max:.0f} MHz")
            print("[cpu] If 'max' here is far below your CPU's actual rated boost clock, "
                  "Windows/BIOS/vendor power management is capping it — that's the bottleneck, "
                  "not this application.")
        else:
            print("[cpu] psutil could not read frequency info on this system.")
        print("=" * 72)
    except Exception as e:
        print(f"[cpu] Could not read CPU info: {e}")


class CpuPerfMonitorThread:
    """
    Every few seconds, on its OWN thread (never touches the vision loop's
    timing), asks Windows for '% Processor Performance' — the same
    hardware-performance-counter-based metric Task Manager's live CPU
    graph is actually built on. This is a materially different, more
    trustworthy source than psutil.cpu_freq(): that call reads a STATIC
    field from WMI that's well-documented to often just report the CPU's
    nominal spec rather than real-time turbo state (which is exactly
    consistent with seeing an unchanging 2200 MHz across dozens of samples
    regardless of load) — this instead reads the live counter Windows
    itself uses, so a real answer either way: if this also never rises
    above ~100% of base clock while the app is visibly slow, that's real,
    strong evidence of an actual power-limit cap, not a reporting quirk.
    """

    def __init__(self, interval_s=3.0):
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self.interval_s)

    def _poll_once(self):
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Counter '\\Processor Information(_Total)\\% Processor Performance')"
                 ".CounterSamples.CookedValue"],
                capture_output=True, text=True, timeout=5,
            )
        except FileNotFoundError:
            print("[cpu-perf-counter] powershell not found — Windows-only diagnostic, skipping.")
            self._stop_event.set()
            return
        except subprocess.TimeoutExpired:
            print("[cpu-perf-counter] PowerShell query timed out (>5s) — skipping this round.")
            return
        except Exception as e:
            print(f"[cpu-perf-counter] Could not launch PowerShell: {e}")
            return

        # From here on, ALWAYS print something — the previous version had a
        # silent gap right here (empty stdout + empty stderr fell through
        # both branches and printed nothing at all), which is exactly the
        # kind of failure this whole diagnostic effort keeps running into.
        # Raw returncode/stdout/stderr are shown on ANY path that isn't a
        # clean successful parse, so the next round can't go silent again.
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            try:
                pct = float(stdout)
                print(f"[cpu-perf-counter] % Processor Performance (vs. base clock): {pct:.0f}%  "
                      f"(100% = running at rated base clock, >100% = turbo boost is actually engaging)")
            except ValueError:
                print(f"[cpu-perf-counter] Got output but couldn't parse it as a number — "
                      f"returncode={result.returncode}  stdout={stdout!r}  stderr={stderr!r}")
        else:
            print(f"[cpu-perf-counter] No stdout from PowerShell — "
                  f"returncode={result.returncode}  stdout={stdout!r}  stderr={stderr!r}")


class YoloWorker:
    """
    Runs object detection on its OWN thread, on the GPU — see
    yolo_detector.py. Always processes the LATEST submitted frame (same
    drop-old-keep-newest pattern as camera_pipeline.py): detection does
    not need to keep up with the full vision-loop frame rate, and this
    must never become a bottleneck for the safety-critical loop, which
    never waits on this thread for anything. Purely an overlay — its
    output never reaches braking, steering, or the perception watchdog.
    """

    def __init__(self, shared_state: SharedState):
        self.state = shared_state
        self.detector = AgriYoloDetector()
        self._latest_frame = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._new_frame_event.set()  # wake the thread so it can see the stop

    def submit_frame(self, frame_bgr):
        """Non-blocking — called from the vision loop every frame. Cheap:
        stores a reference only, never blocks on the GPU or waits for a
        previous detection to finish."""
        with self._lock:
            self._latest_frame = frame_bgr
        self._new_frame_event.set()

    def _run(self):
        if not self.detector.available:
            self.state.update_yolo([], status=f"unavailable: {self.detector.error}")
            return
        self.state.update_yolo([], status=f"running ({self.detector.mode})")

        last_run = 0.0
        while not self._stop_event.is_set():
            self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                frame = self._latest_frame

            if frame is None:
                continue

            now = time.time()
            if now - last_run < config.YOLO_MIN_INTERVAL_S:
                continue
            last_run = now

            detections = self.detector.detect(frame)
            self.state.update_yolo(detections, status=f"running ({self.detector.mode})")


class VisionWorker:
    def __init__(self, shared_state: SharedState, yolo_worker: YoloWorker = None):
        self.state = shared_state
        self.yolo_worker = yolo_worker
        self.camera = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def zero_imu(self):
        if self.camera is not None:
            self.camera.zero_imu()

    def _run(self):
        try:
            self.camera = RealSenseCapture()
        except Exception as e:
            self.state.update(status_text=f"Camera error: {e}")
            return

        self.state.update(status_text="running")
        _print_cpu_startup_info()

        # Camera capture runs on its OWN thread (see camera_pipeline.py) —
        # capture (confirmed ~25ms/frame) and processing (confirmed
        # ~75-100ms/frame — see the [timing] log) genuinely overlap
        # instead of running strictly sequentially. This is real,
        # measured: verified separately before being wired in here.
        capture_pipeline = CameraCapturePipeline(self.camera)
        capture_pipeline.start()

        ground_segmenter = GroundSegmenter()
        speed_estimator = SpeedEstimator()
        brake_smoother = SlewRateLimiter(config.MAX_BRAKE_CHANGE_PCT_PER_S)
        steer_smoother = SlewRateLimiter(config.MAX_STEER_CHANGE_DEG_PER_S)
        obstacle_tracker = ObstacleTracker()
        ar_ribbon = ArRibbon()  # stateful: anti-jitter across frames
        last_frame_time = time.time()

        # Perception watchdog: tracks how long perception has been
        # continuously degraded — either no ground plane found, OR a plane
        # WAS found but land coverage in the frame is suspiciously low
        # (camera fogged/obstructed). If this persists past
        # WATCHDOG_MAX_DEGRADED_S, force full brake + straight steering
        # rather than acting on stale/absent data.
        degraded_since = None

        # Minimal, inline timing — no separate module, just enough to see
        # where time is actually going. CPU throttling is conclusively
        # ruled out (confirmed via the real Windows performance counter,
        # not the unreliable static WMI value), and the full stage
        # breakdown showed cost distributed fairly evenly across every
        # stage below — real work, not one fixable bug. This block is
        # deliberately fully sequential: an earlier attempt at running
        # speed_est/obstacles/path_plan concurrently was found, on review,
        # to rest on an incorrect premise (path_plan actually depends on
        # raised_mask, which comes FROM the obstacles step below — they
        # are not independent), and was removed rather than shipped
        # half-finished with a comment that didn't match what it did.
        _cap_time_sum = 0.0
        _seg_time_sum = 0.0
        _speed_time_sum = 0.0
        _obs_time_sum = 0.0
        _path_time_sum = 0.0
        _overlay_time_sum = 0.0
        _state_update_time_sum = 0.0
        _frame_count = 0
        _last_seen_capture_time = None

        while not self._stop_event.is_set():
            _t0 = time.perf_counter()
            frames, capture_time = capture_pipeline.get_latest()
            if frames is None:
                if capture_pipeline.last_error:
                    self.state.update(status_text=f"Frame error: {capture_pipeline.last_error}")
                time.sleep(0.002)
                continue
            if capture_time == _last_seen_capture_time:
                # No NEW frame since the last loop iteration — the capture
                # thread is still working on the next one.
                time.sleep(0.002)
                continue
            _last_seen_capture_time = capture_time
            color_image, depth_image_m, xyz, ir_image, raw_valid_mask, imu_state = frames
            _t1 = time.perf_counter()
            _cap_time_sum += (_t1 - _t0)

            if imu_state is not None:
                self.state.update_imu(imu_state)

            if color_image is None:
                continue

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            # Reduces the resolution the expensive geometric math runs at
            # (see _downsample_for_processing's docstring) — capture stays
            # full-res (IMU depends on it), and the frame that actually
            # gets displayed stays full-res too; only the working arrays
            # for segmentation/obstacles/path get smaller.
            pd = config.PROCESSING_DOWNSCALE
            full_h, full_w = depth_image_m.shape[:2]
            depth_proc, xyz_proc, valid_proc, ir_proc = _downsample_for_processing(
                depth_image_m, xyz, raw_valid_mask, ir_image, pd
            )

            seg = ground_segmenter.segment(
                depth_proc, xyz_proc, ir_image=ir_proc, raw_valid_mask=valid_proc
            )
            _t2 = time.perf_counter()
            _seg_time_sum += (_t2 - _t1)
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
                degraded_since = None
                degraded_duration = 0.0
            else:
                if degraded_since is None:
                    degraded_since = now
                degraded_duration = now - degraded_since
            watchdog_triggered = degraded_duration >= config.WATCHDOG_MAX_DEGRADED_S

            # Speed estimation deliberately stays on the ORIGINAL full-
            # resolution color_image/xyz, not the downscaled versions:
            # optical-flow feature tracking needs precise, consistent
            # pixel-to-3D correspondence frame to frame (SpeedEstimator
            # keeps its own previous-frame arrays internally, so mixing
            # resolutions across calls would silently corrupt that), and
            # the resulting number feeds directly into braking — not
            # worth trading any of its accuracy for a stage that was
            # never the dominant cost to begin with.
            land_mask_full = _upsample_mask(land_mask_proc, full_w, full_h, pd)
            speed_mps = speed_estimator.update(color_image, xyz, land_mask_full, dt, imu_state=imu_state)
            _t3 = time.perf_counter()
            _speed_time_sum += (_t3 - _t2)

            corridor_mask_proc = obstacle_decision.build_corridor_mask(xyz_proc, valid_depth_proc)
            # Distance-scaled per-pixel height check (depth noise grows ~z^2)
            raised_mask_proc = obstacle_decision.build_raised_mask(
                dist_to_plane_proc, valid_depth_proc, land_mask_proc, xyz=xyz_proc)
            # Glare (reflections / puddles / sun) = untrusted depth
            untrusted_proc = (obstacle_decision.glare_mask(color_image, land_mask_proc.shape)
                              if config.OBSTACLE_REJECT_GLARE else None)
            # Real obstacles ANYWHERE in view, judged by REAL-WORLD size
            # (metres from the 3D points), not pixel area — so floor texture,
            # glossy patches and noise specks don't count.
            blobs, blob_labels = obstacle_decision.detect_obstacle_blobs(
                raised_mask_proc, xyz_proc, dist_to_plane_proc, untrusted_proc)
            # Tracking + PERSISTENCE (see object_tracker.py): confirmed after
            # OBSTACLE_CONFIRM_FRAMES frames, or instantly if big and close.
            tracked = obstacle_tracker.update(blobs, now)
            confirmed = [b for b in tracked if b["confirmed"]]
            # ONE definition of "obstacle" everywhere: this confirmed mask
            # drives the path planner and the AR occlusion, and the braking
            # list below is the subset of these same objects in the corridor.
            obstacle_mask_proc = obstacle_decision.blobs_mask(blob_labels, confirmed)
            obstacles = obstacle_decision.corridor_obstacles(
                confirmed, blob_labels, corridor_mask_proc, xyz_proc)

            def _bbox_to_full(lst):
                # bbox is in PROC-resolution pixels -> scale for the full-res
                # display. distance/centroid/height/width are real metres and
                # need no scaling.
                out = []
                for o in lst:
                    o = dict(o)
                    bx, by, bw, bh = o["bbox"]
                    o["bbox"] = (bx * pd, by * pd, bw * pd, bh * pd)
                    out.append(o)
                return out
            display_obstacles = _bbox_to_full(confirmed)
            obstacles = _bbox_to_full(obstacles)

            nearest_distance = obstacles[0]["distance_m"] if obstacles else None
            closing_speed_mps = obstacles[0]["closing_speed_mps"] if obstacles else 0.0

            brake_raw = obstacle_decision.compute_brake_percent(
                nearest_distance, speed_mps, closing_speed_mps=closing_speed_mps
            )
            steer_raw = obstacle_decision.compute_steer_suggestion(
                obstacles[0] if obstacles else None
            )

            if watchdog_triggered:
                # Perception has been unreliable for too long — force full
                # brake and straight steering immediately (bypass the slew
                # limiter going UP, since this is a safety trip, not
                # routine control). Recovery still ramps down smoothly
                # afterward through the normal .update() path once
                # ground_ok again.
                brake_percent = brake_smoother.force(100.0)
                steer_deg = steer_smoother.force(0.0)
            else:
                brake_percent = brake_smoother.update(brake_raw, dt)
                steer_deg = steer_smoother.update(steer_raw, dt)
            _t4 = time.perf_counter()
            _obs_time_sum += (_t4 - _t3)

            # Real drivable path ahead — see path_planner.py. Its search
            # happens entirely in real-world meters (resolution-
            # independent); the ONLY pixel-space step is the final
            # projection to screen coordinates, which uses the ORIGINAL
            # full-res camera intrinsics directly — so waypoints_px lands
            # correctly on the full-res display frame with no separate
            # scaling step needed here.
            path_result = path_planner.plan_path(
                land_mask_proc, obstacle_mask_proc, valid_depth_proc, xyz_proc, seg["plane"], self.camera.intrinsics
            )
            _t5 = time.perf_counter()
            _path_time_sum += (_t5 - _t4)

            corridor_mask_full = _upsample_mask(corridor_mask_proc, full_w, full_h, pd)
            confidence_mask_full = _upsample_mask_nearest(confidence_mask_proc, full_w, full_h, pd)
            boundary_contour_full = (
                boundary_contour_proc * pd
                if (pd > 1 and boundary_contour_proc is not None)
                else boundary_contour_proc
            )

            # AR ribbon geometry (display only — never feeds brake/steer)
            ribbon = None
            if config.AR_ENABLED:
                ribbon = ar_ribbon.update(path_result, xyz_proc, valid_depth_proc,
                                          obstacle_mask_proc, seg["plane"], self.camera.intrinsics)

            annotated = overlay.draw_annotations(
                color_image, land_mask_full, boundary_contour_full, corridor_mask_full,
                display_obstacles, brake_percent, steer_deg, path_result, speed_mps=speed_mps,
                confidence_mask=confidence_mask_full, raw_valid_mask=raw_valid_mask,
                ribbon=ribbon,
            )
            if watchdog_triggered:
                overlay.draw_watchdog_banner(annotated, degraded_duration)

            # Object detection (see yolo_detector.py) runs on its OWN GPU
            # thread, fully decoupled from this loop — submit_frame() is a
            # cheap, non-blocking reference store, never waits on the GPU.
            # Draws whatever detections are LATEST available (from a
            # recent-but-not-necessarily-this-exact frame — normal for an
            # async overlay) on top of everything else, as the final layer.
            if self.yolo_worker is not None:
                self.yolo_worker.submit_frame(color_image)
                draw_detections(annotated, self.state.get_yolo_detections())

            _t6 = time.perf_counter()
            _overlay_time_sum += (_t6 - _t5)

            if watchdog_triggered:
                status_text = f"PERCEPTION DEGRADED {degraded_duration:.1f}s — safety brake engaged"
            elif ground_ok:
                status_text = "running"
            elif plane_found:
                status_text = "low ground visibility (coasting)"
            else:
                status_text = "no ground plane found (coasting)"

            self.state.update(
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
            _t7 = time.perf_counter()
            _state_update_time_sum += (_t7 - _t6)

            _frame_count += 1
            if _frame_count >= 10:
                cap_ms = _cap_time_sum / _frame_count * 1000
                seg_ms = _seg_time_sum / _frame_count * 1000
                speed_ms = _speed_time_sum / _frame_count * 1000
                obs_ms = _obs_time_sum / _frame_count * 1000
                path_ms = _path_time_sum / _frame_count * 1000
                overlay_ms = _overlay_time_sum / _frame_count * 1000
                state_ms = _state_update_time_sum / _frame_count * 1000
                total_ms = cap_ms + seg_ms + speed_ms + obs_ms + path_ms + overlay_ms + state_ms
                line = (f"[timing] capture={cap_ms:.1f}  segment={seg_ms:.1f}  speed_est={speed_ms:.1f}  "
                        f"obstacles={obs_ms:.1f}  path_plan={path_ms:.1f}  overlay={overlay_ms:.1f}  "
                        f"state_update={state_ms:.1f}  (all ms/frame)  TOTAL={total_ms:.1f}ms/frame "
                        f"(~{1000/total_ms:.1f} FPS)")
                if _PSUTIL_AVAILABLE:
                    try:
                        freq = psutil.cpu_freq()
                        cpu_pct = psutil.cpu_percent(interval=None)
                        if freq is not None:
                            line += f"  |  CPU: {freq.current:.0f} MHz (max {freq.max:.0f} MHz)  {cpu_pct:.0f}% util"
                    except Exception as e:
                        line += f"  |  CPU freq/util read failed: {e}"
                line += f"  running_on_core={_current_core_index()}"
                print(line)
                _cap_time_sum = 0.0
                _seg_time_sum = 0.0
                _speed_time_sum = 0.0
                _obs_time_sum = 0.0
                _path_time_sum = 0.0
                _overlay_time_sum = 0.0
                _state_update_time_sum = 0.0
                _frame_count = 0

        capture_pipeline.stop()
        if self.camera is not None:
            self.camera.stop()


def main():
    # Runs BEFORE anything else, including YOLO/CUDA setup — caps
    # OpenCV's own internal per-call threading so it doesn't compete with
    # the separate threads this app already runs (camera capture, YOLO).
    _limit_cv2_threads()

    # Restore the vehicle size saved from the UI last session (falls back
    # to config.py defaults if none / invalid) — BEFORE any worker starts,
    # so the very first frame already uses the right width.
    vehicle_profile.load_profile()

    shared_state = SharedState()

    yolo_worker = YoloWorker(shared_state)
    yolo_worker.start()

    vision_worker = VisionWorker(shared_state, yolo_worker=yolo_worker)
    vision_worker.start()

    cpu_perf_monitor = CpuPerfMonitorThread(interval_s=3.0)
    cpu_perf_monitor.start()

    def on_close():
        vision_worker.stop()
        yolo_worker.stop()
        cpu_perf_monitor.stop()

    app = TractorVisionApp(shared_state, on_close=on_close, on_imu_zero=vision_worker.zero_imu)
    app.run()


if __name__ == "__main__":
    main()

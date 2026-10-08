"""
playback/bag_source.py — a recorded RealSense sequence presented with the
SAME interface a live RealSenseCapture has.

Deliberately a drop-in shape match: .intrinsics, .depth_scale, .width,
.height, .get_frames() returning the same 6-tuple, .zero_imu(), .stop().
Everything downstream of capture (segmentation, obstacle detection, path
planning, overlay) therefore runs completely unchanged on recorded data —
no "is this a file?" branches anywhere in the perception code.

Playback runs through librealsense's own playback device
(config.enable_device_from_file), not a hand-rolled bag reader, so the
frames that come out are the real thing: recorded intrinsics, recorded
depth scale, the SDK's own rs.align, and the same spatial/temporal depth
filters a live run would apply.

Three things about playback are genuinely different from a live camera,
and each one is handled here rather than left to the caller:

  * STREAM RESOLUTION. A recording can ADVERTISE a stream it holds no
    data for. Confirmed on this project's own data: 20261007_145608
    advertises accel+gyro but contains zero motion messages, and a
    pipeline that enables them against it produces NO framesets at all —
    it waits forever for motion frames that never arrive, with no error.
    So streams are resolved with a fallback ladder (richest first, each
    candidate actually tried for real frames before being accepted),
    mirroring what RealSenseCapture already does for live stream combos.

  * FRAME BUFFER LIFETIME. np.asanyarray(frame.get_data()) is a VIEW into
    a librealsense-owned buffer that returns to the SDK's pool as soon as
    the frame object is released. A live loop mostly gets away with that
    because it consumes each frame immediately; a player hands frames to
    a consumer that may still be working on the previous one, so every
    returned array is COPIED out (options.copy_frames). Cheap next to the
    per-frame processing cost, and it removes a whole class of
    use-after-free corruption that would look like random depth noise.

  * COLOR FORMAT. Recordings are typically RGB8 (these are), while the
    whole OpenCV-side pipeline is BGR. Converted once here.

Non-realtime mode (set_real_time(False)) is always used: the SDK then
hands over EVERY recorded frame on demand instead of dropping frames to
chase the wall clock. Pacing and transport policy belong to
PlaybackController, not here — this class only ever answers "give me the
next recorded frameset".
"""

import datetime

import numpy as np
import cv2
import pyrealsense2 as rs

from .options import PlaybackOptions
from .discovery import probe_sequence


class PlaybackEnded(Exception):
    """Raised by read_frames() when the recording ran out and looping is
    off. Not an error condition — the normal end of a sequence."""


class _StreamRequest:
    """One enable_stream() call, described in plain data so a combination
    of them can be built, logged and compared before being tried."""

    def __init__(self, stream, index, fmt, fps, width=0, height=0, label=""):
        self.stream = stream
        self.index = index
        self.fmt = fmt
        self.fps = fps
        self.width = width
        self.height = height
        self.label = label

    def apply(self, cfg):
        if self.width and self.height:
            cfg.enable_stream(self.stream, self.index, self.width, self.height,
                              self.fmt, self.fps)
        else:
            cfg.enable_stream(self.stream, self.index, self.fmt, self.fps)

    def __str__(self):
        geom = f" {self.width}x{self.height}" if self.width else ""
        return f"{self.label}{geom}@{self.fps}"


_RS_STREAMS = {
    "depth": rs.stream.depth,
    "color": rs.stream.color,
    "infrared": rs.stream.infrared,
    "accel": rs.stream.accel,
    "gyro": rs.stream.gyro,
}


def _rs_format(format_name):
    """"rgb8" -> rs.format.rgb8. None if this SDK has no such format."""
    return getattr(rs.format, format_name, None)


class RealSenseBagSource:
    """
    A recorded sequence, read frame by frame.

    imu_fusion: anything exposing update_from_rs_frames(accel, gyro) ->
    state dict, plus (optionally) set_sample_time(seconds) and
    reset_timebase(). Injected rather than constructed so this package
    never has to import the host project's IMU code — see
    playback.imu_adapter for the standard adapter.

    unavailable_imu_state: callable(reason) -> dict, used when the
    recording has no usable motion data. Injected for the same reason.
    """

    def __init__(self, options: PlaybackOptions, imu_fusion=None,
                 unavailable_imu_state=None, info=None, verbose=True):
        if not options.path:
            raise ValueError("PlaybackOptions.path is empty — nothing to play")

        self.options = options
        self.verbose = verbose
        self.info = info if info is not None else probe_sequence(options.path)
        if self.info.error:
            raise RuntimeError(f"Cannot read {options.path!r}: {self.info.error}")

        self.imu_fusion = imu_fusion
        self._unavailable_imu_state = unavailable_imu_state or _default_unavailable_state

        self.motion_available = False
        self.ir_available = False
        self._imu_unavailable_reason = None
        self._ir_unavailable_reason = None

        # Transport bookkeeping. frame_index counts frames delivered on the
        # CURRENT pass; pass_index counts how many times the recording has
        # looped. Both are what a UI needs and neither is available from
        # the SDK.
        self.frame_index = 0
        self.pass_index = 0
        # Seeded from the container's own message count where that could
        # be read, so a UI can show "frame 40/673" from the first frame
        # rather than only after a full pass. Replaced by the measured
        # count once a pass has actually been played start to finish —
        # but ONLY such a pass: one that was seeked into has a
        # meaningless count, and publishing it would put a wrong
        # "frame 40/77" under the timeline for the rest of the session.
        self.frames_per_pass = self.info.video_frame_count
        self._pass_from_start = True
        self._last_position_ns = -1
        self._first_frame_time_s = None
        self._last_frame_time_s = None

        self.pipeline = None
        self._resolve_streams()
        self._configure_processing()

    # ------------------------------------------------------------- startup

    def _candidate_combinations(self):
        """
        Stream combinations to try, richest first. Built from what the
        recording itself advertises — never from hardcoded resolutions —
        so this works on any recorded sequence, not just this project's.
        """
        required = []
        for name in ("depth", "color"):
            entry = self.info.stream(name)
            if entry is None:
                raise RuntimeError(
                    f"Recording {self.info.name!r} has no {name} stream "
                    f"(found: {', '.join(s.stream_type for s in self.info.streams) or 'none'})"
                )
            required.append(self._request_for(entry))

        ir_entry = self.info.stream("infrared") if self.options.want_ir else None
        motion_entries = []
        if self.options.want_motion:
            accel, gyro = self.info.stream("accel"), self.info.stream("gyro")
            if accel is not None and gyro is not None:
                motion_entries = [self._request_for(accel), self._request_for(gyro)]

        ir_requests = [self._request_for(ir_entry)] if ir_entry is not None else []

        # Motion varies slowest: losing the IMU matters more than losing
        # the IR confidence check, which is the same priority ordering
        # realsense_capture.py uses for the live camera.
        combos = []
        for motion in ([motion_entries, []] if motion_entries else [[]]):
            for ir in ([ir_requests, []] if ir_requests else [[]]):
                combos.append(required + ir + motion)
        return combos

    def _request_for(self, entry):
        stream = _RS_STREAMS.get(entry.stream_type)
        fmt = _rs_format(entry.format_name)
        if stream is None or fmt is None:
            raise RuntimeError(
                f"Recorded stream {entry.stream_type}/{entry.format_name} is not "
                f"something this pyrealsense2 build understands"
            )
        return _StreamRequest(stream, entry.index, fmt, entry.fps,
                              entry.width, entry.height, entry.stream_type)

    def _resolve_streams(self):
        """
        Try each candidate combination for REAL frames, keep the first
        that delivers. "Real frames" is the whole point: a combination
        that merely starts without raising is not proof of anything —
        the empty-motion-stream case starts perfectly and then never
        produces a single frameset.
        """
        attempts = []
        chosen = None
        for combo in self._candidate_combinations():
            names = {r.label for r in combo}
            ok, failure = self._try_combination(combo)
            if ok:
                chosen = combo
                self.motion_available = {"accel", "gyro"} <= names
                self.ir_available = "infrared" in names
                break
            attempts.append(f"{{{', '.join(str(r) for r in combo)}}}: {failure}")

        if chosen is None:
            details = "\n".join(f"  - {line}" for line in attempts)
            raise RuntimeError(
                f"No stream combination produced frames from {self.info.name!r}:\n{details}"
            )

        if attempts and self.verbose:
            print(f"[playback] Playing {self.info.name} with "
                  f"{{{', '.join(str(r) for r in chosen)}}}; "
                  f"{len(attempts)} richer combination(s) produced no frames first:")
            for line in attempts:
                print(f"[playback]   tried and rejected — {line}")

        if self.options.want_motion and not self.motion_available:
            advertised = self.info.stream("accel") is not None and self.info.stream("gyro") is not None
            self._imu_unavailable_reason = (
                "this recording advertises accel+gyro but no motion frames ever "
                "arrived from it (it was recorded without IMU data) — IMU-dependent "
                "readouts are inert for this sequence"
                if advertised else
                f"{self.info.name} was recorded without accel/gyro streams"
            )
            if self.verbose:
                print(f"[playback] IMU NOT available: {self._imu_unavailable_reason}")

        if self.options.want_ir and not self.ir_available:
            self._ir_unavailable_reason = (
                f"{self.info.name} holds no usable infrared stream, so the "
                "depth-confidence (IR reflectivity) check runs without it"
            )
            if self.verbose:
                print(f"[playback] IR confidence NOT available: {self._ir_unavailable_reason}")

        if self.imu_fusion is None or not self.motion_available:
            self.motion_available = False
            if self._imu_unavailable_reason is None:
                # Name the ACTUAL cause. "No fusion was supplied" is true
                # but useless to someone looking at an inert attitude
                # readout and wondering whether the player is broken.
                if not self.info.has_motion:
                    self._imu_unavailable_reason = (
                        f"{self.info.name} was recorded without usable accel/gyro data, "
                        f"so attitude and IMU-fused speed are inert for this sequence"
                    )
                elif not self.options.want_motion:
                    self._imu_unavailable_reason = "IMU playback is turned off in the playback options"
                else:
                    self._imu_unavailable_reason = "no IMU fusion object was supplied to the player"

    def _try_combination(self, combo):
        """
        Start a pipeline for `combo` and wait for one real frameset. On
        success the pipeline is KEPT (seeked back to the start, so the
        probe frame is not lost from the run); on failure it is torn down.
        """
        pipeline = rs.pipeline()
        cfg = rs.config()
        # repeat_playback=True unconditionally: with it off, the SDK stops
        # the device at the end and neither seek() nor further reads
        # recover it (verified). Looping at the SDK level and detecting the
        # wrap ourselves gives BOTH behaviours from one setting — see
        # read_frames().
        cfg.enable_device_from_file(self.options.path, repeat_playback=True)
        for request in combo:
            request.apply(cfg)

        try:
            profile = pipeline.start(cfg)
        except Exception as e:
            _safe_stop(pipeline)
            return False, f"{type(e).__name__}: {e}"

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        try:
            got, _ = pipeline.try_wait_for_frames(self.options.probe_timeout_ms)
        except Exception as e:
            _safe_stop(pipeline)
            return False, f"{type(e).__name__}: {e}"

        if not got:
            _safe_stop(pipeline)
            return False, (f"no frameset within {self.options.probe_timeout_ms} ms "
                           f"(status={playback.current_status()})")

        self.pipeline = pipeline
        self.profile = profile
        self.playback = playback
        playback.seek(datetime.timedelta(0))
        self._last_position_ns = -1
        return True, None

    def _configure_processing(self):
        device = self.profile.get_device()
        self.depth_scale = device.first_depth_sensor().get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()
        self.width = self.intrinsics.width
        self.height = self.intrinsics.height

        self.align = rs.align(rs.stream.color) if self.options.align_to_color else None

        self._build_depth_filters()

        # Pixel grid for vectorized deprojection — built once, reused.
        uu, vv = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self._uu = uu.astype(np.float32)
        self._vv = vv.astype(np.float32)

        if self.verbose:
            print(f"[playback] {self.info.name}: color {self.width}x{self.height}, "
                  f"depth_scale={self.depth_scale:.6f} m/unit, "
                  f"duration={self.duration_s:.2f}s, "
                  f"IMU={'yes' if self.motion_available else 'no'}, "
                  f"IR={'yes' if self.ir_available else 'no'}")

    def _build_depth_filters(self):
        """
        Same filters, same settings, same deliberate no-hole-filling
        choice as the live path: hole-filling would copy nearby ground
        depth into holes left by IR-absorbing objects, which is exactly
        the failure the perception code is built to avoid.

        Rebuildable because the temporal filter carries frame-to-frame
        state that must not survive a seek — see seek().
        """
        o = self.options
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, o.spatial_filter_alpha)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, o.spatial_filter_delta)
        self.spatial_filter.set_option(rs.option.holes_fill, o.spatial_holes_fill)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, o.temporal_filter_alpha)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, o.temporal_filter_delta)

    # ----------------------------------------------------------- transport

    @property
    def duration_s(self):
        try:
            return self.playback.get_duration().total_seconds()
        except Exception:
            return self.info.duration_s

    @property
    def position_s(self):
        try:
            return self.playback.get_position() / 1e9
        except Exception:
            return 0.0

    @property
    def recorded_time_s(self):
        """Seconds of recording elapsed since the first delivered frame,
        from the frames' OWN timestamps. This — not the wall clock — is
        the clock the host should use for dt, so results don't depend on
        how fast playback happens to be running."""
        if self._first_frame_time_s is None or self._last_frame_time_s is None:
            return 0.0
        return self._last_frame_time_s - self._first_frame_time_s

    def seek(self, seconds):
        """Jump to `seconds` into the recording. Clamped to the sequence.
        Callers must treat this as a discontinuity: anything holding
        frame-to-frame state (speed estimate, trackers, temporal filters)
        should be reset — PlaybackController does that for you."""
        duration = self.duration_s
        seconds = max(0.0, min(float(seconds), max(0.0, duration - 1e-3)))

        self._drain_queue()
        self.playback.seek(datetime.timedelta(seconds=seconds))

        self._last_position_ns = -1
        self._first_frame_time_s = None
        self._last_frame_time_s = None
        # Only a seek to the very start leaves the frame counter
        # meaningful; any other lands mid-sequence, so this pass can no
        # longer tell us how many frames a full pass holds.
        self._pass_from_start = seconds <= 0.0
        if seconds <= 0.0:
            self.frame_index = 0
        if self.imu_fusion is not None and hasattr(self.imu_fusion, "reset_timebase"):
            self.imu_fusion.reset_timebase()
        # The SDK's temporal depth filter blends each frame with the
        # previous one. Across a jump cut that previous frame is from a
        # different part of the recording entirely, so keeping it would
        # smear one scene into another for several frames. Rebuilt, not
        # merely "cleared" — the filter exposes no reset.
        self._build_depth_filters()

    def _drain_queue(self, max_frames=8):
        """
        Throw away anything already sitting in the pipeline's queue.

        This must happen before every seek, and it is not a tidiness
        measure — it is the difference between a seek that works and one
        that does not. Measured on this project's own recording, 10 seeks
        each:

          plain seek()               2/10 took exactly 10.0 s (an internal
                                     librealsense timeout) AND landed on
                                     the wrong position — one target behind
          pause / seek / resume      8/10 bad; the widely suggested
                                     workaround is markedly WORSE here
          drain, then seek           0/10 bad, 9 ms worst case

        The pattern fits a playback device in non-realtime mode blocking
        on a frame nobody has collected: with the queue empty the seek
        completes immediately every time.
        """
        for _ in range(max_frames):
            if not self.pipeline.poll_for_frames():
                return

    def restart(self):
        self.seek(0.0)
        self.frame_index = 0

    def stop(self):
        if self.pipeline is not None:
            _safe_stop(self.pipeline)
            self.pipeline = None

    def zero_imu(self):
        if self.imu_fusion is not None and hasattr(self.imu_fusion, "zero_on_current_orientation"):
            self.imu_fusion.zero_on_current_orientation()

    # -------------------------------------------------------------- frames

    def read_frames(self):
        """
        The next recorded frameset, as (frames_tuple, meta).

        frames_tuple is exactly what a live RealSenseCapture.get_frames()
        returns: (color_image, depth_image_m, xyz, ir_image,
        raw_valid_mask, imu_state).

        meta carries what only a player knows: position_s, frame_time_s
        (the recording's own clock), frame_index, pass_index, wrapped.

        Raises PlaybackEnded when the recording wrapped and
        options.loop is False.
        """
        if self.pipeline is None:
            raise PlaybackEnded("playback source is stopped")

        got, frameset = self.pipeline.try_wait_for_frames(self.options.frame_timeout_ms)
        if not got:
            # In non-realtime mode with looping enabled this should not
            # happen; if it does, the sequence is over as far as we are
            # concerned.
            raise PlaybackEnded(
                f"no frameset within {self.options.frame_timeout_ms} ms "
                f"(status={self.playback.current_status()})"
            )

        # Wrap detection: the SDK loops silently, so the only signal is
        # the playback position jumping BACKWARDS. The frame that comes
        # with the wrap is the first frame of the next pass.
        position_ns = self.playback.get_position()
        wrapped = 0 <= position_ns < self._last_position_ns
        self._last_position_ns = position_ns

        if wrapped:
            if self._pass_from_start:
                self.frames_per_pass = self.frame_index
            self.pass_index += 1
            self.frame_index = 0
            self._pass_from_start = True       # the new pass starts at 0
            self._first_frame_time_s = None
            if not self.options.loop:
                raise PlaybackEnded("reached the end of the recording")

        frames = self._prepare(frameset, wrapped)
        self.frame_index += 1

        meta = {
            "position_s": position_ns / 1e9,
            "frame_time_s": self._last_frame_time_s,
            "recorded_time_s": self.recorded_time_s,
            "frame_index": self.frame_index,
            "pass_index": self.pass_index,
            "wrapped": wrapped,
        }
        return frames, meta

    # RealSenseCapture-compatible alias, so this class can be swapped in
    # anywhere the live capture object is used.
    def get_frames(self):
        frames, _meta = self.read_frames()
        return frames

    def _prepare(self, frameset, wrapped):
        """Recorded frameset -> the same 6-tuple the live capture returns."""
        # Recording clock, in seconds (frame timestamps are milliseconds).
        # Tracked for EVERY sequence, IMU or not: this is the clock the
        # host uses for per-frame dt, so it has to exist even when the
        # recording has no motion data at all.
        timestamp_s = frameset.get_timestamp() / 1000.0
        if self._first_frame_time_s is None or wrapped:
            self._first_frame_time_s = timestamp_s
        self._last_frame_time_s = timestamp_s

        imu_state = self._read_imu(frameset, timestamp_s, wrapped)

        aligned = self.align.process(frameset) if self.align is not None else frameset
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None, None, None, imu_state

        color_image = self._to_array(color_frame)
        if self.options.color_to_bgr and color_frame.profile.format() == rs.format.rgb8:
            # cvtColor allocates its own output, so this also satisfies the
            # copy-out requirement regardless of options.copy_frames.
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        raw_depth = self._to_array(depth_frame)
        raw_valid_mask = raw_depth > 0        # a fresh array, never a view

        if self.options.enable_depth_filters:
            filtered = self.spatial_filter.process(depth_frame)
            filtered = self.temporal_filter.process(filtered)
            depth_units = np.asanyarray(filtered.as_depth_frame().get_data()).astype(np.float32)
        else:
            depth_units = raw_depth.astype(np.float32)
        depth_image_m = depth_units * self.depth_scale

        ir_image = None
        if self.ir_available:
            ir_frame = aligned.get_infrared_frame(self.options.ir_stream_index)
            if ir_frame:
                ir_image = self._to_array(ir_frame)
                # rs.align reprojects depth into color's resolution but not
                # IR, so match shapes here the way the live path does.
                if ir_image.shape[:2] != depth_image_m.shape[:2]:
                    ir_image = cv2.resize(
                        ir_image, (depth_image_m.shape[1], depth_image_m.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

        xyz = self._deproject(depth_image_m)
        return color_image, depth_image_m, xyz, ir_image, raw_valid_mask, imu_state

    def _to_array(self, frame):
        """
        Frame data as a numpy array that stays valid after the frame is
        released. See this module's docstring: without the copy these are
        views into SDK-pooled memory, and a consumer still working on the
        previous frame can have it recycled underneath it.
        """
        array = np.asanyarray(frame.get_data())
        return array.copy() if self.options.copy_frames else array

    def _read_imu(self, frameset, timestamp_s, wrapped):
        if not self.motion_available:
            return self._unavailable_imu_state(self._imu_unavailable_reason)

        if self.options.use_recorded_timebase and hasattr(self.imu_fusion, "set_sample_time"):
            # Gyro integration must advance by RECORDED time, not wall-clock
            # time, or a playback that runs at a different speed than the
            # recording integrates the wrong dt and the attitude estimate
            # drifts by exactly that ratio.
            if wrapped and hasattr(self.imu_fusion, "reset_timebase"):
                self.imu_fusion.reset_timebase()
            self.imu_fusion.set_sample_time(timestamp_s)

        accel = frameset.first_or_default(rs.stream.accel)
        gyro = frameset.first_or_default(rs.stream.gyro)
        return self.imu_fusion.update_from_rs_frames(accel, gyro)

    def _deproject(self, depth_image_m):
        """Per-pixel XYZ in meters, camera frame — one vectorized pass,
        the same math the live capture uses."""
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
        z = depth_image_m
        x = (self._uu - cx) * z / fx
        y = (self._vv - cy) * z / fy
        return np.dstack((x, y, z)).astype(np.float32)

    # ------------------------------------------------------------ reporting

    def status_lines(self):
        """Human-readable notes about what this sequence can and cannot
        exercise — worth showing once at startup so an inert IMU readout
        is never mistaken for a broken one."""
        lines = [f"sequence: {self.info.summary()}"]
        if self._imu_unavailable_reason:
            lines.append(f"IMU: {self._imu_unavailable_reason}")
        if self._ir_unavailable_reason:
            lines.append(f"IR: {self._ir_unavailable_reason}")
        return lines


def _safe_stop(pipeline):
    try:
        pipeline.stop()
    except Exception:
        pass


def _default_unavailable_state(reason=None):
    """Fallback for hosts that do not supply their own — same shape the
    rest of this project's IMU state uses."""
    return {
        "available": False,
        "status": reason or "no IMU data in this recording",
        "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "tilt_deg": 0.0,
        "accel_mps2": (0.0, 0.0, 0.0), "gyro_dps": (0.0, 0.0, 0.0),
        "rollover_risk": "ok",
    }

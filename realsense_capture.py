"""
realsense_capture.py — wraps the Intel RealSense pipeline.

Provides aligned color + depth frames, plus a vectorized function to turn
the whole depth image into a per-pixel XYZ point map (meters, camera frame)
in one numpy pass — no per-pixel Python loops.

IMU note: depth + color + (if present) accel/gyro are all enabled on ONE
rs.pipeline / rs.config here, and the IMU's fused pitch/roll/yaw is read out
of the same frameset every call to get_frames(). Earlier versions opened a
second, independent rs.pipeline for motion data — that's a well-known way to
get an IMU that silently "just doesn't work": many USB controllers/driver
stacks only let one pipeline claim a physical device at a time, so the
second pipeline.start() either throws or never yields frames. A single
combined pipeline (what Intel's own multi-stream examples do) doesn't have
that failure mode, and as a bonus means one USB read per loop iteration
instead of two independent blocking reads competing with each other.

Startup fallback ladder: "Couldn't resolve requests" from pipeline.start()
doesn't mean the IMU itself is broken — it means the SPECIFIC COMBINATION of
streams/rates requested together isn't one the camera's firmware advertises
as a valid joint profile. Two independent, well-documented RealSense quirks
can cause this:
  - Adding an IR stream alongside color+depth+motion can conflict, even at
    framerates that work individually.
  - Explicitly requesting a fixed accel/gyro FPS (e.g. 250/200) can itself
    fail to resolve on some SDK/firmware combinations, even though those
    are documented-valid rates — letting the SDK pick its own default
    motion profile (by omitting the FPS argument) is a known fix, and
    nothing downstream (realsense_imu.py) assumes a fixed rate — it times
    itself from wall-clock deltas between frames, whatever rate they arrive.
So instead of one all-or-nothing attempt, __init__ tries progressively
simpler combinations and keeps the richest one that actually resolves:
  1. color + depth + IR + motion@fixed-fps   (everything, as configured)
  2. color + depth + IR + motion@auto-fps    (let the SDK pick the rate)
  3. color + depth + motion@fixed-fps        (drop IR)
  4. color + depth + motion@auto-fps         (drop IR, auto rate)
  5. color + depth + IR                      (drop motion, keep IR)
  6. color + depth                           (baseline)
Only combinations for features actually turned on in config.py are tried
(e.g. if IMU_ENABLED is False, no motion variant is ever attempted).
"""

import numpy as np
import cv2
import pyrealsense2 as rs

import config
from realsense_imu import ImuFusion, unavailable_state


class RealSenseCapture:
    def __init__(self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS):
        self.width = width
        self.height = height
        self.imu_fusion = None
        self.motion_available = False
        self._imu_unavailable_reason = None
        self._ir_unavailable_reason = None

        want_motion = config.IMU_ENABLED
        want_ir = config.ENABLE_IR_CONFIDENCE

        # Priority order, richest first. "motion_mode" is tri-state:
        #   "fps"  — accel/gyro at the fixed rates in config.py (preferred:
        #            matches the documented/intended rate)
        #   "auto" — accel/gyro with no FPS specified, SDK picks its own
        #            default motion profile (fallback if "fps" won't resolve)
        #   "off"  — no motion streams at all
        # IR varies fastest within each motion setting, since dropping IR
        # is the smaller sacrifice than dropping motion.
        motion_options = ["fps", "auto", "off"] if want_motion else ["off"]
        ir_options = [True, False] if want_ir else [False]
        combos = [(m, ir) for m in motion_options for ir in ir_options]

        self.profile = None
        attempt_log = []
        chosen_motion_mode = "off"
        chosen_ir = False

        for motion_mode, with_ir in combos:
            self.pipeline = rs.pipeline()
            cfg = self._build_config(motion_mode=motion_mode, with_ir=with_ir,
                                      width=width, height=height, fps=fps)
            try:
                self.profile = self.pipeline.start(cfg)
                chosen_motion_mode = motion_mode
                chosen_ir = with_ir
                break
            except Exception as e:
                attempt_log.append(f"motion={motion_mode}, ir={with_ir}: {e}")
                continue

        if self.profile is None:
            # Nothing resolved, not even the color+depth baseline — this is
            # a real camera/connection problem, not a stream-combo issue.
            details = "\n".join(f"  - {line}" for line in attempt_log)
            raise RuntimeError(
                "RealSense pipeline failed to start with every stream "
                f"combination tried:\n{details}"
            )

        self._ir_enabled = chosen_ir
        chosen_motion = chosen_motion_mode != "off"

        # Diagnostic: print the USB link speed the camera actually
        # negotiated. "Couldn't resolve requests" on combos that include
        # motion (accel/gyro), while color+depth (+IR) alone succeed, is a
        # known symptom of the camera being negotiated at USB 2.1 instead of
        # USB 3.x — this can happen even with a USB3-rated cable if it's
        # plugged into a USB2 port, hub, or extension, and D435i's IMU
        # commonly fails to resolve at USB2 link speed even though video
        # alone keeps working there.
        try:
            usb_type = self.profile.get_device().get_info(rs.camera_info.usb_type_descriptor)
            print(f"[realsense_capture] USB connection type reported by camera: {usb_type}")
            if not usb_type.startswith("3."):
                print(
                    "[realsense_capture]   NOTE: this is NOT a USB3 link. Even a "
                    "USB3-rated cable negotiates at USB2 speed if the PORT, hub, or "
                    "extension in the chain is USB2 — and that alone is enough to make "
                    "the D435i's IMU (accel/gyro) fail to resolve while color+depth "
                    "keep working fine. Try a different USB3 port directly on the "
                    "machine (no hub) if motion keeps failing below."
                )
        except Exception:
            pass

        if len(attempt_log) > 0:
            # Something had to be dropped to get a working profile — say so
            # loudly, since it's easy to miss otherwise.
            print(f"[realsense_capture] Started with motion={chosen_motion_mode}, ir={chosen_ir} "
                  f"(requested motion={want_motion}, ir={want_ir}); "
                  f"{len(attempt_log)} richer combination(s) failed to resolve first:")
            for line in attempt_log:
                print(f"[realsense_capture]   tried and failed — {line}")

        if want_ir and not chosen_ir:
            self._ir_unavailable_reason = (
                "IR stream dropped — the combination of color+depth+IR"
                + ("+motion" if chosen_motion else "")
                + " isn't a valid joint profile on this camera/firmware at "
                f"{width}x{height}@{fps}. Depth-confidence (IR reflectivity "
                "check) is running without it, so it's less robust against "
                "dark/IR-absorbing objects being misread as ground."
            )
            print(f"[realsense_capture] IR confidence NOT available: {self._ir_unavailable_reason}")

        if want_motion:
            if chosen_motion:
                self.motion_available = self._streams_present(rs.stream.accel) and self._streams_present(rs.stream.gyro)
                if self.motion_available:
                    rate_note = "fixed configured rate" if chosen_motion_mode == "fps" else "SDK-auto rate (fixed rate didn't resolve)"
                    print(f"[realsense_capture] IMU: motion streams (accel+gyro) confirmed present on combined pipeline ({rate_note}).")
                else:
                    self._imu_unavailable_reason = (
                        "pipeline.start() succeeded but accel/gyro streams aren't in the "
                        "resulting profile — camera likely has no motion module "
                        "(requires D435i / D455, not D415/other)"
                    )
                    print(f"[realsense_capture] IMU NOT available: {self._imu_unavailable_reason}")
            else:
                self.motion_available = False
                self._imu_unavailable_reason = (
                    "no stream combination that included motion resolved on this "
                    "camera/firmware at the requested resolution (tried both the "
                    "configured fixed accel/gyro rate AND letting the SDK auto-pick "
                    "its own rate) — see the 'tried and failed' lines above for "
                    "exactly what was attempted"
                )
                print(f"[realsense_capture] IMU NOT available: {self._imu_unavailable_reason}")

        if self.motion_available:
            self.imu_fusion = ImuFusion()

        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # meters per depth unit

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Denoise only — deliberately no hole-filling filter. Hole-filling would
        # copy nearby valid (ground) depth into holes left by IR-absorbing
        # objects like black cloth, which is exactly the failure we're avoiding.
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, config.SPATIAL_FILTER_ALPHA)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, config.SPATIAL_FILTER_DELTA)
        self.spatial_filter.set_option(rs.option.holes_fill, 0)  # explicitly off

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, config.TEMPORAL_FILTER_ALPHA)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, config.TEMPORAL_FILTER_DELTA)

        # Precompute pixel coordinate grid once — reused every frame for vectorized deprojection
        u = np.arange(self.width)
        v = np.arange(self.height)
        self._uu, self._vv = np.meshgrid(u, v)  # shape (H, W)

        self._printed_first_motion_frame = not self.motion_available

    def _build_config(self, motion_mode, with_ir, width, height, fps):
        """
        width/height here are the COLOR stream's resolution (config.FRAME_
        WIDTH/HEIGHT — see the note there). Depth and IR request their OWN
        resolution (config.DEPTH_WIDTH/HEIGHT) — confirmed in the RealSense
        Viewer that depth and color need to be DIFFERENT resolutions for
        this camera to resolve a combined profile that includes motion.
        """
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, config.DEPTH_WIDTH, config.DEPTH_HEIGHT, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if with_ir:
            # IR shares the depth sensor's native resolution (it's one of
            # the two physical stereo imagers depth is computed from).
            cfg.enable_stream(
                rs.stream.infrared, config.IR_STREAM_INDEX,
                config.DEPTH_WIDTH, config.DEPTH_HEIGHT, rs.format.y8, fps
            )
        if motion_mode == "fps":
            # Fixed, documented rates — preferred when they resolve.
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, config.IMU_ACCEL_FPS)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, config.IMU_GYRO_FPS)
        elif motion_mode == "auto":
            # No FPS specified — let the SDK pick its own default motion
            # profile. Known workaround for "Couldn't resolve requests" on
            # some SDK/firmware combos where the fixed rate above doesn't
            # resolve even though it's a documented-valid rate. Safe here:
            # realsense_imu.py times itself from wall-clock deltas between
            # frames, not from an assumed fixed rate.
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
        # motion_mode == "off": no motion streams requested at all.
        return cfg

    def _streams_present(self, stream_type):
        return any(s.stream_type() == stream_type for s in self.profile.get_streams())

    def zero_imu(self):
        if self.imu_fusion is not None:
            self.imu_fusion.zero_on_current_orientation()

    def get_frames(self):
        """
        Returns (color_image, depth_image_m, xyz, ir_image, raw_valid_mask, imu_state):
          color_image     : (H, W, 3) uint8 BGR
          depth_image_m   : (H, W) float32 meters (0 = invalid), denoised (no hole-fill)
          xyz             : (H, W, 3) float32 meters, camera-frame X/Y/Z per pixel
          ir_image        : (H, W) uint8, or None if IR stream disabled/unavailable
          raw_valid_mask  : (H, W) bool — depth was non-zero *before* denoise filtering.
                            This is the ground truth for "did the stereo matcher
                            actually resolve this pixel", independent of any
                            smoothing that happens afterward.
          imu_state       : fused IMU dict (see realsense_imu.ImuFusion.get_state /
                            unavailable_state), read from the SAME frameset as
                            the video above.
        """
        frames = self.pipeline.wait_for_frames()

        if self.motion_available:
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if not self._printed_first_motion_frame and (accel_frame or gyro_frame):
                print("[realsense_capture] First motion frame received — IMU is streaming data.")
                self._printed_first_motion_frame = True
            imu_state = self.imu_fusion.update_from_rs_frames(accel_frame, gyro_frame)
        else:
            imu_state = unavailable_state(self._imu_unavailable_reason)

        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None, None, None, imu_state

        color_image = np.asanyarray(color_frame.get_data())

        raw_depth_raw = np.asanyarray(depth_frame.get_data())
        raw_valid_mask = raw_depth_raw > 0

        if config.ENABLE_DEPTH_FILTERS:
            filtered = self.spatial_filter.process(depth_frame)
            filtered = self.temporal_filter.process(filtered)
            depth_raw = np.asanyarray(filtered.as_depth_frame().get_data()).astype(np.float32)
        else:
            # Skips the RealSense SDK's own spatial+temporal denoising
            # entirely — real, measured native-SDK cost per frame, not
            # Python overhead. Useful as a direct test: if disabling this
            # meaningfully drops the "capture" time in main.py's [timing]
            # log, that confirms the filters (not hardware wait, not our
            # own code) are the real cost there. Depth will be noisier
            # without them — ground_segmentation.py's own temporal
            # smoothing and normal-consistency check absorb some of that,
            # but this is a genuine quality/speed trade-off, not free.
            depth_raw = raw_depth_raw.astype(np.float32)
        depth_image_m = depth_raw * self.depth_scale

        ir_image = None
        if self._ir_enabled:
            ir_frame = aligned.get_infrared_frame(config.IR_STREAM_INDEX)
            if ir_frame:
                ir_image = np.asanyarray(ir_frame.get_data())
                # IR is requested at DEPTH_WIDTH/HEIGHT (native stereo-sensor
                # resolution), but rs.align() only reprojects the depth
                # frame into color's resolution — it does not also reproject
                # IR. Since depth_image_m above IS already at color's
                # resolution post-align, resize IR to match so
                # ground_segmentation.py's confidence mask (which combines
                # ir_image with the depth-derived valid mask) never hits a
                # shape mismatch. Nearest-neighbor since this only feeds a
                # coarse low-confidence threshold, not sub-pixel geometry.
                if ir_image.shape[:2] != depth_image_m.shape[:2]:
                    ir_image = cv2.resize(
                        ir_image, (depth_image_m.shape[1], depth_image_m.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

        xyz = self._deproject_vectorized(depth_image_m)
        return color_image, depth_image_m, xyz, ir_image, raw_valid_mask, imu_state

    def _deproject_vectorized(self, depth_image_m):
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        z = depth_image_m
        x = (self._uu - cx) * z / fx
        y = (self._vv - cy) * z / fy
        xyz = np.dstack((x, y, z)).astype(np.float32)
        return xyz

    def stop(self):
        self.pipeline.stop()

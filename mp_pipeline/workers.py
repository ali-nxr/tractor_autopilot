"""
mp_pipeline/workers.py — the entry point of every pipeline process.

    capture ──► semantic ──► geometry ──► render ──► UI (main process)
       │  ▲                    ▲     │       │  ╲
       │  │                    │     │       │   ╲──► yolo ──► (detections back to render)
       └──┼──────► speed ──────┘◄────┘ (ground mask)
          └────────── free slot index ◄──────┘

Speed estimation (optical flow, ~40 ms) runs in its own process, in
parallel with SegFormer and segmentation of the same frame. Geometry waits
for that frame's speed before deciding brake/steer, so braking always uses
the current frame's speed. The one difference from the single-process
loop: the ground mask that decides WHERE features are tracked is the most
recent one geometry has published (usually the previous frame's), since
the current frame's mask does not exist yet when speed starts. The speed
math itself is unchanged.

Each stage is its own OS process, so stages run truly in parallel — the
single-process design serialized them all on Python's GIL (measured: the
whole pipeline headless, no UI at all, ran at ~4 FPS while its stages
summed to ~300 ms). Frames stay in shared memory (shm.FrameRing); only
slot indices and small/proc-resolution results travel through queues.

Ordering and completeness: every stage is a single process reading one
FIFO queue, so frames stay in capture order. In playback the capture
process waits for a free slot (never drops a frame); live, it drops the
newest frame when every slot is busy, like the old capture thread did.

Shutdown: the capture process sends None down the chain; each stage
forwards it and exits.
"""

import os
import queue
import threading
import time
import traceback
import types

import numpy as np


# ----------------------------------------------------------------- common

def _bootstrap(config_snapshot, name):
    """Spawned processes import config fresh — re-apply the parent's exact
    values (recorded-mode overrides, --debug, runtime vehicle size, ...)."""
    import config
    import config_lane
    import config_segformer
    for module_name, values in config_snapshot.items():
        module = {"config": config, "config_segformer": config_segformer, "config_lane": config_lane}[module_name]
        for key, value in values.items():
            setattr(module, key, value)
    import cv2
    cv2.setNumThreads(config.CV2_NUM_THREADS)
    return config


def _trace(text):
    if os.environ.get("MP_TRACE"):
        print(f"[mp-trace {time.time() % 1000:7.2f}] {text}", flush=True)


def _release_queues(*queues):
    """
    A process that put data on a multiprocessing.Queue waits at exit until
    that data has been read. At shutdown the reader may already be gone
    (speed exits before geometry stops sending it masks; the UI stops
    reading before render stops sending frames) — so without this, workers
    hang on exit and have to be terminated. ONLY for such side queues:
    skipping the wait also drops whatever is still unsent, and on the data
    chain the last item is the None that tells the next stage to stop.
    """
    for q in queues:
        if q is not None:
            try:
                q.cancel_join_thread()
            except Exception:
                pass


def _report_error(ui_q, stage):
    text = traceback.format_exc()
    print(f"[mp:{stage}] FAILED:\n{text}", flush=True)
    try:
        ui_q.put({"type": "error", "stage": stage, "text": text.strip().splitlines()[-1]})
    except Exception:
        pass


def _intrinsics_ns(intr, grid_dtype):
    return types.SimpleNamespace(fx=float(intr.fx), fy=float(intr.fy), ppx=float(intr.ppx),
                                 ppy=float(intr.ppy), width=int(intr.width), height=int(intr.height),
                                 grid_dtype=str(grid_dtype))


def _frame_view(ring, slot, meta):
    """The slot's arrays as the dict perception_core expects (views, no copy)."""
    s = ring[slot]
    return {"color": s["color"], "depth": s["depth"], "xyz": None, "raw_valid": s["raw_valid"],
            "ir": s["ir"] if meta.get("has_ir") else None, "imu_state": meta.get("imu_state")}


# ---------------------------------------------------------------- capture

def capture_main(config_snapshot, mode, source_args, ring_count, q_out, free_q, cmd_q, info_q, ui_q):
    """
    mode "recorded": source_args = (PlaybackOptions, SequenceInfo).
    mode "live": source_args = None (RealSenseCapture with config defaults).
    """
    config = _bootstrap(config_snapshot, "capture")
    from mp_pipeline.shm import FrameRing
    ring = None
    stop = threading.Event()
    try:
        controller = None
        if mode == "recorded":
            from playback import PlaybackController, RealSenseBagSource, playback_fusion
            from realsense_imu import ImuFusion, unavailable_state
            options, info = source_args
            fusion = playback_fusion(ImuFusion)() if options.want_motion else None
            source = RealSenseBagSource(options, imu_fusion=fusion, unavailable_imu_state=unavailable_state, info=info)
            controller = PlaybackController(source)
            for line in source.status_lines():
                print(f"[playback] {line}", flush=True)
            intr = source.intrinsics
            color_stream = info.stream("color")
            nominal_dt = 1.0 / (color_stream.fps if color_stream and color_stream.fps else config.FPS)
            read = controller.next_frame
            zero_imu = source.zero_imu
            grid_dtype = source._uu.dtype
            # Downstream deprojects at processing resolution (bit-identical,
            # see perception_core.deproject_proc) — skip the full-res pass
            # here (measured 16 ms/frame of this process's ~80 ms).
            source._deproject = lambda depth_image_m: None
        else:
            from realsense_capture import RealSenseCapture
            source = RealSenseCapture()
            intr = source.intrinsics
            nominal_dt = 1.0 / config.FPS

            def read():
                frames = source.get_frames()
                return frames, {"frame_time_s": time.time()}
            zero_imu = source.zero_imu
            grid_dtype = source._uu.dtype
            source._deproject_vectorized = lambda depth_image_m: None   # see the playback branch

        width, height = int(intr.width), int(intr.height)
        ring = FrameRing.create(height, width, ring_count)
        for i in range(ring_count):
            free_q.put(i)
        info_q.put({
            "type": "info", "ring": ring.handle(), "width": width, "height": height,
            "intrinsics": _intrinsics_ns(intr, grid_dtype), "nominal_dt": nominal_dt,
            "duration_s": controller.duration_s if controller else 0.0,
            "summary": info.summary() if controller else "live camera",
            "name": info.name if controller else "live",
        })

        # Playback commands arrive from the UI process. Controller methods
        # are thread-safe and wake its reader, so they are applied from this
        # thread even while next_frame() is blocked waiting (paused).
        def command_loop():
            while not stop.is_set():
                try:
                    cmd = cmd_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if cmd is None:
                    stop.set()
                    if controller:
                        controller.stop()
                    return
                name, args = cmd
                try:
                    if name == "zero_imu":
                        zero_imu()
                    elif controller is not None:
                        getattr(controller, name)(*args)
                except Exception as e:
                    print(f"[mp:capture] command {name}{args} failed: {e}", flush=True)

        def status_loop():
            while not stop.is_set():
                if controller is not None:
                    ui_q.put({"type": "playback_status", "status": controller.status()})
                stop.wait(0.15)

        threading.Thread(target=command_loop, daemon=True, name="commands").start()
        threading.Thread(target=status_loop, daemon=True, name="status").start()

        seq = 0
        dropped = 0
        slot = None
        # Streaming playback (PlaybackOptions.drop_frames): the recording
        # plays at its paced speed and only frames the pipeline can take
        # RIGHT NOW are decoded; the rest are skipped like a live camera
        # drops them. A seek/loop marker on a skipped frame carries over to
        # the next processed one, so every stage still resets.
        streaming = controller is not None and controller.options.drop_frames
        pending_discontinuity = None
        while not stop.is_set():
            if streaming:
                try:
                    slot = free_q.get_nowait() if slot is None else slot
                except queue.Empty:
                    slot = None
                result = controller.next_frame(decode=slot is not None)
                if result is None:
                    time.sleep(0.02)
                    continue
                if slot is None:
                    pending_discontinuity = result[1].get("discontinuity") or pending_discontinuity
                    dropped += 1
                    continue
                t_read = time.time()
                frames, meta = result
                if pending_discontinuity and not meta.get("discontinuity"):
                    meta["discontinuity"] = pending_discontinuity
                pending_discontinuity = None
                color, depth, xyz, ir, raw_valid, imu_state = frames
                if color is None:
                    continue
                capture_time = time.time()
                s = ring[slot]
                s["color"][...] = color
                s["depth"][...] = depth
                s["raw_valid"][...] = raw_valid
                if ir is not None:
                    s["ir"][...] = ir
                for q in q_out:
                    q.put((slot, {
                        "seq": seq, "capture_time": capture_time,
                        "frame_time_s": meta.get("frame_time_s", capture_time),
                        "discontinuity": meta.get("discontinuity"),
                        "frame_index": meta.get("frame_index"),
                        "imu_state": imu_state, "has_ir": ir is not None, "dropped": dropped,
                        "timing": {"capture": time.time() - capture_time},
                    }))
                slot = None
                seq += 1
                continue

            # Playback: take a free slot FIRST, then read, so a frame is only
            # read when the pipeline can take it — it never sits waiting
            # (pure added latency) and none is dropped. Live: read first (the
            # camera keeps producing regardless) and drop the frame if every
            # slot is busy.
            if controller is not None and slot is None:
                try:
                    slot = free_q.get(timeout=0.2)
                except queue.Empty:
                    continue
            t_read = time.time()
            result = read()
            if result is None:
                time.sleep(0.02)                  # paused at end / stopped: idle
                continue
            frames, meta = result
            color, depth, xyz, ir, raw_valid, imu_state = frames
            if color is None:
                continue
            capture_time = time.time()
            if controller is None:
                try:
                    slot = free_q.get_nowait()
                except queue.Empty:
                    dropped += 1
                    continue
            s = ring[slot]
            s["color"][...] = color
            s["depth"][...] = depth
            s["raw_valid"][...] = raw_valid
            if ir is not None:
                s["ir"][...] = ir
            out_meta = {
                "seq": seq, "capture_time": capture_time,
                "frame_time_s": meta.get("frame_time_s", capture_time),
                "discontinuity": meta.get("discontinuity"),
                "frame_index": meta.get("frame_index"),
                "imu_state": imu_state, "has_ir": ir is not None, "dropped": dropped,
                "timing": {"capture": time.time() - t_read},
            }
            for q in q_out:
                q.put((slot, out_meta))
            slot = None
            seq += 1
    except Exception:
        _report_error(ui_q, "capture")
    finally:
        _trace("capture: loop ended, shutting down")
        stop.set()
        for q in q_out:
            q.put(None)
        _release_queues(ui_q, info_q)
        try:
            source.stop()
        except Exception:
            pass
        # Give downstream a moment to finish with the shared slots, then
        # release the block (the creator unlinks it).
        time.sleep(0.5)
        if ring is not None:
            ring.close()


# --------------------------------------------------------------- semantic

def semantic_main(config_snapshot, ring_handle, q_in, q_out, ui_q):
    config = _bootstrap(config_snapshot, "semantic")
    from mp_pipeline.shm import FrameRing
    ring = FrameRing.attach(ring_handle)
    try:
        from segformer_onnx import SegformerOnnx
        model = SegformerOnnx()
        if not model.available:
            print(f"[segformer] unavailable — pure-depth ground segmentation: {model.error}", flush=True)
        info = {"available": model.available, "error": model.error, "mode": model.mode,
                "id2label": model.id2label}
        pd = config.PROCESSING_DOWNSCALE
        while True:
            item = q_in.get()
            if item is None:
                break
            slot, meta = item
            t0 = time.time()
            if meta.get("discontinuity"):
                model.reset()
            color = ring[slot]["color"]
            h, w = color.shape[:2]
            size = (max(1, w // pd), max(1, h // pd))
            prob = model.driveable_probability(color, size, want_class_ids=config.DEBUG)
            sky = model.sky_mask(size) if prob is not None else None
            class_ids = model.last_class_ids
            semantic = {
                "prob": prob, "sky": sky,
                "class_ids": None if class_ids is None else class_ids.astype(np.uint8),
                "info": dict(info, last_infer_ms=model.last_infer_ms),
            }
            meta["timing"]["semantic"] = time.time() - t0
            q_out.put((slot, meta, semantic))
    except Exception:
        _report_error(ui_q, "semantic")
    finally:
        _trace("semantic: loop ended, shutting down")
        q_out.put(None)
        _release_queues(ui_q)
        ring.close()


# ------------------------------------------------------------------ speed

def speed_main(config_snapshot, ring_handle, nominal_dt, q_in, q_out, mask_q, ui_q):
    """Optical-flow + IMU speed per frame, in parallel with semantic/geometry."""
    config = _bootstrap(config_snapshot, "speed")
    from mp_pipeline.shm import FrameRing
    ring = FrameRing.attach(ring_handle)
    try:
        from perception_core import upsample_mask, z_as_xyz
        from speed_estimator import SpeedEstimator
        estimator, last_time, epoch, mask = SpeedEstimator(), None, 0, None
        pd = config.PROCESSING_DOWNSCALE
        while True:
            item = q_in.get()
            if item is None:
                break
            slot, meta = item
            t0 = time.time()
            if meta.get("discontinuity"):
                estimator, last_time, epoch, mask = SpeedEstimator(), None, epoch + 1, None
            # Newest ground mask geometry has published — from THIS scene only
            # (masks from before a seek/loop carry an older epoch).
            while True:
                try:
                    m_epoch, m = mask_q.get_nowait()
                    if m_epoch == epoch:
                        mask = m
                except queue.Empty:
                    break
            now = meta["frame_time_s"]
            dt = nominal_dt if last_time is None else max(1e-4, now - last_time)
            last_time = now
            s = ring[slot]
            h, w = s["depth"].shape[:2]
            land_full = (upsample_mask(mask, w, h, pd) if mask is not None
                         else np.zeros((h, w), dtype=bool))
            speed = estimator.update(s["color"], z_as_xyz(s["depth"]), land_full, dt,
                                     imu_state=meta.get("imu_state"))
            q_out.put((meta["seq"], speed, time.time() - t0))
    except Exception:
        _report_error(ui_q, "speed")
    finally:
        _trace("speed: loop ended, shutting down")
        q_out.put(None)
        _release_queues(ui_q)
        ring.close()


# ------------------------------------------------------------------- lane

def lane_main(config_snapshot, ring_handle, intrinsics, q_in, q_out, plane_q, enabled, ui_q):
    """Lane model (lane_assist plugin) per frame, in parallel with semantic/speed.
    Calibrated from the ground planes geometry publishes; runs the model only
    while lane assist is switched on (shared flag `enabled`)."""
    _bootstrap(config_snapshot, "lane")
    import config_lane as lcfg
    from mp_pipeline.shm import FrameRing
    ring = FrameRing.attach(ring_handle)
    try:
        from lane_assist.base import create_model
        from lane_assist.camera import LaneCamera
        camera = LaneCamera(intrinsics)
        model = create_model(lcfg.LANE_MODEL, lcfg.LANE_MODELS[lcfg.LANE_MODEL], camera)
        if not model.available:
            print(f"[lane] {model.error}", flush=True)
        status_error = None if model.available else model.error
        was_on, reported = False, False
        while True:
            item = q_in.get()
            if item is None:
                break
            slot, meta = item
            t0 = time.time()
            while True:
                try:
                    camera.update_from_plane(plane_q.get_nowait())
                except queue.Empty:
                    break
            if meta.get("discontinuity"):
                model.reset()
            on = bool(enabled.value)
            est = None
            if on and model.available:
                est = model.infer(ring[slot]["color"], meta["frame_time_s"])
                if est is not None and "view_from_road" not in est.info:
                    est.info["view_from_road"] = camera.view_from_road()
                    est.info["camera_height_m"] = camera.height_m
                if est is not None and not reported:
                    print(f"[lane] {camera.describe()}", flush=True)
                    reported = True
            elif was_on and not on:
                model.reset()
            was_on = on
            q_out.put((meta["seq"], est, time.time() - t0, status_error))
    except Exception:
        _report_error(ui_q, "lane")
    finally:
        _trace("lane: loop ended, shutting down")
        q_out.put(None)
        _release_queues(ui_q)
        ring.close()


# --------------------------------------------------------------- geometry

def geometry_main(config_snapshot, ring_handle, intrinsics, nominal_dt, q_in, q_out, cmd_q, ui_q,
                  speed_q, mask_q, lane_q=None, plane_q=None, lane_enabled=None):
    config = _bootstrap(config_snapshot, "geometry")
    from mp_pipeline.shm import FrameRing
    ring = FrameRing.attach(ring_handle)
    try:
        from perception_core import Perception
        perception = Perception()
        epoch = 0
        while True:
            item = q_in.get()
            if item is None:
                break
            # Runtime vehicle-size changes from the UI (vehicle_profile listener).
            while True:
                try:
                    width_m, length_m = cmd_q.get_nowait()
                    config.TRACTOR_WIDTH_M, config.TRACTOR_LENGTH_M = width_m, length_m
                except queue.Empty:
                    break
            slot, meta, semantic = item
            if meta.get("discontinuity"):
                # A seek / loop / restart: nothing carried over from the
                # previous frame applies (same as main_recorded's rebuild).
                perception.reset()
                epoch += 1
                print(f"[playback] {meta['discontinuity']}: per-frame state reset", flush=True)
            # This frame's speed, computed in parallel by speed_main (results
            # arrive in frame order, exactly one per frame).
            speed_item = speed_q.get()
            if speed_item is None:
                break
            speed_seq, speed_mps, speed_s = speed_item
            if speed_seq != meta["seq"]:
                raise RuntimeError(f"speed result for frame {speed_seq} arrived for frame {meta['seq']}")
            lane_est, lane_error = None, None
            if lane_q is not None:
                lane_item = lane_q.get()
                if lane_item is None:
                    break
                lane_seq, lane_est, lane_s, lane_error = lane_item
                if lane_seq != meta["seq"]:
                    raise RuntimeError(f"lane result for frame {lane_seq} arrived for frame {meta['seq']}")
                stage_lane_s = lane_s
            t0 = time.time()
            frame = _frame_view(ring, slot, meta)
            stage_timing = {}
            result, hud = perception.step(frame, meta["frame_time_s"], nominal_dt, intrinsics,
                                          semantic, stage_timing, speed_mps=speed_mps,
                                          lane_estimate=lane_est,
                                          lane_enabled=bool(lane_enabled.value) if lane_enabled is not None else False,
                                          lane_model_error=lane_error)
            if plane_q is not None and result["plane"] is not None:
                plane_q.put(result["plane"])
            if lane_q is not None:
                meta["timing"]["lane"] = stage_lane_s
            mask_q.put((epoch, result["land_mask"]))
            stage_timing.pop("speed", None)
            meta["timing"].update(stage_timing)
            meta["timing"]["speed"] = speed_s
            meta["timing"]["geometry"] = time.time() - t0
            if not config.DEBUG:
                semantic = {"info": semantic["info"]}   # render only needs the prob/class map for the debug view
            q_out.put((slot, meta, semantic, result, hud))
    except Exception:
        _report_error(ui_q, "geometry")
    finally:
        _trace("geometry: loop ended, shutting down")
        q_out.put(None)
        _release_queues(ui_q, mask_q, plane_q)
        ring.close()


# ------------------------------------------------------------------ render

_STAGES = ("capture", "semantic", "lane", "segment", "speed", "obstacles", "path", "render", "overlay", "debug")


def render_main(config_snapshot, ring_handle, display_handle, intrinsics, q_in, free_q, ui_q, yolo_in, yolo_out):
    config = _bootstrap(config_snapshot, "render")
    from mp_pipeline.shm import DisplayRing, FrameRing
    ring = FrameRing.attach(ring_handle)
    display = DisplayRing.attach(display_handle)
    try:
        from perception_core import render_debug, render_frame
        detections, yolo_status, last_yolo_submit = [], "not started", 0.0
        sums = {k: 0.0 for k in _STAGES}
        n, latency_sum, window_start = 0, 0.0, time.time()
        while True:
            item = q_in.get()
            if item is None:
                break
            slot, meta, semantic, result, hud = item
            t0 = time.time()
            frame = _frame_view(ring, slot, meta)

            # Object detection: hand the latest frame to the YOLO process at
            # most every YOLO_MIN_INTERVAL_S, never wait for it; draw whatever
            # detections are newest (an informational overlay, as before).
            if yolo_in is not None:
                try:
                    while True:
                        detections, yolo_status = yolo_out.get_nowait()
                except queue.Empty:
                    pass
                if t0 - last_yolo_submit >= config.YOLO_MIN_INTERVAL_S:
                    try:
                        yolo_in.put_nowait(frame["color"].copy())
                        last_yolo_submit = t0
                    except queue.Full:
                        pass

            annotated = render_frame(frame, result, detections, intrinsics)
            t_overlay = time.time()
            debug_img, probe = None, None
            if config.DEBUG:
                info = semantic["info"]
                semantic_info = types.SimpleNamespace(
                    available=info["available"], error=info["error"], mode=info["mode"],
                    last_infer_ms=info["last_infer_ms"], id2label=info["id2label"],
                    last_class_ids=semantic.get("class_ids"))
                debug_img, probe = render_debug(frame, result, semantic, semantic_info, intrinsics)

            # Done with the shared slot: hand it back to capture.
            free_q.put(slot)

            seq = meta["seq"]
            buf = seq % display.count
            display.write(buf, seq, annotated, debug_img)
            meta["timing"]["render"] = time.time() - t0
            meta["timing"]["overlay"] = t_overlay - t0
            meta["timing"]["debug"] = time.time() - t_overlay if config.DEBUG else 0.0
            ui_q.put({"type": "frame", "seq": seq, "buf": buf, "hud": hud,
                      "imu_state": meta.get("imu_state"), "yolo": (detections, yolo_status),
                      "probe": probe})

            # Pipeline report: per-stage cost (each in its own process) and
            # the THROUGHPUT actually achieved, which is no longer their sum.
            n += 1
            for k in _STAGES:
                sums[k] += meta["timing"].get(k, 0.0)
            latency_sum += time.time() - meta["capture_time"]
            if n >= 20:
                el = time.time() - window_start
                stage_txt = "  ".join(f"{k}={sums[k] / n * 1000:.1f}" for k in _STAGES)
                print(f"[timing:mp] {stage_txt}  (ms/frame, each stage in its own process)  "
                      f"THROUGHPUT={n / el:.1f} FPS  latency={latency_sum / n * 1000:.0f} ms  "
                      f"frame={meta.get('frame_index')}", flush=True)
                sums = {k: 0.0 for k in _STAGES}
                n, latency_sum, window_start = 0, 0.0, time.time()
    except Exception:
        _report_error(ui_q, "render")
    finally:
        _trace("render: loop ended, shutting down")
        if yolo_in is not None:
            try:
                yolo_in.put(None, timeout=1.0)
            except Exception:
                pass
        ui_q.put({"type": "ended"})
        _release_queues(ui_q, free_q)
        ring.close()
        display.close()


# -------------------------------------------------------------------- yolo

def yolo_main(config_snapshot, yolo_in, yolo_out):
    _bootstrap(config_snapshot, "yolo")
    from yolo_detector import AgriYoloDetector
    detector = AgriYoloDetector()
    status = f"running ({detector.mode})" if detector.available else f"unavailable: {detector.error}"

    def publish(dets):
        try:
            yolo_out.put_nowait((dets, status))
        except queue.Full:
            try:
                yolo_out.get_nowait()
            except queue.Empty:
                pass
            yolo_out.put_nowait((dets, status))

    publish([])
    while True:
        frame = yolo_in.get()
        if frame is None:
            break
        if detector.available:
            publish(detector.detect(frame))
    _trace("yolo: loop ended, shutting down")
    _release_queues(yolo_out)

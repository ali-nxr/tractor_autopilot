# `playback` — recorded RealSense sequences as a live-camera frame source

Plays an Intel RealSense recording (`.db3` / `.bag`) through a perception
pipeline that was written for a live camera, **without changing that
pipeline**. The frame source has the same shape as a live
`RealSenseCapture`, so segmentation, obstacle detection, path planning and
overlay code run unmodified and never need an "is this a file?" branch.

Frames come from librealsense's own playback device
(`config.enable_device_from_file`), not a hand-rolled bag reader, so what
comes out is the real thing: recorded intrinsics, recorded depth scale,
the SDK's own `rs.align`, and the same spatial/temporal depth filters a
live run applies.

## Reusing this in another project

The package imports **nothing** from its host — only `pyrealsense2`,
`numpy` and `cv2` (plus `tkinter`, in the one optional UI module). Copy
the `playback/` folder across; there is nothing to edit inside it.

The whole contract is:

1. Build a `PlaybackOptions` (`options.py` — every knob, with defaults).
2. Optionally inject your own IMU fusion object and your own
   "IMU unavailable" state factory.
3. Read frames.

```python
from playback import (PlaybackOptions, RealSenseBagSource, PlaybackController,
                      PlaybackCapturePipeline, select_sequence)

info = select_sequence(r"C:\data\recordings")          # or a direct path
options = PlaybackOptions(path=info.path, want_motion=info.has_motion)

source = RealSenseBagSource(options, info=info)
controller = PlaybackController(source, on_discontinuity=reset_my_frame_state)
pipeline = PlaybackCapturePipeline(controller)
pipeline.start()

while running:
    frames, capture_time = pipeline.get_latest()       # live-compatible
    if frames is None or capture_time == last_seen:
        time.sleep(0.002); continue
    last_seen = capture_time
    color, depth_m, xyz, ir, raw_valid_mask, imu_state = frames
    ...
```

`get_frames()` also exists on the source, returning the same 6-tuple, if
you want to drop it straight in where a live capture object was used and
skip the threading entirely.

## The modules

| Module | Role |
| --- | --- |
| `options.py` | `PlaybackOptions` — the entire configuration surface |
| `discovery.py` | Find and inspect sequences in a folder without playing them |
| `bag_source.py` | `RealSenseBagSource` — frames, shaped like a live capture |
| `controller.py` | `PlaybackController` — play/pause/step/seek/speed/loop |
| `pipeline.py` | `PlaybackCapturePipeline` — reader thread + `get_latest()` |
| `imu_adapter.py` | Re-time an existing IMU fusion onto the recording's clock |
| `ui_controls.py` | `PlaybackBar` — optional Tk transport bar (the only `tkinter` import) |

`ui_controls` is deliberately **not** imported by `__init__.py`, so the
package works headless with no Tk installed.

## Four things that are not obvious

**A recording can advertise a stream it holds no data for.** Confirmed on
real data: `20261007_145608.db3` advertises a Motion Module with
accel+gyro profiles and contains zero motion messages. A pipeline that
enables accel/gyro against it produces *no framesets at all* — it waits
forever for motion frames that never arrive, with no error and no
timeout. `RealSenseBagSource` therefore resolves streams with a fallback
ladder (richest first) and accepts a combination only after it has
actually delivered a frame. `discovery.py` additionally reads per-topic
message counts straight out of the `.db3` so this is visible before
playback even starts.

**Frame buffers are borrowed, not owned.**
`np.asanyarray(frame.get_data())` is a view into librealsense-owned memory
that returns to the SDK's pool when the frame is released. A live loop
mostly gets away with that by consuming each frame immediately; a player
hands frames to a consumer that may still be working on the previous one.
Every returned array is copied out (`options.copy_frames`). Cheap next to
per-frame processing, and it removes a class of use-after-free corruption
that would look like random depth noise.

**A player must not drop frames; a live pipeline must.** Perception
typically costs more per frame than the recording's frame interval, so
keep-newest/drop-old would skip most of the sequence — and *which* frames
got skipped would depend on how fast the machine is, making runs
unreproducible. The default is backpressure: the reader prepares exactly
one frame ahead and waits. One frame ahead, not zero, keeps capture and
processing overlapping, so determinism costs no throughput. Set
`drop_frames=True` for live-like behaviour.

**Playback has a second clock, and it is the one that matters.** Anything
computed per frame from `dt` — speed from optical flow, slew-rate limits,
track ages, watchdog timers, gyro integration — is wrong if `dt` comes
from the wall clock while the player runs at a different rate than the
recording. Use `meta["frame_time_s"]` (the frames' own timestamps) as
your clock. For IMU fusion specifically, `imu_adapter.playback_fusion()`
subclasses your existing fusion class and swaps only its time source, so
the filter math stays identical:

```python
from realsense_imu import ImuFusion
from playback.imu_adapter import playback_fusion

fusion = playback_fusion(ImuFusion)()
source = RealSenseBagSource(options, imu_fusion=fusion, info=info)
```

If the class does not expose the internals needed, it is returned
unchanged with a warning rather than silently half-wrapped.

## Discontinuities

Seeking, stepping, restarting and looping all mean the next frame does
not follow the previous one. Anything holding frame-to-frame state —
optical flow, object tracks, temporal plane/mask smoothing, slew limiters
— carries nonsense across such a cut (a loop wrap looks like the world
teleporting). `PlaybackController(on_discontinuity=...)` fires once per
cut with `"seek"`, `"restart"` or `"loop"`.

The callback runs on the **reader** thread, so do not mutate the
processing thread's objects from inside it — set a flag and rebuild at
the top of the next processing iteration. `main_recorded.py` shows the
pattern.

The player resets what it owns itself: the SDK temporal depth filter is
rebuilt on a seek (it would otherwise blend two unrelated scenes), and
the IMU timebase is restarted.

## Threading

`next_frame()` is called from exactly one reader thread. Every control
method is safe from any other thread. The SDK playback object is touched
**only** by the reader: controls record an intent for it to apply, and
`controller.status()` serves a UI from the last frame's metadata rather
than querying the device.

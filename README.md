# Tractor Vision V2 — Perception Suite

## V2.2 changes (FPS overhaul, segmentation accuracy, land analyzer removed)

- **Land analyzer removed.** `agri_land_analyzer.py`, `models/agri_cnn.py`,
  and the Land Analysis page are gone. This was adding cost to every frame
  for a vegetation-index heuristic that was never the point of this rig —
  the actual ask is a fast, reliable perception/decision pipeline. If crop
  condition analysis comes back later, it should be its own separate pass,
  not bolted onto the main loop.

- **~4x faster ground segmentation** — this was the single biggest thing
  capping FPS. Profiling found `segment_ground()` costing ~87ms/frame
  (640x480), dominated by two things:
  - RANSAC ran as a 150-iteration **Python for-loop**, each iteration
    recomputing inlier counts over the full ~138k-point seed set (~49ms).
  - The surface-normal consistency check ran `np.gradient` at full image
    resolution (~23ms).

  Both are rebuilt in `ground_segmentation.py` (now a stateful
  `GroundSegmenter` class instead of a bare function):
  - RANSAC is **fully vectorized** — every candidate plane, for every
    iteration, is generated and scored in one batch of numpy matrix ops, no
    per-iteration Python loop. This means running *more* iterations (150 →
    400 by default) costs *less* time than the old 150 did, not more.
  - The winning RANSAC plane is then **refined with a least-squares fit
    (SVD) through all of its inliers**. This is the real accuracy
    improvement, not just more RANSAC tries — a raw 3-point RANSAC sample is
    noisy by construction; a least-squares fit through thousands of inlier
    points is dramatically more accurate and stable. Verified against a
    known ground-truth plane: **fitted normal within 0.01° of true**.
  - The normal-consistency check now runs on a downsampled grid
    (`NORMAL_COMPUTE_DOWNSCALE`) using **area-averaging** (`cv2.INTER_AREA`),
    then upsamples with nearest-neighbor. Area-averaging matters here, not
    just speed: naive strided subsampling keeps per-pixel depth noise fully
    intact and actually made the noise-amplification problem *worse* — it
    was caught mid-build when it fragmented a flat, gently-noisy synthetic
    ground plane into ~350 disconnected speckles instead of one region.
    Area-averaging suppresses that noise the way real depth-noise reduction
    should, restoring full (verified: ~100%) coverage on the same test scene
    while still being faster than the old full-resolution version.
  - The fitted plane is now **smoothed across frames** (hemisphere-aware
    EMA, `PLANE_TEMPORAL_SMOOTHING_ALPHA`) to remove depth-noise flicker in
    the segmentation boundary. A short "coast" window
    (`PLANE_COAST_MAX_MISSES`) means a single bad/occluded frame reuses the
    last good plane instead of blanking the segmentation for one miss —
    verified it coasts through the configured window and then correctly
    reports "no plane" if the loss is sustained, and re-locks immediately
    once a good frame comes back.
  - The extracted land-boundary contour is lightly smoothed
    (`BOUNDARY_SMOOTHING_EPSILON_PX`, `cv2.approxPolyDP`) to remove
    pixel-level jaggies without losing real shape.

  **Net result** (benchmarked on a 640x480 synthetic frame with realistic
  depth noise): **86.75ms → ~20.7ms per frame**, i.e. the segmentation-side
  FPS ceiling went from **~11.5 FPS to ~48 FPS** — while the plane fit is
  measurably more accurate (sub-0.01° vs. a single noisy 3-point sample) and
  the mask is temporally stable instead of flickering frame to frame.

  Everything from the depth-confidence work (IR reflectivity check, no
  hole-filling upstream, confidence surviving morphological cleanup) is
  unchanged in *behavior* — verified with the same black-cloth and
  raised-obstacle test cases as before, both still correctly excluded from
  "land."

## V2.1 changes (IMU fix, IMU-fused speed, CUDA, more FPS)

- **IMU fix — two bugs, both fixed:**
  1. *Architecture*: the IMU used to run on its own independent
     `rs.pipeline()`, separate from the depth/color pipeline. Two pipelines
     opened against the same physical RealSense device is a well-known way
     to get an IMU that silently doesn't come up — many USB
     controllers/driver stacks only let one pipeline claim the device, so
     the second `pipeline.start()` either throws or never yields frames.
     Motion streams are now enabled in the SAME `rs.config`/`rs.pipeline` as
     depth+color (`realsense_capture.py`), which is the pattern Intel's own
     multi-stream examples use, and there's no longer a separate IMU thread
     at all — IMU data comes out of the same frameset as video every loop
     iteration.
  2. *Math*: the accelerometer-derived pitch/roll formulas in
     `realsense_imu.py` had the axes crossed — a level, stationary camera
     converged toward reporting **90° of roll** (a false rollover-danger
     reading) instead of 0°. Fixed to the standard 3-axis tilt-from-gravity
     formulas for this sensor's axis convention (X: right, Y: down, Z:
     forward): `pitch = atan2(az, sqrt(ax²+ay²))`,
     `roll = atan2(-ax, sqrt(ay²+az²))`. Verified against level, pure-roll,
     and pure-pitch synthetic inputs.
- **Speed now fused with the IMU**, not vision-only. Raw accelerometer
  integration for velocity isn't usable alone — bias drifts the estimate
  into nonsense within seconds — so this isn't a straight swap. Instead
  `speed_estimator.py` integrates the IMU's gravity-compensated forward
  acceleration between frames for responsiveness, and continuously leaks
  that estimate back toward the vision (optical-flow) speed
  (`config.SPEED_IMU_VISION_CORRECTION_RATE`) so drift can't run away —
  the same complementary-filter idea already used for pitch/roll, one
  derivative up. Falls back to vision-only automatically if the IMU isn't
  available, matching old behavior exactly.
- **CUDA**: `device_utils.py` picks `cuda` if `config.CUDA_ENABLED` and a
  CUDA-capable GPU/torch build are actually present, else `cpu` — used by
  the YOLO detector. No code changes needed if you don't have an NVIDIA GPU;
  it just runs on CPU exactly as before. If `torch.cuda.is_available()`
  returns `False` on a machine that does have an NVIDIA GPU, it's almost
  always because `pip install torch` picked a CPU-only wheel — see "GPU /
  CUDA setup" below.
- **More FPS**:
  - Camera stream FPS default raised 30 → 60 (`config.FPS`; needs USB3 —
    drop it back down if `pipeline.start()` fails to negotiate that rate).
  - YOLO detection moved off the main vision thread onto its own
    (`YoloWorker` in `main.py`). It always processes the freshest submitted
    frame and drops stale ones rather than queuing — so it runs at whatever
    rate the device can sustain (fast on a CUDA GPU) without ever blocking
    the main perception loop, replacing the old fixed "every Nth frame"
    throttle that existed to protect the main loop from a slow synchronous
    CPU call.
  - The per-point Python loop in the optical-flow speed estimator is now a
    vectorized numpy computation.
  - UI redraw interval matched to the new camera FPS
    (`config.UPDATE_INTERVAL_MS`).

### GPU / CUDA setup
`pip install torch` alone sometimes resolves to a CPU-only wheel depending
on your platform/Python version. If `device_utils.get_torch_device()` (or
the Detections page) reports `cpu` on a machine that does have an NVIDIA
GPU: check `nvidia-smi` works first (driver installed), then install the
CUDA build of torch from the selector at
https://pytorch.org/get-started/locally/ (pick your OS + CUDA version) in
place of the plain `pip install torch` in `requirements.txt`.

At startup the app now prints a specific diagnostic line to the console,
e.g. `[device_utils] GPU ACTIVE: using CUDA GPU: NVIDIA RTX 4070 Laptop GPU
(torch 2.4.0, CUDA 12.1)` or `[device_utils] GPU NOT IN USE: <specific
reason>` — check that line first; it tells you exactly which of the causes
below applies instead of you having to guess. The Detections page's status
line mirrors the same thing.

#### Laptop GPU switching (Optimus / hybrid graphics)
Laptops with both an integrated GPU and an NVIDIA GPU often don't run every
process on the NVIDIA one by default, even with `torch.cuda.is_available()`
returning `True` at the driver level — this shows up as low/no GPU
utilization in Task Manager's Performance tab or `nvidia-smi` while the app
is running, even though the diagnostic line above says GPU is active
(is_available() checks the driver/runtime, not which physical GPU actually
executes the work on every platform/config).
- **Windows**: Settings → System → Display → Graphics → find `python.exe`
  (or add it) → set to **High performance**. Also check the NVIDIA Control
  Panel → Manage 3D settings → Program Settings → add python.exe → set
  "Preferred graphics processor" to the NVIDIA GPU.
- **Linux (Optimus laptops, e.g. NVIDIA Prime)**: run with
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py`,
  or check `prime-select` / `nvidia-prime` is set to the NVIDIA GPU.
Once set, `nvidia-smi` should show python.exe/main.py's GPU utilization
climb while the Detections page is active.

## Quick start

1. Plug in the RealSense camera (D435i or D455 if you want the IMU page to
   actually report data — D415 and other non-`i` cameras have no motion
   module, and the app will just say so on that page and keep working).
2. Run the launcher for your OS:
   - **Windows**: double-click `run.bat`
   - **Linux / Mac**: `./run.sh`

That's it — it creates a virtual environment, installs everything from
`requirements.txt`, and opens the app. Only needs to install once; later runs
reuse the same `venv` folder and start straight away.

If you'd rather run it manually:

```bash
pip install -r requirements.txt
python main.py
```

**First run will be slower** — `ultralytics` auto-downloads the YOLOv8n
weights (~6 MB) the first time the Detections page runs, and `torch` itself
is a large install. Both are optional in the sense that the rest of the app
works fine without them (see "Optional dependencies" below); they're only
required for the Detections page.

---

Vision-only test bed for the MF 385 autonomy project. **No actuator control.**
This just proves out the perception + decision-logic pipeline on a screen
before it's wired to steering/brakes.

## What it does, frame by frame

1. **Capture** — RealSense color + depth (+ IMU, if present), aligned,
   deprojected to a per-pixel 3D point map (vectorized, no Python pixel
   loops).
2. **Ground segmentation** (`ground_segmentation.py`, `GroundSegmenter`) —
   fits a plane via fully-vectorized RANSAC + least-squares refinement using
   points from the bottom ~45% of the frame (safest "definitely ground"
   region), classifies every pixel in the whole frame as land / not-land
   based on distance to that plane and local surface-normal consistency,
   smooths the plane across frames, and extracts the **land boundary
   contour**. See "Segmentation: speed and accuracy" below for the full
   breakdown.
3. **Forward corridor** (`obstacle_decision.py`) — a real-world lateral band
   (`CORRIDOR_HALF_WIDTH_M` = tractor width/2 + margin) computed directly from
   3D X coordinates, not just image columns — so it stays correct regardless
   of camera tilt.
4. **Obstacle detection** — pixels inside the corridor that are *not* land and
   rise more than `OBSTACLE_HEIGHT_ABOVE_PLANE_M` above the ground plane are
   grouped into blobs; each blob's nearest depth (5th percentile, robust to
   noise) becomes its reported distance.
5. **Decision layer**, physics-based (tune in `config.py`):
   - **Brake %**: the deceleration actually required to stop before the
     obstacle at the current speed (`v² / 2d`), expressed as a percentage of
     `MAX_DECELERATION_MPS2` — see "Speed estimation & physics-based
     braking" below.
   - **Steer suggestion (deg)**: proportional to how far the nearest
     obstacle's centroid sits off-center inside the corridor, capped at
     `MAX_STEER_SUGGESTION_DEG`.
6. **Overlay + UI** — draws land fill (green), boundary (yellow), corridor
   outline (blue), obstacle boxes + distance labels (red), and a suggested
   path curve colored green/orange/red by brake severity, across separate
   Live Feed / Detections / IMU / Analytics pages (`ui/app.py`).

## Segmentation: speed and accuracy

See the "V2.2 changes" section at the top for the full before/after
breakdown of what was slow, why, and what changed. Short version:

- RANSAC: 150-iteration Python loop → fully vectorized batch (numpy matrix
  ops), with iterations raised 150 → 400 since it's now nearly free, plus a
  least-squares refit through all inliers for real accuracy (not just more
  RANSAC tries).
- Normal-consistency check: full-resolution `np.gradient` →
  `cv2.INTER_AREA`-downsampled + nearest-neighbor upsample. Area-averaging
  (not naive strided subsampling) is what makes this both faster *and* more
  robust to depth noise — this was caught as an actual bug during
  optimization, not a theoretical concern.
- Plane fit is now smoothed across frames with a short miss-tolerant "coast"
  window, instead of being refit from scratch (and re-flickering) every
  single frame.

**Tunables that trade speed for accuracy** if you want to push further in
either direction: `RANSAC_ITERATIONS`, `RANSAC_SUBSAMPLE_POINTS`,
`RANSAC_REFINE_MAX_POINTS`, `NORMAL_COMPUTE_DOWNSCALE` (set to `1` for
full-resolution normals, slower), `PLANE_TEMPORAL_SMOOTHING_ALPHA` (lower =
smoother but laggier).

## Depth confidence (why black cloth was being called "land")

RealSense stereo depth works by matching an IR speckle pattern between two
IR sensors. Black/dark, IR-absorbing material reflects almost nothing back,
so the stereo matcher gets no real match there — a depth "hole." Left alone
(or worse, hole-filled), those holes can end up with a depth value close to
the ground plane by coincidence, and a pure "distance to plane" classifier
will call that "land."

Fixes in the pipeline (`realsense_capture.py` + `ground_segmentation.py`):

- **No hole-filling filter** — only spatial + temporal denoise, which never
  invents depth by copying nearby ground values into a hole.
- **IR reflectivity check** — the raw infrared frame is read alongside depth;
  pixels with IR return below `IR_LOW_REFLECTIVITY_THRESHOLD` are marked
  low-confidence and excluded from land (and obstacle) classification.
- **Pre-filter validity tracking** — `raw_valid_mask` captures whether the
  stereo matcher resolved a pixel *before* any smoothing, so smoothing can't
  hide an unresolved pixel as a "confident" one.
- **Surface-normal consistency** — even where depth is valid, a pixel only
  counts as land if its local surface patch is actually flat and faces the
  same way as the fitted ground plane (`NORMAL_MAX_ANGLE_DEG`).
- **Morphology can't resurrect excluded pixels** — the land mask is
  re-intersected with the confidence/validity mask *after* cleanup, so
  small-gap-closing morphology can't quietly patch an untrusted hole back
  into "land."
- **Honest UI** — untrusted regions render as gray hatch instead of silently
  becoming green land or a red obstacle, so you can see where the sensor
  itself isn't sure.

If it's still tagging dark objects as land after this, the next lever is
`IR_LOW_REFLECTIVITY_THRESHOLD` — raise it if too much real ground (e.g. dark
wet soil) is being excluded as "low confidence" too.

## Speed estimation & physics-based braking

Speed is estimated from the camera (`speed_estimator.py`, fused with the IMU
if present — see "V2.1 changes" above): texture points on the segmented
ground are tracked frame-to-frame with optical flow, and the RealSense depth
at each tracked point tells us how much closer it got. For a static ground
point and a forward-translating camera, that closing distance *is* the
distance the tractor moved — divide by the frame's `dt` and you have m/s.
Median across all tracked points (not mean) keeps one bad match from skewing
it, and the result is EMA-smoothed since raw frame-to-frame flow is
naturally noisy.

Braking (`obstacle_decision.compute_brake_percent`) is not a fixed distance
ladder — it's the deceleration actually required to stop before the obstacle
at the current speed (`v² / 2d`), expressed as a percentage of the tractor's
assumed max braking capability (`MAX_DECELERATION_MPS2`). Same distance at
1 km/h and at 8 km/h produce very different brake %, the way it should. A
reaction-time buffer (`REACTION_TIME_S`) and a hard minimum stand-off
(`STOP_SAFETY_MARGIN_M`, always 100% inside this regardless of speed) are
built into the distance used for the calculation.

Both brake % and the steering suggestion are passed through a slew-rate
limiter (`smoothing.py`) so they ramp at a capped rate
(`MAX_BRAKE_CHANGE_PCT_PER_S`, `MAX_STEER_CHANGE_DEG_PER_S`) instead of
jumping straight from 0 to 100 in one frame — closer to how a real
hydraulic/mechanical system responds.

**Tuning on the real tractor**: `MAX_DECELERATION_MPS2` is the one that
matters most and can only really be set from real braking tests — it's a
placeholder (3.0 m/s²) right now. `SPEED_EMA_ALPHA` trades responsiveness vs.
noise in the speed readout; lower it if the speed number jitters too much,
raise it if it lags real changes in speed.

## What to tune first on the real mount

- `GROUND_SEED_ROW_FRACTION` / `RANSAC_INLIER_THRESHOLD_M` — how strict the
  "this is flat ground" fit is. Loosen if the field is bumpy and it keeps
  losing the plane; tighten if it's calling crop rows "ground".
- `TRACTOR_WIDTH_M` / `CORRIDOR_MARGIN_M` — must match the real tractor.
- `OBSTACLE_HEIGHT_ABOVE_PLANE_M` — raise this if tall grass/crop is
  triggering false obstacles.
- `MAX_DECELERATION_MPS2` / `REACTION_TIME_S` — set against real braking
  tests once you know the tractor's actual stopping performance.
- `IMU_MOUNT_PITCH_OFFSET_DEG` / `IMU_MOUNT_ROLL_OFFSET_DEG` — zero the IMU
  page's reading against the real mount (or use its "Zero on current
  orientation" button).

## Object detection (Detections page)

`yolo_detector.py` wraps a stock Ultralytics YOLOv8n model — **no
agriculture-specific weights ship with this project**, because training a
real crop-row/weed/disease detector needs a labeled agricultural dataset
this repo doesn't have. What it does instead: runs the COCO-pretrained model
and filters its output down to `config.YOLO_ALLOWED_CLASSES` — people,
livestock/animals, and other vehicles — which is genuinely useful for a
"something's in the corridor that isn't ground" safety check, but it will
**not** identify crops, weeds, or field boundaries. `ground_segmentation.py`
already handles land/not-land geometrically and is a much better tool for
that than an object detector.

To upgrade to a real agriculture detector: train or download YOLO weights on
an agricultural dataset (Roboflow Universe has several public row-crop/weed/
livestock sets), point `config.YOLO_WEIGHTS_PATH` at the resulting `.pt`
file, and update `YOLO_ALLOWED_CLASSES` to match its label set. Nothing else
in `yolo_detector.py` needs to change.

Detection runs on its own thread (`YoloWorker` in `main.py`), always
processing the freshest submitted frame — it naturally runs at whatever rate
the device (GPU if available) can sustain without ever blocking the main
vision loop. If `ultralytics` isn't installed, the Detections page just shows
"unavailable" and everything else in the app keeps working.

## IMU / attitude (IMU page)

`realsense_imu.py` reads the accelerometer + gyroscope from a RealSense
motion module — **only present on D435i / D455**, not D415 or other
non-`i` cameras. Motion streams are enabled on the SAME pipeline as
depth+color (`realsense_capture.py`) and fused with a complementary filter
(`IMU_COMPLEMENTARY_ALPHA` in `config.py`) into:

- **Pitch / roll** — drift-corrected by the gravity vector, shown on an
  artificial-horizon widget.
- **Yaw / heading** — gyro-integrated only, **no magnetometer correction
  exists in this pipeline, so it will drift** — treat it as a relative
  turn-rate integral, not a compass heading.
- **Tilt off vertical** (combined pitch+roll lean) — compared against
  `IMU_TILT_WARNING_DEG` / `IMU_TILT_DANGER_DEG` for a rollover-risk banner.
  These are placeholder numbers (15°/25°); replace them with the tractor's
  actual stability/ROPS rating before relying on this operationally.
- **Raw acceleration + angular rate**, for bump/impact or turn-rate reference.

Mounting is never perfectly square to the chassis. Drive on genuinely flat
ground, then either use the IMU page's **"Zero on current orientation"**
button (a runtime calibration) or bake the offset into
`IMU_MOUNT_PITCH_OFFSET_DEG` / `IMU_MOUNT_ROLL_OFFSET_DEG` in `config.py` (a
persistent one) so the dashboard reads ~0°/0° on flat ground.

If the camera has no motion module, or the combined pipeline can't start
with motion streams enabled, the app automatically retries video-only and
the IMU page reports "unavailable" with the specific reason instead of
crashing anything else.

## Optional dependencies

| Missing dependency | What happens |
|---------------------|--------------|
| `ultralytics` / no YOLO weights | Detections page shows "unavailable"; everything else works normally. |
| `torch` | Same as above for YOLO — nothing else in the app depends on it. |
| Camera has no IMU (D415, etc.) | IMU page reports "unavailable" with the reason; rest of the app unaffected. |

## Known limitations (by design — this is still an early pass)

- Single dominant ground plane — won't handle sloped or terraced fields well;
  a patch-wise RANSAC grid is the natural upgrade path once this base
  pipeline is validated.
- YOLO detection is COCO-class-filtered, not a trained agricultural model —
  see "Object detection" above.
- Yaw has no magnetometer correction and will drift — see "IMU / attitude"
  above.
- Steering "path" is an illustrative curve for the dashboard, not a real
  planned trajectory — that comes later with the control layer.

## File map

| File                        | Responsibility                                          |
|------------------------------|----------------------------------------------------------|
| `config.py`                  | every tunable parameter                                  |
| `realsense_capture.py`       | camera color/depth/IMU I/O + vectorized 3D deprojection   |
| `realsense_imu.py`           | complementary-filter fusion of accel/gyro into pitch/roll/yaw |
| `ground_segmentation.py`     | `GroundSegmenter` — vectorized RANSAC+refinement plane fit, land mask, boundary |
| `speed_estimator.py`         | optical-flow + depth (+ IMU-fused) forward speed estimate  |
| `obstacle_decision.py`       | corridor mask, obstacle blobs, physics-based brake %, steer deg |
| `smoothing.py`               | slew-rate limiter for brake %/steer ramping                |
| `overlay.py`                 | OpenCV drawing for land/corridor/obstacles onto the frame   |
| `yolo_detector.py`           | agriculture-filtered YOLOv8 wrapper + box drawing           |
| `device_utils.py`            | CUDA/CPU device selection + diagnostics for YOLO            |
| `shared_state.py`            | thread-safe handoff between workers and UI, + history buffers |
| `ui/app.py`                  | main window shell + sidebar page navigation                |
| `ui/live_view.py`            | Live Feed page                                             |
| `ui/detection_view.py`       | Detections page                                             |
| `ui/imu_view.py`             | IMU / Motion page                                            |
| `ui/analytics_view.py`       | Analytics (time-series graphs) page                          |
| `ui/widgets.py`               | reusable stat card / bar gauge / attitude-indicator widgets |
| `ui/theme.py`                 | shared colors/fonts                                          |
| `main.py`                    | wires camera + IMU + YOLO workers together, entry point     |

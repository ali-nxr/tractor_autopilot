"""
config.py — every tunable number the core system uses: camera/IMU,
ground segmentation, obstacle detection + avoidance, drivable path,
braking, and steering. Nothing else.
"""

# ---------------- Camera stream settings ----------------
# Depth and color are requested at DIFFERENT resolutions on purpose — this
# specific combination is the one confirmed (via the RealSense Viewer) to
# actually resolve on the D435i together with the IMU motion streams;
# requesting them at the same resolution failed to resolve at all.
# FRAME_WIDTH/HEIGHT is the COLOR stream's resolution, and — since
# get_frames() aligns everything to the color frame — also the resolution
# every array returned from get_frames() actually has.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEPTH_WIDTH = 848
DEPTH_HEIGHT = 480
FPS = 30
DEPTH_UNITS_TO_METERS = 0.001   # RealSense depth is uint16 in mm by default

# ---------------- IMU ----------------
IMU_ENABLED = True
IMU_ACCEL_FPS = 100          # confirmed valid for the D435i's BMI085 IMU chip
IMU_GYRO_FPS = 200
# Complementary-filter blend weight (see realsense_imu.py): how much of
# each step's pitch/roll comes from integrating the gyro (fast, responsive,
# drifts) vs. how much gets pulled from the accelerometer's absolute
# gravity-vector reading (noisy, but doesn't drift). 0.98 is a standard
# starting point for a filter running at the gyro's own update rate.
IMU_COMPLEMENTARY_ALPHA = 0.98
# The camera is never mounted perfectly square to the chassis — these
# correct for that. LAB PLACEHOLDERS: 0.0 (no correction) until set from
# the IMU page's "zero on flat ground" reading with the camera actually
# mounted on the real tractor (see calibration_placeholders() below).
IMU_MOUNT_PITCH_OFFSET_DEG = 0.0
IMU_MOUNT_ROLL_OFFSET_DEG = 0.0
# Rollover-risk thresholds (combined pitch+roll tilt off vertical). These
# are LAB PLACEHOLDERS — set from the tractor's actual ROPS/stability
# rating before relying on them operationally (see calibration_placeholders()).
IMU_TILT_WARNING_DEG = 15.0
IMU_TILT_DANGER_DEG = 25.0
# How much the IMU-integrated speed estimate is pulled back toward the
# vision-based one per second, to correct accelerometer drift without
# fully discarding the IMU's own smoother, faster-reacting estimate.
SPEED_IMU_VISION_CORRECTION_RATE = 1.5

# ---------------- Speed estimation (optical flow + IMU fusion) ----------------
# See speed_estimator.py. Vision speed comes from tracking sparse features
# on the segmented ground with Lucas-Kanade optical flow and reading how
# their depth changed frame to frame.
LK_WIN_SIZE = (21, 21)             # standard pyramidal-LK search window size
LK_MAX_PYRAMID_LEVEL = 3
SPEED_FEATURE_MAX_CORNERS = 150    # matches the "150 tracked points/frame" this module is written against
SPEED_FEATURE_QUALITY = 0.1        # goodFeaturesToTrack qualityLevel
SPEED_FEATURE_MIN_DISTANCE = 15    # pixels between tracked features
SPEED_MIN_TRACK_POINTS = 20        # below this, vision speed isn't trusted this frame
SPEED_MAX_REASONABLE_MPS = 15.0    # ~54 km/h — covers field work through road transport, clamps sensor-noise spikes
SPEED_EMA_ALPHA = 0.3              # final smoothing on the fused speed estimate

# ---------------- Depth confidence ----------------
# Guards against black/IR-absorptive material reading as false "ground".
ENABLE_IR_CONFIDENCE = True
IR_STREAM_INDEX = 1
IR_LOW_REFLECTIVITY_THRESHOLD = 15
SPATIAL_FILTER_ALPHA = 0.5
SPATIAL_FILTER_DELTA = 20
TEMPORAL_FILTER_ALPHA = 0.4
TEMPORAL_FILTER_DELTA = 20
# Set False to skip the RealSense SDK's own spatial+temporal denoising
# entirely — this is real, native-SDK processing cost per frame (measured,
# not assumed), separate from anything in this project's own Python code.
# A direct, isolated test: if turning this off meaningfully drops the
# "capture" time in main.py's [timing] log, the filters (not USB/hardware
# wait, not our own processing) are the real cost there. Off = noisier
# depth — a real quality trade-off, not a free win.
ENABLE_DEPTH_FILTERS = True

# ---------------- Processing resolution (main.py) ----------------
# The full 1280x720 color / 848x480 depth capture resolution has to stay
# exactly as-is — that specific pairing is what makes the IMU motion
# streams resolve on the D435i at all (see the note in FRAME_WIDTH/HEIGHT
# above), and the DISPLAYED video stays full resolution too. This affects
# only the WORKING arrays the expensive geometric math (segmentation,
# obstacles, path search) runs on: main.py downsamples depth/xyz once
# right after capture, runs that math on the smaller arrays, then scales
# just the results (masks, obstacle boxes) back up for display. Real-
# world X/Y/Z values are unaffected — downscaling only reduces sample
# DENSITY, not the correctness of the samples that remain. 2 = half
# linear resolution = a quarter of the pixels for every one of those
# stages. See NORMAL_COMPUTE_DOWNSCALE and SEGMENTATION_DOWNSCALE below,
# both deliberately lowered from their old values so their EFFECTIVE
# total reduction (this value x theirs) lands back at what was actually
# validated, not a new, untested, more aggressive combination.
PROCESSING_DOWNSCALE = 2

# ---------------- Ground segmentation (RANSAC plane fit) ----------------
GROUND_SEED_ROW_FRACTION = 0.45      # bottom fraction of the frame used to seed the fit
RANSAC_ITERATIONS = 400              # fully vectorized — see ground_segmentation.py
RANSAC_INLIER_THRESHOLD_M = 0.04     # a point within 4cm of the fitted plane = "ground"
RANSAC_MIN_SAMPLE_POINTS = 300
RANSAC_SUBSAMPLE_POINTS = 4000
RANSAC_REFINE_MAX_POINTS = 8000
RANSAC_REFINE_CANDIDATE_POINTS = 40000

# Smooths the fitted plane across frames (hemisphere-aware EMA).
PLANE_TEMPORAL_SMOOTHING_ENABLED = True
PLANE_TEMPORAL_SMOOTHING_ALPHA = 0.35
PLANE_COAST_MAX_MISSES = 6           # frames to keep the last good plane through a brief dropout

# Smooths the classified MASK itself (not just the plane) — removes the
# frame-to-frame flicker at the land boundary.
LAND_MASK_TEMPORAL_SMOOTHING_ALPHA = 0.45

MORPH_KERNEL_SIZE = 7
MIN_LAND_BLOB_AREA_PX = 4000
BOUNDARY_SMOOTHING_EPSILON_PX = 2.0

# Local surface-normal consistency: a pixel only counts as land if its
# local patch is actually flat and facing the same way as the fitted
# plane — rejects things that happen to sit at ground height but aren't
# actually flat ground. Reduced 4 -> 2: this is now applied ON TOP OF
# config.PROCESSING_DOWNSCALE (see main.py), which already reduces the
# incoming xyz's resolution once — the EFFECTIVE total reduction this
# receives is PROCESSING_DOWNSCALE x NORMAL_COMPUTE_DOWNSCALE, and this
# value was lowered specifically so that total lands back at roughly the
# same effective resolution this was originally validated at (4x on full
# res), not a new, untested, more aggressive reduction.
NORMAL_CONSISTENCY_ENABLED = True
NORMAL_MAX_ANGLE_DEG = 30.0
NORMAL_COMPUTE_DOWNSCALE = 2

# Runs the expensive morphology + connectedComponents labeling step on a
# downscaled copy of the mask, then upscales the result back — "which
# pixels belong to the largest blob" doesn't meaningfully change with
# resolution for a coherent ground region. Tested bumping this to 4 as a
# speed lever on its own — reverted: it measured SLOWER, not faster, in a
# real test, and caused real obstacle pixels to be misclassified as land
# (a safety regression). Reduced 2 -> 1 for a DIFFERENT reason: this now
# runs on input ALREADY reduced by config.PROCESSING_DOWNSCALE (see
# main.py) — stacking this on top would push the EFFECTIVE total
# reduction for this specific step to the same over-aggressive level that
# already measured unsafe above, just reached a different way. 1 keeps
# the effective total here equal to what was actually validated.
SEGMENTATION_DOWNSCALE = 1

# ---------------- Forward path corridor (real-world, meters) ----------------
# Vehicle size — DEFAULTS only. Can be changed at runtime from the UI's
# "Vehicle Size" panel (see vehicle_profile.py), which updates these two
# attributes live and saves them to VEHICLE_PROFILE_PATH so they survive a
# restart. Every consumer reads config.TRACTOR_WIDTH_M at call time (never
# cached at import), so a change takes effect on the very next frame.
TRACTOR_WIDTH_M = 3.6576             # 12 ft = 12 * 0.3048 m
# NOTE: length is stored/shown but NOT yet used by any perception or
# braking calculation — this is a forward-looking system (corridor width,
# path gap-fitting and the AR surface all depend on WIDTH only). It would
# matter for future turning/swept-path planning.
TRACTOR_LENGTH_M = 4.0               # placeholder — set to the real vehicle
VEHICLE_WIDTH_RANGE_M = (1.0, 6.0)   # accepted input range (sanity limits)
VEHICLE_LENGTH_RANGE_M = (1.5, 15.0)
VEHICLE_PROFILE_PATH = "vehicle_profile.json"
CORRIDOR_MARGIN_M = 0.5
MIN_RANGE_M = 0.5
MAX_RANGE_M = 15.0


def corridor_half_width_m():
    return (TRACTOR_WIDTH_M / 2.0) + CORRIDOR_MARGIN_M


# ---------------- Drivable path planning ----------------
PATH_NUM_STEPS = 15
PATH_LATERAL_BIN_M = 0.10
PATH_LATERAL_SEARCH_MARGIN_M = 1.0   # how far past the corridor's own width to search for a gap
PATH_LATERAL_GAP_CLOSE_BINS = 3      # closes small no-data gaps so sampling sparsity isn't mistaken for an obstacle

# ---------------- Obstacle detection + tracking ----------------
# Pixel minimum (full-res equivalent) only removes specks; the REAL test of
# "is this an obstacle" is physical size below. Lowered 300 -> 120 so a
# 20 cm object is still a candidate at ~12 m (it is only ~225 px there).
MIN_OBSTACLE_BLOB_AREA_PX = 120
# Per-pixel height threshold grows with distance (depth noise ~ z^2):
# threshold = max(OBSTACLE_HEIGHT_ABOVE_PLANE_M, K * z^2) -> 8 cm to ~5 m,
# 19 cm at 8 m, 30 cm at 10 m. See obstacle_decision.height_threshold_m.
OBSTACLE_HEIGHT_NOISE_K = 0.003
# REAL-WORLD size an obstacle must have (measured from the 3D points).
# LAB PLACEHOLDERS — tune on real footage. Trade-off: raising these ignores
# more false alarms but also smaller real objects (a person lying down is
# ~25-30 cm tall, so keep MIN_HEIGHT well below that).
OBSTACLE_MIN_HEIGHT_M = 0.20
OBSTACLE_MIN_WIDTH_M = 0.20
OBSTACLE_TALL_MIN_HEIGHT_M = 0.50   # tall + thin things (posts, poles, thin trunks) ...
OBSTACLE_THIN_MIN_WIDTH_M = 0.05    # ... still count if at least this wide
# Glare: blobs mostly on blown-out bright pixels (reflections, puddles, sun
# glare) are not trusted. Trade-off: a very bright white object in full sun
# could be ignored while it looks blown-out — set False to disable.
OBSTACLE_REJECT_GLARE = True
OBSTACLE_GLARE_LEVEL = 245          # all colour channels at/above this = glare
OBSTACLE_MAX_UNTRUSTED_FRAC = 0.5
# Persistence: must be seen this many frames before it counts ...
OBSTACLE_CONFIRM_FRAMES = 3
# ... EXCEPT a big object that is already close, confirmed instantly.
OBSTACLE_IMMEDIATE_CONFIRM_M = 4.0
OBSTACLE_IMMEDIATE_MIN_HEIGHT_M = 0.40
OBSTACLE_HEIGHT_ABOVE_PLANE_M = 0.08  # LAB PLACEHOLDER — verify against real crop/stubble height
CLOSING_SPEED_EMA_ALPHA = 0.4
OBSTACLE_TRACK_MAX_MATCH_DISTANCE_M = 1.5
OBSTACLE_TRACK_MAX_MISSED_FRAMES = 5

# ---------------- Object detection (YOLO, GPU-only) ----------------
# Runs on its OWN thread. The neural network MATH runs on the GPU, but
# real CPU-side work still happens around it every call — image
# preprocessing, NMS/box-decode postprocessing, and Python's GIL, which is
# shared with the main safety loop even though the two run on separate
# threads. Measured on real hardware: at a high call rate this genuinely
# slowed the CPU-only segmentation/obstacle/path pipeline by roughly 2x —
# "GPU-only" was true for the matrix multiplication, not for the whole
# cost. YOLO_MIN_INTERVAL_S below exists specifically to bound that
# contention. This is an INFORMATIONAL OVERLAY ONLY either way: it never
# feeds braking, steering, or the perception watchdog — those stay
# pure-geometry decisions from ground_segmentation.py / obstacle_decision.py,
# completely unchanged regardless of any of this.
YOLO_ENABLED = True
YOLO_WEIGHTS_PATH = "yolo26n.pt"      # nano — the lightest YOLO26 variant
# TensorRT gives a real speedup (typically 2-5x faster inference than
# plain CUDA) but the engine file it produces is compiled specifically for
# the GPU it was built on — it cannot be shared between machines, and the
# FIRST run on a new machine has to build it (can take a few minutes).
# yolo_detector.py falls back to plain CUDA automatically, with no crash
# and no manual step, if TensorRT isn't installed or the export fails for
# any reason — this is a speed bonus, never a hard requirement.
YOLO_USE_TENSORRT = True
YOLO_USE_HALF_PRECISION = True
YOLO_CONF_THRESHOLD = 0.4
YOLO_IOU_THRESHOLD = 0.45
# Filtered to safety-relevant classes — people, animals, other vehicles —
# not general COCO categories that don't matter in a field.
YOLO_ALLOWED_CLASSES = {
    "person", "dog", "cat", "horse", "cow", "sheep", "bird",
    "car", "truck", "bus", "motorcycle", "bicycle",
}
# How often detection actually runs. Was 0.05s (~20/s) — measured on real
# hardware to fire roughly 4x per single main-loop frame once the main
# loop itself slowed down, repeatedly stealing CPU/GIL time within the
# span of one frame. Raised to 0.3s (~3/s): nothing relevant appears or
# disappears from a field scene in 300ms, and this cuts the contention
# window by roughly 6x. Raise further (e.g. 0.5-1.0) if CPU contention is
# still visible in the [timing] log's core-pipeline numbers after this.
YOLO_MIN_INTERVAL_S = 0.3
# Caps PyTorch's OWN internal CPU threading for whatever part of detection
# doesn't run on the GPU (pre/postprocessing) — same principle as capping
# cv2's internal threading earlier in this project: PyTorch defaults to
# spinning up threads across every available core, which competes with
# the main vision loop's own CPU-bound numpy/OpenCV work for actual CPU
# cycles, on top of the separate GIL-contention issue above.
YOLO_TORCH_CPU_THREADS = 2

# ---------------- CPU threading limits (avoid oversubscription) ----------------
# See main.py's _limit_cv2_threads(). OpenCV defaults to auto-threading
# across EVERY CPU core for each individual call — this app already runs
# its own separate threads (camera capture, YOLO detection), and letting
# both layers compete uncontrolled for cores is a real, previously
# measured cause of slowdown on real (many-core) hardware.
CV2_NUM_THREADS = 1

# ---------------- Braking ----------------
MAX_DECELERATION_MPS2 = 3.0           # LAB PLACEHOLDER — set from a real braking test
REACTION_TIME_S = 0.4
STOP_SAFETY_MARGIN_M = 1.0
MAX_BRAKE_CHANGE_PCT_PER_S = 150.0    # slew limit so brake % doesn't jump instantly

# ---------------- Steering ----------------
MAX_STEER_CHANGE_DEG_PER_S = 60.0
MAX_STEER_SUGGESTION_DEG = 25.0
STEER_GAIN = 18.0                     # deg of suggested steer per meter of lateral obstacle intrusion

# ---------------- Perception watchdog ----------------
# If the ground plane can't be found (or land coverage is suspiciously
# low — camera fogged/obstructed) for this many CONSECUTIVE seconds, the
# safe default is full brake + straight steering rather than acting on
# stale/absent data.
WATCHDOG_MAX_DEGRADED_S = 1.5
MIN_LAND_COVERAGE_PCT = 8.0            # LAB PLACEHOLDER — verify against real field footage

# ---------------- Overlay colors (BGR, for cv2 drawing) ----------------
COLOR_LAND_FILL = (60, 200, 60)
COLOR_LAND_BOUNDARY = (0, 255, 120)
COLOR_LOW_CONFIDENCE = (0, 140, 255)
COLOR_CORRIDOR = (255, 180, 0)
COLOR_OBSTACLE_BOX = (0, 100, 255)
COLOR_YOLO_BOX = (255, 0, 255)  # magenta — visually distinct from every other overlay color
# AR-style drivable-surface strip (see path_planner.py / overlay.py's
# _draw_drivable_surface) — the real tractor-width ground ahead. A
# distinct, vivid blue so "this is the surface you can drive on" reads at
# a glance, separate from the green land fill (which just means
# "classified as ground", not "verified wide enough for the tractor").
COLOR_DRIVABLE_SURFACE = (255, 120, 20)
DRIVABLE_SURFACE_ALPHA = 0.40

# ---------------- AR overlay (ar_ribbon.py + overlay._draw_ar_ribbon) ----------------
AR_ENABLED = True                 # False = legacy overlay look
AR_RIBBON_SAMPLE_M = 0.2          # ribbon sample spacing along the path
AR_GROUND_CELL_M = 0.25           # measured-ground height grid cell size
AR_TEMPORAL_ALPHA = 0.5           # anti-jitter blend (1 = no smoothing)
AR_SNAP_SHIFT_M = 0.5             # lateral change larger than this = real re-route -> snap instantly
AR_MAX_GROW_M_PER_FRAME = 1.5     # far end grows at most this fast (it SHRINKS instantly)
AR_STOP_GAP_M = 0.4               # ribbon ends this far before a blocking point
AR_ALPHA_NEAR = 0.55              # surface opacity near the tractor ...
AR_ALPHA_FAR = 0.20               # ... fading to this at the far end
AR_WARN_RAMP_M = 3.0              # ribbon turns amber over this distance before a STOP
COLOR_AR_EDGE = (255, 245, 210)   # crisp wheel-line edges + major stripes (BGR)
COLOR_AR_GLOW = (255, 230, 150)
COLOR_AR_STRIPE = (255, 210, 150)
COLOR_AR_WARN = (40, 170, 255)    # amber
COLOR_AR_STOP = (40, 40, 255)     # red
# Declutter switches (AR mode only; the HUD side panels already show this info)
AR_SHOW_LAND_FILL = False
AR_SHOW_LAND_BOUNDARY = False
AR_SHOW_CORRIDOR_OUTLINE = False
AR_SHOW_CENTER_LINE = False
AR_SHOW_STATUS_TEXT = False
COLOR_PATH_SAFE = (0, 255, 0)
COLOR_PATH_WARN = (0, 165, 255)
COLOR_PATH_DANGER = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
SHOW_LOW_CONFIDENCE_OVERLAY = True

# ---------------- App window ----------------
WINDOW_TITLE = "Tractor Vision — Perception & Control"
UPDATE_INTERVAL_MS = 33
HISTORY_LENGTH = 150


def calibration_placeholders():
    """
    Values that are still LAB PLACEHOLDERS, not real-tractor numbers —
    printed once at startup and shown in the UI so this is never
    forgotten. Returns [(name, current_value, why_it_matters)].
    """
    return [
        ("MAX_DECELERATION_MPS2", MAX_DECELERATION_MPS2,
         "set this from a real braking test, not a guess"),
        ("IMU_MOUNT_PITCH_OFFSET_DEG", IMU_MOUNT_PITCH_OFFSET_DEG,
         "set from the IMU page's 'zero on flat ground' reading with the camera mounted on the real tractor"),
        ("IMU_MOUNT_ROLL_OFFSET_DEG", IMU_MOUNT_ROLL_OFFSET_DEG,
         "set from the IMU page's 'zero on flat ground' reading with the camera mounted on the real tractor"),
        ("IMU_TILT_WARNING_DEG", IMU_TILT_WARNING_DEG,
         "set from the tractor's actual ROPS/stability rating"),
        ("IMU_TILT_DANGER_DEG", IMU_TILT_DANGER_DEG,
         "set from the tractor's actual ROPS/stability rating"),
        ("OBSTACLE_MIN_HEIGHT_M", OBSTACLE_MIN_HEIGHT_M,
         "smallest real obstacle height to react to — verify on real footage (a lying person is ~25-30 cm)"),
        ("OBSTACLE_HEIGHT_ABOVE_PLANE_M", OBSTACLE_HEIGHT_ABOVE_PLANE_M,
         "verify against real crop/stubble height so it doesn't false-trigger"),
        ("MIN_LAND_COVERAGE_PCT", MIN_LAND_COVERAGE_PCT,
         "verify against real field footage so normal turns/edges don't false-trigger the watchdog"),
    ]

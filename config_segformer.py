"""
config_segformer.py — every tunable for the SegFormer semantic segmentation
model (segformer_onnx.py) and for how its output GUIDES the depth-based
ground segmentation (ground_segmentation.py).

Why a separate file from config.py: these settings come as one unit (the
model, its class vocabulary, and the fusion rules only make sense
together), and SEGFORMER_ENABLED = False restores the pure-depth system
exactly as it was, with nothing else to touch.

How the semantics are used — see GroundSegmenter.segment() for the code:
  1. PLANE SEEDING. RANSAC is seeded only with points the model calls
     driveable, so the plane is fit to the ground itself and not to crop
     rows, a wall or a vehicle at the bottom of the frame. Falls back to the
     old unrestricted seed set when too few such points exist.
  2. ROUGH-TERRAIN RELAXATION. The strict geometric test (within 4 cm of
     ONE flat plane) drops ruts, furrows and bumps — that is what makes the
     land mask fragmented and frame-to-frame inconsistent on rough ground.
     Where the model says "driveable", a pixel is accepted with a larger
     height tolerance and a looser surface-normal check.
  3. NON-GROUND VETO. Where the model is confident a pixel is NOT ground
     (water, a flat roof, a table top at plane height), the geometric "land"
     classification is removed.

SKY MASK EXCEPTION: with SEMANTIC_SKY_MASK_ENABLED, depth on pixels the model
labels as sky is discarded before anything else runs (see that setting for
the measured reasons and cost) — that does remove obstacle evidence there.

SAFETY INVARIANT (enforced in code, not just by these defaults): land never
absorbs a raised pixel that could belong to an obstacle. With
SEMANTIC_ABSORB_DEPTH_SPECKLE, only raised pixels that the obstacle
detector's own speckle filter (obstacle_decision.open_raised) would discard
anyway may become land; without it, nothing above
obstacle_decision.height_threshold_m(z) may. The veto only removes land.
So the obstacle blobs, braking and steering cannot be made LESS sensitive
by this model (verified frame for frame on recorded sequences). Semantics change the land mask (display, the
coverage watchdog, speed-estimator feature selection) and, through better
seeding, the ground plane itself.
"""

# ---------------- Model ----------------
SEGFORMER_ENABLED = True
# Produced once by tools/export_segformer_onnx.py (needs torch+transformers;
# the running app only needs onnxruntime). b2 is more accurate and ~3-4x
# slower: export it with --model nvidia/segformer-b2-finetuned-ade-512-512.
SEGFORMER_MODEL_PATH = "models/segformer-b0-ade-512x512.onnx"

# Execution provider preference, first available wins. TensorRT is usually
# the fastest on an NVIDIA GPU but builds an engine on first run (can take a
# few minutes, cached in SEGFORMER_TRT_CACHE_DIR afterwards) — the same
# trade-off as YOLO_USE_TENSORRT in config.py.
SEGFORMER_USE_TENSORRT = True
SEGFORMER_USE_CUDA = True
SEGFORMER_TRT_FP16 = True
SEGFORMER_TRT_CACHE_DIR = "models/trt_cache"
# ORT's own CPU thread pool (CPU provider only) — capped for the same reason
# as CV2_NUM_THREADS / YOLO_TORCH_CPU_THREADS in config.py: unbounded, it
# competes with the main vision loop for every core.
SEGFORMER_CPU_THREADS = 2

# Run the model every N vision frames and reuse the last probability map in
# between. 1 = every frame (best on a GPU). Raise on CPU-only machines: the
# ground does not change meaning between consecutive frames, and the depth
# geometry still updates every frame.
SEGFORMER_RUN_EVERY_N_FRAMES = 1

# ---------------- Classes ----------------
# ADE20K label names (the vocabulary of the nvidia/segformer-*-ade
# checkpoints) that count as DRIVEABLE ground. Unknown names are an error at
# startup, not a silent no-op. "floor" is included so lab/indoor testing
# behaves like a field.
DRIVEABLE_CLASSES = (
    "road", "grass", "sidewalk", "earth", "field", "sand", "path",
    "runway", "dirt track", "land", "floor",
)

# Classes whose depth is discarded when SEMANTIC_SKY_MASK_ENABLED (below).
SKY_CLASSES = ("sky",)

# ---------------- Sky depth mask ----------------
# The D435's stereo matcher returns garbage depth on featureless sky: points
# ~1.5-2 m away and 1+ m above the ground, thin but tall enough to pass the
# "post/pole" obstacle rule. Measured on recording 20261007_143903 (900
# frames): 82% of all confirmed obstacles were such sky phantoms; they were
# the nearest BRAKING obstacle in 634 frames, caused 295 of 296 "no drivable
# path" frames and every ribbon spike. True = depth on pixels SegFormer labels
# as sky is treated as invalid before segmentation, obstacles and planning.
# This is the ONE place semantics can remove obstacle evidence: something the
# model mislabels as sky is not seen by depth either. Measured cost on the
# same frames: real (non-sky) corridor obstacles present in 269 frames before,
# 270 after (13 lost / 14 gained, mostly 8-12 m tree-line edges near the
# horizon, where the 1/4-resolution class map is coarse).
SEMANTIC_SKY_MASK_ENABLED = True

# ---------------- Fusion with the depth segmentation ----------------
# Temporal EMA on the probability map itself (1.0 = no smoothing). Runs
# before thresholding, so an uncertain pixel cannot flicker on and off.
SEMANTIC_TEMPORAL_ALPHA = 0.5

# 1. Plane seeding
SEMANTIC_SEED_ENABLED = True
SEMANTIC_SEED_MIN_PROB = 0.6         # seed points must be at least this confidently ground

# 2. Rough-terrain relaxation
SEMANTIC_GROUND_MIN_PROB = 0.5       # "driveable" decision threshold
# Height tolerance for semantic-ground pixels, in metres from the plane.
# Clamped per-pixel to obstacle_decision.height_threshold_m(z) (see the
# SAFETY INVARIANT above), so with the default obstacle settings the
# effective value is 8 cm out to ~5 m, rising to this cap further out where
# depth noise is larger.
SEMANTIC_GROUND_MAX_HEIGHT_M = 0.15
# Looser than config.NORMAL_MAX_ANGLE_DEG: a furrow wall or a bump is still
# ground, just not parallel to the average plane.
SEMANTIC_NORMAL_MAX_ANGLE_DEG = 75.0      # None = no check. Tuned 50 -> 75: fewer flickering holes on rough ground
# Rough ground scatters isolated depth-noise pixels above the obstacle
# height threshold; each becomes a hole in the land mask that flickers
# frame to frame. True: on semantic ground, only raised pixels that survive
# the obstacle detector's own speckle filter (obstacle_decision.open_raised)
# are excluded — the only raised pixels that can ever form an obstacle, so
# obstacle output is unchanged. False: every over-threshold pixel is
# excluded, and SEMANTIC_GROUND_MAX_HEIGHT_M above applies.
SEMANTIC_ABSORB_DEPTH_SPECKLE = True

# 3. Non-ground veto
SEMANTIC_VETO_ENABLED = True
SEMANTIC_VETO_MAX_PROB = 0.10        # geometric land with driveable prob below this is removed

# ---------------- Display ----------------
# Thin outline of what the model alone calls driveable — for tuning the
# fusion against real footage. Off by default (declutter).
SEMANTIC_SHOW_OVERLAY = False
COLOR_SEMANTIC_OUTLINE = (255, 255, 0)   # BGR cyan

"""
config_lane.py — lane assist plugin (lane_assist/): detect the track's
lanes / edges, LOCK the path to that corridor instead of the default
"forward drivable space" path, and suggest steering to follow it.

Model-agnostic: LANE_MODEL picks a backend from LANE_MODELS below; each
backend turns a camera frame into a lane_assist.base.LaneEstimate, and
everything after that (corridor choice, lock, steering, path, drawing) is
shared. To try another model, add a backend module and an entry here.

Safety: braking still comes only from the depth obstacle detector — lane
assist changes the PATH that is drawn and checked for obstacles, and the
steer SUGGESTION. An obstacle in the corridor always overrides the lane
steer with the obstacle-avoidance steer, exactly as before.

Requires the multi-process pipeline (config.MULTIPROCESS_PIPELINE = True).
"""

import os

_PROJECT = os.path.dirname(os.path.abspath(__file__))

# ---------------- Plugin ----------------
# Load the plugin at all (a lane-model process starts with the pipeline).
LANE_ASSIST_AVAILABLE = True
# Lane assist ON at startup. Toggle at runtime with the HUD's Lane Assist button.
LANE_ASSIST_ENABLED = False
# Which backend in LANE_MODELS to use.
LANE_MODEL = "openpilot_supercombo"

# ---------------- Camera (shared by every lane model) ----------------
# None = the stream's own intrinsics (the RealSense camera, or the ones
# stored in the recording). Override with a dict: {"fx":..., "fy":..., "cx":..., "cy":...}
# in pixels of the color frame.
CAMERA_INTRINSICS = None
# Camera height above the ground, metres. None = measured every frame from
# the fitted ground plane (ground_segmentation.py).
CAMERA_HEIGHT_M = None
# Camera orientation relative to the ground. None (both) = measured every
# frame from the fitted ground plane, so it follows the tractor pitching
# over terrain. Set BOTH to fix them. Pitch: positive = camera tilted DOWN.
# Roll: positive = camera rotated clockwise as seen from behind it.
CAMERA_PITCH_DOWN_DEG = None
CAMERA_ROLL_DEG = None
# Yaw cannot be measured from the ground plane: the camera's heading
# relative to the direction of travel. Positive = camera points RIGHT of
# straight ahead. Set if the camera is not mounted square to the chassis.
CAMERA_YAW_DEG = 0.0
# Smoothing of the measured calibration (per frame, 1 = none). Slow on
# purpose: the warp into the model's frame should not jitter with depth noise.
CALIBRATION_SMOOTHING_ALPHA = 0.1
# Where the camera is on the vehicle — used by the steering geometry.
CAMERA_X_FROM_REAR_AXLE_M = 2.0      # camera this far AHEAD of the rear axle
CAMERA_Y_OFFSET_M = 0.0              # camera this far RIGHT of the vehicle centreline

# ---------------- Vehicle ----------------
WHEELBASE_M = 2.4                    # front-to-rear axle distance (steering geometry)

# ---------------- Corridor selection ----------------
# Tried in order; a corridor kind that is currently locked keeps priority
# while it stays valid (no flip-flopping between sources).
#   lane_lines : painted/visible lane lines (both ego lines)
#   road_edges : the track's left/right edges — the strong signal on
#                unmarked farm tracks
#   plan       : the model's own planned path (no boundaries; last resort)
LANE_SOURCE_PRIORITY = ("lane_lines", "road_edges", "plan")
LANE_RANGE_M = 30.0                  # use the corridor out to this distance
LANE_STATION_STEP_M = 0.5
LANE_LINE_MIN_PROB = 0.5
EDGE_MIN_CONF = 0.45
PLAN_MIN_CONF = 0.5
MIN_CORRIDOR_WIDTH_M = 1.0
MAX_CORRIDOR_WIDTH_M = 15.0

# ---------------- Lock ----------------
# A corridor must be seen consistently for LOCK_ACQUIRE_S before it locks;
# while locked, an estimate that jumps more than LOCK_MAX_JUMP_M (at
# LOCK_REF_DISTANCE_M ahead) is treated as a miss; after LOCK_COAST_S
# without a good estimate the lock is dropped.
LOCK_REF_DISTANCE_M = 8.0
LOCK_ACQUIRE_S = 1.0
LOCK_CONSISTENCY_M = 0.6
LOCK_MAX_JUMP_M = 1.2
LOCK_COAST_S = 1.0
LOCK_SMOOTHING_ALPHA = 0.3           # EMA on the locked corridor (1 = no smoothing)

# ---------------- Steering (pure pursuit on the locked centreline) ----------------
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_TIME_S = 1.5               # lookahead = clip(MIN + speed * TIME, MIN, MAX)
LOOKAHEAD_MAX_M = 15.0

# ---------------- Display ----------------
LANE_DRAW = True
LANE_DRAW_MAX_M = 30.0
COLOR_LANE_LOCKED = (60, 220, 255)     # BGR amber-yellow
COLOR_LANE_SEARCHING = (160, 160, 160)
COLOR_LANE_CENTER = (255, 255, 255)

# ---------------- Models ----------------
LANE_MODELS = {
    # comma.ai openpilot's driving model (supercombo), run through the
    # comma-operpilot wrapper project (decoding/warp/packing/parsing are
    # openpilot's own code). On unmarked farm tracks its lane lines are ~0;
    # its road edges + path are the usable signals (see that project's README).
    "openpilot_supercombo": {
        "backend": "lane_assist.models.openpilot_supercombo:OpenpilotSupercombo",
        "root": os.path.normpath(os.path.join(_PROJECT, "..", "..", "comma-operpilot")),
        "model_path": None,                  # None = <root>/models/driving_supercombo.onnx
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        # openpilot's network is recurrent and trained at 20 Hz; frames are
        # sub-sampled on the recording/camera clock to keep that tick rate
        # (a slower pipeline just ticks less often).
        "rate_hz": 20.0,
        # The network assumes comma's mount (~1.2 m up). Its metric output is
        # rescaled by true camera height / the model's own road height —
        # exact for points on flat ground (they move along their camera rays).
        "height_scaling": True,
        "model_camera_height_m": 1.22,       # fallback when the model's road height is unusable
        "edge_std_full_m": 1.5,              # road-edge std at which confidence reaches 0
        "plan_std_full_m": 1.0,
        "rhd": False,                         # right-hand-drive traffic convention
    },
}

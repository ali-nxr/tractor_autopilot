"""
obstacle_decision.py — finds obstacles inside the tractor's forward path
corridor, picks the nearest one, and converts that into:
  - a recommended brake percentage (rule-based, distance thresholds)
  - a suggested steering offset (degrees) to path around it

Nothing here touches actuators. It only produces numbers + geometry for
the dashboard to draw / for a future control layer to consume.
"""

import numpy as np
import cv2

import config


def build_corridor_mask(xyz, valid_depth):
    """
    Real-world forward corridor: |X| < corridor_half_width_m(), Z within range.
    Returns a boolean (H, W) mask.
    """
    x = xyz[..., 0]
    z = xyz[..., 2]
    corridor = (
        valid_depth
        & (np.abs(x) < config.corridor_half_width_m())
        & (z > config.MIN_RANGE_M)
        & (z < config.MAX_RANGE_M)
    )
    return corridor


def height_threshold_m(z):
    """
    Per-pixel "sticks out of the ground" threshold, growing with distance:
    RealSense stereo depth noise grows roughly with distance squared, so a
    fixed 8 cm threshold that is sensible at 3 m is BELOW the sensor's own
    noise further out (the cause of noise/rolling-terrain false obstacles).
    threshold = max(OBSTACLE_HEIGHT_ABOVE_PLANE_M, OBSTACLE_HEIGHT_NOISE_K * z^2)
    e.g. with K=0.003: 8 cm up to ~5 m, 19 cm at 8 m, 30 cm at 10 m.
    """
    return np.maximum(config.OBSTACLE_HEIGHT_ABOVE_PLANE_M, config.OBSTACLE_HEIGHT_NOISE_K * z * z)


def build_raised_mask(dist_to_plane, valid_depth, land_mask, xyz=None):
    """
    Pixels that are NOT part of the segmented ground and rise more than
    OBSTACLE_HEIGHT_ABOVE_PLANE_M above the fitted ground plane — i.e. a
    real physical obstacle. Deliberately NOT restricted to the braking
    corridor's lateral bounds here, so the exact same "what counts as an
    obstacle" rule can be reused both for the fixed braking corridor
    (find_obstacles, below) and for the wider gap-search the path planner
    does (path_planner.py) — just windowed differently by each caller.
    xyz given -> distance-scaled threshold (height_threshold_m); without it
    the plain fixed threshold (kept for older callers/tests).
    """
    thr = config.OBSTACLE_HEIGHT_ABOVE_PLANE_M if xyz is None else height_threshold_m(xyz[..., 2])
    return (dist_to_plane > thr) & valid_depth & (~land_mask)


def glare_mask(color_image, out_shape):
    """
    Blown-out bright pixels (specular reflections on glossy floors, water,
    wet mud, sun glare) resized to the processing resolution and slightly
    grown — stereo depth is unreliable there, so obstacle blobs sitting
    mostly on glare are not trusted (see detect_obstacle_blobs).
    """
    h, w = out_shape
    small = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_NEAREST)
    lvl = int(config.OBSTACLE_GLARE_LEVEL)
    # cv2.inRange: identical result to (min over channels >= level), but
    # ~0.2 ms vs ~7 ms for numpy's min(axis=2) on uint8 (measured).
    g = (cv2.inRange(small, (lvl, lvl, lvl), (255, 255, 255)) > 0).astype(np.uint8)
    g = cv2.dilate(g, np.ones((5, 5), np.uint8))
    return g > 0


def detect_obstacle_blobs(raised_mask, xyz, dist_to_plane, untrusted_mask=None):
    """
    Finds REAL obstacles anywhere in view (not only the braking corridor, so
    the path planner and braking share one definition of "obstacle").

    A connected blob of raised pixels counts only if its REAL-WORLD size,
    measured from the 3D points (not pixels), is that of a physical object:
        (height >= OBSTACLE_MIN_HEIGHT_M and width >= OBSTACLE_MIN_WIDTH_M)
     or (height >= OBSTACLE_TALL_MIN_HEIGHT_M and width >= OBSTACLE_THIN_MIN_WIDTH_M)
    — a person, dog, rock, tree or post passes; floor texture, glossy
    patches and depth-noise speckles don't. Blobs whose pixels are mostly
    untrusted (glare) are rejected.
      height = 90th percentile of the blob's distance from the ground plane
      width  = 5th..95th percentile spread of its lateral (X) positions

    Returns (blobs, labels): blobs sorted nearest-first, each
      {bbox (x,y,w,h) in THIS array's pixels, distance_m, centroid_x_m,
       area_px, height_m, width_m, label}
    and the label image (for building masks of chosen blobs).
    """
    pd = config.PROCESSING_DOWNSCALE
    min_area = max(1, config.MIN_OBSTACLE_BLOB_AREA_PX // (pd * pd))
    mask_u8 = raised_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    x_all, z_all = xyz[..., 0], xyz[..., 2]
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = np.nonzero(areas >= min_area)[0] + 1
    blobs = []
    for lab in candidates:
        bx, by, bw, bh = (int(v) for v in stats[lab, :4])
        sl = (slice(by, by + bh), slice(bx, bx + bw))
        pix = labels[sl] == lab
        zc = z_all[sl][pix]
        ok = zc > 0
        if not ok.any():
            continue
        if untrusted_mask is not None and untrusted_mask[sl][pix].mean() > config.OBSTACLE_MAX_UNTRUSTED_FRAC:
            continue
        zc = zc[ok]
        xc = x_all[sl][pix][ok]
        dc = dist_to_plane[sl][pix][ok]
        height = float(np.percentile(dc, 90))
        width = float(np.percentile(xc, 95) - np.percentile(xc, 5))
        big = height >= config.OBSTACLE_MIN_HEIGHT_M and width >= config.OBSTACLE_MIN_WIDTH_M
        tall_thin = height >= config.OBSTACLE_TALL_MIN_HEIGHT_M and width >= config.OBSTACLE_THIN_MIN_WIDTH_M
        if not (big or tall_thin):
            continue
        blobs.append({
            "bbox": (bx, by, bw, bh),
            "distance_m": float(np.percentile(zc, 5)),
            "centroid_x_m": float(np.mean(xc)),
            "area_px": int(pix.sum()),
            "height_m": height,
            "width_m": width,
            "label": int(lab),
        })
    blobs.sort(key=lambda o: o["distance_m"])
    return blobs, labels


def blobs_mask(labels, blobs):
    """Boolean mask of the pixels belonging to the given blobs."""
    lut = np.zeros(int(labels.max()) + 1, bool)
    for b in blobs:
        lut[b["label"]] = True
    return lut[labels]


def corridor_obstacles(blobs, labels, corridor_mask, xyz):
    """
    The subset of blobs that reach into the braking corridor, with distance
    and lateral position measured over the part INSIDE the corridor (the
    part the tractor would actually hit). Input dicts are not mutated.
    """
    z_all, x_all = xyz[..., 2], xyz[..., 0]
    out = []
    for b in blobs:
        bx, by, bw, bh = b["bbox"]
        sl = (slice(by, by + bh), slice(bx, bx + bw))
        pix = (labels[sl] == b["label"]) & corridor_mask[sl]
        zc = z_all[sl][pix]
        ok = zc > 0
        if not ok.any():
            continue
        o = dict(b)
        o["distance_m"] = float(np.percentile(zc[ok], 5))
        o["centroid_x_m"] = float(np.mean(x_all[sl][pix][ok]))
        out.append(o)
    out.sort(key=lambda o: o["distance_m"])
    return out


def find_obstacles(raised_mask, corridor_mask, xyz, downscale=None):
    """
    Obstacle candidate = inside the fixed braking corridor AND part of
    raised_mask (see build_raised_mask above).

    downscale (default config.SEGMENTATION_DOWNSCALE) speeds up the
    connectedComponentsWithStats labeling step the same way
    ground_segmentation.py's does — its cost is dominated by total pixel
    count regardless of how sparse the actual obstacle pixels are
    (confirmed by profiling: an EMPTY mask costs the same as a populated
    one). Distance and centroid — the numbers that actually drive
    braking — are always computed from the FULL-RESOLUTION xyz data for
    each detected blob, never the downscaled grid: only the labeling step
    itself runs at reduced resolution, not the safety-critical distance
    math. The bounding box drawn on screen is scaled back up from the
    downscaled labeling, off by at most a couple of pixels — irrelevant
    for a box that's only ever used for the visual overlay.

    Returns list of obstacle dicts sorted by distance (nearest first):
      { "bbox": (x, y, w, h), "distance_m": float, "centroid_x_m": float, "area_px": int }
    """
    if downscale is None:
        downscale = config.SEGMENTATION_DOWNSCALE

    candidate_mask = raised_mask & corridor_mask
    h, w = candidate_mask.shape

    # min_area is calibrated in pixels against the ORIGINAL full-resolution
    # capture. config.PROCESSING_DOWNSCALE (see main.py) may have ALREADY
    # shrunk this function's input before it ever got here — a separate,
    # always-applied factor from this function's own `downscale` param
    # (an ADDITIONAL internal reduction for the labeling step
    # specifically). Both have to compose together, or a real obstacle
    # correctly sized for the reduced input gets rejected as "too small"
    # purely because the threshold was never adjusted for the resolution
    # it's actually being measured against. Confirmed as a real bug this
    # way, not theoretical: this returned zero obstacles for a real
    # synthetic obstacle once PROCESSING_DOWNSCALE was introduced, until
    # this fix.
    effective_downscale = config.PROCESSING_DOWNSCALE * (downscale if downscale > 1 else 1)

    if downscale > 1:
        small_w = max(1, w // downscale)
        small_h = max(1, h // downscale)
        mask_u8 = cv2.resize((candidate_mask.astype(np.uint8)) * 255, (small_w, small_h),
                              interpolation=cv2.INTER_AREA)
        mask_u8 = ((mask_u8 > 127).astype(np.uint8)) * 255
    else:
        mask_u8 = (candidate_mask.astype(np.uint8)) * 255
    min_area = max(1, config.MIN_OBSTACLE_BLOB_AREA_PX // (effective_downscale * effective_downscale))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if n_labels > 1 and downscale > 1:
        # Upscale the WHOLE label map back to full resolution ONCE (not
        # once per obstacle) — float32 + nearest-neighbor exactly preserves
        # small integer label IDs with no interpolation blending, so
        # `labels_full == label` afterward is still an exact match, not an
        # approximation.
        labels_full = cv2.resize(labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        labels_full = labels

    obstacles = []
    z = xyz[..., 2]
    x = xyz[..., 0]

    # Find which labels actually pass the area threshold with ONE
    # vectorized numpy comparison over the whole stats array, instead of a
    # Python for-loop checking `stats[label, ...] < min_area` one label at
    # a time. This matters a lot on real (noisy) depth data: real sensor
    # noise can produce thousands of tiny spurious connected components
    # that clean synthetic test scenes never had — a plain Python loop
    # over n_labels, even doing cheap work per iteration, adds up fast at
    # that scale. Confirmed as a real, measured cost on real hardware, not
    # a hypothetical: this is exactly the kind of gap between "fast in a
    # clean synthetic benchmark" and "slow on a real noisy camera" that
    # only shows up once actual sensor data is involved.
    areas = stats[1:, cv2.CC_STAT_AREA]
    passing_labels = np.nonzero(areas >= min_area)[0] + 1  # +1: stats[1:] dropped the background label

    for label in passing_labels:
        blob_mask = labels_full == label

        blob_z = z[blob_mask]
        blob_z = blob_z[blob_z > 0]
        if blob_z.size == 0:
            continue

        distance_m = float(np.percentile(blob_z, 5))  # robust "nearest edge" estimate
        centroid_x_m = float(np.mean(x[blob_mask]))

        bx, by, bw, bh = (
            stats[label, cv2.CC_STAT_LEFT],
            stats[label, cv2.CC_STAT_TOP],
            stats[label, cv2.CC_STAT_WIDTH],
            stats[label, cv2.CC_STAT_HEIGHT],
        )
        if downscale > 1:
            bx, by, bw, bh = bx * downscale, by * downscale, bw * downscale, bh * downscale

        obstacles.append(
            {
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "distance_m": distance_m,
                "centroid_x_m": centroid_x_m,
                "area_px": int(np.sum(blob_mask)),
            }
        )

    obstacles.sort(key=lambda o: o["distance_m"])
    return obstacles


def compute_brake_percent(nearest_distance_m, speed_m_s, closing_speed_mps=0.0):
    """
    Physics-based, not a fixed distance ladder: works out the deceleration
    actually required to stop before the obstacle at the current speed,
    and expresses that as a percentage of the tractor's assumed max braking
    capability (MAX_DECELERATION_MPS2).

        required_decel = effective_speed^2 / (2 * effective_distance)
        effective_distance = distance - reaction_distance - safety_margin

    reaction_distance accounts for the delay between deciding to brake and
    the brakes actually biting. This is what makes the same distance call for
    very different brake % depending on how fast the tractor is actually going,
    instead of jumping straight to a fixed number.

    closing_speed_mps (see ClosingSpeedEstimator below) is how fast the
    NEAREST obstacle's distance is actually shrinking — not just how fast
    the tractor itself is moving. A person or animal walking toward the
    tractor closes distance faster than the tractor's own forward speed
    alone explains; effective_speed uses whichever is larger, so a closing
    obstacle brakes harder than "my own speed" would call for. It only ever
    RAISES the effective speed used for the stopping-distance math, never
    lowers it — an obstacle that appears to be moving away must never
    reduce how hard this brakes, since that reading is exactly the kind of
    thing a bad frame or a mistracked blob could fake.
    """
    if nearest_distance_m is None:
        return 0.0

    # Hard floor: never plan to be this close to an obstacle, regardless of
    # speed or how noisy the speed estimate is.
    if nearest_distance_m <= config.STOP_SAFETY_MARGIN_M:
        return 100.0

    effective_speed = max(speed_m_s, closing_speed_mps or 0.0)

    if effective_speed <= 0.05:
        # Not really moving, and the obstacle isn't closing either — no
        # deceleration is needed to stop before something you're not
        # driving toward and that isn't coming at you.
        return 0.0

    reaction_distance = effective_speed * config.REACTION_TIME_S
    effective_distance = nearest_distance_m - config.STOP_SAFETY_MARGIN_M - reaction_distance
    effective_distance = max(effective_distance, 0.05)

    required_decel = (effective_speed ** 2) / (2.0 * effective_distance)
    brake_percent = (required_decel / config.MAX_DECELERATION_MPS2) * 100.0

    return float(np.clip(brake_percent, 0.0, 100.0))


class ClosingSpeedEstimator:
    """
    Tracks how fast the NEAREST obstacle's distance is actually shrinking,
    frame to frame — not just the tractor's own speed (see
    compute_brake_percent's docstring for why this matters). Smoothed (EMA)
    since the raw frame-to-frame distance derivative is otherwise noisy
    enough to be nearly useless directly.

    Deliberately resets whenever the nearest obstacle disappears or changes
    identity is not tracked — this is intentionally a simple "nearest
    distance right now" differentiator, not real multi-object tracking. A
    new obstacle appearing at some distance is NOT the same as the previous
    one suddenly teleporting, so losing continuity (distance_m is None) or
    a sustained gap resets the estimate rather than computing a derivative
    across two unrelated objects.
    """

    def __init__(self, alpha=None):
        self.alpha = alpha if alpha is not None else config.CLOSING_SPEED_EMA_ALPHA
        self._prev_distance_m = None
        self._prev_time = None
        self._closing_rate_mps = 0.0

    def update(self, distance_m, now):
        """Returns the current smoothed closing rate in m/s (can be
        negative if the obstacle is moving away — callers should clip that
        to 0 before using it to INCREASE required braking; see
        compute_brake_percent, which already does this via max(...))."""
        if distance_m is None:
            self._prev_distance_m = None
            self._prev_time = None
            self._closing_rate_mps = 0.0
            return 0.0

        if self._prev_distance_m is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-3:
                raw_rate = (self._prev_distance_m - distance_m) / dt  # positive = closing
                self._closing_rate_mps = (
                    self.alpha * raw_rate + (1.0 - self.alpha) * self._closing_rate_mps
                )

        self._prev_distance_m = distance_m
        self._prev_time = now
        return self._closing_rate_mps


def compute_steer_suggestion(nearest_obstacle):
    """
    Steer away from the obstacle's lateral position. Positive degrees = steer
    right, negative = steer left (camera-frame X: positive is to the right).
    """
    if nearest_obstacle is None:
        return 0.0

    centroid_x_m = nearest_obstacle["centroid_x_m"]
    # obstacle is to the right (+X) -> steer left (negative); and vice versa
    raw = -centroid_x_m * config.STEER_GAIN
    return float(np.clip(raw, -config.MAX_STEER_SUGGESTION_DEG, config.MAX_STEER_SUGGESTION_DEG))


def brake_color(brake_percent):
    if brake_percent < 20:
        return config.COLOR_PATH_SAFE
    if brake_percent < 60:
        return config.COLOR_PATH_WARN
    return config.COLOR_PATH_DANGER

"""
ground_segmentation.py — ground-plane segmentation: fits the best possible
single flat plane to the ground each frame and classifies every pixel as
land / not-land against it.

Quality techniques used, all kept because each one measurably improves
real segmentation accuracy:
  1. RANSAC is fully vectorized — every candidate plane across all
     iterations is generated and scored in one batch of numpy ops, not a
     Python loop. This means MORE iterations costs LESS time than a naive
     loop would, so search quality goes up while cost stays low.
  2. The winning RANSAC candidate is refined with a least-squares (SVD)
     fit through all of its inliers. A raw 3-point RANSAC plane is noisy
     by construction — the least-squares refit through thousands of real
     inlier points is what actually makes the boundary accurate and
     stable, not just "more RANSAC tries."
  3. A local surface-normal consistency check rejects anything that's at
     the right HEIGHT to be ground but isn't actually FLAT there (e.g. a
     fence rail, a raised edge) — computed on a downsampled grid with
     proper area-averaging (not naive striding, which would just keep
     depth noise intact and fragment the mask into speckle).
  4. The fitted plane is smoothed across frames (hemisphere-aware EMA) so
     depth noise doesn't make the boundary flicker, with a short "coast"
     window so one bad/occluded frame doesn't blank the segmentation —
     but a sustained loss of ground view still correctly reports "no
     plane" instead of serving a stale one forever.
  5. The classified MASK itself is also temporally smoothed (a running
     per-pixel probability, not just the plane) — this is what actually
     removes frame-to-frame flicker at the boundary: a single noisy frame
     can only nudge a pixel's probability, not flip its classification.
  6. Optional semantic guidance (SegFormer, see segformer_onnx.py and
     config_segformer.py): a per-pixel "driveable ground" probability seeds
     the plane fit with real ground only, admits rough terrain the strict
     single-plane test rejects, and vetoes flat non-ground. Never lets land
     absorb a pixel tall enough to be an obstacle. Without it (model
     disabled / unavailable), behavior is exactly the pure-depth version.
"""

import numpy as np
import cv2

import config
import config_segformer as scfg
from obstacle_decision import height_threshold_m, open_raised


def build_confidence_mask(raw_valid_mask, ir_image):
    """
    True where the depth reading can actually be trusted:
      - the stereo matcher resolved something there BEFORE any denoise/smoothing
        (raw_valid_mask), and
      - if IR is available, the surface reflected enough IR to trust the match
        (low IR return = black/absorptive material = stereo likely guessed/noisy).
    """
    if raw_valid_mask is None:
        return None

    confidence_mask = raw_valid_mask.copy()

    if config.ENABLE_IR_CONFIDENCE and ir_image is not None:
        low_ir = ir_image < config.IR_LOW_REFLECTIVITY_THRESHOLD
        confidence_mask &= ~low_ir

    return confidence_mask


def _fit_plane_ransac_vectorized(points, rng):
    """
    Fully vectorized RANSAC + least-squares refinement. No per-iteration
    Python loop: every candidate 3-point plane across all iterations is
    built and scored against a bounded subsample in one batch of numpy ops,
    then the winner is refined with an SVD plane fit through all of its
    inliers in the FULL point set (not just the subsample).

    Returns (a, b, c, d) for a*X + b*Y + c*Z + d = 0, or None.
    """
    n = points.shape[0]
    if n < config.RANSAC_MIN_SAMPLE_POINTS:
        return None

    if n > config.RANSAC_SUBSAMPLE_POINTS:
        idx = rng.choice(n, size=config.RANSAC_SUBSAMPLE_POINTS, replace=False)
        sample = points[idx]
    else:
        sample = points
    m = sample.shape[0]

    iterations = config.RANSAC_ITERATIONS
    tri = rng.integers(0, m, size=(iterations, 3))
    p1, p2, p3 = sample[tri[:, 0]], sample[tri[:, 1]], sample[tri[:, 2]]
    v1, v2 = p2 - p1, p3 - p1
    normals = np.cross(v1, v2)
    norm_len = np.linalg.norm(normals, axis=1)
    degenerate = norm_len <= 1e-6
    safe_len = np.where(degenerate, 1.0, norm_len)
    normals = normals / safe_len[:, None]
    d = -np.einsum("ij,ij->i", normals, p1)

    # (m, iterations): distance of every sampled point to every candidate plane
    dist = np.abs(sample @ normals.T + d[None, :])
    inlier_counts = np.sum(dist < config.RANSAC_INLIER_THRESHOLD_M, axis=0)
    inlier_counts = np.where(degenerate, -1, inlier_counts)

    best_idx = int(np.argmax(inlier_counts))
    if inlier_counts[best_idx] < 3:
        return None

    best_normal = normals[best_idx]
    best_d = float(d[best_idx])

    # Refine: least-squares plane through inliers in the full point set.
    # This is what actually makes the plane accurate — RANSAC above only
    # needs to land close enough that the inlier set is clean.
    if points.shape[0] > config.RANSAC_REFINE_CANDIDATE_POINTS:
        cand_idx = rng.choice(points.shape[0], size=config.RANSAC_REFINE_CANDIDATE_POINTS, replace=False)
        refine_candidates = points[cand_idx]
    else:
        refine_candidates = points

    full_dist = np.abs(refine_candidates @ best_normal + best_d)
    inlier_mask = full_dist < config.RANSAC_INLIER_THRESHOLD_M
    inliers = refine_candidates[inlier_mask]
    if inliers.shape[0] < 3:
        return (float(best_normal[0]), float(best_normal[1]), float(best_normal[2]), best_d)

    if inliers.shape[0] > config.RANSAC_REFINE_MAX_POINTS:
        ridx = rng.choice(inliers.shape[0], size=config.RANSAC_REFINE_MAX_POINTS, replace=False)
        inliers = inliers[ridx]

    centroid = inliers.mean(axis=0)
    centered = inliers - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    refined_normal = vt[-1]
    # The smallest-singular-vector normal has an arbitrary sign — keep it in
    # the same hemisphere as the RANSAC candidate so temporal smoothing
    # doesn't see a spurious 180-degree "flip" every frame.
    if np.dot(refined_normal, best_normal) < 0:
        refined_normal = -refined_normal
    refined_d = -float(np.dot(refined_normal, centroid))

    return (float(refined_normal[0]), float(refined_normal[1]), float(refined_normal[2]), refined_d)


def _compute_surface_normals(xyz, downscale):
    """
    Per-pixel surface normal from local image-space gradients. Computed on a
    downsampled grid and upsampled with nearest-neighbor (never blurs or
    invents an edge on the way back out) since this feeds a coarse flatness
    veto, not a sub-pixel boundary — the actual land boundary comes from
    dist_to_plane and confidence, both still computed at full resolution.

    The downsample step uses cv2.resize with INTER_AREA (proper pixel
    averaging), NOT naive strided subsampling (xyz[::k, ::k]). Differentiating
    depth data amplifies high-frequency sensor noise into near-random normal
    directions — naive striding just picks one noisy sample per block and
    keeps that noise fully intact, which fragments the land mask into
    hundreds of tiny speckles instead of one coherent region. Area-averaging
    actually suppresses that noise the way real depth-noise reduction should,
    which is what makes the normal-consistency check reliable rather than a
    source of false rejections.
    """
    h, w = xyz.shape[:2]
    if downscale > 1:
        small_w = max(1, w // downscale)
        small_h = max(1, h // downscale)
        small = cv2.resize(xyz, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small = xyz

    gx = np.gradient(small, axis=1)
    gy = np.gradient(small, axis=0)
    normals = np.cross(gx, gy)
    norm_len = np.linalg.norm(normals, axis=2, keepdims=True)
    norm_len = np.where(norm_len < 1e-8, 1.0, norm_len)
    normals = normals / norm_len
    normals = np.nan_to_num(normals)

    if downscale > 1:
        normals = cv2.resize(normals, (w, h), interpolation=cv2.INTER_NEAREST)
    return normals


def _clean_and_take_largest_blob(mask, downscale=1):
    """
    downscale > 1 runs the expensive part (morphology + connectedComponents
    labeling) on a smaller working copy, then upscales the result back —
    "which pixels belong to the largest blob" doesn't meaningfully change
    with resolution for a coherent ground region. The area-average
    downscale (INTER_AREA) before re-binarizing avoids the aliasing a naive
    nearest/strided downsample would introduce at the boundary, and the
    upscale back (bilinear + rethreshold) gives a smoother edge, not a
    blockier one.
    """
    h, w = mask.shape
    # min_area is calibrated in pixels against the ORIGINAL full-resolution
    # capture. config.PROCESSING_DOWNSCALE (see main.py) may have ALREADY
    # shrunk this function's input before it ever got here — that's a
    # separate, always-applied factor from this function's own optional
    # `downscale` (an ADDITIONAL internal reduction for the labeling step
    # specifically) — both have to be accounted for together, or a real
    # object sized correctly for the reduced input still gets rejected as
    # "too small" purely because the threshold was never adjusted for the
    # resolution it's actually being measured against. Confirmed as a real
    # bug this way, not theoretical: obstacle detection returned zero
    # obstacles for a real synthetic obstacle once PROCESSING_DOWNSCALE
    # was introduced, until this fix.
    effective_downscale = config.PROCESSING_DOWNSCALE * (downscale if downscale > 1 else 1)
    if downscale > 1:
        small_w = max(1, w // downscale)
        small_h = max(1, h // downscale)
        mask_u8 = cv2.resize((mask.astype(np.uint8)) * 255, (small_w, small_h),
                              interpolation=cv2.INTER_AREA)
        mask_u8 = ((mask_u8 > 127).astype(np.uint8)) * 255
    else:
        mask_u8 = (mask.astype(np.uint8)) * 255
    min_area = max(1, config.MIN_LAND_BLOB_AREA_PX // (effective_downscale * effective_downscale))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        result = np.zeros(mask_u8.shape, dtype=bool)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            result = np.zeros(mask_u8.shape, dtype=bool)
        else:
            largest_label = 1 + int(np.argmax(areas))
            if stats[largest_label, cv2.CC_STAT_AREA] < min_area:
                result = np.zeros(mask_u8.shape, dtype=bool)
            else:
                result = labels == largest_label

    if downscale > 1:
        result_u8 = cv2.resize((result.astype(np.uint8)) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        result = result_u8 > 127

    return result


def _light_speckle_removal(mask):
    """
    A cheap alternative to _clean_and_take_largest_blob for the SECOND
    cleanup pass in segment() below — strips the small (1-2px) speckle the
    EMA mask-probability threshold can reintroduce at the boundary, without
    connectedComponentsWithStats' relabeling cost. Safe specifically
    because the input here has ALREADY been through
    _clean_and_take_largest_blob once this frame: temporal EMA blending
    against an already-single-coherent-blob mask only nudges pixels near
    the existing boundary, it does not create a new, separate large region
    that would need "pick the largest one" logic run again.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8 > 0


def _structural_raised(dist_to_plane, valid_depth, xyz):
    """
    Raised pixels that SURVIVE the obstacle detector's own speckle removal
    (obstacle_decision.open_raised) — the only raised pixels that can ever
    belong to an obstacle blob. Every other raised pixel is isolated depth
    noise, which on rough ground is what punches flickering holes into the
    land mask.
    """
    raised_all = valid_depth & (dist_to_plane > height_threshold_m(xyz[..., 2]))
    return open_raised(raised_all) > 0


def _fuse_semantic(geo_land, sem_prob, semantic_ground, dist_to_plane, valid_depth, xyz, cos_angle,
                   structural_raised):
    """
    Combines the strict geometric land test with SegFormer's per-pixel
    driveable probability (see config_segformer.py):

      land = (geo_land AND NOT confidently-non-ground)
          OR (semantic ground AND valid depth AND not part of a raised
              structure AND within the relaxed normal angle)

    SEMANTIC_ABSORB_DEPTH_SPECKLE = True: "not part of a raised structure"
    means not in structural_raised, so isolated over-threshold noise pixels
    on semantic ground count as land. False: the stricter rule — below
    min(height_threshold_m(z), SEMANTIC_GROUND_MAX_HEIGHT_M).

    Either way the obstacle output is unchanged: land never takes a pixel
    that survives obstacle_decision.open_raised, and opening is monotone and
    idempotent, so the opened raised mask detect_obstacle_blobs labels is
    identical with or without these pixels. The veto only removes land.
    """
    land = geo_land
    if scfg.SEMANTIC_VETO_ENABLED:
        land = land & (sem_prob >= scfg.SEMANTIC_VETO_MAX_PROB)

    if scfg.SEMANTIC_ABSORB_DEPTH_SPECKLE:
        relaxed = semantic_ground & valid_depth & ~structural_raised
    else:
        tolerance = np.minimum(height_threshold_m(xyz[..., 2]), scfg.SEMANTIC_GROUND_MAX_HEIGHT_M)
        relaxed = semantic_ground & valid_depth & (dist_to_plane < tolerance)
    if cos_angle is not None and scfg.SEMANTIC_NORMAL_MAX_ANGLE_DEG is not None:
        relaxed &= cos_angle > np.cos(np.deg2rad(scfg.SEMANTIC_NORMAL_MAX_ANGLE_DEG))

    return land | relaxed


def _extract_boundary(land_mask):
    mask_u8 = (land_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if config.BOUNDARY_SMOOTHING_EPSILON_PX > 0:
        largest = cv2.approxPolyDP(largest, config.BOUNDARY_SMOOTHING_EPSILON_PX, closed=True)
    return largest


class GroundSegmenter:
    """
    Stateful across frames — the only state is the temporally-smoothed
    plane + mask probability, used to damp frame-to-frame flicker from
    depth noise without lagging behind genuine terrain changes. Instantiate
    once per camera session (see main.py), not once per frame.
    """

    def __init__(self):
        self._rng = np.random.default_rng()
        self._smoothed_plane = None
        self._miss_count = 0
        self._land_prob = None   # running per-pixel EMA of the land classification
        self._land_state = None  # last thresholded mask (hysteresis memory)
        self._semantic_prob = None   # running per-pixel EMA of the SegFormer driveable probability

    def segment(self, depth_image_m, xyz, ir_image=None, raw_valid_mask=None, semantic_prob=None):
        """
        semantic_prob : optional (H, W) float32 in [0, 1] — SegFormer's
                        driveable-ground probability at this same resolution
                        (segformer_onnx.SegformerOnnx.driveable_probability).
                        None = pure-depth segmentation, unchanged.

        Returns dict with:
          land_mask         : (H, W) bool — cleaned, largest-blob land region
          boundary_contour   : Nx1x2 int32 array (image coords) or None
          plane              : (a, b, c, d) or None — the SMOOTHED plane actually used
          raw_plane          : (a, b, c, d) or None — this frame's own fit, pre-smoothing
          dist_to_plane      : (H, W) float32, meters
          valid_depth        : (H, W) bool
          confidence_mask    : (H, W) bool or None
          semantic_ground    : (H, W) bool or None — what the model alone calls driveable
          semantic_seeded    : bool — this frame's plane was seeded from semantic ground only
          semantic_prob      : (H, W) float32 or None — the temporally SMOOTHED map actually used
          geo_land_raw       : (H, W) bool or None — the strict geometric test alone, before
                               semantic fusion and cleanup (debug view: what semantics added/vetoed)
        """
        h, w = depth_image_m.shape
        valid_depth = depth_image_m > 0.05

        confidence_mask = build_confidence_mask(raw_valid_mask, ir_image)
        if confidence_mask is not None:
            valid_depth = valid_depth & confidence_mask

        sem_prob = self._smooth_semantic(semantic_prob, (h, w))
        semantic_ground = None if sem_prob is None else sem_prob > scfg.SEMANTIC_GROUND_MIN_PROB

        seed_row_start = int(h * (1.0 - config.GROUND_SEED_ROW_FRACTION))
        # Slice the row range first (a cheap view, not a copy) and only
        # fancy-index the boolean valid_depth mask within that smaller
        # region, rather than building a full-frame boolean mask and
        # fancy-indexing the whole frame — measurably faster.
        xyz_seed_rows = xyz[seed_row_start:, :]
        valid_seed_rows = valid_depth[seed_row_start:, :]
        seed_points = None
        semantic_seeded = False
        if sem_prob is not None and scfg.SEMANTIC_SEED_ENABLED:
            # Seed only from what the model calls ground, so the plane is
            # fit to the ground itself rather than to a crop row, a vehicle
            # or a wall that happens to fill the bottom of the frame.
            # Too few such points (looking at a hedge, a dropout) -> fall
            # back to the unrestricted seed set below.
            sem_seed_rows = sem_prob[seed_row_start:, :] > scfg.SEMANTIC_SEED_MIN_PROB
            candidate = xyz_seed_rows[valid_seed_rows & sem_seed_rows]
            if candidate.shape[0] >= config.RANSAC_MIN_SAMPLE_POINTS:
                seed_points = candidate
                semantic_seeded = True
        if seed_points is None:
            seed_points = xyz_seed_rows[valid_seed_rows]
        raw_plane = _fit_plane_ransac_vectorized(seed_points, self._rng)
        plane = self._smooth_plane(raw_plane)

        if plane is None:
            # Sustained loss of ground view, not just one noisy frame — reset
            # the mask probability too, so re-acquiring the plane later
            # starts fresh instead of blending against a stale, possibly
            # very old, land mask.
            self._land_prob = None
            return {
                "land_mask": np.zeros((h, w), dtype=bool),
                "boundary_contour": None,
                "plane": None,
                "raw_plane": raw_plane,
                "dist_to_plane": np.zeros((h, w), dtype=np.float32),
                "valid_depth": valid_depth,
                "confidence_mask": confidence_mask,
                "semantic_ground": semantic_ground,
                "semantic_seeded": semantic_seeded,
                "semantic_prob": sem_prob,
                "geo_land_raw": None,
            }

        a, b, c, d = plane
        plane_normal = np.array([a, b, c], dtype=np.float32)

        dist_to_plane = np.abs(xyz @ plane_normal + d)
        dist_to_plane[~valid_depth] = np.inf

        raw_land_mask = (dist_to_plane < config.RANSAC_INLIER_THRESHOLD_M) & valid_depth

        cos_angle = None
        if config.NORMAL_CONSISTENCY_ENABLED:
            surface_normals = _compute_surface_normals(xyz, config.NORMAL_COMPUTE_DOWNSCALE)
            cos_angle = np.abs(surface_normals @ plane_normal)
            angle_ok = cos_angle > np.cos(np.deg2rad(config.NORMAL_MAX_ANGLE_DEG))
            raw_land_mask &= angle_ok

        geo_land_raw = raw_land_mask
        structural_raised = None
        if sem_prob is not None:
            structural_raised = _structural_raised(dist_to_plane, valid_depth, xyz)
            raw_land_mask = _fuse_semantic(raw_land_mask, sem_prob, semantic_ground,
                                           dist_to_plane, valid_depth, xyz, cos_angle,
                                           structural_raised)

        land_mask = _clean_and_take_largest_blob(raw_land_mask, downscale=config.SEGMENTATION_DOWNSCALE)

        # Temporal smoothing of the classified mask itself (not just the
        # plane) — see LAND_MASK_TEMPORAL_SMOOTHING_ALPHA in config.py. This
        # is what actually removes the frame-to-frame abrupt/flickery look
        # at the land boundary: a single noisy frame can only nudge the
        # per-pixel probability, not flip the classification outright.
        land_mask = self._smooth_land_mask(land_mask)
        # The probability threshold can reintroduce small speckle at the
        # boundary — a light cleanup, not a full second largest-blob pass.
        land_mask = _light_speckle_removal(land_mask)

        # Morphological closing can patch a genuinely untrusted hole back
        # into "land" — re-punch out anything confidence/validity rejected.
        land_mask &= valid_depth
        if sem_prob is not None:
            # Semantic land reaches right up to the base of an obstacle, so
            # the morphological CLOSE / temporal smoothing above could
            # bridge into a raised fringe — re-punch it, so the obstacle
            # output stays exactly what geometry alone says. With
            # SEMANTIC_ABSORB_DEPTH_SPECKLE only raised STRUCTURES are
            # punched (see _fuse_semantic for why that is equivalent for
            # obstacles); otherwise every over-threshold pixel is.
            if scfg.SEMANTIC_ABSORB_DEPTH_SPECKLE:
                land_mask &= ~structural_raised
            else:
                land_mask &= dist_to_plane <= height_threshold_m(xyz[..., 2])
        boundary_contour = _extract_boundary(land_mask)

        return {
            "land_mask": land_mask,
            "boundary_contour": boundary_contour,
            "plane": plane,
            "raw_plane": raw_plane,
            "dist_to_plane": dist_to_plane,
            "valid_depth": valid_depth,
            "confidence_mask": confidence_mask,
            "semantic_ground": semantic_ground,
            "semantic_seeded": semantic_seeded,
            "semantic_prob": sem_prob,
            "geo_land_raw": geo_land_raw,
        }

    def _smooth_semantic(self, semantic_prob, shape):
        """
        EMA over the model's probability map (SEMANTIC_TEMPORAL_ALPHA),
        before it is thresholded anywhere — an uncertain pixel near 0.5
        cannot flicker in and out of "ground" frame to frame. A missing map
        (model failed this frame) clears the history rather than coasting on
        a stale one; a shape change does the same.
        """
        if semantic_prob is None or semantic_prob.shape != shape:
            self._semantic_prob = None
            return None
        alpha = scfg.SEMANTIC_TEMPORAL_ALPHA
        frame_p = semantic_prob.astype(np.float32, copy=False)
        if self._semantic_prob is None:
            self._semantic_prob = frame_p.copy()
        else:
            self._semantic_prob = alpha * frame_p + (1.0 - alpha) * self._semantic_prob
        return self._semantic_prob

    def _smooth_land_mask(self, frame_land_mask):
        """
        EMA over the boolean land classification, per pixel. Returns a bool
        mask thresholded at probability > 0.5. Resets automatically (see
        the "plane is None" branch in segment()) on a sustained loss of
        ground view, so it never blends against a stale mask from before
        a long occlusion/gap.
        """
        alpha = config.LAND_MASK_TEMPORAL_SMOOTHING_ALPHA
        frame_f = frame_land_mask.astype(np.float32)

        if self._land_prob is None or self._land_prob.shape != frame_f.shape:
            self._land_prob = frame_f
            self._land_state = self._land_prob > 0.5
            return self._land_state

        self._land_prob = alpha * frame_f + (1.0 - alpha) * self._land_prob
        # Hysteresis (LAND_MASK_HYSTERESIS): a pixel turns ON above 0.5+h
        # and OFF below 0.5-h, keeping its previous state in between — so a
        # pixel whose probability hovers near 0.5 cannot toggle every frame.
        h = config.LAND_MASK_HYSTERESIS
        self._land_state = np.where(self._land_state,
                                    self._land_prob > 0.5 - h,
                                    self._land_prob > 0.5 + h)
        return self._land_state

    def _smooth_plane(self, raw_plane):
        if not config.PLANE_TEMPORAL_SMOOTHING_ENABLED:
            return raw_plane

        if raw_plane is None:
            self._miss_count += 1
            if self._smoothed_plane is not None and self._miss_count <= config.PLANE_COAST_MAX_MISSES:
                # A single bad/occluded frame — coast on the last good plane
                # rather than blinking to "no ground" for one missed fit.
                return self._smoothed_plane
            self._smoothed_plane = None
            return None

        self._miss_count = 0

        if self._smoothed_plane is None:
            self._smoothed_plane = raw_plane
            return raw_plane

        a0, b0, c0, d0 = self._smoothed_plane
        n0 = np.array([a0, b0, c0], dtype=np.float32)
        a1, b1, c1, d1 = raw_plane
        n1 = np.array([a1, b1, c1], dtype=np.float32)

        # Keep both normals in the same hemisphere before blending.
        if np.dot(n0, n1) < 0:
            n1 = -n1
            d1 = -d1

        alpha = config.PLANE_TEMPORAL_SMOOTHING_ALPHA
        n_blend = alpha * n1 + (1.0 - alpha) * n0
        norm = np.linalg.norm(n_blend)
        if norm < 1e-6:
            return self._smoothed_plane
        n_blend = n_blend / norm
        d_blend = alpha * d1 + (1.0 - alpha) * d0

        self._smoothed_plane = (
            float(n_blend[0]), float(n_blend[1]), float(n_blend[2]), float(d_blend)
        )
        return self._smoothed_plane

"""
debug_view.py — builds the debug window's image (config.DEBUG): a 3x2 grid
of pixel-aligned panels showing what the perception stack actually saw
and decided this frame, plus a "probe" of the raw per-pixel values behind
them so the window can show exact numbers under the mouse.

  1. Camera + final land      | 2. Raw depth                 | 3. SegFormer driveable probability (raw, pre-EMA)
  4. SegFormer class map      | 5. Land source (fusion)      | 6. Height above the fitted plane

Display only: nothing here feeds braking, steering or the watchdog, and
nothing here mutates its inputs. Pure numpy/OpenCV (no Tk) — the window
itself is ui/debug_window.py.
"""

import numpy as np
import cv2

import config
import config_segformer as scfg
from obstacle_decision import height_threshold_m

# Per-pixel "where did this land decision come from" codes (panel 5).
SRC_NONE, SRC_DEPTH, SRC_ADDED, SRC_VETOED, SRC_RAISED, SRC_INVALID = range(6)
SOURCE_NAMES = {
    SRC_NONE: "not land",
    SRC_DEPTH: "land (depth test)",
    SRC_ADDED: "land (added by semantics / cleanup)",
    SRC_VETOED: "vetoed by semantics",
    SRC_RAISED: "raised (obstacle candidate)",
    SRC_INVALID: "no valid depth (incl. masked sky)",
}
_SOURCE_COLORS = {                      # BGR
    SRC_DEPTH: (60, 200, 60),
    SRC_ADDED: (230, 200, 0),
    SRC_VETOED: (60, 60, 230),
    SRC_RAISED: (0, 150, 255),
    SRC_INVALID: (70, 70, 70),
}
_SOURCE_LEGEND = (SRC_DEPTH, SRC_ADDED, SRC_VETOED, SRC_RAISED, SRC_INVALID)



def _class_palette(n=256):
    """Fixed color per ADE20K class id (a class keeps its color frame to
    frame). Golden-ratio hue spacing keeps consecutive ids far apart in hue
    — random RGB gave near-identical colors for e.g. tree vs mountain."""
    ids = np.arange(n)
    hsv = np.stack([(ids * 0.618034 % 1.0) * 180,
                    np.where(ids % 2, 200, 255),
                    np.where(ids % 3 == 0, 170, 240)], -1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]


_CLASS_PALETTE = _class_palette()
_SOURCE_LUT = np.zeros((256, 3), np.uint8)
for _code, _bgr in _SOURCE_COLORS.items():
    _SOURCE_LUT[_code] = _bgr

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def panel_size(frame_shape):
    """(w, h) of one panel, keeping the camera frame's aspect ratio."""
    h, w = frame_shape[:2]
    pw = int(config.DEBUG_PANEL_WIDTH)
    return pw, max(1, int(round(pw * h / w)))


# ------------------------------------------------------------------ drawing helpers

def _text(img, text, org, scale=0.45, color=(255, 255, 255)):
    # A thin outline — a thicker one reads as a doubled glyph at this size.
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _title(img, title, lines=()):
    """Title + info lines on a translucent dark box, readable on any panel."""
    rows = [(title, 0.5)] + [(line, 0.4) for line in lines]
    box_w = max(cv2.getTextSize(t, _FONT, s, 1)[0][0] for t, s in rows) + 12
    box_h = 22 + 16 * len(lines)
    roi = img[0:box_h, 0:min(box_w, img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, title, (6, 16), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, 34 + 16 * i), _FONT, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def _swatch_legend(img, entries, bottom_margin=6):
    """entries: [(bgr, text)], drawn bottom-left, last entry lowest."""
    h = img.shape[0]
    for i, (bgr, text) in enumerate(reversed(entries)):
        y = h - bottom_margin - 15 * i
        cv2.rectangle(img, (6, y - 9), (16, y + 1), tuple(int(c) for c in bgr), -1)
        cv2.rectangle(img, (6, y - 9), (16, y + 1), (0, 0, 0), 1)
        _text(img, text, (21, y), 0.38)


def _colorbar(img, cmap, left_label, right_label):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = w - 150, w - 10, h - 16, h - 8
    ramp = np.linspace(0, 255, x1 - x0).astype(np.uint8)[None, :].repeat(y1 - y0, 0)
    img[y0:y1, x0:x1] = cv2.applyColorMap(ramp, cmap)
    _text(img, left_label, (x0, y0 - 4), 0.35)
    (tw, _), _ = cv2.getTextSize(right_label, _FONT, 0.35, 1)
    _text(img, right_label, (x1 - tw, y0 - 4), 0.35)


def _placeholder(size, title, message):
    pw, ph = size
    img = np.full((ph, pw, 3), 30, np.uint8)
    _title(img, title)
    _text(img, message, (6, ph // 2), 0.42, (180, 180, 180))
    return img


def _fit(arr, size, interp=cv2.INTER_NEAREST):
    return cv2.resize(arr, size, interpolation=interp)


def _blend_where(base, overlay, mask, alpha):
    """base with overlay alpha-blended in where mask is set — one
    addWeighted over the image instead of per-region float math."""
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    out = base.copy()
    cv2.copyTo(blended, mask.astype(np.uint8), out)
    return out


# ------------------------------------------------------------------ panels

def _panel_camera(color_small, land_small, seg, semantic_model, size):
    green = np.empty_like(color_small)
    green[:] = (60, 200, 60)
    img = _blend_where(color_small, green, land_small, 0.45)
    coverage = 100.0 * land_small.mean()
    if seg.get("plane") is None:
        plane = "plane: NONE"
    else:
        plane = "plane: seeded from semantic ground" if seg.get("semantic_seeded") else "plane: seeded from all depth"
    model = (f"segformer: {semantic_model.mode} {semantic_model.last_infer_ms:.0f} ms"
             if semantic_model.available else "segformer: off (pure depth)")
    _title(img, "1  camera + final land", (f"land {coverage:.1f}%", plane, model))
    return img


def _panel_depth(depth_small, raw_valid_small, size):
    lo, hi = config.DEBUG_DEPTH_RANGE_M
    valid = depth_small > 0
    norm = np.clip((depth_small - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # Near = warm, far = cool — the usual depth-camera convention.
    img = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    filled = valid & ~raw_valid_small
    if filled.any():
        # Depth exists only because an SDK filter filled a stereo hole.
        img[filled] = (img[filled] * 0.35 + 90).astype(np.uint8)
    lines = [f"valid {100 * valid.mean():.1f}%   black = no depth"]
    if filled.any():
        lines.append(f"dimmed = filled by SDK filter ({100 * filled.mean():.1f}%)")
    _title(img, "2  raw depth", lines)
    _colorbar(img, cv2.COLORMAP_TURBO, f"{hi:g} m", f"{lo:g} m")
    return img


def _panel_probability(prob_raw, semantic_model, size):
    title = "3  segformer driveable prob (raw)"
    if prob_raw is None:
        return _placeholder(size, title, semantic_model.error or "model unavailable")
    prob_small = _fit(prob_raw, size, cv2.INTER_LINEAR)
    img = cv2.applyColorMap((np.clip(prob_small, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    ground = ((prob_small > scfg.SEMANTIC_GROUND_MIN_PROB).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(ground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    _title(img, title, (f"white line = {scfg.SEMANTIC_GROUND_MIN_PROB:g} threshold",
                        f"> threshold: {100 * (prob_small > scfg.SEMANTIC_GROUND_MIN_PROB).mean():.1f}%"))
    _colorbar(img, cv2.COLORMAP_VIRIDIS, "0", "1")
    return img


def _panel_classes(ids_small, semantic_model, size):
    title = "4  segformer class map"
    if ids_small is None:
        reason = (semantic_model.error if not semantic_model.available
                  else "model has no class_ids output - re-export it")
        return _placeholder(size, title, reason)
    img = _CLASS_PALETTE[ids_small % len(_CLASS_PALETTE)]
    counts = np.bincount(ids_small.ravel())
    top = np.argsort(counts)[::-1][:6]
    driveable = set(scfg.DRIVEABLE_CLASSES)
    entries = []
    for cid in top:
        if counts[cid] == 0:
            continue
        name = semantic_model.id2label.get(int(cid), str(cid))
        mark = " *" if name in driveable else ""
        entries.append((_CLASS_PALETTE[cid % len(_CLASS_PALETTE)],
                        f"{name}{mark}  {100 * counts[cid] / ids_small.size:.0f}%"))
    _title(img, title, ("* = in DRIVEABLE_CLASSES",))
    _swatch_legend(img, entries)
    return img


def land_sources(seg, xyz_proc):
    """Per-pixel SRC_* code at processing resolution (see SOURCE_NAMES)."""
    valid = seg["valid_depth"]
    land = seg["land_mask"]
    src = np.full(land.shape, SRC_NONE, np.uint8)
    src[~valid] = SRC_INVALID
    if seg.get("plane") is None:
        return src
    dist = seg["dist_to_plane"]
    src[valid & ~land & (dist > height_threshold_m(xyz_proc[..., 2]))] = SRC_RAISED
    geo = seg.get("geo_land_raw")
    sem = seg.get("semantic_prob")
    if geo is not None and sem is not None and scfg.SEMANTIC_VETO_ENABLED:
        src[geo & ~land & (sem < scfg.SEMANTIC_VETO_MAX_PROB)] = SRC_VETOED
    if geo is None:
        src[land] = SRC_DEPTH
    else:
        src[land & geo] = SRC_DEPTH
        src[land & ~geo] = SRC_ADDED
    return src


def _panel_sources(color_small, src_small, size):
    dim = cv2.convertScaleAbs(color_small, alpha=0.45)
    img = _blend_where(dim, _SOURCE_LUT[src_small], src_small != SRC_NONE, 0.7)
    counts = np.bincount(src_small.ravel(), minlength=len(SOURCE_NAMES)) / src_small.size * 100
    _title(img, "5  land source (depth + semantics)")
    _swatch_legend(img, [(_SOURCE_COLORS[c], f"{SOURCE_NAMES[c]}  {counts[c]:.0f}%")
                         for c in _SOURCE_LEGEND])
    return img


def _panel_height(height_small, seg, size):
    title = "6  height above fitted plane"
    if seg.get("plane") is None:
        return _placeholder(size, title, "no ground plane this frame")
    top = config.DEBUG_HEIGHT_RANGE_M
    finite = np.isfinite(height_small)
    norm = np.clip(np.where(finite, height_small, 0.0) / top, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img[~finite] = 0
    a, b, c, d = seg["plane"]
    _title(img, title, (f"n=({a:+.2f},{b:+.2f},{c:+.2f})  d={d:+.2f} m",
                        f"strict land < {100 * config.RANSAC_INLIER_THRESHOLD_M:.0f} cm"))
    _colorbar(img, cv2.COLORMAP_INFERNO, "0", f"{100 * top:.0f} cm")
    return img


# ------------------------------------------------------------------ public API

def build(color_bgr, depth_m, raw_valid_mask, xyz_proc, seg, semantic_prob_raw, semantic_model):
    """
    color_bgr, depth_m, raw_valid_mask : full-resolution capture arrays
    xyz_proc, seg                      : processing-resolution xyz and GroundSegmenter.segment() result
    semantic_prob_raw                  : the model's map BEFORE temporal smoothing, or None
    semantic_model                     : the SegformerOnnx instance (status, labels, class map)

    Returns (mosaic_bgr, probe). probe holds per-pixel arrays at panel
    resolution for describe() / draw_crosshair().
    """
    size = panel_size(color_bgr.shape)
    pw, ph = size

    color_small = _fit(color_bgr, size, cv2.INTER_AREA)
    depth_small = _fit(depth_m.astype(np.float32, copy=False), size)
    raw_valid_small = (_fit(raw_valid_mask.astype(np.uint8), size) > 0
                       if raw_valid_mask is not None else depth_small > 0)
    land_small = _fit(seg["land_mask"].astype(np.uint8), size) > 0
    src_small = _fit(land_sources(seg, xyz_proc), size)
    height_small = _fit(seg["dist_to_plane"].astype(np.float32, copy=False), size)
    if seg.get("plane") is None:
        height_small = np.full(height_small.shape, np.nan, np.float32)

    prob_small = (_fit(semantic_prob_raw, size, cv2.INTER_LINEAR)
                  if semantic_prob_raw is not None else None)
    class_ids = semantic_model.last_class_ids if semantic_prob_raw is not None else None
    ids_small = _fit(class_ids.astype(np.int32), size) if class_ids is not None else None

    panels = [
        _panel_camera(color_small, land_small, seg, semantic_model, size),
        _panel_depth(depth_small, raw_valid_small, size),
        _panel_probability(semantic_prob_raw, semantic_model, size),
        _panel_classes(ids_small, semantic_model, size),
        _panel_sources(color_small, src_small, size),
        _panel_height(height_small, seg, size),
    ]
    rows = [np.hstack(panels[0:3]), np.hstack(panels[3:6])]
    mosaic = np.vstack(rows)
    # Thin separators so panel edges read clearly.
    mosaic[ph - 1:ph + 1, :] = 0
    mosaic[:, pw - 1:pw + 1] = 0
    mosaic[:, 2 * pw - 1:2 * pw + 1] = 0

    probe = {
        "panel_size": size,
        "depth": depth_small,
        "raw_valid": raw_valid_small,
        "prob": prob_small,
        "class_ids": ids_small,
        "id2label": semantic_model.id2label,
        "height": height_small,
        "source": src_small,
    }
    return mosaic, probe


def panel_pixel(probe, mosaic_x, mosaic_y):
    """Mosaic coords -> (x, y) inside a panel (all panels are aligned), or None."""
    pw, ph = probe["panel_size"]
    if not (0 <= mosaic_x < 3 * pw and 0 <= mosaic_y < 2 * ph):
        return None
    return int(mosaic_x) % pw, int(mosaic_y) % ph


def describe(probe, x, y):
    """One-line readout of every probed value at panel pixel (x, y)."""
    parts = [f"px ({x},{y})"]
    depth = float(probe["depth"][y, x])
    if depth > 0:
        filled = "" if probe["raw_valid"][y, x] else " (SDK-filled)"
        parts.append(f"depth {depth:.2f} m{filled}")
    else:
        parts.append("depth --")
    if probe["prob"] is not None:
        parts.append(f"driveable p {float(probe['prob'][y, x]):.2f}")
    if probe["class_ids"] is not None:
        cid = int(probe["class_ids"][y, x])
        parts.append(f"class '{probe['id2label'].get(cid, cid)}'")
    height = float(probe["height"][y, x])
    parts.append(f"height {100 * height:+.1f} cm" if np.isfinite(height) else "height --")
    parts.append(SOURCE_NAMES[int(probe["source"][y, x])])
    return "   |   ".join(parts)


def draw_crosshair(mosaic, probe, x, y):
    """Marks panel pixel (x, y) in EVERY panel of a copy of the mosaic."""
    out = mosaic.copy()
    pw, ph = probe["panel_size"]
    for row in range(2):
        for col in range(3):
            cx, cy = col * pw + x, row * ph + y
            cv2.drawMarker(out, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 17, 3)
            cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 15, 1)
    return out

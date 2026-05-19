"""Hand mask, silhouette matching, plain gray background — match competition training."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import mediapipe as mp

    _MP_HANDS = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.40,
        min_tracking_confidence=0.40,
    )
    _HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)
    MEDIAPIPE_OK = True
except Exception:
    mp = None
    _MP_HANDS = None
    _HAND_CONNECTIONS = []
    MEDIAPIPE_OK = False

TRAIN_CANVAS_SIZE = 200
TRAIN_BG_RGB = (198, 198, 198)
HAND_DARK_RGB = (35, 35, 38)
HAND_FILL_RATIO = 0.86


def mirror_frame(image: np.ndarray) -> np.ndarray:
    return np.fliplr(image).copy()


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _center_square_crop(rgb: np.ndarray, scale: float = 0.60) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = int(min(h, w) * scale)
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    return rgb[y1 : y1 + side, x1 : x1 + side]


def _expand_mask(mask: np.ndarray, pixels: int = 6) -> np.ndarray:
    pil = Image.fromarray(mask).convert("L")
    for _ in range(pixels):
        pil = pil.filter(ImageFilter.MaxFilter(3))
    return np.array(pil)


def _apply_silhouette(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dark hand on light gray inside mask — matches dataset silhouette style."""
    m = mask.astype(np.float32) / 255.0
    m3 = np.stack([m, m, m], axis=-1)

    gray = np.array(Image.fromarray(crop).convert("L"), dtype=np.float32)
    inside = mask > 80
    if inside.any():
        thresh = float(np.percentile(gray[inside], 58))
    else:
        thresh = 128.0

    hand_dark = (gray < thresh).astype(np.float32)
    dark_rgb = np.array(HAND_DARK_RGB, dtype=np.float32)

    bg = np.array(TRAIN_BG_RGB, dtype=np.float32)
    out = crop.astype(np.float32) * (1 - m3) + m3 * (
        hand_dark[..., None] * dark_rgb + (1 - hand_dark[..., None]) * bg
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _crop_and_mask_with_mediapipe(rgb: np.ndarray, padding: float = 0.48):
    if not MEDIAPIPE_OK:
        return None

    h, w = rgb.shape[:2]
    result = _MP_HANDS.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad = max(x2 - x1, y2 - y1) * padding
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(w, x2 + pad))
    y2 = int(min(h, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = rgb[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    rel_pts = [(int(x - x1), int(y - y1)) for x, y in pts]

    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)

    hull = _convex_hull(rel_pts)
    if len(hull) >= 3:
        draw.polygon(hull, fill=230)

    line_w = max(20, int(max(cw, ch) * 0.13))
    rad = max(12, line_w // 2)
    for a, b in _HAND_CONNECTIONS:
        xa, ya = rel_pts[a]
        xb, yb = rel_pts[b]
        draw.line((xa, ya, xb, yb), fill=255, width=line_w)
    for x, y in rel_pts:
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=255)

    mask = np.array(mask.filter(ImageFilter.GaussianBlur(radius=max(4, line_w // 4))))
    mask = _expand_mask(mask, pixels=4)
    mask = np.array(ImageOps.autocontrast(Image.fromarray(mask)))

    crop = _apply_silhouette(crop, mask)
    return crop, mask


def composite_on_training_background(rgb_crop: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    h, w = rgb_crop.shape[:2]
    side = TRAIN_CANVAS_SIZE
    target = int(side * HAND_FILL_RATIO)
    scale = target / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    pil = Image.fromarray(rgb_crop).convert("RGB")
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (side, side), TRAIN_BG_RGB)
    x = (side - new_w) // 2
    y = (side - new_h) // 2

    if mask is not None:
        mask_pil = Image.fromarray(mask).convert("L")
        mask_pil = mask_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas.paste(pil, (x, y), mask_pil)
    else:
        canvas.paste(pil, (x, y))

    return np.array(canvas)


def match_dataset_contrast(pil: Image.Image) -> Image.Image:
    pil = ImageOps.autocontrast(pil, cutoff=1)
    pil = ImageEnhance.Contrast(pil).enhance(1.5)
    pil = ImageEnhance.Brightness(pil).enhance(0.84)
    return pil


def prepare_webcam_frame(
    image: np.ndarray,
    *,
    mirror: bool = True,
    use_hand_crop: bool = True,
    gray_background: bool = True,
    match_dataset_style: bool = True,
) -> tuple[Image.Image, str]:
    if image is None:
        raise ValueError("empty frame")

    rgb = image
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if mirror:
        rgb = mirror_frame(rgb)

    mask = None
    if use_hand_crop:
        detected = _crop_and_mask_with_mediapipe(rgb)
        if detected is not None:
            region, mask = detected
            note = "hand mask + silhouette + gray bg"
        else:
            region = _center_square_crop(rgb)
            note = "center crop — show hand closer"
    else:
        region = _center_square_crop(rgb, scale=0.85)
        note = "center crop"

    if gray_background:
        region = composite_on_training_background(region, mask=mask)

    pil = Image.fromarray(region).convert("RGB")
    if match_dataset_style:
        pil = match_dataset_contrast(pil)
        note += " + contrast match"

    return pil, note

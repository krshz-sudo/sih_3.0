"""
SatQuery AI — Annotation Renderer
====================================
Draws Gemini's bounding boxes and segmentation masks onto the original
image using Pillow.  No AI inference — purely rendering.

Coordinate convention (from Gemini):
    box_2d = [ymin, xmin, ymax, xmax]   normalised 0–1000
    mask   = [[x, y], [x, y], ...]      normalised 0–1000
"""

import logging
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette per category
# ---------------------------------------------------------------------------

CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "water":        (0, 120, 255),
    "vegetation":   (34, 197, 94),
    "forest":       (22, 163, 74),
    "built-up":     (249, 115, 22),
    "building":     (249, 115, 22),
    "urban":        (249, 115, 22),
    "road":         (234, 179, 8),
    "highway":      (234, 179, 8),
    "infrastructure":(234, 179, 8),
    "agricultural": (132, 204, 22),
    "crop":         (132, 204, 22),
    "industrial":   (239, 68, 68),
    "ship":         (168, 85, 247),
    "vehicle":      (168, 85, 247),
    "runway":       (236, 72, 153),
    "airport":      (236, 72, 153),
    "flood":        (56, 189, 248),
    "shadow":       (100, 116, 139),
    "unknown":      (148, 163, 184),
}

FALLBACK_COLORS = [
    (6, 182, 212),    # cyan
    (249, 115, 22),   # orange
    (168, 85, 247),   # purple
    (234, 179, 8),    # amber
    (239, 68, 68),    # red
    (34, 197, 94),    # green
    (236, 72, 153),   # pink
    (59, 130, 246),   # blue
]

BOX_LINE_WIDTH = 3
LABEL_FONT_SIZE = 13
LABEL_PADDING = 4
MASK_ALPHA = 60       # 0-255 transparency for mask overlay


def _get_color(category: str, index: int = 0) -> Tuple[int, int, int]:
    """Get colour for a feature category."""
    cat = category.lower().strip()
    for key, color in CATEGORY_COLORS.items():
        if key in cat:
            return color
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def _confidence_label(conf: float) -> str:
    """Human-readable confidence label."""
    if conf >= 0.90:
        return "Confirmed"
    elif conf >= 0.75:
        return "Likely"
    elif conf >= 0.50:
        return "Possible"
    elif conf >= 0.30:
        return "Uncertain"
    else:
        return "Weak"


def _load_font(size: int = LABEL_FONT_SIZE):
    """Try to load a decent font, fall back to default."""
    for path in [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_annotations(
    pil_image: Image.Image,
    features: List[Dict],
    *,
    show_boxes: bool = True,
    show_masks: bool = True,
    show_labels: bool = True,
    highlight_id: Optional[str] = None,
) -> Image.Image:
    """Draw bounding boxes and masks from Gemini features onto the image.

    Parameters
    ----------
    pil_image : PIL.Image
        The original image (will be copied, not mutated).
    features : list of dict
        Each feature dict should have: label, category, confidence,
        and optionally box_2d ([ymin, xmin, ymax, xmax] 0-1000)
        and mask ([[x,y], ...] 0-1000).
    show_boxes, show_masks, show_labels : bool
        Toggle which annotations to draw.
    highlight_id : str, optional
        If set, highlight this feature's annotation more prominently.

    Returns
    -------
    PIL.Image  — annotated copy of the image.
    """
    if not features:
        return pil_image.copy()

    img = pil_image.copy().convert("RGB")
    img_w, img_h = img.size
    font = _load_font()

    # Draw masks first (underneath boxes)
    if show_masks:
        for i, feat in enumerate(features):
            mask_pts = feat.get("mask")
            if not mask_pts or not isinstance(mask_pts, list) or len(mask_pts) < 3:
                continue

            color = _get_color(feat.get("category", ""), i)
            is_highlight = (highlight_id and feat.get("id") == highlight_id)

            # Convert normalised points to pixel coordinates
            pixel_pts = []
            for pt in mask_pts:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px = int(pt[0] / 1000 * img_w)
                    py = int(pt[1] / 1000 * img_h)
                    pixel_pts.append((px, py))

            if len(pixel_pts) >= 3:
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                alpha = MASK_ALPHA + 40 if is_highlight else MASK_ALPHA
                overlay_draw.polygon(pixel_pts, fill=color + (alpha,), outline=color + (200,))
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Draw bounding boxes
    draw = ImageDraw.Draw(img, "RGBA")

    for i, feat in enumerate(features):
        box = feat.get("box_2d")
        if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        if not show_boxes:
            continue

        # Convert [ymin, xmin, ymax, xmax] normalised 0-1000 to pixels
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin / 1000 * img_w)
        y1 = int(ymin / 1000 * img_h)
        x2 = int(xmax / 1000 * img_w)
        y2 = int(ymax / 1000 * img_h)

        # Clamp
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = _get_color(feat.get("category", ""), i)
        is_highlight = (highlight_id and feat.get("id") == highlight_id)
        line_w = BOX_LINE_WIDTH + 2 if is_highlight else BOX_LINE_WIDTH

        # Semi-transparent fill
        fill_alpha = 50 if not is_highlight else 80
        draw.rectangle([x1, y1, x2, y2], fill=color + (fill_alpha,), outline=color, width=line_w)

        # Label
        if show_labels:
            conf = feat.get("confidence", 0)
            label_text = f"{feat.get('label', 'feature')} {conf:.0%} {_confidence_label(conf)}"

            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Label background
            label_y = max(0, y1 - text_h - 2 * LABEL_PADDING)
            label_bg = [x1, label_y, x1 + text_w + 2 * LABEL_PADDING, label_y + text_h + 2 * LABEL_PADDING]
            draw.rectangle(label_bg, fill=color + (220,))
            draw.text(
                (x1 + LABEL_PADDING, label_y + LABEL_PADDING),
                label_text, fill=(255, 255, 255), font=font
            )

    return img

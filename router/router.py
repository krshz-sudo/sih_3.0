"""
SatQuery AI — Router (Pure Gemini)
====================================
Thin wrapper: takes user query + images → calls gemini_client once →
returns the structured analysis.

No heuristic routing, no VLM-based classification, no specialist tools.
One AI call per analysis:  Image + Query → Gemini → Result.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image modes (kept for UI compatibility)
# ---------------------------------------------------------------------------

IMAGE_MODE_SINGLE = "single"
IMAGE_MODE_BITEMPORAL = "bitemporal"
IMAGE_MODE_SAR_OPTICAL = "sar_optical"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(
    query: str,
    images: list,
    *,
    image_mode: str = IMAGE_MODE_SINGLE,
    timeout: int = 90,
    **kwargs,
) -> dict:
    """Route a user query to Gemini for analysis.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    images : list
        Uploaded image paths or PIL Images.
    image_mode : str
        "single", "bitemporal", or "sar_optical".
    timeout : int
        Seconds for Gemini call.

    Returns
    -------
    dict — full analysis result ready for the GUI.
    """
    from tools.gemini_client import analyze_image

    if not query or not query.strip():
        return {
            "status": "error",
            "error": "Please enter a question.",
            "analysis": None,
            "model_used": "",
            "elapsed_s": 0.0,
        }

    if not images:
        return {
            "status": "error",
            "error": "Please upload an image first.",
            "analysis": None,
            "model_used": "",
            "elapsed_s": 0.0,
        }

    # Resolve image paths to PIL objects
    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        elif isinstance(img, (str, Path)):
            p = Path(img)
            if p.suffix.lower() in {'.tif', '.tiff'}:
                from tools.geotiff_utils import load_geotiff
                pil_images.append(load_geotiff(str(p)).pil_image)
            else:
                pil_images.append(Image.open(p).convert("RGB"))
        else:
            pil_images.append(img)

    # Single Gemini call
    result = analyze_image(
        images=pil_images,
        query=query.strip(),
        image_mode=image_mode,
        timeout=timeout,
    )

    return result

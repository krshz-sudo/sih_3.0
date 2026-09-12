"""
SatQuery AI — Gemini Client (SOLE AI Engine)
==============================================
Pure Gemini multimodal inference.  No Ollama, no CNN, no fallback.

    User Image + Query  →  Gemini  →  Structured JSON  →  UI

Every AI-powered feature in SatQuery AI goes through this module.
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    _st_secrets = st.secrets
except ImportError:
    _st_secrets = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _st_secrets.get("GEMINI_API_KEY", "") or ("AQ." + "Ab8RN6LYpxN4PT2-rhWALU_dnLWok76yxS4zvPXjEpYbFsQVbg")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_IMAGE_DIM = 1024          # resize longest edge
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 16.0
TIMEOUT_S = 90

# ---------------------------------------------------------------------------
# Master System Prompt  (spec §24)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SatQuery AI, a careful remote-sensing and SAR image interpretation assistant.

Analyze ONLY information supported by the provided image and user question.

Treat the image as SAR/remote-sensing imagery unless reliable metadata says otherwise.

Separate direct observations from interpretation.

Never invent:
location, city, country, river names, landmarks, GPS coordinates, sensor,
acquisition date, polarization, resolution, weather, measurements,
or historical facts.

For every important feature, provide:
- label
- evidence (what you actually see)
- status: "observed" (directly visible), "inferred" (interpretation), or "uncertain"
- calibrated confidence (0.00–1.00)
- confidence_reason (why this confidence level)
- bounding box [ymin, xmin, ymax, xmax] with coordinates normalized 0–1000 ONLY when the region is visually localizable
- segmentation polygon [[x,y], ...] normalized 0–1000 when useful for large/irregular regions

Confidence calibration:
- 0.90–1.00 = very strong visual evidence, unambiguous
- 0.75–0.89 = strong evidence but minor ambiguity
- 0.50–0.74 = plausible interpretation, some uncertainty
- 0.30–0.49 = weak evidence
- 0.00–0.29 = highly uncertain

Do NOT generate a bounding box merely because a feature is mentioned in text.
Only return a bounding box when the corresponding region is visually supported.

Do not invent measurements or statistics.
When information cannot be determined from the image alone, explicitly state that.

Provide DETAILED analysis: multi-paragraph, section-structured responses.
For SAR imagery, discuss backscatter patterns, texture, geometric features,
bright/dark regions, speckle, shadows, and spatial relationships.

Use relative directions (upper-left, lower-right) instead of cardinal
directions unless image orientation is genuinely known.

Fluent language is not evidence. Prioritize factual visual evidence over plausible guesses.

The objective is reliable, evidence-based remote-sensing interpretation."""

# ---------------------------------------------------------------------------
# Structured response schema  (spec §5)
# ---------------------------------------------------------------------------

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "direct_answer": {"type": "string", "description": "Direct answer to the user's question"},
        "summary": {"type": "string", "description": "2-3 sentence executive summary"},
        "detailed_analysis": {"type": "string", "description": "Multi-paragraph detailed analysis with sections"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["text", "confidence"]
            }
        },
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "category": {"type": "string"},
                    "status": {"type": "string", "enum": ["observed", "inferred", "uncertain"]},
                    "confidence": {"type": "number"},
                    "confidence_reason": {"type": "string"},
                    "evidence": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[ymin, xmin, ymax, xmax] normalized 0-1000"
                    }
                },
                "required": ["id", "label", "category", "status", "confidence", "confidence_reason", "evidence"]
            }
        },
        "land_cover": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"}
                },
                "required": ["class_name", "confidence", "evidence"]
            }
        },
        "statistics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"}
                },
                "required": ["label", "value", "unit"]
            }
        },
        "chart_data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["bar", "pie"]},
                    "title": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "values": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["chart_type", "title", "labels", "values"]
            }
        },
        "csv_rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "category": {"type": "string"},
                    "status": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"}
                },
                "required": ["feature", "category", "status", "confidence", "evidence"]
            }
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "direct_answer", "summary", "detailed_analysis",
        "observations", "features", "land_cover",
        "statistics", "chart_data", "csv_rows", "limitations"
    ]
}

# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image(image: Union[str, Path, Image.Image, bytes]) -> bytes:
    """Return JPEG bytes of the image, resized if needed."""
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    w, h = pil_img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Connection verification
# ---------------------------------------------------------------------------

def verify_connection() -> Dict[str, Any]:
    """Check if Gemini API is reachable. Returns status dict."""
    if not GEMINI_API_KEY:
        return {"connected": False, "error": "GEMINI_API_KEY not set", "model": GEMINI_MODEL}

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # Quick test — list models
        models = genai.list_models()
        available = [m.name for m in models if "generateContent" in [s for s in getattr(m, 'supported_generation_methods', [])]]
        return {
            "connected": True,
            "error": None,
            "model": GEMINI_MODEL,
            "available_models": len(available),
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc), "model": GEMINI_MODEL}


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_image(
    images: List[Union[str, Path, Image.Image]],
    query: str,
    *,
    image_mode: str = "single",
    timeout: int = TIMEOUT_S,
) -> Dict[str, Any]:
    """Send image(s) + query to Gemini, return structured analysis.

    This is the ONE AND ONLY AI inference call in SatQuery AI.

    Parameters
    ----------
    images : list
        One or more PIL Images / file paths.
    query : str
        The user's natural-language question.
    image_mode : str
        "single", "bitemporal", or "sar_optical"
    timeout : int
        Seconds per attempt.

    Returns
    -------
    dict with keys:
        status, model_used, elapsed_s, error,
        analysis (the structured JSON from Gemini)
    """
    if not GEMINI_API_KEY:
        return _error("GEMINI_API_KEY is not configured. Set it in .env")

    if not images:
        return _error("No image provided")

    if not query or not query.strip():
        return _error("No question provided")

    try:
        import google.generativeai as genai
    except ImportError:
        return _error("google-generativeai package not installed. Run: pip install google-generativeai")

    genai.configure(api_key=GEMINI_API_KEY)

    # Build context based on image mode
    if image_mode == "bitemporal":
        context = (
            "You are analyzing a BI-TEMPORAL image pair (before/after of the same area). "
            "Image 1 is the BEFORE image, Image 2 is the AFTER image. "
            "Focus on identifying and describing changes between the two time periods. "
            "Provide bounding boxes around regions where changes are most evident."
        )
    elif image_mode == "sar_optical":
        context = (
            "You are analyzing an OPTICAL + SAR image pair of the same area. "
            "Image 1 is the OPTICAL image, Image 2 is the SAR (radar) image. "
            "Compare what is visible in both modalities. Discuss features unique to each."
        )
    else:
        context = (
            "You are analyzing a single satellite/aerial/SAR image. "
            "Treat it as remote-sensing imagery. Analyze backscatter, texture, "
            "geometry, and spatial patterns if it appears to be SAR."
        )

    user_prompt = (
        f"{context}\n\n"
        f"USER QUESTION: {query}\n\n"
        "Provide a thorough, evidence-based analysis. "
        "Return bounding boxes [ymin, xmin, ymax, xmax] normalized 0-1000 for every "
        "visually localizable feature. Only include a bounding box when the feature "
        "is genuinely visible in the image.\n\n"
        "Respond with a JSON object matching the required schema."
    )

    # Encode images as PIL objects for Gemini
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

    # Resize for Gemini
    resized = []
    for pil in pil_images:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        w, h = pil.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        resized.append(pil)

    # Build Gemini content parts
    parts = [f"[System Instructions]\n{SYSTEM_PROMPT}\n\n[User Query]\n{user_prompt}"]
    for pil_img in resized:
        parts.append(pil_img)

    # Try candidate models
    candidate_models = list(dict.fromkeys([
        GEMINI_MODEL,
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash",
    ]))

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        for mdl_name in candidate_models:
            t0 = time.perf_counter()
            try:
                model = genai.GenerativeModel(mdl_name)
                response = model.generate_content(
                    parts,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.15,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        response_schema=ANALYSIS_SCHEMA,
                    ),
                    request_options={"timeout": timeout},
                )

                elapsed = time.perf_counter() - t0

                if not response or not response.text:
                    raise RuntimeError(f"Empty response from {mdl_name}")

                # Parse JSON
                raw_text = response.text.strip()
                try:
                    analysis = json.loads(raw_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    if match:
                        analysis = json.loads(match.group())
                    else:
                        raise RuntimeError(f"Invalid JSON from Gemini: {raw_text[:200]}")

                logger.info(
                    "Gemini analysis OK: model=%s  elapsed=%.1fs  features=%d",
                    mdl_name, elapsed, len(analysis.get("features", []))
                )

                return {
                    "status": "ok",
                    "model_used": mdl_name,
                    "elapsed_s": round(elapsed, 2),
                    "error": None,
                    "analysis": analysis,
                    "raw_response": raw_text,
                }

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                last_exc = exc
                error_str = str(exc).lower()

                # Classify error
                if "404" in str(exc) or "not found" in error_str:
                    logger.warning("Model %s not found, trying next", mdl_name)
                    continue  # try next model
                elif "429" in str(exc) or "quota" in error_str or "rate" in error_str:
                    logger.warning("Rate limit on %s (attempt %d): %s", mdl_name, attempt, exc)
                    break  # backoff then retry
                elif "403" in str(exc) or "permission" in error_str or "api key" in error_str.lower():
                    return _error(f"Invalid Gemini API key. Check GEMINI_API_KEY in .env. ({exc})")
                else:
                    logger.warning("Gemini error: model=%s attempt=%d: %s", mdl_name, attempt, exc)
                    break  # backoff then retry

        # Backoff before next attempt
        if attempt < MAX_RETRIES:
            delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
            logger.info("Backing off %.1fs before retry %d", delay, attempt + 1)
            time.sleep(delay)

    # All retries exhausted
    error_msg = str(last_exc) if last_exc else "Unknown error"
    if "429" in error_msg or "quota" in error_msg.lower():
        return _error(f"Gemini rate limit reached. Please retry shortly. ({error_msg})")
    elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
        return _error(f"Gemini request timed out after {timeout}s. Try again. ({error_msg})")
    else:
        return _error(f"Gemini analysis failed after {MAX_RETRIES} attempts. ({error_msg})")


def _error(msg: str) -> Dict[str, Any]:
    """Standard error response."""
    return {
        "status": "error",
        "model_used": "",
        "elapsed_s": 0.0,
        "error": msg,
        "analysis": None,
        "raw_response": "",
    }

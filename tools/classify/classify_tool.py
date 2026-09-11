"""
SatQuery AI — Scene Classification Specialist Tool
=====================================================
Classifies a remote-sensing image into one of the EuroSAT land-cover
classes and returns the predicted label + confidence.

Two backends, tried in order:
  1. CNN — a ResNet18 fine-tuned on EuroSAT, loaded via timm from
     HuggingFace (hf_hub:cm93/resnet18-eurosat).  Tiny model (~44 MB),
     runs on CPU in <1 s, leaves VRAM free for VLM calls.
  2. VLM — if PyTorch/timm are missing or the model fails to load,
     falls back to prompting qwen2.5vl with the EuroSAT label set,
     using structured JSON output for consistent confidence reporting.

Both backends now report a unified `confidence_level` field
(high/medium/low) so the GUI can display the same badge format
regardless of which backend answered.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image

# Project imports
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.geotiff_utils import load_geotiff, LoadedImage

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EuroSAT class labels (canonical order used by most pretrained checkpoints)
# ---------------------------------------------------------------------------

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Human-readable versions for the final answer
EUROSAT_LABELS_PRETTY = {
    "AnnualCrop":            "Annual Crop",
    "Forest":                "Forest",
    "HerbaceousVegetation":  "Herbaceous Vegetation",
    "Highway":               "Highway",
    "Industrial":            "Industrial",
    "Pasture":               "Pasture",
    "PermanentCrop":         "Permanent Crop",
    "Residential":           "Residential",
    "River":                 "River",
    "SeaLake":               "Sea / Lake",
}

# ---------------------------------------------------------------------------
# Supported image formats (same as VQA tool)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _validate_image_path(path: Path) -> None:
    """Raise ValueError if the image format is not supported."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _softmax_to_confidence_level(prob: float) -> str:
    """Convert a softmax probability to a human-readable confidence level."""
    if prob >= 0.8:
        return "high"
    elif prob >= 0.5:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Backend 1: CNN classifier (timm + EuroSAT ResNet18)
# ---------------------------------------------------------------------------

# Lazy-loaded singleton — avoids reloading the model on every call
_cnn_model = None
_cnn_transform = None
_cnn_available = None  # None = not checked yet, True/False after first check


def _load_cnn():
    """Attempt to load the EuroSAT ResNet18 model via timm.

    Returns True if successful, False otherwise.
    Sets module-level _cnn_model and _cnn_transform.
    """
    global _cnn_model, _cnn_transform, _cnn_available

    if _cnn_available is not None:
        return _cnn_available

    try:
        import torch
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        logger.info("Loading EuroSAT ResNet18 via timm …")

        model = timm.create_model(
            "hf_hub:cm93/resnet18-eurosat",
            pretrained=True,
        )
        model.eval()

        # Build the preprocessing transform from the model's config
        config = resolve_data_config(model.pretrained_cfg)
        transform = create_transform(**config)

        _cnn_model = model
        _cnn_transform = transform
        _cnn_available = True

        logger.info(
            "EuroSAT ResNet18 loaded successfully (%.1f MB)",
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
        )
        return True

    except Exception as exc:
        logger.warning("CNN backend unavailable: %s", exc)
        _cnn_available = False
        return False


def _classify_cnn(pil_image: Image.Image) -> Tuple[str, float, list]:
    """Run CNN inference. Returns (class_name, confidence, all_probs)."""
    import torch

    inp = _cnn_transform(pil_image).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = _cnn_model(inp)  # (1, num_classes)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    confidence = probs[top_idx].item()

    # Build all-classes breakdown (for diagnostics / GUI display)
    all_probs = [
        {"class": EUROSAT_CLASSES[i], "probability": round(probs[i].item(), 4)}
        for i in range(len(EUROSAT_CLASSES))
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    class_name = EUROSAT_CLASSES[top_idx]
    return class_name, confidence, all_probs


# ---------------------------------------------------------------------------
# Backend 2: VLM-based classification (fallback) — with structured output
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = (
    "You are a remote sensing image analyst specializing in land-cover "
    "classification. You classify satellite images into exactly one of "
    "these EuroSAT categories: " + ", ".join(EUROSAT_CLASSES) + ".\n\n"
    "RULES:\n"
    "1. Choose the single most appropriate category.\n"
    "2. Assess your confidence honestly (high/medium/low).\n"
    "3. Explain briefly what visual features led to your classification.\n"
    "4. If the image is ambiguous, say so in your reasoning and set "
    "confidence to 'low'.\n"
    "5. You MUST respond with valid JSON matching the schema provided."
)

CLASSIFY_PROMPT_TEMPLATE = (
    "Classify this satellite image into exactly one land-cover category.\n\n"
    "Available categories: {classes}\n\n"
    "Respond with a JSON object containing:\n"
    "- \"class_name\": the category name (must be one of the listed categories)\n"
    "- \"confidence\": one of \"high\", \"medium\", or \"low\"\n"
    "- \"reasoning\": brief explanation of what visual features led to this classification"
)

CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "class_name": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    "required": ["class_name", "confidence", "reasoning"],
}


def _classify_vlm(
    pil_image: Image.Image,
    model: Optional[str],
    timeout: int,
    max_retries: int,
) -> Tuple[str, str, str, str, str, float]:
    """Classify via VLM with structured output.
    
    Returns (class_name, confidence_level, reasoning, raw_answer, model_used, elapsed).
    """
    from tools.ollama_client import query_vlm

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(
        classes=", ".join(EUROSAT_CLASSES)
    )

    result = query_vlm(
        prompt=prompt,
        image=pil_image,
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        format_schema=CLASSIFY_RESPONSE_SCHEMA,
    )

    raw = result["answer"]
    model_used = result["model_used"]
    model_tier = result.get("model_tier", "primary")
    elapsed = result["elapsed_s"]

    clean_raw = raw.strip()
    if clean_raw.startswith("```"):
        clean_raw = re.sub(r"^```(?:json)?\s*", "", clean_raw, flags=re.IGNORECASE)
        clean_raw = re.sub(r"\s*```$", "", clean_raw)

    # Try to parse structured response
    try:
        parsed = json.loads(clean_raw)
        vlm_class = parsed.get("class_name", "")
        confidence_level = parsed.get("confidence", "medium")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        vlm_class = raw
        confidence_level = "medium"
        reasoning = f"Classification response from VLM model ({model_used})."

    # Match to EuroSAT class
    matched_class = _match_vlm_output(vlm_class)

    return matched_class, confidence_level, reasoning, raw, model_used, elapsed, model_tier


def _match_vlm_output(text: str) -> str:
    """Best-effort match of free-form VLM text to an EuroSAT class name."""
    text_lower = text.strip().lower().replace("_", "").replace(" ", "")

    # Exact match (case-insensitive, ignoring spaces/underscores)
    for cls in EUROSAT_CLASSES:
        if cls.lower().replace(" ", "") == text_lower:
            return cls

    # Substring match — model might say "This is Forest" instead of "Forest"
    for cls in EUROSAT_CLASSES:
        if cls.lower() in text.lower():
            return cls

    # Keyword heuristics for common VLM paraphrases
    keyword_map = {
        "crop":        "AnnualCrop",
        "farm":        "AnnualCrop",
        "agricult":    "AnnualCrop",
        "forest":      "Forest",
        "tree":        "Forest",
        "wood":        "Forest",
        "herbaceous":  "HerbaceousVegetation",
        "grass":       "HerbaceousVegetation",
        "shrub":       "HerbaceousVegetation",
        "vegetation":  "HerbaceousVegetation",
        "highway":     "Highway",
        "road":        "Highway",
        "motorway":    "Highway",
        "industrial":  "Industrial",
        "factory":     "Industrial",
        "warehouse":   "Industrial",
        "pasture":     "Pasture",
        "meadow":      "Pasture",
        "grazing":     "Pasture",
        "permanent":   "PermanentCrop",
        "orchard":     "PermanentCrop",
        "vineyard":    "PermanentCrop",
        "residential": "Residential",
        "urban":       "Residential",
        "house":       "Residential",
        "building":    "Residential",
        "city":        "Residential",
        "town":        "Residential",
        "river":       "River",
        "stream":      "River",
        "canal":       "River",
        "sea":         "SeaLake",
        "lake":        "SeaLake",
        "ocean":       "SeaLake",
        "water":       "SeaLake",
    }
    for keyword, cls in keyword_map.items():
        if keyword in text.lower():
            return cls

    # Couldn't match — return the raw text as-is, caller handles gracefully
    logger.warning("Could not match VLM output '%s' to EuroSAT class", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API  (same shape as ask_vqa)
# ---------------------------------------------------------------------------

def classify_scene(
    image: Union[str, Path, Image.Image, LoadedImage],
    *,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
    prefer_vlm: bool = False,
) -> dict:
    """Classify a remote-sensing image into a land-cover scene class.

    Parameters
    ----------
    image : str | Path | PIL.Image | LoadedImage
        The input image. Accepts the same types as ``ask_vqa``.
    model : str, optional
        Force a specific Ollama model (only used when VLM backend is active).
    timeout : int
        Seconds per VLM attempt (ignored for CNN backend).
    max_retries : int
        Retries per model (ignored for CNN backend).
    prefer_vlm : bool
        If True, skip the CNN and go straight to VLM classification.
        Useful for testing the VLM path.

    Returns
    -------
    dict
        {
            "answer": str,            # pretty class label
            "raw_answer": str,        # unprocessed output (CNN class or VLM text)
            "model_used": str,        # "eurosat-resnet18" or Ollama model name
            "model_tier": str,        # "primary", "local-fallback", "cloud-fallback"
            "elapsed_s": float,       # time taken
            "status": str,            # "ok" or "error"
            "error": str | None,      # error message if status != "ok"
            "confidence": float,      # softmax probability (CNN) or 0.0 (VLM)
            "confidence_level": str,  # "high", "medium", "low" (unified)
            "reasoning": str,         # explanation for the classification
            "backend": str,           # "cnn" or "vlm"
            "all_classes": list,      # top-N class probabilities (CNN only)
        }
    """
    # --- Resolve image to PIL -----------------------------------------------
    pil_image = None
    try:
        if isinstance(image, LoadedImage):
            pil_image = image.pil_image
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (str, Path)):
            path = Path(image)
            _validate_image_path(path)
            if path.suffix.lower() in {'.tif', '.tiff'}:
                loaded = load_geotiff(path)
                pil_image = loaded.pil_image
            else:
                pil_image = Image.open(path).convert("RGB")
        else:
            return _error_result(
                f"Unsupported image type: {type(image).__name__}"
            )
    except (FileNotFoundError, ValueError) as exc:
        return _error_result(str(exc))

    # --- Choose backend ----------------------------------------------------
    t0 = time.perf_counter()

    cloud_vlm_enabled = os.environ.get("CLOUD_VLM_ENABLED", "false").lower() == "true"
    cloud_api_key = os.environ.get("CLOUD_VLM_API_KEY", "")
    use_vlm_first = prefer_vlm or (cloud_vlm_enabled and bool(cloud_api_key))

    if not use_vlm_first and _load_cnn():
        # ----- CNN path -----
        try:
            class_name, confidence, all_probs = _classify_cnn(pil_image)
            elapsed = time.perf_counter() - t0

            pretty = EUROSAT_LABELS_PRETTY.get(class_name, class_name)
            conf_level = _softmax_to_confidence_level(confidence)

            # Build reasoning from CNN probabilities
            top3 = all_probs[:3]
            top3_str = ", ".join(
                f"{p['class']} ({p['probability']:.1%})" for p in top3
            )
            reasoning = (
                f"CNN classifier (ResNet18-EuroSAT) predicted '{pretty}' "
                f"with {confidence:.1%} softmax confidence. "
                f"Top 3 predictions: {top3_str}."
            )

            logger.info(
                "Classify (CNN): class=%s  conf=%.3f  level=%s  time=%.2fs",
                class_name, confidence, conf_level, elapsed,
            )

            return {
                "answer": pretty,
                "raw_answer": class_name,
                "model_used": "eurosat-resnet18",
                "model_tier": "primary",
                "elapsed_s": round(elapsed, 2),
                "status": "ok",
                "error": None,
                "confidence": round(confidence, 4),
                "confidence_level": conf_level,
                "reasoning": reasoning,
                "backend": "cnn",
                "all_classes": all_probs,
            }
        except Exception as exc:
            logger.warning("CNN inference failed (%s), falling back to VLM", exc)
            # fall through to VLM

    # ----- VLM path -----
    try:
        class_name, conf_level, reasoning, raw, model_used, elapsed, model_tier = _classify_vlm(
            pil_image, model, timeout, max_retries
        )
        elapsed_total = time.perf_counter() - t0

        # Check if the match is a known EuroSAT class
        is_valid = class_name in EUROSAT_CLASSES
        pretty = EUROSAT_LABELS_PRETTY.get(class_name, class_name)

        if not is_valid:
            logger.warning(
                "VLM classify output '%s' not in EuroSAT classes, "
                "returning as-is", class_name
            )

        logger.info(
            "Classify (VLM): class=%s  level=%s  model=%s  tier=%s  time=%.2fs",
            class_name, conf_level, model_used, model_tier, elapsed_total,
        )

        return {
            "answer": pretty,
            "raw_answer": raw,
            "model_used": model_used,
            "model_tier": model_tier,
            "elapsed_s": round(elapsed_total, 2),
            "status": "ok",
            "error": None,
            "confidence": 0.0,  # VLM doesn't give calibrated probabilities
            "confidence_level": conf_level,
            "reasoning": reasoning,
            "backend": "vlm",
            "all_classes": [],
        }
    except Exception as exc:
        elapsed_total = time.perf_counter() - t0
        logger.error("Classification failed: %s", exc)
        return _error_result(f"Classification failed: {exc}", elapsed_total)


def _error_result(msg: str, elapsed: float = 0.0) -> dict:
    """Return a standardised error dict."""
    return {
        "answer": "",
        "raw_answer": "",
        "model_used": "",
        "model_tier": "",
        "elapsed_s": round(elapsed, 2),
        "status": "error",
        "error": msg,
        "confidence": 0.0,
        "confidence_level": "low",
        "reasoning": "",
        "backend": "",
        "all_classes": [],
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(_sys.argv) < 2:
        print("Usage: python classify_tool.py <image_path> [--vlm]")
        print('Example: python classify_tool.py sample.tif')
        _sys.exit(1)

    img_path = _sys.argv[1]
    use_vlm = "--vlm" in _sys.argv

    print(f"\n> Image   : {img_path}")
    print(f"> Backend : {'VLM (forced)' if use_vlm else 'CNN (with VLM fallback)'}\n")

    res = classify_scene(img_path, prefer_vlm=use_vlm)

    print(f"  Status     : {res['status']}")
    print(f"  Backend    : {res['backend']}")
    print(f"  Model      : {res['model_used']} ({res['model_tier']})")
    print(f"  Time       : {res['elapsed_s']}s")
    print(f"  Class      : {res['answer']}")
    print(f"  Confidence : {res['confidence']} ({res['confidence_level']})")
    print(f"  Reasoning  : {res['reasoning']}")
    if res['all_classes']:
        print("  Top 5:")
        for entry in res['all_classes'][:5]:
            print(f"    {entry['class']:25s} {entry['probability']:.4f}")
    if res['error']:
        print(f"  Error      : {res['error']}")

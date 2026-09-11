"""
SatQuery AI -- VQA Specialist Tool
====================================
Takes a remote-sensing image + natural-language question, returns a
detailed text answer with confidence scoring, reasoning, and optional
bounding-box visual grounding by calling the Ollama VLM with RS-tuned
structured prompts.

This is a *specialist tool* invoked by the router, not a general chatbot.

Key improvements over base version:
  - Structured JSON output with confidence scoring (high/medium/low)
  - Non-terse answers with reasoning explanations
  - Itemized breakdowns for counting/comparison questions
  - Bounding-box overlay for spatial/counting queries
  - Short-answer validation/retry
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

# Project imports
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.ollama_client import query_vlm, MAX_IMAGE_DIM
from tools.geotiff_utils import load_geotiff, LoadedImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts (Problem 1 & 2: Confidence + Non-terse answers)
# ---------------------------------------------------------------------------

VQA_SYSTEM_PROMPT = (
    "You are an expert remote sensing image analyst specializing in satellite "
    "and aerial imagery interpretation. You analyze satellite images with "
    "scientific rigor and provide thorough, comprehensive analyses.\n\n"
    "RULES:\n"
    "1. Provide a DETAILED, COMPREHENSIVE answer to the question asked. "
    "Do NOT give one-line answers. Always explain what you observe in depth.\n"
    "2. Include a thorough plain-language explanation of HOW you arrived at "
    "the answer — describe the specific visual features, spatial patterns, "
    "color signatures, textures, shapes, and contextual indicators you used.\n"
    "3. For counting or comparison questions, provide an ITEMIZED breakdown "
    "(list each item separately with its location/description, not just a total).\n"
    "4. Assess your own confidence with this calibration:\n"
    "   - 'high': features are clearly visible and your identification is definitive — "
    "USE THIS whenever the image clearly shows what is being asked about\n"
    "   - 'medium': ONLY when there is genuine ambiguity or partial occlusion\n"
    "   - 'low': ONLY when you truly cannot determine the answer\n"
    "   Default to 'high' when features are visible. Most satellite images with "
    "adequate resolution should yield 'high' confidence answers.\n"
    "5. If you genuinely CANNOT determine something from the image (resolution "
    "too low, feature not visible, ambiguous scene), say so explicitly in your "
    "reasoning rather than guessing.\n"
    "6. Never fabricate details that are not visible in the image.\n"
    "7. IMPORTANT: Provide LONGER, MORE DETAILED answers for longer or more "
    "complex questions. Match the depth of your analysis to the specificity "
    "of the question. A detailed question deserves a multi-paragraph answer.\n"
    "8. You MUST respond with valid JSON matching the schema provided. "
    "Do NOT wrap your JSON in markdown code fences."
)

VQA_PROMPT_TEMPLATE = (
    "Analyze this satellite/aerial image thoroughly and answer the following question.\n\n"
    "Question: {question}\n\n"
    "{detail_instruction}\n\n"
    "Respond with a JSON object (do NOT wrap in ```json``` fences) containing:\n"
    "- \"answer\": your comprehensive, detailed answer to the question. "
    "Include specific observations about visible features, their spatial "
    "arrangement, and any relevant context. Aim for at least 3-5 sentences.\n"
    "- \"confidence\": one of \"high\", \"medium\", or \"low\" — default to "
    "\"high\" when features are clearly visible\n"
    "- \"reasoning\": a detailed explanation of how you determined the answer "
    "(what visual features, patterns, colors, textures, and spatial "
    "relationships you observed). Be specific and technical.\n"
    "- \"items\": an array of objects with \"label\" and \"count\" for "
    "counting/comparison questions (empty array if not applicable)"
)

# Detail level instructions based on query length
DETAIL_INSTRUCTION_SHORT = (
    "Provide a focused but substantive analysis. Include at least 3-4 sentences "
    "describing what you observe."
)
DETAIL_INSTRUCTION_LONG = (
    "This is a detailed question that requires a comprehensive, multi-paragraph "
    "response. Provide an exhaustive analysis covering all relevant aspects of "
    "the image. Include specific observations about geography, land use, "
    "infrastructure, vegetation, water bodies, and any other features relevant "
    "to the question. Aim for at least 6-8 sentences with technical detail."
)

def _get_detail_instruction(question: str) -> str:
    """Return detail-level instruction based on query complexity."""
    word_count = len(question.split())
    if word_count >= 10:
        return DETAIL_INSTRUCTION_LONG
    return DETAIL_INSTRUCTION_SHORT

# Schema for structured output (Ollama format parameter)
VQA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["label", "count"],
            },
        },
    },
    "required": ["answer", "confidence", "reasoning", "items"],
}

# ---------------------------------------------------------------------------
# Spatial/Counting query detection (Problem 3)
# ---------------------------------------------------------------------------

SPATIAL_KEYWORDS = re.compile(
    r'\b(how\s+many|count|locate|find|where\s+(is|are)|position|'
    r'identify\s+(the|all|each)|number\s+of|detect|spot|'
    r'bounding\s+box|bbox|mark\s+(the|all))\b',
    re.IGNORECASE
)

VQA_SPATIAL_SYSTEM_PROMPT = (
    "You are an expert remote sensing image analyst. You detect and locate "
    "objects in satellite imagery with high precision.\n\n"
    "RULES:\n"
    "1. Answer the question thoroughly about what you see in the image. "
    "Provide a detailed, multi-sentence description.\n"
    "2. For each distinct object you identify, provide a bounding box as "
    "[x1, y1, x2, y2] where coordinates are pixel values relative to the "
    "image dimensions shown to you.\n"
    "3. Assess your confidence with this calibration:\n"
    "   - 'high': objects are clearly visible — USE THIS as the default "
    "when you can identify objects\n"
    "   - 'medium': only when objects are partially occluded or ambiguous\n"
    "   - 'low': only when you truly cannot identify objects\n"
    "4. If objects are too small or unclear to reliably locate, say so.\n"
    "5. Provide a comprehensive answer with descriptions of object locations, "
    "sizes, orientations, and spatial relationships.\n"
    "6. You MUST respond with valid JSON matching the schema provided. "
    "Do NOT wrap your JSON in markdown code fences."
)

VQA_SPATIAL_PROMPT_TEMPLATE = (
    "Analyze this satellite image thoroughly and answer the question. For each "
    "object you identify, provide its bounding box location.\n\n"
    "Question: {question}\n\n"
    "{detail_instruction}\n\n"
    "Respond with a JSON object (do NOT wrap in ```json``` fences) containing:\n"
    "- \"answer\": your comprehensive text answer with detailed descriptions "
    "of each identified object, its location, orientation, and characteristics. "
    "Aim for at least 4-6 sentences.\n"
    "- \"confidence\": one of \"high\", \"medium\", or \"low\" — default to "
    "\"high\" when objects are clearly visible\n"
    "- \"reasoning\": detailed explanation of how you identified each object\n"
    "- \"items\": array of {{\"label\": str, \"count\": int}} for counting\n"
    "- \"detections\": array of {{\"label\": str, \"bbox\": [x1, y1, x2, y2]}} "
    "for each detected object with bounding box coordinates"
)

VQA_SPATIAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["label", "count"],
            },
        },
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["label", "bbox"],
            },
        },
    },
    "required": ["answer", "confidence", "reasoning", "items", "detections"],
}


def _is_spatial_query(question: str) -> bool:
    """Check if the question involves spatial/counting that benefits from bounding boxes."""
    return bool(SPATIAL_KEYWORDS.search(question))


# ---------------------------------------------------------------------------
# Bounding box overlay rendering (Problem 3)
# ---------------------------------------------------------------------------

# Detection colors — distinct, high-visibility palette for dark satellite imagery
DETECTION_COLORS = [
    (0, 255, 127),    # spring green
    (255, 106, 0),    # vivid orange
    (0, 191, 255),    # deep sky blue
    (255, 255, 0),    # yellow
    (255, 0, 127),    # rose
    (127, 255, 0),    # chartreuse
    (0, 255, 255),    # cyan
    (255, 165, 0),    # orange
]

BBOX_LINE_WIDTH = 3
LABEL_FONT_SIZE = 14
LABEL_PADDING = 4


def _draw_bounding_boxes(
    pil_image: Image.Image,
    detections: List[Dict],
    model_image_dims: Tuple[int, int],
) -> Image.Image:
    """Draw bounding boxes with labels on the image.

    Parameters
    ----------
    pil_image : PIL.Image
        The original full-resolution image.
    detections : list of dict
        Each dict has {"label": str, "bbox": [x1, y1, x2, y2]}.
        Coordinates are relative to the model's input resolution.
    model_image_dims : tuple
        (width, height) of the image as the model saw it (after resizing).

    Returns
    -------
    PIL.Image
        Annotated image with bounding boxes drawn.
    """
    if not detections:
        return pil_image

    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    orig_w, orig_h = img.size
    model_w, model_h = model_image_dims

    # Scale factors from model resolution to original image
    scale_x = orig_w / model_w if model_w > 0 else 1.0
    scale_y = orig_h / model_h if model_h > 0 else 1.0

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", LABEL_FONT_SIZE)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        label = det.get("label", f"object-{i}")

        if len(bbox) != 4:
            continue

        # Scale coordinates to original image dimensions
        x1 = max(0, int(bbox[0] * scale_x))
        y1 = max(0, int(bbox[1] * scale_y))
        x2 = min(orig_w, int(bbox[2] * scale_x))
        y2 = min(orig_h, int(bbox[3] * scale_y))

        # Validate box dimensions
        if x2 <= x1 or y2 <= y1:
            continue

        color = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        fill_color = color + (50,)  # semi-transparent fill

        # Draw filled rectangle (semi-transparent)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=BBOX_LINE_WIDTH)

        # Draw label background + text
        label_text = f"{label}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_bg = [
            x1, max(0, y1 - text_h - 2 * LABEL_PADDING),
            x1 + text_w + 2 * LABEL_PADDING, y1
        ]
        draw.rectangle(label_bg, fill=color + (200,))
        draw.text(
            (x1 + LABEL_PADDING, label_bg[1] + LABEL_PADDING),
            label_text, fill=(0, 0, 0), font=font
        )

    return img


def _compute_model_dims(original_dims: Tuple[int, int]) -> Tuple[int, int]:
    """Compute what dimensions the image was resized to for the model."""
    w, h = original_dims
    if max(w, h) <= MAX_IMAGE_DIM:
        return (w, h)
    scale = MAX_IMAGE_DIM / max(w, h)
    return (int(w * scale), int(h * scale))


# ---------------------------------------------------------------------------
# Answer post-processing
# ---------------------------------------------------------------------------

def _clean_answer(raw: str) -> str:
    """Clean up VLM output to extract a concise VQA answer.

    Handles common VLM quirks: repeating the question, adding
    explanations after the answer, markdown formatting, etc.
    """
    text = raw.strip()

    # Remove markdown bold/italic
    text = re.sub(r'[*_]{1,3}', '', text)

    # If the model repeated the question, take only what comes after "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # Remove trailing periods and common filler
    text = text.rstrip('.')
    text = re.sub(r'^(The answer is|It is|I see|This is)\s+', '', text, flags=re.IGNORECASE)

    # Remove leading bullet / numbering
    text = re.sub(r'^[-\d.)\s]+', '', text).strip()

    # Collapse whitespace
    text = ' '.join(text.split())

    return text if text else "unknown"


def _is_garbled(answer: str) -> bool:
    """Detect if the model output is garbled / nonsensical."""
    if not answer or answer == "unknown":
        return False  # empty is handled, not garbled

    # Too long for a VQA answer (raised to allow detailed responses)
    if len(answer) > 3000:
        return True

    # Mostly non-alphanumeric characters
    alnum = sum(1 for c in answer if c.isalnum() or c.isspace())
    if len(answer) > 5 and alnum / len(answer) < 0.5:
        return True

    return False


def _parse_structured_response(raw: str) -> Optional[Dict]:
    """Try to parse the VLM response as structured JSON.
    
    Uses multiple strategies:
    1. Direct JSON parse
    2. Strip markdown fences then parse
    3. Regex extraction of JSON object
    4. Field-level regex fallback
    """
    # Strategy 1: Direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Strip markdown code fences
    stripped = raw.strip()
    if stripped.startswith('```'):
        # Remove opening fence (```json or ```)
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped, flags=re.IGNORECASE)
        # Remove closing fence
        stripped = re.sub(r'\s*```\s*$', '', stripped)
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Extract JSON object from surrounding text
    json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: Field-level regex fallback — extract individual fields
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).replace('\\"', '"').replace('\\n', '\n')
        conf_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', raw, re.I)
        reason_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        
        result = {
            "answer": answer_text,
            "confidence": conf_match.group(1).lower() if conf_match else "high",
            "reasoning": reason_match.group(1).replace('\\"', '"').replace('\\n', '\n') if reason_match else "",
            "items": [],
        }
        
        # Try to extract items array
        items_match = re.search(r'"items"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if items_match:
            try:
                result["items"] = json.loads(items_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try to extract detections array
        det_match = re.search(r'"detections"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if det_match:
            try:
                result["detections"] = json.loads(det_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        
        logger.info("Parsed response via field-level regex fallback")
        return result

    return None


# ---------------------------------------------------------------------------
# Supported formats
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_vqa(
    image: Union[str, Path, Image.Image, LoadedImage],
    question: str,
    *,
    model: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 2,
    force_spatial: bool = False,
) -> dict:
    """Ask a visual question about a remote sensing image.

    Parameters
    ----------
    image : str | Path | PIL.Image | LoadedImage
        The input image. Can be a file path (GeoTIFF or standard image),
        a PIL Image object, or a LoadedImage from geotiff_utils.
    question : str
        The natural-language question to ask about the image.
    model : str, optional
        Force a specific Ollama model (skips fallback chain).
    timeout : int
        Seconds per VLM attempt.
    max_retries : int
        Retries per model before fallback.
    force_spatial : bool
        If True, always use the spatial/bounding-box prompt.

    Returns
    -------
    dict
        {
            "answer": str,           # cleaned answer text
            "raw_answer": str,       # unprocessed VLM output
            "model_used": str,       # which model actually responded
            "model_tier": str,       # "primary", "local-fallback", "cloud-fallback"
            "elapsed_s": float,      # time taken
            "status": str,           # "ok", "garbled", "error"
            "error": str | None,     # error message if status != "ok"
            "confidence": str,       # "high", "medium", "low"
            "reasoning": str,        # explanation of how answer was determined
            "items": list,           # itemized breakdown for counting questions
            "detections": list,      # bounding box detections [{"label", "bbox"}]
            "annotated_image": PIL.Image | None,  # image with bounding boxes drawn
        }
    """
    # --- Validate question ---
    question = (question or "").strip()
    if not question:
        return _error_result("Empty question provided")

    # --- Resolve image to PIL ---
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
            return _error_result(f"Unsupported image type: {type(image).__name__}")
    except (FileNotFoundError, ValueError) as exc:
        return _error_result(str(exc))

    # --- Determine if this is a spatial/counting query ---
    is_spatial = force_spatial or _is_spatial_query(question)

    # --- Choose prompt and schema ---
    detail_instruction = _get_detail_instruction(question)
    if is_spatial:
        system_prompt = VQA_SPATIAL_SYSTEM_PROMPT
        prompt = VQA_SPATIAL_PROMPT_TEMPLATE.format(
            question=question, detail_instruction=detail_instruction
        )
        schema = VQA_SPATIAL_RESPONSE_SCHEMA
    else:
        system_prompt = VQA_SYSTEM_PROMPT
        prompt = VQA_PROMPT_TEMPLATE.format(
            question=question, detail_instruction=detail_instruction
        )
        schema = VQA_RESPONSE_SCHEMA

    # --- Call VLM ---
    try:
        result = query_vlm(
            prompt=prompt,
            image=pil_image,
            system_prompt=system_prompt,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            format_schema=schema,
            validate_length=False,  # structured output handles this
        )
    except Exception as exc:
        logger.error("VQA call failed: %s", exc)
        return _error_result(f"VLM call failed: {exc}")

    raw_answer = result["answer"]
    model_used = result["model_used"]
    model_tier = result.get("model_tier", "primary")
    original_dims = result.get("original_image_dims", pil_image.size)

    # --- Parse structured response ---
    parsed = _parse_structured_response(raw_answer)

    if parsed:
        answer_text = parsed.get("answer", "")
        confidence = parsed.get("confidence", "high")
        reasoning = parsed.get("reasoning", "")
        items = parsed.get("items", [])
        detections = parsed.get("detections", [])
    else:
        # Fallback: treat raw answer as plain text
        answer_text = _clean_answer(raw_answer)
        # Estimate confidence based on answer quality instead of always "medium"
        word_count = len(answer_text.split())
        if word_count >= 30:
            confidence = "high"
        elif word_count >= 10:
            confidence = "medium"
        else:
            confidence = "low"
        reasoning = "Response was not in structured format; confidence estimated from answer quality."
        items = []
        detections = []
        logger.warning("VQA response was not valid JSON, falling back to plain text parsing")

    # --- Check for garbled output ---
    status = "ok"
    error = None
    if _is_garbled(answer_text):
        status = "garbled"
        error = f"Model output appears garbled (length={len(answer_text)})"
        answer_text = "Unable to analyze this image clearly. The resolution may be too low or the features may not be distinguishable."
        confidence = "low"
        logger.warning("Garbled VQA output: '%s...'", raw_answer[:100])

    # --- Draw bounding boxes if detections exist ---
    annotated_image = None
    if detections and is_spatial:
        try:
            model_dims = _compute_model_dims(original_dims)
            annotated_image = _draw_bounding_boxes(pil_image, detections, model_dims)
            logger.info("Drew %d bounding boxes on image", len(detections))
        except Exception as draw_exc:
            logger.warning("Failed to draw bounding boxes: %s", draw_exc)

    # --- Note if cloud fallback was used for spatial queries ---
    if model_tier == "cloud-fallback" and detections:
        reasoning += (" Note: bounding-box coordinates from cloud fallback may be "
                      "less reliable than local model localization.")

    logger.info(
        "VQA: q='%s'  a='%s'  conf=%s  model=%s  tier=%s  time=%.1fs  "
        "detections=%d  status=%s",
        question[:50], answer_text[:80], confidence, model_used,
        model_tier, result["elapsed_s"], len(detections), status,
    )

    return {
        "answer": answer_text,
        "raw_answer": raw_answer,
        "model_used": model_used,
        "model_tier": model_tier,
        "elapsed_s": result["elapsed_s"],
        "status": status,
        "error": error,
        "confidence": confidence,
        "reasoning": reasoning,
        "items": items,
        "detections": detections,
        "annotated_image": annotated_image,
    }


def _error_result(msg: str) -> dict:
    """Return a standardised error dict."""
    return {
        "answer": "",
        "raw_answer": "",
        "model_used": "",
        "model_tier": "",
        "elapsed_s": 0.0,
        "status": "error",
        "error": msg,
        "confidence": "low",
        "reasoning": "",
        "items": [],
        "detections": [],
        "annotated_image": None,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(_sys.argv) < 3:
        print("Usage: python vqa_tool.py <image_path> <question>")
        print('Example: python vqa_tool.py sample.tif "How many buildings are visible?"')
        _sys.exit(1)

    img_path = _sys.argv[1]
    q = _sys.argv[2]

    print(f"\n> Image   : {img_path}")
    print(f"> Question: {q}")
    print(f"> Spatial : {_is_spatial_query(q)}\n")

    res = ask_vqa(img_path, q)

    print(f"  Status     : {res['status']}")
    print(f"  Model      : {res['model_used']} ({res['model_tier']})")
    print(f"  Time       : {res['elapsed_s']}s")
    print(f"  Confidence : {res['confidence']}")
    print(f"  Answer     : {res['answer']}")
    print(f"  Reasoning  : {res['reasoning']}")
    if res['items']:
        print(f"  Items      : {res['items']}")
    if res['detections']:
        print(f"  Detections : {len(res['detections'])} objects")
    if res['annotated_image']:
        res['annotated_image'].save("vqa_annotated.jpg", quality=90)
        print(f"  Annotated  : vqa_annotated.jpg")
    if res['error']:
        print(f"  Error      : {res['error']}")

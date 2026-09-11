"""
SatQuery AI — VLM Client
=========================
Sends image + text queries to a VLM.

When CLOUD_VLM_ENABLED=true (default):
  Primary:  Gemini 2.0 Flash (cloud, via google-generativeai)
  Fallback: qwen2.5vl:7b -> qwen2.5vl:3b -> moondream (local Ollama)

When CLOUD_VLM_ENABLED=false:
  Primary:  qwen2.5vl:7b (local Ollama)
  Fallback: qwen2.5vl:3b -> moondream

Design decisions:
  - A threading Lock ensures only ONE VLM call runs at a time (VRAM OOM guard).
  - Retries with exponential backoff on transient failures.
  - Timeout is per-attempt, not cumulative.
  - Structured JSON output via Ollama's `format` parameter when a schema is provided.
  - Short-answer validation/retry for non-yes/no questions.
"""

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ollama
from PIL import Image

# Load .env file (must happen before reading os.environ below)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PRIMARY_MODEL = "qwen2.5vl:7b"
DEFAULT_MODEL_FALLBACKS = [
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "moondream",
]
PRIMARY_MODEL = DEFAULT_PRIMARY_MODEL
FALLBACK_MODELS = DEFAULT_MODEL_FALLBACKS

DEFAULT_TIMEOUT_S = 120          # seconds per attempt
DEFAULT_MAX_RETRIES = 2          # retries on the *same* model before fallback
MAX_IMAGE_DIM = 768              # resize longest edge — reduces VRAM usage significantly
DEFAULT_NUM_CTX = 4096           # context window — increased for detailed answers

# Exponential backoff for transient failures
BACKOFF_BASE_S = 2.0             # base delay (doubles each retry)
BACKOFF_MAX_S = 16.0             # cap on delay

# Short-answer validation
MIN_ANSWER_WORDS = 15            # re-prompt if answer is shorter (non-yes/no)

# Gemini Cloud (primary when enabled)
CLOUD_VLM_ENABLED = os.environ.get("CLOUD_VLM_ENABLED", "false").lower() == "true"
CLOUD_VLM_API_KEY = os.environ.get("CLOUD_VLM_API_KEY", "")
CLOUD_VLM_MODEL = os.environ.get("CLOUD_VLM_MODEL", "gemini-3.6-flash")
GEMINI_MAX_RETRIES = 3           # retries on Gemini before falling to local

# Global lock — prevents parallel VLM calls (VRAM OOM protection)
_vlm_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(image: Union[str, Path, Image.Image, bytes]) -> str:
    """Return a base64-encoded JPEG string suitable for the Ollama API.

    Accepts a file path, a PIL Image, or raw bytes.
    Resizes large images so the VLM doesn't choke on huge GeoTIFFs.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB (drop alpha / palette issues)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize if any dimension exceeds MAX_IMAGE_DIM
    w, h = pil_img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_image_dimensions(image: Union[str, Path, Image.Image, bytes]) -> tuple:
    """Return (width, height) of the original image before any resizing."""
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        return (0, 0)
    return pil_img.size


def _call_model(
    model: str,
    prompt: str,
    images_b64: Union[str, list],
    timeout: int,
    system_prompt: Optional[str] = None,
    format_schema: Optional[Dict] = None,
) -> str:
    """Single attempt to call an Ollama model. Raises on failure/timeout.

    images_b64 can be a single base64 string or a list of them.
    format_schema: if provided, passed as `format=` to ollama.chat for
    structured JSON output.
    """
    if isinstance(images_b64, str):
        images_b64 = [images_b64]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": prompt,
        "images": images_b64,
    })

    # Build options
    chat_kwargs = {
        "model": model,
        "messages": messages,
        "options": {"num_ctx": DEFAULT_NUM_CTX},
    }
    if format_schema is not None:
        chat_kwargs["format"] = format_schema

    # ollama.chat is synchronous; we enforce an external timeout via threading
    result_container = {"response": None, "error": None}

    def _run():
        try:
            resp = ollama.chat(**chat_kwargs)
            result_container["response"] = resp["message"]["content"]
        except Exception as exc:
            result_container["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        raise TimeoutError(
            f"Ollama call to '{model}' timed out after {timeout}s"
        )
    if result_container["error"]:
        raise result_container["error"]

    response = result_container["response"]
    # Handle None and whitespace-only responses
    if response is None or (isinstance(response, str) and not response.strip()):
        raise RuntimeError(f"Empty response from '{model}'")

    return response.strip()


def _is_transient_error(exc: Exception) -> bool:
    """Check if an error is transient and worth retrying with backoff."""
    transient_types = (TimeoutError, ConnectionError, ConnectionRefusedError, OSError)
    if isinstance(exc, transient_types):
        return True
    exc_str = str(exc).lower()
    return any(kw in exc_str for kw in ["timeout", "connection refused", "connection reset", "temporarily unavailable"])


def _is_yes_no_question(prompt: str) -> bool:
    """Heuristic: check if the question is inherently yes/no."""
    prompt_lower = prompt.lower().strip()
    return (
        prompt_lower.startswith("is there") or
        prompt_lower.startswith("are there") or
        prompt_lower.startswith("does ") or
        prompt_lower.startswith("do ") or
        prompt_lower.startswith("can ") or
        prompt_lower.startswith("has ") or
        prompt_lower.startswith("have ") or
        prompt_lower.startswith("will ") or
        prompt_lower.startswith("was ") or
        prompt_lower.startswith("were ")
    )


# ---------------------------------------------------------------------------
# Cloud VLM — Gemini
# ---------------------------------------------------------------------------

def _call_cloud_vlm(
    prompt: str,
    images_b64: Union[str, list],
    system_prompt: Optional[str] = None,
    timeout: int = 60,
) -> tuple:
    """Call Google Gemini Vision API.

    Used as the PRIMARY model when CLOUD_VLM_ENABLED=true.
    Falls back to local Ollama only if Gemini fails after all retries.
    Returns (answer_text, model_name_used).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai package not installed. "
            "Install with: pip install google-generativeai"
        )

    if not CLOUD_VLM_API_KEY:
        raise RuntimeError("CLOUD_VLM_API_KEY not set")

    genai.configure(api_key=CLOUD_VLM_API_KEY)

    if isinstance(images_b64, str):
        images_b64 = [images_b64]

    # Build content parts with decoded PIL images
    parts = []
    if system_prompt:
        parts.append(f"[System Instructions]\n{system_prompt}\n\n[User Query]\n")
    parts.append(prompt)

    for img_b64 in images_b64:
        try:
            img_bytes = base64.b64decode(img_b64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            parts.append(pil_img)
        except Exception as exc:
            logger.warning("Failed to decode image b64 for Gemini: %s", exc)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                }
            })

    candidate_models = list(dict.fromkeys([
        CLOUD_VLM_MODEL,
        "gemini-3.6-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    ]))

    last_exc = None
    for mdl_name in candidate_models:
        try:
            model = genai.GenerativeModel(mdl_name)
            response = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
                request_options={"timeout": timeout},
            )
            if response and response.text:
                return response.text.strip(), mdl_name
            else:
                raise RuntimeError(f"Empty response from Gemini ({mdl_name})")
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini candidate model %s attempt failed: %s", mdl_name, exc)

    raise last_exc or RuntimeError("Gemini API failed for all model candidates")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_vlm(
    prompt: str,
    image: Union[str, Path, Image.Image, bytes],
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    format_schema: Optional[Dict] = None,
    validate_length: bool = False,
) -> dict:
    """Send an image + text query to the VLM and return the answer.

    When CLOUD_VLM_ENABLED=true:
      1. Try Gemini cloud (up to GEMINI_MAX_RETRIES attempts with backoff)
      2. If Gemini fails -> fall back to local Ollama chain

    When CLOUD_VLM_ENABLED=false:
      1. Try local Ollama chain only (primary -> fallback models)

    Returns
    -------
    dict
        {"answer": str, "model_used": str, "elapsed_s": float, "model_tier": str}
    """
    image_b64 = _encode_image(image)
    original_dims = _get_image_dimensions(image)

    with _vlm_lock:
        last_error = None

        # ── STEP 1: Try Gemini FIRST when cloud is enabled ──────────────
        if CLOUD_VLM_ENABLED and CLOUD_VLM_API_KEY and not model:
            for attempt in range(1, GEMINI_MAX_RETRIES + 1):
                logger.info(
                    "Gemini cloud call: attempt=%d/%d  tier=primary",
                    attempt, GEMINI_MAX_RETRIES,
                )
                t0 = time.perf_counter()
                try:
                    answer, used_model = _call_cloud_vlm(
                        prompt, image_b64, system_prompt,
                        timeout=min(timeout, 60),
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info("Gemini cloud success: model=%s elapsed=%.1fs", used_model, elapsed)
                    return {
                        "answer": answer,
                        "model_used": used_model,
                        "elapsed_s": round(elapsed, 2),
                        "model_tier": "primary",
                        "original_image_dims": original_dims,
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "Gemini attempt failed: attempt=%d  elapsed=%.1fs  error=%s",
                        attempt, elapsed, exc,
                    )
                    # Backoff before retrying Gemini
                    if attempt < GEMINI_MAX_RETRIES:
                        delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                        logger.info("Gemini transient error, backing off %.1fs", delay)
                        time.sleep(delay)

            logger.warning(
                "All %d Gemini attempts exhausted, falling back to local Ollama.",
                GEMINI_MAX_RETRIES,
            )

        # ── STEP 2: Local Ollama chain (fallback when cloud is on, primary when off)
        if model:
            models_to_try = [model]
        else:
            models_to_try = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))

        for mdl_idx, mdl in enumerate(models_to_try):
            # Tier depends on whether cloud was tried first
            if CLOUD_VLM_ENABLED and CLOUD_VLM_API_KEY:
                tier = "local-fallback"  # cloud was primary
            elif mdl_idx == 0:
                tier = "primary"         # no cloud, first local = primary
            else:
                tier = "local-fallback"

            for attempt in range(1, max_retries + 1):
                logger.info(
                    "VLM call: model=%s  attempt=%d/%d  tier=%s",
                    mdl, attempt, max_retries, tier,
                )
                t0 = time.perf_counter()
                try:
                    answer = _call_model(
                        mdl, prompt, image_b64, timeout,
                        system_prompt, format_schema,
                    )
                    elapsed = time.perf_counter() - t0

                    # Short-answer validation/retry (Problem 2)
                    if (validate_length and
                            not _is_yes_no_question(prompt) and
                            len(answer.split()) < MIN_ANSWER_WORDS and
                            format_schema is None):
                        logger.info(
                            "Answer too short (%d words), re-prompting with elaboration",
                            len(answer.split()),
                        )
                        elaboration_prompt = (
                            f"Your previous answer was too brief. "
                            f"Please provide a more detailed response.\n\n"
                            f"Original question: {prompt}\n\n"
                            f"Your brief answer was: {answer}\n\n"
                            f"Now give a fuller answer with explanation and reasoning."
                        )
                        try:
                            answer = _call_model(
                                mdl, elaboration_prompt, image_b64, timeout,
                                system_prompt, format_schema,
                            )
                        except Exception:
                            pass  # keep original short answer

                    logger.info(
                        "VLM success: model=%s  elapsed=%.1fs", mdl, elapsed
                    )
                    return {
                        "answer": answer.strip(),
                        "model_used": mdl,
                        "elapsed_s": round(elapsed, 2),
                        "model_tier": tier,
                        "original_image_dims": original_dims,
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "VLM attempt failed: model=%s  attempt=%d  "
                        "elapsed=%.1fs  error=%s",
                        mdl, attempt, elapsed, exc,
                    )
                    if _is_transient_error(exc) and attempt < max_retries:
                        delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                        logger.info("Transient error, backing off %.1fs", delay)
                        time.sleep(delay)

            logger.warning(
                "All %d attempts exhausted for model '%s', trying next fallback.",
                max_retries, mdl,
            )

        # All models failed
        raise RuntimeError(
            f"All VLM models failed. Last error: {last_error}"
        ) from last_error


def query_vlm_multi_image(
    prompt: str,
    images: list,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    format_schema: Optional[Dict] = None,
) -> dict:
    """Send multiple images + a text query to the VLM.

    Same contract as query_vlm but accepts a list of images.
    Uses the same Gemini-first -> Ollama-fallback order.
    """
    if not images:
        raise ValueError("At least one image is required")

    images_b64 = [_encode_image(img) for img in images]

    with _vlm_lock:
        last_error = None

        # ── STEP 1: Try Gemini FIRST when cloud is enabled ──────────────
        if CLOUD_VLM_ENABLED and CLOUD_VLM_API_KEY and not model:
            for attempt in range(1, GEMINI_MAX_RETRIES + 1):
                logger.info(
                    "Gemini cloud multi-image call: images=%d  attempt=%d/%d",
                    len(images_b64), attempt, GEMINI_MAX_RETRIES,
                )
                t0 = time.perf_counter()
                try:
                    answer, used_model = _call_cloud_vlm(
                        prompt, images_b64, system_prompt,
                        timeout=min(timeout, 60),
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info("Gemini cloud success (multi-image): model=%s elapsed=%.1fs", used_model, elapsed)
                    return {
                        "answer": answer,
                        "model_used": used_model,
                        "elapsed_s": round(elapsed, 2),
                        "model_tier": "primary",
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "Gemini multi-image attempt failed: attempt=%d  error=%s",
                        attempt, exc,
                    )
                    if attempt < GEMINI_MAX_RETRIES:
                        delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                        time.sleep(delay)

            logger.warning(
                "All %d Gemini attempts exhausted (multi-image), falling back to local Ollama.",
                GEMINI_MAX_RETRIES,
            )

        # ── STEP 2: Local Ollama chain ──────────────────────────────────
        if model:
            models_to_try = [model]
        else:
            models_to_try = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))

        for mdl_idx, mdl in enumerate(models_to_try):
            if CLOUD_VLM_ENABLED and CLOUD_VLM_API_KEY:
                tier = "local-fallback"
            elif mdl_idx == 0:
                tier = "primary"
            else:
                tier = "local-fallback"

            for attempt in range(1, max_retries + 1):
                logger.info(
                    "VLM multi-image call: model=%s  images=%d  attempt=%d/%d",
                    mdl, len(images_b64), attempt, max_retries,
                )
                t0 = time.perf_counter()
                try:
                    answer = _call_model(
                        mdl, prompt, images_b64, timeout,
                        system_prompt, format_schema,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "VLM success: model=%s  elapsed=%.1fs", mdl, elapsed
                    )
                    return {
                        "answer": answer.strip(),
                        "model_used": mdl,
                        "elapsed_s": round(elapsed, 2),
                        "model_tier": tier,
                    }
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_error = exc
                    logger.warning(
                        "VLM attempt failed: model=%s  attempt=%d  "
                        "elapsed=%.1fs  error=%s",
                        mdl, attempt, elapsed, exc,
                    )
                    if _is_transient_error(exc) and attempt < max_retries:
                        delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                        time.sleep(delay)

            logger.warning(
                "All %d attempts exhausted for model '%s', trying next fallback.",
                max_retries, mdl,
            )

        raise RuntimeError(
            f"All VLM models failed. Last error: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python ollama_client.py <image_path> [question]")
        print('Example: python ollama_client.py sample.jpg "What do you see?"')
        sys.exit(1)

    img_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."

    print(f"\n> Image : {img_path}")
    print(f"> Query : {question}\n")

    result = query_vlm(prompt=question, image=img_path)

    print(f"[OK] Model : {result['model_used']}")
    print(f"[OK] Tier  : {result['model_tier']}")
    print(f"[OK] Time  : {result['elapsed_s']}s")
    print(f"[OK] Answer:\n{result['answer']}")

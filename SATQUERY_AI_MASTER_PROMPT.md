# SatQuery AI — Master Improvement Prompt (Aligned to Actual Project Scope)
**Target: SIH 2026 / ISRO Problem Statement 26167 — for Claude in Antigravity**

This replaces the generic spec you pasted. That spec assumed a **Gemini API + React + city-weather** stack — yours is **local Ollama (qwen2.5vl) + Streamlit + remote-sensing GeoTIFF/EuroSAT/SAR**. The good ideas from that spec (confidence scoring, no-terse-answers, visual overlays, robust fallback, design-system rigor, thorough testing) are kept and re-targeted at your real codebase. The parts that don't fit your problem statement are dropped — see the note at the end.

Read all sections before writing code. Several sections modify tools that already exist (`vqa_tool.py`, `classify_tool.py`, `change_tool.py`, `sar_tool.py`, `ollama_client.py`, `gui/app.py`) — patch them, don't rewrite from scratch where not necessary. Test each mode (`single`, `bitemporal`, `sar_optical`) before and after every change so a fix in one tool can't silently break another.

---

## Context: what already exists (do not re-architect)

- Router: 2-stage heuristic keyword scoring → VLM few-shot fallback for ambiguous queries
- VLM: `qwen2.5vl:7b` primary, with `qwen2.5vl:3b` → `moondream` local fallback chain via Ollama, serialized behind a `threading.Lock()` to avoid VRAM OOM
- CNN: ResNet-18 (timm) for 10-class EuroSAT land-cover classification, CPU, <1s
- Three modes: `single` (VQA + classify), `bitemporal` (ORB align → Otsu diff → overlay), `sar_optical` (false-color composite → Canny edge overlap → SSIM/Pearson)
- `_clean_answer()` / `_is_garbled()` sanitize VLM output today
- Streamlit GUI (`gui/app.py`), Cloudflare tunnel for public demo access

---

## Problem 1 — Confidence is inconsistent or missing

**Symptom:** Some VQA/classification answers give no signal on how reliable they are; a wrong or low-confidence answer looks identical to a solid one.

**Fix:**
- Use Ollama's structured-output mode (`format=<json schema>` on the `chat`/`generate` call — supported by `qwen2.5vl` since it's a standard Ollama vision model) so every VQA/classification response comes back as a schema-validated object: `{ "answer": ..., "confidence": "high" | "medium" | "low", "reasoning": ... }`, instead of free text that might be empty or malformed.
- If the model genuinely cannot determine something from the image (e.g. resolution too low, feature not visible, ambiguous scene), it must say so explicitly in `reasoning` rather than returning a bare answer or silently failing.
- For `classify_scene`, pair the CNN's softmax confidence with the VLM fallback's own stated confidence when the CNN path isn't used, so both classification paths report confidence the same way.
- Surface confidence in the Streamlit UI as a clear colored badge (e.g. green/amber/red) next to every answer — don't bury it in an expander.

## Problem 2 — Terse or occasionally fabricated answers

**Symptom:** The VLM sometimes returns a clipped non-answer regardless of the question, or states something with unwarranted certainty instead of admitting uncertainty.

**Fix:**
- Rewrite the RS-tuned system prompt in `vqa_tool.py` to require: a direct answer to the specific question asked (not a generic yes/no unless the question literally is yes/no); a short, plain-language explanation of how the model arrived at the answer; an itemized breakdown for counting/comparison questions (per-item, not just a total) so it can be charted; and an explicit guess-vs-confident flag.
- Add a validation/retry step in `ollama_client.py`: if a response is suspiciously short (under ~15 words) for a question that isn't inherently yes/no, automatically re-prompt once with a stronger elaboration instruction before it reaches `_clean_answer()`.
- For countable questions ("how many buildings/roads/water bodies"), always pair the text answer with a bar/count chart built from the itemized breakdown (Streamlit's `st.bar_chart` or a small Plotly chart) so the user can visually sanity-check the number.
- Keep logging raw model responses in dev mode so prompt/answer mismatches are easy to catch while tuning.

## Problem 3 — No visual grounding for spatial/counting questions

**Symptom:** "How many buildings/roads/water bodies" returns a number with no way to see what was actually counted.

**Fix:**
- `qwen2.5vl` natively supports visual localization (bounding boxes/points), so for any question involving counting, locating, or identifying discrete objects, switch the prompt to request a JSON list of bounding boxes and labels for each detected instance, using the model's documented coordinate format. Confirm the exact current format (normalization range, box ordering) against Qwen2.5-VL's own docs before wiring it up — don't assume the numbers from an old prompt template.
- For linear features (roads, rivers, coastlines) where a box is less meaningful than a path, ask for a best-fit polyline/segmentation where the model supports it, falling back to per-segment bounding boxes otherwise.
- In `gui/app.py`, draw the returned boxes/lines over the uploaded image (PIL `ImageDraw` server-side, or an HTML/canvas component if you want client-side interactivity) — scale normalized coordinates to the actual rendered image size, with each region's label visible on hover or as a small tag.
- Trigger this automatically whenever the router classifies the query as spatial/countable (extend the existing heuristic keyword scoring — "how many," "where," "locate," "count" — rather than adding a second classifier).
- Always show both the written answer (Problem 2) and the visual overlay together — the overlay never replaces the text.

## Problem 4 — UI should look intentional, not templated

**Fix:**
- Before touching `gui/app.py`, read `/mnt/skills/public/frontend-design/SKILL.md` in full and follow its process: define a token system (color, type, layout, principles) grounded in this project's actual subject matter — remote sensing, satellite/SAR imagery, geospatial data — then review that plan against the brief before building. Streamlit's theming is more constrained than raw HTML/React (you're working through `st.set_page_config`, custom CSS injection via `st.markdown(unsafe_allow_html=True)`, and component styling rather than free-form markup), so treat the skill's *token system and principles* as authoritative and adapt the *implementation mechanics* to Streamlit's constraints.
- Avoid generic AI-default styling (cream + terracotta, identical rounded-corner cards everywhere, tracked-out all-caps labels, arrow-suffixed buttons) unless deliberately chosen for a reason tied to this project.
- The single-image view, the confidence badges (Problem 1), the bounding-box overlay (Problem 3), and the change/SAR visualizations should all read as one consistent design system, not bolted-together widgets from different points in the project's history.
- Responsive down to mobile-width viewports (your Cloudflare tunnel is used for evaluator phone/tablet testing), visible focus states, and adequate color contrast are required, not optional polish.

## Problem 5 — Single point of failure on local Ollama

**Fix (extends the existing fallback chain, doesn't replace it):**
- Keep the existing `qwen2.5vl:7b → qwen2.5vl:3b → moondream` local cascade and its VRAM lock/retry logic as-is.
- Wrap each Ollama call with retry + exponential backoff for transient failures (timeout, connection refused) in addition to the existing 2-retries-per-model policy, so a momentarily slow local server doesn't fall through the whole chain unnecessarily.
- Add one **cloud-hosted vision API** as the final fallback rung below `moondream`, gated behind a config flag/env var so offline demo runs are unaffected. Use it for descriptive Q&A and classification only — bounding-box/localization output (Problem 3) should be explicitly marked unavailable when the cloud fallback answered, since you can't guarantee it supports the same spatial format.
- If every rung fails, the GUI must show a clear message ("Analysis temporarily unavailable — retrying" or similar), never a silent crash or an empty result.
- Always show the user which model tier actually answered (primary / local fallback / cloud fallback), consistent with the "why this tool" transparency theme already in your architecture.

## Problem 6 — Dropped from this pass: city recognition / weather dashboard

The original spec's Gemini → Nominatim → Rainbow Weather pipeline doesn't fit ISRO problem statement 26167 (remote-sensing image analysis, not city/weather lookup) and isn't in scope for this hackathon deliverable. I've left it out rather than bolt it onto SatQuery AI. If you actually want a bonus mode that correlates a classified scene (e.g. `AnnualCrop`, `Forest`, `Residential`) with regional context, that's a different, smaller feature — say so and I'll scope it separately rather than folding an unrelated app in here.

---

## Tech constraints (apply to the whole task)

- Backend stays Python — don't introduce a Node/Express layer; extend the existing `router/`, `tools/`, and `gui/` structure.
- Frontend stays Streamlit — don't switch to React mid-hackathon. If overlay interactivity genuinely can't be done well in Streamlit, propose the specific limitation before switching anything.
- Every model call wrapped in error handling with a user-facing message — no silent failures anywhere in the pipeline.
- Confirm current Qwen2.5-VL and Ollama structured-output/localization request formats against their own documentation before implementing — these APIs move fast enough that assuming an old format risks broken output.
- Env vars for any new keys (e.g. `CLOUD_VLM_API_KEY`) — never hardcoded, and add/update `.env.example`.

## What I need from you (Claude in Antigravity)

1. Read `/mnt/skills/public/frontend-design/SKILL.md` before writing any UI code.
2. Confirm current Ollama structured-output syntax and Qwen2.5-VL's bounding-box/localization output format against up-to-date docs.
3. Fix the VQA pipeline per Problems 1–3 (confidence scoring, non-terse answers, bounding-box overlay) without breaking `classify`, `bitemporal`, or `sar_optical` — re-run the existing test suite after each change.
4. Add the fallback hardening per Problem 5.
5. Rebuild `gui/app.py` styling per Problem 4 across all three modes as one consistent system.
6. Add/update `.env.example` with any new keys.
7. Test end-to-end: all three modes, the overlay, the fallback path (force a failure to confirm graceful degradation), and confidence display — report what you tested and any cases where the bounding-box overlay isn't reliable (e.g. very small objects, low-resolution tiles).

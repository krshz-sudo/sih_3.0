"""
SatQuery AI — Interactive GUI (Pure Gemini)
=============================================
Evidence-based remote-sensing analysis powered exclusively by Gemini.

Features:
  - Single image / bi-temporal / SAR+optical upload
  - One Gemini call per analysis → structured JSON
  - Side-by-side original vs AI-annotated view
  - Interactive feature cards with observed/inferred/uncertain status
  - Numeric confidence (0.00–1.00) with calibrated labels
  - Real bounding boxes and segmentation masks from Gemini
  - Downloadable annotated PNG
  - Downloadable CSV from current analysis
  - Real charts from Gemini's structured data
  - No Ollama, no CNN, no fallback — Gemini only

Run:  streamlit run gui/app.py
"""

import csv
import io
import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"  # Prevent grpcio segfault on Streamlit Cloud
import time
import logging
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

from router.router import route_query, IMAGE_MODE_SINGLE, IMAGE_MODE_BITEMPORAL, IMAGE_MODE_SAR_OPTICAL
from tools.geotiff_utils import load_geotiff
from tools.annotation_renderer import render_annotations

logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SatQuery AI — Remote Sensing Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, p, div, span, label, input, button, textarea, select {
    font-family: 'Outfit', system-ui, sans-serif;
}
code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

#MainMenu { visibility: hidden; }
/* header { visibility: hidden; } */
footer { visibility: hidden; }

.stApp {
    background: linear-gradient(160deg, #0B0F19 0%, #111827 40%, #0F172A 100%);
}

/* Hero */
.hero-header { text-align: center; padding: 1rem 0 0.6rem; }
.hero-header h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #06B6D4, #22D3EE, #10B981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem; letter-spacing: -0.02em;
}
.hero-header p { color: #94A3B8; font-size: 0.9rem; margin: 0; }

/* Glass card */
.glass-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(6, 182, 212, 0.15);
    border-radius: 12px; padding: 1.1rem; margin-bottom: 0.8rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 32px rgba(0,0,0,0.3);
}

/* Status badges */
.status-observed {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    background: rgba(16,185,129,0.15); color: #34D399;
    border: 1px solid rgba(16,185,129,0.3);
}
.status-inferred {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    background: rgba(245,158,11,0.15); color: #FBBF24;
    border: 1px solid rgba(245,158,11,0.3);
}
.status-uncertain {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    background: rgba(244,63,94,0.15); color: #FB7185;
    border: 1px solid rgba(244,63,94,0.3);
}

/* Confidence bar */
.conf-bar-bg {
    height: 6px; border-radius: 3px; background: rgba(255,255,255,0.08);
    margin-top: 0.3rem; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.5s ease;
}

/* Feature card */
.feature-card {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid rgba(6, 182, 212, 0.12);
    border-radius: 10px; padding: 0.85rem; margin-bottom: 0.6rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.feature-card:hover {
    border-color: rgba(34, 211, 238, 0.4);
    transform: translateY(-1px);
}
.feature-card.highlighted {
    border-color: #22D3EE;
    box-shadow: 0 0 12px rgba(34, 211, 238, 0.25);
}
.feature-card .fc-label {
    color: #E2E8F0; font-weight: 600; font-size: 0.9rem;
}
.feature-card .fc-category {
    color: #64748B; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.04em;
}
.feature-card .fc-evidence {
    color: #94A3B8; font-size: 0.82rem; line-height: 1.4;
    margin-top: 0.3rem;
}

/* Answer sections */
.analysis-section {
    background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(16,185,129,0.04));
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 12px; padding: 1.2rem; margin: 0.7rem 0;
}
.analysis-section h3 {
    color: #22D3EE; font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.5rem;
}
.analysis-section p, .analysis-section div {
    color: #F1F5F9; font-size: 0.92rem; line-height: 1.65;
}

/* Stat box */
.stat-box {
    background: rgba(6, 182, 212, 0.06);
    border: 1px solid rgba(6, 182, 212, 0.15);
    border-radius: 12px; padding: 0.75rem; text-align: center;
}
.stat-box .stat-value { font-size: 1.2rem; font-weight: 700; color: #22D3EE; }
.stat-box .stat-label {
    font-size: 0.68rem; color: #64748B; text-transform: uppercase;
    letter-spacing: 0.04em; margin-top: 0.1rem;
}

/* Error */
.error-banner {
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-radius: 12px; padding: 1rem 1.25rem; text-align: center;
}
.error-banner h4 { color: #FB7185; font-weight: 600; margin: 0 0 0.3rem; }
.error-banner p { color: #94A3B8; font-size: 0.85rem; margin: 0; }

/* Limitations */
.limitation-item {
    color: #94A3B8; font-size: 0.82rem; padding: 0.3rem 0;
    border-bottom: 1px solid rgba(71,85,105,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0F19, #111827);
    border-right: 1px solid rgba(6, 182, 212, 0.1);
}

img { border-radius: 10px; max-width: 100%; height: auto; }

/* Buttons */
.stButton > button { border-radius: 10px !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #06B6D4, #0891B2) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
}

/* Mobile */
@media (max-width: 768px) {
    .hero-header h1 { font-size: 1.5rem !important; }
    .glass-card { padding: 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_uploaded_image(uploaded_file):
    """Load uploaded file → (PIL Image, temp path)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / uploaded_file.name
    tmp.write_bytes(uploaded_file.getvalue())

    if suffix in {'.tif', '.tiff'}:
        loaded = load_geotiff(str(tmp))
        return loaded.pil_image, str(tmp)
    else:
        pil = Image.open(tmp).convert("RGB")
        return pil, str(tmp)


def _conf_color(conf: float) -> str:
    if conf >= 0.75: return "#34D399"
    elif conf >= 0.50: return "#FBBF24"
    else: return "#FB7185"

def _conf_label(conf: float) -> str:
    if conf >= 0.90: return "Confirmed"
    elif conf >= 0.75: return "Likely"
    elif conf >= 0.50: return "Possible"
    elif conf >= 0.30: return "Uncertain"
    else: return "Weak"

def _status_badge(status: str) -> str:
    s = status.lower()
    return f'<span class="status-{s}">{s}</span>'


# ── SAR Quick Questions ────────────────────────────────────────────────────

EXAMPLE_QUERIES = {
    IMAGE_MODE_SINGLE: [
        "Identify the major SAR-visible features",
        "Describe the backscatter and texture patterns",
        "Identify possible built-up areas",
        "Identify water-like low-backscatter regions",
        "Identify major linear infrastructure",
        "Describe the dominant land-cover patterns",
    ],
    IMAGE_MODE_BITEMPORAL: [
        "What changed between these two images?",
        "Which regions show the strongest evidence of change?",
        "Has any construction or urban expansion occurred?",
        "Has vegetation cover decreased or increased?",
    ],
    IMAGE_MODE_SAR_OPTICAL: [
        "Compare the optical and SAR images",
        "What features are visible in both modalities?",
        "What does the radar reveal that optical doesn't?",
        "Explain spatial relationships between major features",
    ],
}


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
        <div style="font-size:2.2rem; margin-bottom:0.2rem;">🛰️</div>
        <div style="font-size:1rem; font-weight:700; color:#22D3EE;">SatQuery AI</div>
        <div style="font-size:0.7rem; color:#475569; font-family:'JetBrains Mono',monospace;">SIH 2026 · ISRO PS-26167</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Input mode
    st.markdown("##### Analysis Mode")
    input_mode = st.radio(
        "Select input type", label_visibility="collapsed",
        options=[IMAGE_MODE_SINGLE, IMAGE_MODE_BITEMPORAL, IMAGE_MODE_SAR_OPTICAL],
        format_func=lambda x: {
            IMAGE_MODE_SINGLE: "Single Image Analysis",
            IMAGE_MODE_BITEMPORAL: "Bi-Temporal Change Detection",
            IMAGE_MODE_SAR_OPTICAL: "Optical + SAR Fusion",
        }[x],
    )
    st.markdown("---")

    # File uploads
    st.markdown("##### Upload Images")
    st.caption("Supports GeoTIFF, TIFF, PNG, JPG")

    ftypes = ["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"]
    if input_mode == IMAGE_MODE_SINGLE:
        uploaded_1 = st.file_uploader("Upload satellite image", type=ftypes, key="u1")
        uploaded_2 = None
    elif input_mode == IMAGE_MODE_BITEMPORAL:
        uploaded_1 = st.file_uploader("Before image (T1)", type=ftypes, key="u_b")
        uploaded_2 = st.file_uploader("After image (T2)", type=ftypes, key="u_a")
    else:
        uploaded_1 = st.file_uploader("Optical image", type=ftypes, key="u_o")
        uploaded_2 = st.file_uploader("SAR image", type=ftypes, key="u_s")

    st.markdown("---")

    # Gemini status — real verification
    st.markdown("##### AI Engine")
    from tools.gemini_client import verify_connection, GEMINI_MODEL
    gemini_status = verify_connection()
    if gemini_status["connected"]:
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="width:8px;height:8px;border-radius:50%;background:#34D399;"></div>
            <span style="color:#34D399;font-size:0.82rem;font-weight:500;">Gemini Connected</span>
        </div>
        <div style="color:#64748B;font-size:0.72rem;margin-top:0.2rem;">
            Model: <strong style="color:#94A3B8">{gemini_status["model"]}</strong>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="width:8px;height:8px;border-radius:50%;background:#FB7185;"></div>
            <span style="color:#FB7185;font-size:0.82rem;font-weight:500;">Gemini Unavailable</span>
        </div>
        <div style="color:#64748B;font-size:0.72rem;margin-top:0.2rem;">{gemini_status.get("error","")[:80]}</div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("About SatQuery AI"):
        st.markdown("""
        **SatQuery AI** is an evidence-based remote-sensing
        analysis assistant powered exclusively by **Google Gemini**.

        **Architecture:** Pure Gemini multimodal inference.
        One AI call per analysis → structured JSON → UI.

        **Built for:** SIH 2026 | ISRO Problem 26167
        """)


# ── Main Content ────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>SatQuery AI</h1>
    <p>Evidence-Based Remote Sensing Analysis · Powered by Gemini</p>
</div>
""", unsafe_allow_html=True)

# Load images
images_loaded = []
image_paths = []

if uploaded_1:
    try:
        pil_1, path_1 = _load_uploaded_image(uploaded_1)
        images_loaded.append(pil_1)
        image_paths.append(path_1)
    except Exception as e:
        st.error(f"Failed to load image: {e}")

if uploaded_2:
    try:
        pil_2, path_2 = _load_uploaded_image(uploaded_2)
        images_loaded.append(pil_2)
        image_paths.append(path_2)
    except Exception as e:
        st.error(f"Failed to load second image: {e}")

# Image preview
if images_loaded:
    if len(images_loaded) == 1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(images_loaded[0], caption=uploaded_1.name, use_container_width=True)
    else:
        labels = {
            IMAGE_MODE_BITEMPORAL: ("Before (T1)", "After (T2)"),
            IMAGE_MODE_SAR_OPTICAL: ("Optical", "SAR"),
        }
        l1, l2 = labels.get(input_mode, ("Image 1", "Image 2"))
        c1, c2 = st.columns(2)
        with c1:
            st.image(images_loaded[0], caption=f"{l1} — {uploaded_1.name}", use_container_width=True)
        with c2:
            st.image(images_loaded[1], caption=f"{l2} — {uploaded_2.name}", use_container_width=True)


# ── Query Input ─────────────────────────────────────────────────────────────

st.markdown("---")
examples = EXAMPLE_QUERIES.get(input_mode, EXAMPLE_QUERIES[IMAGE_MODE_SINGLE])

col_q, col_b = st.columns([5, 1])
with col_q:
    query = st.text_input("Ask a question about your satellite image",
                          placeholder="e.g., " + examples[0], key="query_input")
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("Analyze", use_container_width=True, type="primary")

# Quick questions
def _set_example(t): st.session_state["query_input"] = t

st.markdown("**SAR Quick Analysis:**")
cols = st.columns(min(len(examples), 3))
for i, ex in enumerate(examples[:3]):
    with cols[i]:
        st.button(ex, key=f"ex_{i}", use_container_width=True,
                  on_click=_set_example, args=(ex,))
if len(examples) > 3:
    cols2 = st.columns(min(len(examples) - 3, 3))
    for i, ex in enumerate(examples[3:6]):
        with cols2[i]:
            st.button(ex, key=f"ex2_{i}", use_container_width=True,
                      on_click=_set_example, args=(ex,))


# ── Processing ──────────────────────────────────────────────────────────────

def _validate():
    if not query or not query.strip():
        st.warning("Please enter a question.")
        return False
    if not images_loaded:
        st.warning("Please upload an image.")
        return False
    if input_mode in (IMAGE_MODE_BITEMPORAL, IMAGE_MODE_SAR_OPTICAL) and len(images_loaded) < 2:
        st.warning("This mode requires two images.")
        return False
    return True


if submit and _validate():
    with st.spinner("🛰️ Analyzing with Gemini..."):
        progress = st.progress(0, text="Sending to Gemini...")
        t_start = time.perf_counter()

        progress.progress(20, text="Gemini processing image...")
        result = route_query(
            query=query, images=image_paths,
            image_mode=input_mode, timeout=90,
        )
        elapsed = time.perf_counter() - t_start
        progress.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress.empty()

    # ── Handle errors ───────────────────────────────────────────────────
    if result["status"] == "error":
        st.markdown(f"""
        <div class="error-banner">
            <h4>Analysis Failed</h4>
            <p>{result.get('error', 'Unknown error')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.pop("current_analysis", None)
        st.session_state.pop("current_result", None)
        st.stop()

    analysis = result.get("analysis", {})
    if not analysis:
        st.error("Gemini returned an empty analysis. Please try again.")
        st.session_state.pop("current_analysis", None)
        st.session_state.pop("current_result", None)
        st.stop()

    result["elapsed_s"] = result.get("elapsed_s", elapsed)

    # Store in session for interactivity
    st.session_state["current_analysis"] = analysis
    st.session_state["current_result"] = result
    st.session_state["analyzed_images"] = images_loaded

analysis = st.session_state.get("current_analysis", {})
result = st.session_state.get("current_result", {})
analyzed_images = st.session_state.get("analyzed_images", images_loaded)

if analysis and result:
    st.markdown("---")
    model_used = result.get("model_used", "gemini")
    elapsed_s = result.get("elapsed_s", 0.0)

    # ── Metrics Row ─────────────────────────────────────────────────────
    features = analysis.get("features", [])
    avg_conf = np.mean([f.get("confidence", 0) for f in features]) if features else 0

    mc = st.columns(4)
    with mc[0]:
        st.markdown(f'<div class="stat-box"><div class="stat-value" style="font-size:1rem;">{model_used}</div><div class="stat-label">Gemini Model</div></div>', unsafe_allow_html=True)
    with mc[1]:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{elapsed_s:.1f}s</div><div class="stat-label">Response Time</div></div>', unsafe_allow_html=True)
    with mc[2]:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{len(features)}</div><div class="stat-label">Features Detected</div></div>', unsafe_allow_html=True)
    with mc[3]:
        c = _conf_color(avg_conf)
        st.markdown(f'<div class="stat-box"><div class="stat-value" style="color:{c}">{avg_conf:.0%}</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)

    # ── Side-by-Side: Original vs Annotated ─────────────────────────────
    if features and analyzed_images:
        st.markdown("#### 🖼️ Image Analysis")

        # Annotation controls
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1: show_boxes = st.checkbox("Show Boxes", value=True)
        with ac2: show_masks = st.checkbox("Show Masks", value=True)
        with ac3: show_labels = st.checkbox("Show Labels", value=True)
        with ac4:
            highlight_id = st.selectbox(
                "Highlight Feature",
                options=["None"] + [f.get("id", f"f{i}") for i, f in enumerate(features)],
                format_func=lambda x: x if x == "None" else next(
                    (f.get("label", x) for f in features if f.get("id") == x), x
                ),
            )
            if highlight_id == "None":
                highlight_id = None

        annotated = render_annotations(
            analyzed_images[0], features,
            show_boxes=show_boxes, show_masks=show_masks,
            show_labels=show_labels, highlight_id=highlight_id,
        )

        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown('<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
            st.markdown('<div style="color:#64748B;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">Original</div>', unsafe_allow_html=True)
            st.image(analyzed_images[0], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ann:
            st.markdown('<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
            st.markdown('<div style="color:#22D3EE;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">AI Annotated</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download annotated PNG
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "📥 Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"satquery_annotated_{ts}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Direct Answer ───────────────────────────────────────────────────
    direct = analysis.get("direct_answer", "")
    summary = analysis.get("summary", "")
    if direct:
        st.markdown(f"""
        <div class="analysis-section">
            <h3>Direct Answer</h3>
            <p>{direct}</p>
        </div>
        """, unsafe_allow_html=True)

    if summary and summary != direct:
        st.markdown(f"""
        <div class="analysis-section" style="border-color: rgba(16,185,129,0.2);">
            <h3>Summary</h3>
            <p>{summary}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed Analysis ───────────────────────────────────────────────
    detailed = analysis.get("detailed_analysis", "")
    if detailed:
        with st.expander("📋 Detailed Analysis", expanded=True):
            st.markdown(detailed)

    # ── Feature Cards ───────────────────────────────────────────────────
    if features:
        st.markdown("#### 🔍 Detected Features")

        for i, feat in enumerate(features):
            fid = feat.get("id", f"feature_{i}")
            label = feat.get("label", "Unknown")
            category = feat.get("category", "")
            status = feat.get("status", "observed")
            conf = feat.get("confidence", 0)
            conf_reason = feat.get("confidence_reason", "")
            evidence = feat.get("evidence", "")
            interpretation = feat.get("interpretation", "")
            has_box = bool(feat.get("box_2d"))

            color = _conf_color(conf)
            conf_pct = int(conf * 100)

            st.markdown(f"""
            <div class="feature-card" id="{fid}">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.3rem;">
                    <div>
                        <span class="fc-label">{label}</span>
                        <span class="fc-category" style="margin-left:0.5rem;">{category}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        {_status_badge(status)}
                        <span style="color:{color};font-weight:700;font-size:0.85rem;">{conf:.0%} {_conf_label(conf)}</span>
                        {'<span style="color:#22D3EE;font-size:0.65rem;">📍 Located</span>' if has_box else ''}
                    </div>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf_pct}%;background:{color};"></div>
                </div>
                <div class="fc-evidence" style="margin-top:0.4rem;">
                    <strong style="color:#64748B;">Evidence:</strong> {evidence}
                </div>
                {f'<div class="fc-evidence"><strong style="color:#64748B;">Interpretation:</strong> {interpretation}</div>' if interpretation else ''}
                {f'<div style="color:#475569;font-size:0.72rem;margin-top:0.2rem;">Confidence reason: {conf_reason}</div>' if conf_reason else ''}
            </div>
            """, unsafe_allow_html=True)

    # ── Observations ────────────────────────────────────────────────────
    observations = analysis.get("observations", [])
    if observations:
        with st.expander("👁️ Visual Observations"):
            for obs in observations:
                conf = obs.get("confidence", 0)
                c = _conf_color(conf)
                st.markdown(f"""
                <div style="display:flex;gap:0.6rem;align-items:flex-start;padding:0.4rem 0;border-bottom:1px solid rgba(71,85,105,0.2);">
                    <span style="color:{c};font-weight:700;font-size:0.82rem;min-width:3rem;">{conf:.0%}</span>
                    <span style="color:#CBD5E1;font-size:0.88rem;">{obs.get('text','')}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Land Cover ──────────────────────────────────────────────────────
    land_cover = analysis.get("land_cover", [])
    if land_cover:
        with st.expander("🗺️ Land Cover Classification"):
            for lc in land_cover:
                conf = lc.get("confidence", 0)
                c = _conf_color(conf)
                st.markdown(f"""
                <div style="padding:0.5rem 0;border-bottom:1px solid rgba(71,85,105,0.2);">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#E2E8F0;font-weight:600;">{lc.get('class_name','')}</span>
                        <span style="color:{c};font-weight:700;">{conf:.0%}</span>
                    </div>
                    <div style="color:#94A3B8;font-size:0.82rem;">{lc.get('evidence','')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────
    chart_data = analysis.get("chart_data", [])
    if chart_data:
        st.markdown("#### 📊 Analysis Charts")
        for chart in chart_data:
            labels = chart.get("labels", [])
            values = chart.get("values", [])
            title = chart.get("title", "Chart")
            chart_type = chart.get("chart_type", "bar")

            if not labels or not values or len(labels) != len(values):
                continue

            try:
                import plotly.graph_objects as go

                if chart_type == "pie":
                    fig = go.Figure(data=[go.Pie(
                        labels=labels, values=values,
                        marker=dict(colors=['#06B6D4', '#10B981', '#F59E0B', '#EF4444',
                                           '#8B5CF6', '#EC4899', '#14B8A6', '#F97316']),
                        textinfo='label+percent',
                    )])
                else:
                    fig = go.Figure(data=[go.Bar(
                        x=labels, y=values,
                        marker_color='#06B6D4',
                    )])

                fig.update_layout(
                    title=dict(text=title, font=dict(color='#E2E8F0', size=14)),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(17,24,39,0.6)',
                    font=dict(color='#94A3B8'),
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart rendering failed: {e}")

    elif not chart_data:
        pass  # Don't show "unavailable" — only show charts when data exists

    # ── Statistics ──────────────────────────────────────────────────────
    statistics = analysis.get("statistics", [])
    if statistics:
        st.markdown("#### 📏 Statistics")
        stat_cols = st.columns(min(len(statistics), 4))
        for i, stat in enumerate(statistics[:4]):
            with stat_cols[i]:
                val = stat.get("value", "N/A")
                unit = stat.get("unit", "")
                display_val = f"{val}{' ' + unit if unit else ''}" if val != "N/A" else "N/A"
                st.markdown(f'<div class="stat-box"><div class="stat-value" style="font-size:1rem;">{display_val}</div><div class="stat-label">{stat.get("label","")}</div></div>', unsafe_allow_html=True)

    # ── CSV Download ────────────────────────────────────────────────────
    csv_rows = analysis.get("csv_rows", [])
    if csv_rows:
        st.markdown("#### 📋 Export Data")
        # Show preview
        df = pd.DataFrame(csv_rows)
        st.dataframe(df, use_container_width=True, height=200)

        # Generate CSV
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "📥 Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"satquery_analysis_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Limitations ─────────────────────────────────────────────────────
    limitations = analysis.get("limitations", [])
    if limitations:
        with st.expander("⚠️ Limitations & Caveats"):
            for lim in limitations:
                st.markdown(f'<div class="limitation-item">• {lim}</div>', unsafe_allow_html=True)

    # ── Raw Debug ───────────────────────────────────────────────────────
    with st.expander("🔧 Debug / Raw Gemini Output"):
        st.json(analysis)

elif not analysis:
    # ── Welcome state ───────────────────────────────────────────────────
    if not images_loaded:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0; color:#475569;">
            <div style="font-size:3rem; margin-bottom:0.4rem; opacity:0.8;">🛰️</div>
            <h3 style="color:#22D3EE; font-weight:600; font-size:1.1rem;">Upload Satellite Image to Begin</h3>
            <p style="max-width:480px; margin:0 auto; line-height:1.5; font-size:0.85rem;">
                Upload GeoTIFF / PNG / JPG satellite or SAR imagery using the sidebar.<br>
                Every analysis is powered by Google Gemini with evidence-based reasoning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        fc = st.columns(4)
        feats = [
            ("🔍", "Visual QA", "Ask any question about satellite/SAR imagery with evidence-based answers"),
            ("🗺️", "Land Cover", "Identify land-cover patterns with calibrated confidence and evidence"),
            ("📍", "Feature Detection", "Detect and locate features with real bounding boxes from Gemini"),
            ("📊", "Export & Charts", "Download annotated images, CSV data, and analysis charts"),
        ]
        for col, (icon, title, desc) in zip(fc, feats):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; min-height:140px;">
                    <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
                    <div style="font-weight:600; color:#E2E8F0; font-size:0.82rem;">{title}</div>
                    <div style="font-size:0.72rem; color:#64748B; line-height:1.4; margin-top:0.2rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:0.5rem 0; color:#334155; font-size:0.7rem; font-family:'JetBrains Mono',monospace;">
    SatQuery AI v3.0 &middot; SIH 2026 &middot; ISRO Problem Statement 26167<br>
    Powered exclusively by Google Gemini &middot; Evidence-Based SAR Analysis
</div>
""", unsafe_allow_html=True)

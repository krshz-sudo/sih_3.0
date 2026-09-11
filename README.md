# SatQuery AI — Vision-Language SAR & Satellite Image Analysis

> **SIH 2026 / ISRO Problem Statement 26167 Solution**  
> An evidence-based, visual-grounding AI application for Synthetic Aperture Radar (SAR) and optical satellite imagery interpretation, powered exclusively by **Google Gemini 3.6 Flash Multimodal VLM**.

---

## 📌 Project Overview

**SatQuery AI** transforms complex satellite and SAR imagery into structured, evidence-grounded insights. Built specifically to handle radar backscatter characteristics, land-cover classification, change detection, and spatial feature localization, the system guarantees high-precision output with zero hallucinated locations or arbitrary facts.

### 🌟 Key Capabilities & Functional Highlights

1. **Pure Gemini 3.6 Flash Integration**
   - Single unified AI engine handling image reasoning, Visual QA, object localization, land-cover estimation, and structured JSON output.
   - **No Ollama, CNNs, or secondary AI fallbacks** required.

2. **Evidence-Based Interpretation Guardrails**
   - Distinguishes strictly between **Direct Visual Evidence** and **Inferred Interpretation**.
   - Every detected feature carries an explicit status (`observed`, `inferred`, or `uncertain`) along with a calibrated numeric confidence score (`0.00` – `1.00`).
   - Prevents inventing city names, river designations, acquisition dates, or unverified sensor metadata.

3. **Visual Grounding & Automatic Bounding Boxes**
   - Automatically parses normalized coordinates (`0–1000`) returned by Gemini.
   - Draws clear category-coded bounding boxes and labels onto the image (via PIL/OpenCV).

4. **Side-by-Side Verification Interface**
   - Interactive toggle to compare the original raw satellite image side-by-side with the annotated feature detection layer.

5. **Quantitative Analytics & Plotly Charts**
   - Dynamic charts rendering land-cover percentages, confidence distributions, and spatial metrics directly from Gemini's structured output.

6. **One-Click Export Capabilities**
   - Download high-resolution annotated satellite maps in PNG format.
   - Export full feature extraction logs, evidence, and confidence ratings as CSV reports.

7. **Multi-Modal Operational Modes**
   - **Single Image Analysis**: Deep inspection of SAR backscatter, texture, geometry, and land cover.
   - **Bi-Temporal Change Detection**: Before/after image analysis identifying structural or environmental changes.
   - **Optical + SAR Fusion**: Cross-modal comparative evaluation leveraging complementary sensor characteristics.

8. **SAR Quick Analysis Presets**
   - Instant query shortcuts for common remote-sensing tasks:
     - *Identify major SAR-visible features*
     - *Describe backscatter and texture patterns*
     - *Identify possible built-up areas*
     - *Identify water-like low-backscatter regions*
     - *Identify major linear infrastructure*
     - *Describe dominant land-cover patterns*

---

## 🏗️ Architecture & Data Flow

```
┌─────────────────┐       ┌─────────────────┐       ┌────────────────────────┐
│  User Upload &  │  ───> │   Router Module │  ───> │  Gemini Client         │
│  Query (UI)     │       │  (router.py)    │       │  (gemini_client.py)    │
└─────────────────┘       └─────────────────┘       └────────────────────────┘
                                                                │
                                                        Google Gemini VLM
                                                      (gemini-3.6-flash API)
                                                                │
                                                                ▼
┌─────────────────┐       ┌─────────────────┐       ┌────────────────────────┐
│  Streamlit UI   │  <─── │ Plotly Charts & │  <─── │  Structured JSON       │
│  Display        │       │ Annotation Render│      │  Analysis Output       │
└─────────────────┘       └─────────────────┘       └────────────────────────┘
```

---

## 📁 Repository Structure

```
SIH/Demo/
├── .env                       # Environment variables (Gemini API Key & Model)
├── requirements.txt           # Required Python packages
├── README.md                  # System documentation & setup guide
├── gui/
│   └── app.py                 # Streamlit UI dashboard
├── router/
│   └── router.py              # Query routing layer to Gemini engine
├── tools/
│   ├── gemini_client.py       # Sole AI inference client (System prompt + JSON schema)
│   ├── annotation_renderer.py # Bounding box & mask visual grounding renderer
│   ├── geotiff_utils.py       # GeoTIFF / TIFF satellite image loader
│   ├── change/                # Bi-temporal change detection tool wrapper
│   └── sar_fusion/            # Optical + SAR fusion tool wrapper
└── data/                      # Sample satellite imagery & test datasets
```

---

## ⚡ Quick Start & Installation Guide

### Prerequisites
- Python **3.10** or higher
- A valid **Google Gemini API Key**

### 1. Clone the Repository
```bash
git clone https://github.com/krshz-sudo/sih_2.0.git
cd sih_2.0
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create or edit the `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

### 4. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run gui/app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🛠️ Tech Stack

- **Frontend / Dashboard**: Streamlit, Plotly
- **Multimodal AI Engine**: Google Gemini API (`gemini-3.6-flash`)
- **Image Processing & Visual Grounding**: Pillow, OpenCV, NumPy, GeoTIFF / GDAL Utilities
- **Data Export**: Pandas, CSV, Pillow JPEG/PNG Encoders

---

## 📊 Summary of Recent Updates

- **Ollama Deprecation**: Completely removed local Ollama dependencies for faster, consistent cloud inference.
- **Model Upgrade**: Switched to `gemini-3.6-flash` for guaranteed structured JSON response compliance and multi-token visual reasoning.
- **Structured Schema Enforcer**: Gemini now returns strict `ANALYSIS_SCHEMA` including observations, features, land cover, statistics, CSV rows, and limitations.
- **Improved Grounding**: Bounding boxes scale accurately to high-resolution satellite inputs.

---

## 🤝 Contribution & License

Developed for **Smart India Hackathon (SIH 2026)** — Problem Statement 26167 (ISRO).  
Maintained by the SatQuery AI Team.

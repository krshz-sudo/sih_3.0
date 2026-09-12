<div align="center">
  <img src="assets/banner.svg" alt="SatQuery AI Banner" width="100%" />

  <h1>SatQuery AI v3.0</h1>
  <p><strong>Evidence-based, visual-grounding AI application for Synthetic Aperture Radar (SAR) and optical satellite imagery.</strong></p>
  
  <p>
    <a href="https://satquery-ai-v3.vercel.app"><img src="https://img.shields.io/badge/Live_Auth_Portal-Vercel-black?style=for-the-badge&logo=vercel" alt="Vercel App" /></a>
    <a href="https://krshz-sudo-sih-3-0-guiapp-0esz5j.streamlit.app/"><img src="https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit App" /></a>
  </p>

  <p><em>Developed for SIH 2026 | ISRO Problem Statement 26167</em></p>
</div>

---

## 📌 Project Overview

**SatQuery AI** transforms complex satellite and SAR imagery into structured, evidence-grounded insights. By leveraging the advanced multimodal capabilities of **Google Gemini 3.6 Flash**, the platform delivers high-precision output with zero hallucinated locations or arbitrary facts. 

Version 3.0 introduces a premium, cinematic **Vercel-hosted Landing & Authentication portal** seamlessly integrated with our powerful **Streamlit-hosted AI Engine**.

<div align="center">
  <img src="assets/gui-mockup.svg" alt="GUI Mockup" width="80%" />
</div>

---

## 🌟 Key Capabilities & Functional Highlights

1. **Dual-Node Architecture**
   - **Auth Portal (Vercel)**: A cinematic, ultra-responsive React/Vite frontend secured by **Supabase OAuth** (Google Login).
   - **AI Dashboard (Streamlit)**: Python-based visual grounding engine that processes user queries and satellite images.
2. **Pure Gemini 3.6 Flash Integration**
   - Single unified AI engine handling image reasoning, Visual QA, object localization, land-cover estimation, and structured JSON output.
   - **No Ollama, CNNs, or secondary AI fallbacks** required.
3. **Evidence-Based Interpretation Guardrails**
   - Distinguishes strictly between **Direct Visual Evidence** and **Inferred Interpretation**.
   - Every detected feature carries an explicit status (`observed`, `inferred`, or `uncertain`) along with a calibrated numeric confidence score (`0.00` – `1.00`).
4. **Visual Grounding & Automatic Bounding Boxes**
   - Automatically parses normalized coordinates (`0–1000`) returned by Gemini.
   - Draws clear category-coded bounding boxes and labels onto the image.
5. **Multi-Modal Operational Modes**
   - **Single Image Analysis**: Deep inspection of SAR backscatter, texture, geometry, and land cover.
   - **Bi-Temporal Change Detection**: Before/after image analysis identifying structural or environmental changes.
   - **Optical + SAR Fusion**: Cross-modal comparative evaluation leveraging complementary sensor characteristics.

---

## 🏗️ Architecture & Data Flow

Our platform is split into two primary environments to ensure high-performance UI and heavy-duty data processing are handled optimally.

<div align="center">
  <img src="assets/architecture.svg" alt="System Architecture" width="80%" />
</div>

### 1. Frontend: Auth & Landing (Vercel)
The entry point to the application is a lightweight Vite frontend that handles user authentication.
- **Tech Stack**: HTML, CSS, JavaScript, Vite, Supabase.
- **Flow**: User lands on the cinematic video background page → Clicks "Continue with Google" → Authenticates via Supabase → Redirects securely to the Streamlit Dashboard.

### 2. Backend: AI Engine (Streamlit)
The core analysis engine where satellite imagery is uploaded and processed.
- **Tech Stack**: Streamlit, Plotly, Pillow, OpenCV, Google Gemini API.
- **Flow**: User uploads GeoTIFF/PNG/JPG → Selects Analysis Mode (Single, Bi-temporal, Fusion) → Gemini API processes the image → Structured JSON is parsed into charts, maps, and CSVs.

---

## ⚡ Quick Start & Local Setup

### Option 1: Running the AI Engine (Streamlit)
```bash
# 1. Clone the repository
git clone https://github.com/krshz-sudo/sih_3.0.git
cd sih_3.0

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Environment Variables
# Create a .env file in the root directory:
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
echo "GEMINI_MODEL=gemini-3.6-flash" >> .env

# 5. Launch the Streamlit dashboard
streamlit run gui/app.py
```
*Open your browser and navigate to: `http://localhost:8501`*

### Option 2: Running the Auth Portal (Vite)
```bash
# 1. Navigate to the web directory
cd web

# 2. Install dependencies
npm install

# 3. Run the development server
npm run dev
```
*Open your browser and navigate to: `http://localhost:5173`*

---

## 📱 Mobile Responsiveness

Version 3.0 has been meticulously optimized for both Desktop and Mobile experiences.
- The **Auth Portal** features dynamic media queries, seamless scrolling, and touch-optimized buttons that guarantee zero overlapping or clipped elements on mobile displays.
- The **Streamlit Dashboard** automatically collapses sidebars and restructures Plotly charts to fit small viewports natively.

---

## 🤝 Contribution & License

Developed for **Smart India Hackathon (SIH 2026)** — Problem Statement 26167 (ISRO).  
Maintained by the **SatQuery AI Team**.

<p align="center">
  <sub>Built with ❤️ and 🛰️</sub>
</p>

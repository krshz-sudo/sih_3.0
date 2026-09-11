# SatQuery AI — Design System & Visual Specification (DESIGN.md)

> **Design Read**: Vision-Language AI assistant for multimodal remote sensing image analysis (ISRO PS-26167 / SIH 2026).
> **Aesthetic Family**: Deep Dark Tech OLED + Scientific Precision + Glassmorphic Materiality.
> **Dials**: `DESIGN_VARIANCE: 7` | `MOTION_INTENSITY: 6` | `VISUAL_DENSITY: 4`

---

## 1. Visual Theme & Atmosphere

- **Theme & Palette Strategy**: Dark OLED base (`#0B0F19` -> `#020617`) with vibrant cyan (`#06B6D4` / `#22D3EE`) accents, emerald (`#10B981` / `#34D399`) high-confidence signals, amber (`#F59E0B` / `#FBBF24`) warnings, and rose (`#F43F5E` / `#FB7185`) low-confidence badges.
- **Materiality**: Frosted glassmorphism (`backdrop-filter: blur(16px)`) with specular top-edge highlights (`inset 0 1px 0 rgba(255, 255, 255, 0.08)`).
- **Typography Tone**: Clean scientific readability using **Outfit** for headers/body and **JetBrains Mono** for coordinates, task metadata, confidence scores, and raw model outputs.

---

## 2. Color Palette & CSS Variables

```css
:root {
  /* Surface Layers */
  --bg-app: #0B0F19;
  --bg-surface-dark: #0F172A;
  --bg-card: rgba(17, 24, 39, 0.75);
  --bg-card-hover: rgba(30, 41, 59, 0.85);

  /* Accents & Brand */
  --accent-cyan: #06B6D4;
  --accent-cyan-light: #22D3EE;
  --accent-emerald: #10B981;
  --accent-emerald-light: #34D399;
  --accent-amber: #F59E0B;
  --accent-purple: #8B5CF6;

  /* Borders & Glass Dividers */
  --border-glass: rgba(6, 182, 212, 0.18);
  --border-glass-hover: rgba(34, 211, 238, 0.35);

  /* Text Hierarchy */
  --text-primary: #F8FAFC;
  --text-secondary: #94A3B8;
  --text-muted: #64748B;
  --text-dim: #475569;

  /* Status Colors */
  --status-high-bg: rgba(16, 185, 129, 0.18);
  --status-high-border: rgba(16, 185, 129, 0.3);
  --status-medium-bg: rgba(245, 158, 11, 0.18);
  --status-medium-border: rgba(245, 158, 11, 0.3);
  --status-low-bg: rgba(244, 63, 94, 0.18);
  --status-low-border: rgba(244, 63, 94, 0.3);
}
```

---

## 3. Typography Rules

- **Display & Hero Titles**: `Outfit`, Weight 800, `font-size: 2.2rem`, `letter-spacing: -0.02em`, Cyan-to-Emerald gradient text.
- **Section Headers & Subtitles**: `Outfit`, Weight 600, `font-size: 1.1rem`, color `#22D3EE`.
- **Body & Paragraphs**: `Outfit`, Weight 400, `font-size: 0.95rem`, line height `1.7`, color `#F1F5F9`.
- **Code & Metadata**: `JetBrains Mono`, Weight 400/500, `font-size: 0.82rem`, color `#94A3B8`.
- **Google Fonts Import**:
  ```html
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">
  ```

---

## 4. Component Stylings

### 4.1 Primary CTA Buttons
- **Default**: `background: linear-gradient(135deg, #06B6D4, #0891B2)`, `border-radius: 10px`, `box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3)`, `color: #FFFFFF`.
- **Hover**: `background: linear-gradient(135deg, #22D3EE, #06B6D4)`, `transform: translateY(-1px)`, `box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45)`.
- **Active**: `transform: scale(0.98)`.
- **Disabled**: `opacity: 0.5`, `cursor: not-allowed`.

### 4.2 Glass Cards & Containers
- `background: rgba(17, 24, 39, 0.75)`
- `backdrop-filter: blur(16px)`
- `border: 1px solid rgba(6, 182, 212, 0.15)`
- `box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 8px 32px rgba(0, 0, 0, 0.3)`

### 4.3 Task Badges
- **VQA**: Background `rgba(6, 182, 212, 0.15)`, text `#22D3EE`, border `1px solid rgba(6,182,212,0.25)`.
- **Classification**: Background `rgba(16, 185, 129, 0.15)`, text `#34D399`, border `1px solid rgba(16,185,129,0.25)`.
- **Change Detection**: Background `rgba(245, 158, 11, 0.15)`, text `#FBBF24`, border `1px solid rgba(245,158,11,0.25)`.
- **SAR Fusion**: Background `rgba(139, 92, 246, 0.15)`, text `#A78BFA`, border `1px solid rgba(139,92,246,0.25)`.

### 4.4 Model Tier Badges
- **Primary Gemini**: Color `#34D399`, background `rgba(16, 185, 129, 0.08)`.
- **Local Fallback**: Color `#FBBF24`, background `rgba(245, 158, 11, 0.08)`.

---

## 5. Layout Principles

- **Maximum Container Width**: `1400px` centered with margin `0 auto`.
- **Viewport Height**: `min-h-[100dvh]` to prevent jumpy layout on mobile address bar.
- **Hero Section Stack**:
  1. Header Icon & Badge
  2. Main Title (`SatQuery AI`)
  3. Subtitle ("Interactive Vision-Language Assistant for Remote Sensing Image Analysis")
- **Multi-Image Split Layout**: 50/50 two-column glass grid for bi-temporal and optical+SAR image pairs.

---

## 6. Depth & Elevation

- **Level 0**: Base canvas `#0B0F19`.
- **Level 1**: Sidebar & glass card background `rgba(17, 24, 39, 0.75)`.
- **Level 2**: Answer box highlight `linear-gradient(135deg, rgba(6,182,212,0.1), rgba(16,185,129,0.05))`.
- **Level 3**: Hover cards with `translateY(-1px)` lift and elevated specular border.

---

## 7. Animation & Interaction

- **Hover Transitions**: `transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1)`.
- **Pulse Indicators**: `@keyframes pulse-dot` (2s infinite ease-in-out) for active server connectivity.
- **Tactile Button Press**: `:active { transform: scale(0.98); }`.
- **Accessibility**: Reduced motion fallback support via standard CSS.

---

## 8. Do's and Don'ts (Design Guardrails)

1. **DO** use SVG icons from official icon sets (Phosphor, Radix, Lucide).
2. **DON'T** use emoji characters as functional action icons.
3. **DO** maintain `cursor: pointer` on all interactive buttons, chips, and upload triggers.
4. **DON'T** hardcode text colors without verifying WCAG AA contrast (minimum 4.5:1 ratio).
5. **DO** display exact model tier (`Gemini 3.6 Flash` vs `Local Fallback`) on every response.
6. **DON'T** wrap button text across multiple lines.
7. **DO** maintain dark-mode consistency across all page components.
8. **DON'T** allow non-responsive side scrolling on screen widths down to 375px.

---

## 9. Responsive Behavior

- **Breakpoints**:
  - `sm`: 640px
  - `md`: 768px (column stacking triggered)
  - `lg`: 1024px
  - `xl`: 1280px
- **Touch Target**: Minimum `44px x 44px` on all buttons and upload areas for mobile devices.

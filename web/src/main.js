/**
 * SatQuery AI — Main Application Bootstrap
 * Wires SceneManager, loading screen, HUD, sidebar, and interactions.
 */
import './style.css';
import { SceneManager } from './scene/SceneManager.js';
import { checkSession, handleLogout } from './auth/session.js';

// ── DOM References ─────────────────────────────────────────
const loadingScreen = document.getElementById('loading-screen');
const loadingBarFill = document.getElementById('loading-bar-fill');
const nav = document.getElementById('nav');
const hudTelemetry = document.getElementById('hud-telemetry');
const hudControls = document.getElementById('hud-controls');
const sidebar = document.getElementById('sidebar');
const hero = document.getElementById('hero');
const bottomBar = document.getElementById('bottom-bar');
const satViewCard = document.getElementById('sat-view-card');
const canvasContainer = document.getElementById('canvas-container');
const miniCanvas = document.getElementById('mini-canvas');

// Loading step elements
const steps = {
  scene:     document.querySelector('[data-step="scene"]'),
  satellite: document.querySelector('[data-step="satellite"]'),
  moon:      document.querySelector('[data-step="moon"]'),
  ready:     document.querySelector('[data-step="ready"]'),
};

// HUD dynamic elements
const hudLatlon = document.getElementById('hud-latlon');

// ── Loading Screen Logic ───────────────────────────────────
const stepOrder = ['scene', 'satellite', 'moon', 'ready'];

function updateLoadingStep(stepName, progress) {
  loadingBarFill.style.width = `${progress}%`;
  const idx = stepOrder.indexOf(stepName);
  for (let i = 0; i <= idx; i++) {
    const key = stepOrder[i];
    const el = steps[key];
    if (!el) continue;
    if (i < idx) {
      el.classList.remove('active');
      el.classList.add('done');
      el.querySelector('.step-status').textContent = '✓';
    } else if (i === idx) {
      el.classList.add('active');
      el.querySelector('.step-status').textContent = '⏳';
      if (progress >= 100 && key === 'ready') {
        el.classList.remove('active');
        el.classList.add('done');
        el.querySelector('.step-status').textContent = '✓';
      }
    }
  }
}

function hideLoadingScreen() {
  for (const key of stepOrder) {
    const el = steps[key];
    if (el) {
      el.classList.remove('active');
      el.classList.add('done');
      el.querySelector('.step-status').textContent = '✓';
    }
  }
  loadingBarFill.style.width = '100%';

  setTimeout(() => {
    loadingScreen.classList.add('fade-out');
    setTimeout(() => nav.classList.remove('hidden'), 200);
    setTimeout(() => hudTelemetry.classList.remove('hidden'), 350);
    setTimeout(() => hudControls.classList.remove('hidden'), 450);
    setTimeout(() => sidebar.classList.remove('hidden'), 500);
    setTimeout(() => hero.classList.remove('hidden'), 650);
    setTimeout(() => bottomBar.classList.remove('hidden'), 800);
    setTimeout(() => satViewCard.classList.remove('hidden'), 900);
    setTimeout(() => { loadingScreen.style.display = 'none'; }, 1200);
  }, 400);
}

// ── HUD Telemetry Updates ──────────────────────────────────
function startHUD(sceneManager) {
  setInterval(() => {
    // Simulated lat/lon drift (decorative)
    const t = performance.now() * 0.0001;
    if (hudLatlon) {
      const lat = (28.6 + Math.sin(t * 0.7) * 15).toFixed(3);
      const lon = (77.2 + Math.cos(t * 0.5) * 30).toFixed(3);
      hudLatlon.textContent = `${lat} / ${lon}`;
    }
  }, 300);
}

// ── Sidebar Interaction ────────────────────────────────────
function setupSidebar(sceneManager) {
  const btns = document.querySelectorAll('.sidebar-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active from all
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const target = btn.dataset.target;
      switch (target) {
        case 'satellite':
          sceneManager.focusSatellite();
          break;
        case 'moon':
          sceneManager.focusMoon();
          break;
        case 'focus':
          sceneManager.resetView();
          break;
        case 'layers':
          // Toggle wireframe or other layer mode — future
          break;
      }
    });
  });
}

// ── Initialize ─────────────────────────────────────────────
async function init() {
  const sceneManager = new SceneManager(canvasContainer);

  sceneManager.onProgress = (step, pct) => {
    updateLoadingStep(step, pct);
  };

  sceneManager.onReady = () => {
    hideLoadingScreen();
    startHUD(sceneManager);
    setupSidebar(sceneManager);

    // Render mini view
    if (miniCanvas) {
      sceneManager.renderMiniView(miniCanvas);
    }
  };

  sceneManager.start();
  await sceneManager.loadAssets();

  // ── Auth Handling ─────────────────────────────────────────
  const session = await checkSession();
  const launchBtns = document.querySelectorAll('#launch-btn, #hero-launch-btn');
  
  if (session) {
    // Authenticated state
    launchBtns.forEach(btn => {
      btn.href = 'https://krshz-sudo-sih-3-0-guiapp-0esz5j.streamlit.app/';
    });
    
    // Add logout button to nav right
    const navRight = document.querySelector('.nav-right');
    const logoutBtn = document.createElement('button');
    logoutBtn.className = 'nav-cta';
    logoutBtn.style.marginLeft = '12px';
    logoutBtn.innerHTML = 'LOGOUT <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>';
    logoutBtn.addEventListener('click', handleLogout('/index.html'));
    navRight.appendChild(logoutBtn);
    
  } else {
    // Unauthenticated state
    launchBtns.forEach(btn => {
      btn.href = '/login.html';
      btn.target = '_self'; // Open login in same tab
    });
  }
}

init().catch((err) => {
  console.error('SatQuery AI initialization failed:', err);
  hideLoadingScreen();
});

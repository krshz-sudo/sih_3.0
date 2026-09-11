/**
 * SceneManager — Three.js scene orchestrator
 * Uses SPACE.jpg as equirect environment background,
 * loads real satellite and moon GLB models,
 * cinematic camera with orbit interaction.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer, RenderPass, EffectPass, BloomEffect, VignetteEffect, BlendFunction } from 'postprocessing';

export class SceneManager {
  constructor(container) {
    this.container = container;
    this._startTime = performance.now() * 0.001;
    this._prevTime = this._startTime;
    this.isReady = false;
    this.satellite = null;
    this.moon = null;
    this.frameCount = 0;
    this.lastFpsTime = 0;
    this.fps = 60;
    this.onProgress = null;
    this.onReady = null;

    // Interaction state
    this._hoveredObject = null;
    this._focusTarget = null; // 'satellite' | 'moon' | null
    this._idleCamPos = new THREE.Vector3(2.5, 1.2, 4.0);
    this._idleCamTarget = new THREE.Vector3(0.3, 0.0, 0.0);
    this._lerpSpeed = 0.02;

    this._init();
  }

  _init() {
    const w = window.innerWidth;
    const h = window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();

    // Load SPACE.jpg as equirect environment
    const texLoader = new THREE.TextureLoader();
    texLoader.load('/SPACE.jpg', (tex) => {
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.colorSpace = THREE.SRGBColorSpace;
      this.scene.background = tex;
    });

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Camera — matching reference composition
    // Satellite center-left, Moon large on right
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 500);
    this.camera.position.copy(this._idleCamPos);
    this.camera.lookAt(this._idleCamTarget);

    // OrbitControls — cinematic defaults
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this._idleCamTarget);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.rotateSpeed = 0.4;
    this.controls.zoomSpeed = 0.6;
    this.controls.panSpeed = 0.3;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 20;
    this.controls.enablePan = true;
    this.controls.autoRotate = false;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.update();

    // Lighting — matching reference: strong sun from upper-left, rim light
    this._setupLighting();

    // Post-processing (no chromatic aberration)
    this._setupPostProcessing();

    // Raycaster for hover/click
    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();
    this._setupInteraction();

    // Resize
    this._onResize = this._handleResize.bind(this);
    window.addEventListener('resize', this._onResize);
  }

  _setupLighting() {
    // Key light — strong sun from upper-left (matching reference shadows)
    const sun = new THREE.DirectionalLight(0xfff8f0, 3.0);
    sun.position.set(-5, 6, 4);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.near = 0.1;
    sun.shadow.camera.far = 40;
    sun.shadow.camera.left = -6;
    sun.shadow.camera.right = 6;
    sun.shadow.camera.top = 6;
    sun.shadow.camera.bottom = -6;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun);

    // Fill hemisphere
    const hemi = new THREE.HemisphereLight(0x1a2040, 0x050508, 0.25);
    this.scene.add(hemi);

    // Rim light from behind-right (cold blue for edge definition)
    const rim = new THREE.DirectionalLight(0x6688bb, 0.5);
    rim.position.set(4, 2, -5);
    this.scene.add(rim);

    // Subtle ambient
    const ambient = new THREE.AmbientLight(0x0c1020, 0.15);
    this.scene.add(ambient);
  }

  _setupPostProcessing() {
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    const bloomEffect = new BloomEffect({
      intensity: 0.5,
      luminanceThreshold: 0.7,
      luminanceSmoothing: 0.3,
      mipmapBlur: true,
    });

    const vignetteEffect = new VignetteEffect({
      darkness: 0.4,
      offset: 0.3,
    });

    this.composer.addPass(new EffectPass(this.camera, bloomEffect, vignetteEffect));
  }

  _setupInteraction() {
    const canvas = this.renderer.domElement;

    canvas.addEventListener('mousemove', (e) => {
      this._mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      this._mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    canvas.addEventListener('click', (e) => {
      if (!this.isReady) return;
      this._raycaster.setFromCamera(this._mouse, this.camera);

      // Check satellite
      if (this.satellite) {
        const hits = this._raycaster.intersectObject(this.satellite, true);
        if (hits.length > 0) {
          this._focusOnSatellite();
          return;
        }
      }
      // Check moon
      if (this.moon) {
        const hits = this._raycaster.intersectObject(this.moon, true);
        if (hits.length > 0) {
          this._focusOnMoon();
          return;
        }
      }
    });

    canvas.addEventListener('dblclick', () => {
      if (this.satellite) {
        this._cinematicCloseFocus();
      }
    });

    // ESC to return to hero composition
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this._returnToHero();
      }
    });
  }

  _focusOnSatellite() {
    this._focusTarget = 'satellite';
    const targetPos = new THREE.Vector3(1.5, 0.5, 2.5);
    const targetLookAt = new THREE.Vector3(0, 0, 0);
    this._animateCamera(targetPos, targetLookAt);
  }

  _focusOnMoon() {
    this._focusTarget = 'moon';
    const moonPos = this.moon ? this.moon.position.clone() : new THREE.Vector3(6, 0, -4);
    const targetPos = moonPos.clone().add(new THREE.Vector3(-3, 1, 3));
    this._animateCamera(targetPos, moonPos);
  }

  _cinematicCloseFocus() {
    const targetPos = new THREE.Vector3(0.8, 0.3, 1.5);
    const targetLookAt = new THREE.Vector3(0, 0, 0);
    this._animateCamera(targetPos, targetLookAt);
  }

  _returnToHero() {
    this._focusTarget = null;
    this._animateCamera(this._idleCamPos.clone(), this._idleCamTarget.clone());
  }

  _animateCamera(targetPos, targetLookAt) {
    // Smoothly animate via controls target
    this._camDestPos = targetPos;
    this._camDestTarget = targetLookAt;
    this._isCameraAnimating = true;
  }

  async loadAssets() {
    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
    const gltfLoader = new GLTFLoader();
    gltfLoader.setDRACOLoader(dracoLoader);

    try {
      this._reportProgress('scene', 15);

      // Load satellite
      const satGltf = await new Promise((resolve, reject) => {
        gltfLoader.load('/models/satellite.glb', resolve, undefined, reject);
      });
      this.satellite = satGltf.scene;
      this.satellite.name = 'Satellite';
      // Scale and position matching reference: satellite center-left, slightly above center
      this.satellite.scale.setScalar(0.8);
      this.satellite.position.set(0, 0, 0);
      this.satellite.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      this.scene.add(this.satellite);
      this._reportProgress('satellite', 50);

      // Load moon
      const moonGltf = await new Promise((resolve, reject) => {
        gltfLoader.load('/models/moon.glb', resolve, undefined, reject);
      });
      this.moon = moonGltf.scene;
      this.moon.name = 'Moon';
      // Moon: large, right side, partially off-screen (matching reference)
      this.moon.scale.setScalar(6.0);
      this.moon.position.set(7, -1, -5);
      this.moon.traverse((child) => {
        if (child.isMesh) {
          child.receiveShadow = true;
        }
      });
      this.scene.add(this.moon);
      this._reportProgress('moon', 80);

      dracoLoader.dispose();
      await new Promise(r => setTimeout(r, 300));
      this._reportProgress('ready', 100);
      this.isReady = true;
      if (this.onReady) this.onReady();

    } catch (err) {
      console.error('Asset loading failed:', err);
      dracoLoader.dispose();
      this._reportProgress('ready', 100);
      this.isReady = true;
      if (this.onReady) this.onReady();
    }
  }

  _reportProgress(step, pct) {
    if (this.onProgress) this.onProgress(step, pct);
  }

  start() {
    this._animate();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    const now = performance.now() * 0.001;
    const delta = Math.min(now - this._prevTime, 0.1);
    const elapsed = now - this._startTime;
    this._prevTime = now;

    // FPS counter
    this.frameCount++;
    if (elapsed - this.lastFpsTime >= 1.0) {
      this.fps = this.frameCount / (elapsed - this.lastFpsTime);
      this.frameCount = 0;
      this.lastFpsTime = elapsed;
    }

    // Satellite animation: slow rotation + subtle float
    if (this.satellite) {
      this.satellite.rotation.y += 0.05 * delta;
      this.satellite.position.y = Math.sin(elapsed * 0.4) * 0.04;
      this.satellite.rotation.x = Math.sin(elapsed * 0.25) * 0.01;
      this.satellite.rotation.z = Math.cos(elapsed * 0.18) * 0.008;
    }

    // Moon: very slow rotation
    if (this.moon) {
      this.moon.rotation.y += 0.003 * delta;
    }

    // Smooth camera animation
    if (this._isCameraAnimating && this._camDestPos && this._camDestTarget) {
      this.camera.position.lerp(this._camDestPos, 0.03);
      this.controls.target.lerp(this._camDestTarget, 0.03);
      if (this.camera.position.distanceTo(this._camDestPos) < 0.01) {
        this._isCameraAnimating = false;
      }
    }

    // Hover detection
    if (this.isReady) {
      this._raycaster.setFromCamera(this._mouse, this.camera);
      let hovering = false;
      if (this.satellite) {
        const hits = this._raycaster.intersectObject(this.satellite, true);
        if (hits.length > 0) hovering = true;
      }
      if (!hovering && this.moon) {
        const hits = this._raycaster.intersectObject(this.moon, true);
        if (hits.length > 0) hovering = true;
      }
      this.container.classList.toggle('hovering-object', hovering);
    }

    this.controls.update();
    this.composer.render(delta);
  }

  // Get mini-view renderer for satellite card
  renderMiniView(canvas) {
    if (!this.satellite || !canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    // Just draw a dark frame — the mini renderer is complex; we'll use main renderer readback
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = '#080c16';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#1a2030';
    ctx.font = '8px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('LIVE FEED', w/2, h/2);
  }

  focusSatellite() { this._focusOnSatellite(); }
  focusMoon() { this._focusOnMoon(); }
  resetView() { this._returnToHero(); }

  _handleResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    this.controls.dispose();
    this.renderer.dispose();
    this.composer.dispose();
  }
}

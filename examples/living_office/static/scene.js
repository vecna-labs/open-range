// OpenRange — Living Office
// Interior office scene (not a town). Wood-floor, low walls, room labels,
// per-role desk grid auto-sized for N personas. No trees. No grass.
// Three.js r163 + OrbitControls, flat-shaded low-poly Sims look.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

window.__orSceneBoot = true;
function diag(step) { try { (window.__orDiag ||= {steps:[]}).steps.push(step); } catch {} console.log('[or]', step); }
diag('scene.js loaded · THREE=' + THREE.REVISION);
document.getElementById('scene-boot')?.remove();

// ---------- palette (warm, cream office) ----------
const PAL = {
  // Warm wood-and-carpet office (static base tones)
  floor:     0xdbb988,
  floorAlt:  0xc4a06e,
  wall:      0xf3e9cd,
  wallTrim:  0x8f6a3a,
  red:  0xa64040,
  blue: 0x3d7aad,
  ink:  0x1d1a14,
  skin: 0xe8c9a8,
  hair: 0x3d2818,
};

// Role aliases normalize synonyms: engineer/engineering, it/it_admin, soc/it_admin.
// Any role not in this map uses its own string verbatim for hashing + override lookup.
const ROLE_ALIASES = {
  engineering: 'engineer',
  it:          'it_admin',
  soc:         'it_admin',
  people:      'hr',
  r_and_d:     'engineer',
};
function canonicalRole(role) {
  const key = String(role || '').toLowerCase();
  return ROLE_ALIASES[key] || key;
}

// Hand-tuned palette for the roles we've seen. Anything missing falls through
// to the procedural warm-office sweep below.
const ROLE_OVERRIDES = {
  engineer:  { desk: 0x3d7aad, carpet: 0xbecbd6 },
  sales:     { desk: 0xc9881c, carpet: 0xedce78 },
  finance:   { desk: 0xa64040, carpet: 0xe5aba2 },
  it_admin:  { desk: 0x7a5bae, carpet: 0xccb5d4 },
  hr:        { desk: 0x4d8548, carpet: 0xb4cba8 },
  legal:     { desk: 0x8d7648, carpet: 0xd6c096 },
  ops:       { desk: 0x5e7c3d, carpet: 0xc3d19b },
  external:  { desk: 0x8f6a3a, carpet: 0xd6a878 },
  dmz:       { desk: 0xc79a18, carpet: 0xe8c77a },
  data:      { desk: 0x4f6d88, carpet: 0xa0bcd8 },
  management:{ desk: 0x7a5bae, carpet: 0xbfa8d8 },
};

// Deterministic string hash → 32-bit int (FNV-1a). Used to pick hues for
// unknown role names so the same manifest always gets the same palette.
function _hashString(s) {
  let h = 2166136261;
  const str = String(s || '');
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return Math.abs(h);
}

// HSL → RGB packed 0xRRGGBB. h in [0,1], s/l in [0,1].
function _hslHex(h, s, l) {
  const c = new THREE.Color();
  c.setHSL(h, s, l);
  return (Math.round(c.r * 255) << 16) | (Math.round(c.g * 255) << 8) | Math.round(c.b * 255);
}

// paletteForRole(role) → { desk, carpet } packed hex. Warm-office family only.
// Hues sweep the warm/gold/green/plum range (skip saturated blue-green and neon).
function paletteForRole(role) {
  const key = canonicalRole(role);
  if (ROLE_OVERRIDES[key]) return ROLE_OVERRIDES[key];
  // Warm sweep: 8 anchor hues across copper/gold/olive/sage/rose/plum/teal/brown.
  const warmHues = [0.05, 0.10, 0.13, 0.27, 0.33, 0.94, 0.75, 0.08];
  const hIdx = _hashString(key) % warmHues.length;
  const hue = warmHues[hIdx];
  return {
    desk:   _hslHex(hue, 0.45, 0.48),
    carpet: _hslHex(hue, 0.28, 0.82),
  };
}

// ---------- boot ----------
const canvas = document.getElementById('canvas');
const sceneEl = document.getElementById('scene');
const labelLayer = document.getElementById('labels');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'default' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0xf3e9cd, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 220);
camera.position.set(15, 23, 20);
camera.lookAt(0, 0, 4);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 4);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.minDistance = 12;
controls.maxDistance = 46;
controls.maxPolarAngle = Math.PI * 0.46;
controls.rotateSpeed = 0.6;

// ---------- lights (warm sunlit daylight) ----------
const hemi = new THREE.HemisphereLight(0xfff0cc, 0x8a7a55, 1.05);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff0c8, 1.3);
sun.position.set(16, 26, 12); sun.castShadow = true;
sun.shadow.mapSize.set(3072, 3072);
sun.shadow.camera.left = -24; sun.shadow.camera.right = 24;
sun.shadow.camera.top = 24; sun.shadow.camera.bottom = -14;
sun.shadow.bias = -0.00025;
sun.shadow.radius = 3.2;  // soft shadows
scene.add(sun);

// Window-side fill for ambient bounce
const fillN = new THREE.DirectionalLight(0xcfe3f5, 0.38);
fillN.position.set(-10, 14, -10);
scene.add(fillN);
// Warm-floor bounce (pretend-radiosity)
const fillS = new THREE.DirectionalLight(0xffd9a8, 0.22);
fillS.position.set(0, -2, 8);
scene.add(fillS);

// ---------- camera auto-fit ----------
// Reframe the orbit camera around whatever ROOM_PLAN we rendered. Called after
// computeRoomPlan(...) so the same code handles a 3-role snapshot and a
// 12-role snapshot without hardcoded positions.
function fitCameraToPlan(roomPlan) {
  const rects = Object.values(roomPlan || {}).filter(r => r && Number.isFinite(r.cx));
  if (!rects.length) return;
  let minX =  Infinity, maxX = -Infinity;
  let minZ =  Infinity, maxZ = -Infinity;
  for (const r of rects) {
    minX = Math.min(minX, r.cx - r.w / 2);
    maxX = Math.max(maxX, r.cx + r.w / 2);
    minZ = Math.min(minZ, r.cz - r.d / 2);
    maxZ = Math.max(maxZ, r.cz + r.d / 2);
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxZ)) return;
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;
  const dx = maxX - minX, dz = maxZ - minZ;
  // Clamp the diagonal so tiny scenes don't push the camera into the floor.
  const diag = Math.max(14, Math.sqrt(dx * dx + dz * dz));

  camera.position.set(
    cx + diag * 0.45,
    Math.max(14, diag * 0.85),
    cz + diag * 0.75,
  );
  const lookZ = cz + 2;
  camera.lookAt(cx, 0, lookZ);
  controls.target.set(cx, 0, lookZ);
  controls.minDistance = Math.max(10, diag * 0.35);
  controls.maxDistance = Math.max(30, diag * 1.6);
  controls.update();
}

// ---------- dynamic containers (per-scene_init rebuild) ----------
const npcs = new Map();
const services = new Map();
const agents = { red: null, blue: null };
const speechBubbles = new Map();
const attackLines = [];
const attackPackets = [];
const activityPulses = [];
let dynamicNodes = [];  // non-NPC scene objects that rebuild per layout

// Most-recent mutation_stages event keyed by world_id. Feeds the admission
// report panel. Entries are replaced in place, so the map stays small.
const lastMutationStages = new Map();

function clearDynamic() {
  for (const n of dynamicNodes) scene.remove(n);
  dynamicNodes = [];
}

// ---------- time-of-day ----------
let simTime = 0;
// Mutable so handleSceneInit can recompute from msg.tick_interval_s. Default
// 60 covers a 30-tick episode at 2s cadence; reassigned on scene_init.
let EP_HORIZON_TICKS = 60;
const DAY_START_HOURS = 9.0;
const DAY_SPAN_HOURS  = 4.0;
function timeOfDay(sim) {
  const t = Math.max(0, Math.min(1, sim / EP_HORIZON_TICKS));
  const hh = DAY_START_HOURS + t * DAY_SPAN_HOURS;
  const h = Math.floor(hh), m = Math.floor((hh - h) * 60);
  const phase = t < 0.18 ? 'morning'
              : t < 0.45 ? 'midmorning'
              : t < 0.55 ? 'noon'
              : t < 0.78 ? 'afternoon'
              : 'late afternoon';
  return { t, hh: h, mm: m, phase };
}
const SUN_A = [-12, 20, 14, 0xffd4a5, 0.6];
const SUN_B = [  6, 28, 12, 0xfff0c8, 1.2];
const SUN_C = [ 18, 24, 10, 0xffcf6e, 1.0];
function lerpColor(a, b, t) { return new THREE.Color(a).lerp(new THREE.Color(b), t); }
function updateDayCycle(sim) {
  const t = Math.max(0, Math.min(1, sim / EP_HORIZON_TICKS));
  const seg = t < 0.5 ? 0 : 1;
  const lt = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const a = seg === 0 ? SUN_A : SUN_B;
  const b = seg === 0 ? SUN_B : SUN_C;
  sun.position.set(
    a[0] + (b[0] - a[0]) * lt,
    a[1] + (b[1] - a[1]) * lt,
    a[2] + (b[2] - a[2]) * lt,
  );
  sun.color.copy(lerpColor(a[3], b[3], lt));
  sun.intensity = a[4] + (b[4] - a[4]) * lt;
  // Warm hemi drift across the day
  hemi.color.copy(lerpColor(0xfff0cc, 0xffd09a, t));
  hemi.intensity = 1.0 + Math.sin(t * Math.PI) * 0.2;
}

// ---------- building blocks ----------
function makeBox(w, h, d, color, opts = {}) {
  const mat = new THREE.MeshStandardMaterial({
    color, roughness: opts.rough ?? 0.82, metalness: 0.02, flatShading: true,
  });
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.castShadow = opts.cast ?? true;
  m.receiveShadow = opts.receive ?? true;
  return m;
}

function makeFloor(w, d) {
  const g = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.25, d),
    new THREE.MeshStandardMaterial({ color: PAL.floor, roughness: 1.0, flatShading: true })
  );
  base.position.y = 0.12; base.receiveShadow = true;
  g.add(base);
  // subtle plank stripes
  const plankW = 1.1;
  const planks = Math.floor(w / plankW);
  for (let i = 0; i < planks; i++) {
    const p = new THREE.Mesh(
      new THREE.BoxGeometry(plankW * 0.95, 0.01, d * 0.995),
      new THREE.MeshStandardMaterial({ color: i % 2 ? PAL.floorAlt : PAL.floor, roughness: 1.0, flatShading: true })
    );
    p.position.set(-w/2 + plankW * (i + 0.5), 0.251, 0);
    p.receiveShadow = true;
    g.add(p);
  }
  return g;
}

function makeExteriorWalls(w, d, h = 1.4) {
  const g = new THREE.Group();
  const wallMat = new THREE.MeshStandardMaterial({ color: PAL.wall, roughness: 0.9, flatShading: true });
  const t = 0.22;
  // north
  const n = new THREE.Mesh(new THREE.BoxGeometry(w, h, t), wallMat);
  n.position.set(0, h/2 + 0.24, -d/2); n.castShadow = true; g.add(n);
  // south
  const s = new THREE.Mesh(new THREE.BoxGeometry(w, h, t), wallMat);
  s.position.set(0, h/2 + 0.24, d/2); s.castShadow = true; g.add(s);
  // east
  const e = new THREE.Mesh(new THREE.BoxGeometry(t, h, d), wallMat);
  e.position.set(w/2, h/2 + 0.24, 0); e.castShadow = true; g.add(e);
  // west
  const wl = new THREE.Mesh(new THREE.BoxGeometry(t, h, d), wallMat);
  wl.position.set(-w/2, h/2 + 0.24, 0); wl.castShadow = true; g.add(wl);

  // wall trim (top cap)
  const capMat = new THREE.MeshStandardMaterial({ color: PAL.wallTrim, roughness: 0.7, flatShading: true });
  for (const [px, pz, sx, sz] of [
    [0, -d/2, w, t], [0, d/2, w, t], [-w/2, 0, t, d], [w/2, 0, t, d],
  ]) {
    const cap = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.12, sz), capMat);
    cap.position.set(px, h + 0.3, pz);
    cap.castShadow = true;
    g.add(cap);
  }
  return g;
}

function makeRoomCarpet(cx, cz, w, d, color) {
  const g = new THREE.Group();
  const rug = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.02, d),
    new THREE.MeshStandardMaterial({ color, roughness: 1.0, flatShading: true })
  );
  rug.position.set(cx, 0.26, cz); rug.receiveShadow = true;
  g.add(rug);
  return g;
}

function makeRoomWalls(cx, cz, w, d, h = 1.0) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0xe8d9b4, roughness: 0.92, flatShading: true });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x8f6a3a, roughness: 0.55, flatShading: true });
  const t = 0.15;
  // full north wall
  const north = new THREE.Mesh(new THREE.BoxGeometry(w, h, t), mat);
  north.position.set(cx, h/2 + 0.24, cz - d/2);
  north.castShadow = true; north.receiveShadow = true; g.add(north);
  // north trim
  const nTrim = new THREE.Mesh(new THREE.BoxGeometry(w, 0.06, t * 1.5), trimMat);
  nTrim.position.set(cx, h + 0.24, cz - d/2);
  nTrim.castShadow = true; g.add(nTrim);
  // partial east+west walls (back half only — keeps room open to center)
  for (const sign of [-1, 1]) {
    const ew = new THREE.Mesh(new THREE.BoxGeometry(t, h, d * 0.55), mat);
    ew.position.set(cx + sign * (w/2), h/2 + 0.24, cz - d * 0.22);
    ew.castShadow = true; ew.receiveShadow = true; g.add(ew);
  }
  // front corner stubs so camera reads "room"
  const stubW = Math.min(1.4, w / 3);
  for (const sign of [-1, 1]) {
    const st = new THREE.Mesh(new THREE.BoxGeometry(stubW, h, t), mat);
    st.position.set(cx + sign * (w/2 - stubW/2), h/2 + 0.24, cz + d/2);
    st.castShadow = true; st.receiveShadow = true; g.add(st);
  }
  // nameplate on the north wall (tiny sign above the door)
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 0.14, 0.04),
    new THREE.MeshStandardMaterial({ color: 0x3e2f1a, roughness: 0.5, flatShading: true })
  );
  plate.position.set(cx, h - 0.08, cz - d/2 + t/2 + 0.03);
  g.add(plate);
  return g;
}

function makeDesk(color) {
  const g = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.08, 0.55),
    new THREE.MeshStandardMaterial({ color: 0xa87345, roughness: 0.55, flatShading: true })
  );
  top.position.y = 0.54; top.castShadow = true; top.receiveShadow = true;
  g.add(top);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x4a3323, flatShading: true });
  for (const [dx, dz] of [[0.42, 0.22], [-0.42, 0.22], [0.42, -0.22], [-0.42, -0.22]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.5, 0.07), legMat);
    leg.position.set(dx, 0.27, dz); leg.castShadow = true;
    g.add(leg);
  }
  // monitor
  const mon = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.32, 0.04),
    new THREE.MeshStandardMaterial({ color: 0x0e1118, emissive: color, emissiveIntensity: 0.35, flatShading: true })
  );
  mon.position.set(0, 0.78, -0.18); g.add(mon);
  // keyboard
  const kb = new THREE.Mesh(
    new THREE.BoxGeometry(0.48, 0.02, 0.14),
    new THREE.MeshStandardMaterial({ color: 0x2a2824, flatShading: true })
  );
  kb.position.set(0, 0.59, 0.12); g.add(kb);
  // paper pad
  const pad = new THREE.Mesh(
    new THREE.BoxGeometry(0.22, 0.01, 0.16),
    new THREE.MeshStandardMaterial({ color: 0xfaf3dd, roughness: 1.0, flatShading: true })
  );
  pad.position.set(0.35, 0.585, 0.05);
  pad.rotation.y = 0.18;
  g.add(pad);
  // coffee mug (tiny cylinder)
  const mug = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.055, 0.11, 12),
    new THREE.MeshStandardMaterial({ color: 0x6a4a2a, roughness: 0.5, flatShading: true })
  );
  mug.position.set(-0.36, 0.635, 0.04);
  mug.castShadow = true;
  g.add(mug);
  // chair
  const chair = new THREE.Mesh(
    new THREE.BoxGeometry(0.38, 0.06, 0.38),
    new THREE.MeshStandardMaterial({ color: 0x2e2a26, flatShading: true })
  );
  chair.position.set(0, 0.32, 0.48); chair.castShadow = true;
  g.add(chair);
  const backrest = new THREE.Mesh(
    new THREE.BoxGeometry(0.38, 0.42, 0.05),
    new THREE.MeshStandardMaterial({ color: 0x2e2a26, flatShading: true })
  );
  backrest.position.set(0, 0.54, 0.65); backrest.castShadow = true;
  g.add(backrest);
  return g;
}

// Warm-office skin + hair palettes (no blue/green). Chosen deterministically
// by persona_id so the same person looks the same across re-renders.
const SKIN_TONES = [0xf0d4b0, 0xe8c9a8, 0xd9ae85, 0xb98860, 0x8a5e3c];
const HAIR_TONES = [0x2a1a10, 0x3d2818, 0x5a3a20, 0x7a5230, 0xa87345, 0xd4b078];
function makeCharacter(color, seed) {
  const group = new THREE.Group();
  const skin = seed != null ? SKIN_TONES[_hashString(seed + 'skin') % SKIN_TONES.length] : PAL.skin;
  const hairColor = seed != null ? HAIR_TONES[_hashString(seed + 'hair') % HAIR_TONES.length] : PAL.hair;
  // Slightly larger for visibility at the current camera distance
  const bodyGeo = new THREE.BoxGeometry(0.7, 1.05, 0.48);
  const body = new THREE.Mesh(
    bodyGeo,
    new THREE.MeshStandardMaterial({ color, roughness: 0.68, flatShading: true })
  );
  body.position.y = 0.58; body.castShadow = true; body.receiveShadow = true;
  group.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 14, 10),
    new THREE.MeshStandardMaterial({ color: skin, roughness: 0.6, flatShading: true })
  );
  head.position.y = 1.30; head.castShadow = true;
  group.add(head);
  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.35, 14, 8, 0, Math.PI * 2, 0, Math.PI * 0.55),
    new THREE.MeshStandardMaterial({ color: hairColor, flatShading: true })
  );
  hair.position.y = 1.30; hair.castShadow = true;
  group.add(hair);
  // Dark cartoon-outline shell around body
  const shell = new THREE.Mesh(bodyGeo.clone(),
    new THREE.MeshBasicMaterial({ color: PAL.ink, side: THREE.BackSide })
  );
  shell.scale.set(1.06, 1.03, 1.1);
  shell.position.copy(body.position);
  group.add(shell);
  // Role-colored ground ring — always visible, marks the character from above
  const ringMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.62, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.48, 0.62, 28), ringMat);
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.29;
  group.add(ring);
  group.userData = { body, head, hair, ringMat, pulse: 0 };
  return group;
}

// ---------- service kiosks (interior, not exterior buildings) ----------
// Shared material helpers — warm wood base, metal stand, glowing screen.
const KIOSK_WOOD  = 0x8a6a4a;
const KIOSK_METAL = 0x5a5046;
const KIOSK_DARK  = 0x1d1a14;
function _kioskBase(w = 1.1, d = 0.9) {
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.08, d),
    new THREE.MeshStandardMaterial({ color: KIOSK_WOOD, roughness: 0.7, flatShading: true })
  );
  base.position.y = 0.3; base.castShadow = true;
  return base;
}
function _kioskScreen(w, h, color, emissive = 0.6) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.04),
    new THREE.MeshStandardMaterial({ color: KIOSK_DARK, emissive: color, emissiveIntensity: emissive, flatShading: true })
  );
}
function _kioskGlow(w, h, color) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.02),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.95 })
  );
}

// Monitor on a thin stand — web app / HTTP service.
function _makeWebKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.7));
  const stand = new THREE.Mesh(
    new THREE.BoxGeometry(0.14, 0.55, 0.14),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.6, flatShading: true })
  );
  stand.position.y = 0.62; stand.castShadow = true; g.add(stand);
  const screen = _kioskScreen(0.86, 0.58, color, 0.7);
  screen.position.set(0, 1.05, 0); screen.castShadow = true; g.add(screen);
  const bezel = new THREE.Mesh(
    new THREE.BoxGeometry(0.24, 0.14, 0.02),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9 })
  );
  bezel.position.set(0, 1.05, 0.03); g.add(bezel);
  return g;
}

// Short rack tower with three blinking LED stripes — database.
function _makeDbKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.8));
  const tower = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 1.0, 0.55),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true })
  );
  tower.position.y = 0.85; tower.castShadow = true; g.add(tower);
  for (let i = 0; i < 3; i++) {
    const led = _kioskGlow(0.55, 0.06, color);
    led.position.set(0, 0.58 + i * 0.21, 0.29);
    g.add(led);
  }
  return g;
}

// Envelope on a stand — email / SMTP.
function _makeMailKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.8));
  const stand = new THREE.Mesh(
    new THREE.BoxGeometry(0.1, 0.45, 0.1),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.6, flatShading: true })
  );
  stand.position.y = 0.57; stand.castShadow = true; g.add(stand);
  const env = new THREE.Mesh(
    new THREE.BoxGeometry(0.78, 0.5, 0.06),
    new THREE.MeshStandardMaterial({ color: 0xfaf3dd, roughness: 0.9, flatShading: true })
  );
  env.position.set(0, 0.95, 0); env.rotation.x = -0.22; env.castShadow = true; g.add(env);
  // Flap as a smaller tilted panel on top
  const flap = new THREE.Mesh(
    new THREE.BoxGeometry(0.78, 0.28, 0.03),
    new THREE.MeshStandardMaterial({ color: 0xedddb5, roughness: 0.9, flatShading: true })
  );
  flap.position.set(0, 1.12, 0.05); flap.rotation.x = -0.45; g.add(flap);
  // Wax seal glow
  const seal = _kioskGlow(0.12, 0.12, color);
  seal.position.set(0, 0.98, 0.07); g.add(seal);
  return g;
}

// Low file cabinet with two drawers and pull handles — fileshare / SMB / NFS.
function _makeFilesKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.95, 0.7));
  const cab = new THREE.Mesh(
    new THREE.BoxGeometry(0.8, 0.95, 0.6),
    new THREE.MeshStandardMaterial({ color: KIOSK_WOOD, roughness: 0.7, flatShading: true })
  );
  cab.position.y = 0.82; cab.castShadow = true; g.add(cab);
  for (let i = 0; i < 2; i++) {
    const drawer = new THREE.Mesh(
      new THREE.BoxGeometry(0.72, 0.38, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x6a4a2a, roughness: 0.6, flatShading: true })
    );
    drawer.position.set(0, 0.55 + i * 0.45, 0.3); g.add(drawer);
    const pull = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.04, 0.04),
      new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.4, flatShading: true })
    );
    pull.position.set(0, 0.55 + i * 0.45, 0.33); g.add(pull);
  }
  // role-tint LED strip
  const led = _kioskGlow(0.8, 0.04, color);
  led.position.set(0, 1.3, 0.31); g.add(led);
  return g;
}

// Three shallow boxes offset like a card stack — LDAP / IdP / directory.
function _makeDirectoryKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.8));
  for (let i = 0; i < 3; i++) {
    const card = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, 0.14, 0.55),
      new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true })
    );
    card.position.set(i * 0.06 - 0.06, 0.52 + i * 0.18, -i * 0.04);
    card.castShadow = true; g.add(card);
    const tab = _kioskGlow(0.22, 0.06, color);
    tab.position.set(0.32 + i * 0.06, 0.52 + i * 0.18, 0.28 - i * 0.04);
    g.add(tab);
  }
  return g;
}

// Wide dual-screen console — SIEM / monitoring / observability.
function _makeSiemKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.3, 0.75));
  const desk = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 0.05, 0.55),
    new THREE.MeshStandardMaterial({ color: KIOSK_WOOD, roughness: 0.6, flatShading: true })
  );
  desk.position.y = 0.58; desk.castShadow = true; g.add(desk);
  for (const dx of [-0.32, 0.32]) {
    const stand = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.34, 0.08),
      new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true })
    );
    stand.position.set(dx, 0.78, 0); g.add(stand);
    const screen = _kioskScreen(0.52, 0.4, color, 0.7);
    screen.position.set(dx, 1.1, 0); screen.castShadow = true; g.add(screen);
  }
  return g;
}

// Short drum / cylinder — cache / Redis / Memcached.
function _makeCacheKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.9));
  const drum = new THREE.Mesh(
    new THREE.CylinderGeometry(0.38, 0.38, 0.75, 20),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.5, flatShading: true })
  );
  drum.position.y = 0.72; drum.castShadow = true; g.add(drum);
  // two glow rings around the drum
  for (let i = 0; i < 2; i++) {
    const band = new THREE.Mesh(
      new THREE.CylinderGeometry(0.40, 0.40, 0.04, 20),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9, flatShading: true })
    );
    band.position.y = 0.5 + i * 0.35; g.add(band);
  }
  return g;
}

// Fallback: equipment crate with a role-colored LED strip.
function _makeGenericKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.85));
  const crate = new THREE.Mesh(
    new THREE.BoxGeometry(0.78, 0.9, 0.7),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.6, flatShading: true })
  );
  crate.position.y = 0.8; crate.castShadow = true; g.add(crate);
  const strip = _kioskGlow(0.6, 0.05, color);
  strip.position.set(0, 1.18, 0.36); g.add(strip);
  return g;
}

// Kind normalization — strip common prefixes, squash aliases.
const KIOSK_ALIASES = {
  web:         'web_app',
  http:        'web_app',
  website:     'web_app',
  mail:        'email',
  smtp:        'email',
  pop3:        'email',
  imap:        'email',
  database:    'db',
  mysql:       'db',
  postgres:    'db',
  postgresql:  'db',
  mssql:       'db',
  mongo:       'db',
  mongodb:     'db',
  files:       'fileshare',
  smb:         'fileshare',
  nfs:         'fileshare',
  share:       'fileshare',
  ldap:        'idp',
  ad:          'idp',
  activedirectory: 'idp',
  directory:   'idp',
  monitoring:  'siem',
  observability: 'siem',
  logs:        'siem',
  redis:       'cache',
  memcached:   'cache',
  memcache:    'cache',
  queue:       'cache',
  rabbitmq:    'cache',
  kafka:       'cache',
};
function canonicalKind(kind) {
  const key = String(kind || '').toLowerCase();
  return KIOSK_ALIASES[key] || key;
}

function makeKiosk(kind, color) {
  const k = canonicalKind(kind);
  switch (k) {
    case 'web_app':   return _makeWebKiosk(color);
    case 'db':        return _makeDbKiosk(color);
    case 'email':     return _makeMailKiosk(color);
    case 'fileshare': return _makeFilesKiosk(color);
    case 'idp':       return _makeDirectoryKiosk(color);
    case 'siem':      return _makeSiemKiosk(color);
    case 'cache':     return _makeCacheKiosk(color);
    default:          return _makeGenericKiosk(color);
  }
}

function makePlant() {
  const g = new THREE.Group();
  const pot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.28, 0.35, 12),
    new THREE.MeshStandardMaterial({ color: 0x8a5a3a, roughness: 0.8, flatShading: true })
  );
  pot.position.y = 0.17; pot.castShadow = true; g.add(pot);
  const leaves = new THREE.Mesh(
    new THREE.SphereGeometry(0.38, 10, 7),
    new THREE.MeshStandardMaterial({ color: 0x4e7e3e, roughness: 0.8, flatShading: true })
  );
  leaves.position.y = 0.6; leaves.scale.set(1.1, 1.2, 1.1); leaves.castShadow = true;
  g.add(leaves);
  return g;
}

// ---------- scalable office layout ----------
// Rooms roughly match server-side ROOM_LAYOUT but we derive them from personas
// so N changes automatically. One room per role present.
// Room plan is COMPUTED at scene_init from the actual persona roles + services.
// We build a rectangular department grid with rooms sized by headcount. No hard
// limit on number of roles — supports HR, Legal, Ops, Marketing, anything.
let ROOM_PLAN = {};   // role → {cx, cz, w, d, label}

// Clean casing overrides for known roles. Fall-through replaces underscores
// with spaces and uppercases — so "design", "marketing", "observability" all
// look tidy without being hardcoded.
const ROLE_LABEL_OVERRIDES = {
  engineer:   'ENGINEERING',
  it_admin:   'IT · SOC',
  r_and_d:    'R&D',
  qa:         'QA',
  devops:     'DEVOPS',
  executive:  'EXECUTIVE',
};
function labelForRole(role) {
  const key = canonicalRole(role);
  if (ROLE_LABEL_OVERRIDES[key]) return ROLE_LABEL_OVERRIDES[key];
  return String(role || '').replace(/_/g, ' ').toUpperCase();
}

// seatFor — given role + (1-indexed seat, total seats), return [x, z] inside room
function seatFor(role, seatIndex, totalSeats) {
  const plan = ROOM_PLAN[role];
  if (!plan) return [0, 0];
  const n = Math.max(1, totalSeats);
  const cols = Math.max(1, Math.min(4, Math.ceil(Math.sqrt(n))));
  const rows = Math.ceil(n / cols);
  const innerW = Math.max(1.0, plan.w - 1.8);
  const innerH = Math.max(1.0, plan.d - 1.8);
  const i = seatIndex - 1;
  const col = i % cols;
  const row = Math.floor(i / cols);
  const x = plan.cx + (col + 0.5 - cols / 2) * (innerW / cols);
  const z = plan.cz + (row + 0.5 - rows / 2) * (innerH / rows);
  return [x, z];
}

function computeRoomPlan(roleCounts) {
  // roleCounts: { engineer: 3, sales: 5, finance: 2, ... }
  // Rooms are packed into a 2-column grid; size scales with seat count.
  const roles = Object.keys(roleCounts)
    .filter(r => roleCounts[r] > 0)
    .sort((a, b) => roleCounts[b] - roleCounts[a]);  // biggest first

  const plan = {};
  const cols = 2;
  const gap = 0.8;
  let rowStartZ = -3.0;
  let colWidths = [0, 0];

  // Pair rooms into rows of 2
  for (let i = 0; i < roles.length; i += cols) {
    const pair = roles.slice(i, i + cols);
    // Size each room by seats in the pair (balanced)
    const rowMaxN = Math.max(...pair.map(r => roleCounts[r]));
    const w = Math.max(5.5, Math.min(10, 4.5 + rowMaxN * 0.8));
    const d = Math.max(4.0, Math.min(6.5, 3.5 + rowMaxN * 0.35));

    pair.forEach((role, idx) => {
      const sign = idx === 0 ? -1 : 1;
      const cx = sign * (w / 2 + gap / 2);
      plan[role] = {
        cx, cz: rowStartZ + d / 2, w, d,
        label: labelForRole(role),
      };
    });
    rowStartZ += d + gap;
  }

  // Always attach external street (where red spawns) and service corridor
  plan.external = { cx: 0, cz: -13.0, w: 24.0, d: 2.5, label: 'EXTERNAL' };
  return plan;
}
// Where each service KIND wants to live. Multiple instances of the same kind
// get spread around the room's perimeter. Overrides map canonical kinds onto
// preferred rooms; anything unmapped falls through to zone → hash.
const SERVICE_ROOM_OVERRIDES = {
  web_app:   'external',
  email:     'external',
  fileshare: 'finance',
  db:        'finance',
  idp:       'it_admin',
  siem:      'it_admin',
  cache:     'it_admin',
};

// Resolve a room key for a service. Preference order:
//   1. Canonical-kind override (web_app → external, etc.)
//   2. Service's zone field matches an existing ROOM_PLAN room
//   3. Hash the kind onto whichever interior room we actually have
//   4. 'corp' as a last-ditch fallback
function roomForService(svc) {
  const k = canonicalKind(svc.kind);
  const roomKeys = Object.keys(ROOM_PLAN).filter(r => r !== 'corp');
  if (SERVICE_ROOM_OVERRIDES[k] && ROOM_PLAN[SERVICE_ROOM_OVERRIDES[k]]) {
    return SERVICE_ROOM_OVERRIDES[k];
  }
  if (svc.zone && ROOM_PLAN[svc.zone]) return svc.zone;
  const canonZone = canonicalRole(svc.zone);
  if (canonZone && ROOM_PLAN[canonZone]) return canonZone;
  const interior = roomKeys.filter(r => r !== 'external');
  if (interior.length) {
    const idx = _hashString(k || svc.id || '') % interior.length;
    return interior[idx];
  }
  if (roomKeys.length) return roomKeys[0];
  return 'corp';
}

function serviceLabel(svcId) {
  return String(svcId || '').replace(/^svc-/, '');
}

// Compute a placement for each service inside its preferred room, laying them
// out along the wall opposite the door so characters stay visible.
function computeServicePlacements(servicesArr) {
  const byRoom = new Map();
  for (const s of servicesArr) {
    const room = roomForService(s);
    if (!byRoom.has(room)) byRoom.set(room, []);
    byRoom.get(room).push(s);
  }
  const placements = {};
  for (const [roomKey, list] of byRoom.entries()) {
    const plan = ROOM_PLAN[roomKey] || { cx: 0, cz: -12, w: 20, d: 4 };
    const n = list.length;
    const wallX = plan.cx;
    // north-interior wall for inside rooms, south strip for external
    const wallZ = roomKey === 'external'
      ? plan.cz    // external is a strip along south
      : plan.cz - plan.d / 2 + 0.7;
    const span = Math.max(3.0, plan.w - 2.0);
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : (i / (n - 1) - 0.5);
      const x = (roomKey === 'external')
        ? plan.cx + t * Math.max(18, plan.w * 1.1)
        : wallX + t * span * 0.8;
      const z = (roomKey === 'external')
        ? plan.cz + ((i % 2 === 0) ? -0.2 : 0.6)
        : wallZ + ((i % 2 === 0) ? 0 : 0.9);
      placements[list[i].id] = { x, z, label: serviceLabel(list[i].id) };
    }
  }
  return placements;
}

function buildOfficeShell(activeRoles) {
  clearDynamic();
  // Compute a bounding box from the rooms actually in use so the floor
  // auto-resizes when more departments are added.
  const interiorRooms = activeRoles.filter(r => ROOM_PLAN[r] && r !== 'external');
  let minX = -8, maxX = 8, minZ = -5, maxZ = 5;
  for (const r of interiorRooms) {
    const p = ROOM_PLAN[r];
    minX = Math.min(minX, p.cx - p.w / 2);
    maxX = Math.max(maxX, p.cx + p.w / 2);
    minZ = Math.min(minZ, p.cz - p.d / 2);
    maxZ = Math.max(maxZ, p.cz + p.d / 2);
  }
  const padX = 3, padZ = 3;
  const floorW = (maxX - minX) + padX * 2;
  const floorD = (maxZ - minZ) + padZ * 2;
  const floorCX = (minX + maxX) / 2;
  const floorCZ = (minZ + maxZ) / 2;

  const floor = makeFloor(floorW, floorD);
  floor.position.set(floorCX, 0, floorCZ);
  scene.add(floor); dynamicNodes.push(floor);

  const outer = makeExteriorWalls(floorW, floorD, 1.35);
  outer.position.set(floorCX, 0, floorCZ);
  scene.add(outer); dynamicNodes.push(outer);

  // External "street" — wider than the building, south of the entrance
  const streetW = Math.max(24, floorW + 4);
  const street = new THREE.Mesh(
    new THREE.BoxGeometry(streetW, 0.14, 4.2),
    new THREE.MeshStandardMaterial({ color: 0xc8b79a, roughness: 1.0, flatShading: true })
  );
  street.position.set(floorCX, 0.07, minZ - padZ - 3.8);
  street.receiveShadow = true;
  scene.add(street); dynamicNodes.push(street);
}

function buildRooms(activeRoles) {
  for (const role of activeRoles) {
    const plan = ROOM_PLAN[role];
    if (!plan) continue;
    if (role === 'external' || role === 'corp') continue;  // no carpet/walls for outdoor/public
    // carpet
    const carpet = makeRoomCarpet(plan.cx, plan.cz, plan.w - 0.8, plan.d - 0.8, paletteForRole(role).carpet);
    scene.add(carpet); dynamicNodes.push(carpet);
    // low interior walls
    const walls = makeRoomWalls(plan.cx, plan.cz, plan.w, plan.d, 0.7);
    scene.add(walls); dynamicNodes.push(walls);
    // door-side room label (HTML nametag)
    addRoomLabel(plan.label, plan.cx, plan.cz - plan.d / 2 + 0.3);
    // decor: corner plant
    const plant = makePlant();
    plant.position.set(plan.cx - plan.w/2 + 0.5, 0, plan.cz - plan.d/2 + 0.5);
    scene.add(plant); dynamicNodes.push(plant);
  }
}

function addRoomLabel(text, x, z) {
  const el = document.createElement('div');
  el.className = 'nametag zone';
  el.textContent = text;
  el.dataset.worldX = x;
  el.dataset.worldY = 1.0;
  el.dataset.worldZ = z;
  labelLayer.appendChild(el);
}

// ---------- derived scene anchors ----------
// All waypoints + agent spawns are computed from ROOM_PLAN so layouts of any
// size stay self-consistent. Never returns (0,0,0) for an agent.
function computeSceneAnchors(roomPlan) {
  const rooms = Object.entries(roomPlan)
    .filter(([k, v]) => k !== 'external' && k !== 'corp' && v);
  if (rooms.length === 0) {
    // No interior rooms yet — fall back to a sensible default frame.
    return {
      redSpawn:   { x: -8, z: -12 },
      blueSpawn:  { x:  4, z:   2 },
      coffeeHub:  { x:  0, z:   0 },
      chatHub:    { x:  0, z:   4 },
      coolerHub:  { x: -9, z:   0 },
    };
  }
  let minX =  Infinity, maxX = -Infinity;
  let minZ =  Infinity, maxZ = -Infinity;
  let cxSum = 0, czSum = 0;
  let biggest = null, biggestArea = -1;
  for (const [, p] of rooms) {
    minX = Math.min(minX, p.cx - p.w / 2);
    maxX = Math.max(maxX, p.cx + p.w / 2);
    minZ = Math.min(minZ, p.cz - p.d / 2);
    maxZ = Math.max(maxZ, p.cz + p.d / 2);
    cxSum += p.cx;
    czSum += p.cz;
    const area = p.w * p.d;
    if (area > biggestArea) { biggestArea = area; biggest = p; }
  }
  const avgCx = cxSum / rooms.length;
  const avgCz = czSum / rooms.length;

  // Red spawns outside the building, south of the southernmost edge, west side.
  const redSpawn = {
    x: minX + (maxX - minX) * 0.12,  // west-leaning
    z: maxZ + 3.5,                   // south of building by 3–4 units
  };

  // Blue spawns inside the it_admin / it / soc room; fallback to largest.
  const itRoom = roomPlan.it_admin || roomPlan.it || roomPlan.soc || biggest;
  const blueSpawn = { x: itRoom.cx, z: itRoom.cz };

  // Coffee hub: central corridor, between stacked room pairs.
  const coffeeHub = { x: avgCx, z: (minZ + maxZ) / 2 - 1.8 };

  // Chat hub: sales/marketing centroid if present, else central corridor south.
  const chatRoom = roomPlan.sales || roomPlan.marketing;
  const chatHub = chatRoom
    ? { x: chatRoom.cx, z: chatRoom.cz }
    : { x: avgCx, z: (minZ + maxZ) / 2 + 1.8 };

  // Water cooler: against west wall interior, minus 1 unit.
  const coolerHub = { x: minX - 1, z: avgCz };

  return { redSpawn, blueSpawn, coffeeHub, chatHub, coolerHub };
}

// ---------- HUD renderers (metric strip, zone chips, roster) ----------
// These consume the full scene_init payload so the shell stays generalizable:
// nothing is hardcoded to a specific role or card count; everything derives
// from the manifest we were handed.

function _makeMetricCard({ label, value, sub, valueClass, liveDot }) {
  const card = document.createElement('div');
  card.className = 'metric-card';
  card.tabIndex = 0;
  const labelNode = document.createElement('div');
  labelNode.className = 'label';
  if (liveDot) {
    const dot = document.createElement('span');
    dot.className = 'live-dot' + (liveDot === 'offline' ? ' offline' : '');
    labelNode.appendChild(dot);
  }
  labelNode.appendChild(document.createTextNode(label));
  const valueNode = document.createElement('div');
  valueNode.className = 'value' + (valueClass ? ' ' + valueClass : '');
  valueNode.textContent = value;
  card.appendChild(labelNode);
  card.appendChild(valueNode);
  if (sub) {
    const subNode = document.createElement('div');
    subNode.className = 'sub';
    subNode.textContent = sub;
    card.appendChild(subNode);
  }
  return card;
}

function renderMetricStrip(msg) {
  const strip = document.getElementById('metric-strip');
  if (!strip) return;
  strip.innerHTML = '';

  // Card 1 (always): snapshot short_id + model badge.
  const snapId = (msg.snapshot_id || '').slice(-10) || 'awaiting';
  strip.appendChild(_makeMetricCard({
    label: 'Snapshot',
    value: snapId,
    sub: msg.llm_model || 'grok-4.1-fast',
    valueClass: 'cool',
  }));

  // Card 2 (always): total personas + role variety count.
  const roleSet = new Set((msg.personas || []).map(p => canonicalRole(p.role)));
  strip.appendChild(_makeMetricCard({
    label: 'Personas',
    value: String((msg.personas || []).length),
    sub: `${(msg.personas || []).length} personas · ${roleSet.size} roles`,
  }));

  // Card 3 (always): services + zones.
  const zoneSet = new Set((msg.services || []).map(s => s.zone).filter(Boolean));
  const zoneCount = Math.max(zoneSet.size, Array.isArray(msg.zones) ? msg.zones.length : 0);
  strip.appendChild(_makeMetricCard({
    label: 'Services',
    value: String((msg.services || []).length),
    sub: `${(msg.services || []).length} services · ${zoneCount} zones`,
  }));

  // Card 4 (conditional): runtime presence.
  if (msg.docker && msg.docker.present) {
    strip.appendChild(_makeMetricCard({
      label: 'Runtime',
      value: `kind · ${msg.docker.short_id || 'live'}`,
      sub: 'container runtime',
      liveDot: 'live',
    }));
  } else {
    strip.appendChild(_makeMetricCard({
      label: 'Runtime',
      value: msg.live_backend || 'offline',
      sub: 'stub oracle',
      valueClass: 'offline',
      liveDot: 'offline',
    }));
  }

  // Card 5 (conditional): pods/containers count, only when Kind is live.
  if (msg.live_backend === 'kind') {
    const podCount = msg.docker?.container_count ?? msg.docker?.pod_count ?? msg.pod_count;
    if (Number.isFinite(+podCount)) {
      strip.appendChild(_makeMetricCard({
        label: 'Pods',
        value: String(podCount),
        sub: 'kind cluster',
      }));
    }
  }
}

// ---------- LLM traces (disk-cache tail) ----------
async function fetchAndRenderTraces() {
  try {
    const res = await fetch('/traces?limit=30');
    if (!res.ok) return;
    const d = await res.json();
    const body = document.getElementById('traces-body');
    const tot  = document.getElementById('traces-totals');
    if (!body) return;
    body.innerHTML = (d.items || []).map(it => {
      const p = (it.preview || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
      return `<div class="trace-row" title="${it.key} · seed ${it.seed}">` +
             `<span class="trace-preview">${p || '—'}</span>` +
             `<span class="trace-tok">${it.prompt_tokens}+${it.completion_tokens}</span>` +
             `<span class="trace-lat">${it.latency_ms}ms</span>` +
             `</div>`;
    }).join('');
    if (tot) {
      const t = d.totals || {};
      tot.textContent = `${t.calls || 0} calls · ${(t.prompt_tokens||0)+(t.completion_tokens||0)} tok · p50 ${t.median_latency_ms||0}ms`;
    }
  } catch {}
}

// poll every 5s once the app is booted
setInterval(fetchAndRenderTraces, 5000);

function renderZoneStrip(msg) {
  const strip = document.getElementById('zone-strip');
  if (!strip) return;
  strip.innerHTML = '';

  // Derive { zone → serviceCount } from scene_init.services. Unknown-zone
  // services are grouped under 'other' so they still surface.
  const counts = new Map();
  for (const s of (msg.services || [])) {
    const z = (s.zone && String(s.zone).trim()) || 'other';
    counts.set(z, (counts.get(z) || 0) + 1);
  }
  // Also include any zones the manifest declared even if empty of services.
  if (Array.isArray(msg.zones)) {
    for (const z of msg.zones) {
      const key = String(z).trim();
      if (key && !counts.has(key)) counts.set(key, 0);
    }
  }
  if (counts.size === 0) return;

  const label = document.createElement('span');
  label.className = 'zone-strip-label';
  label.textContent = 'Zones';
  strip.appendChild(label);

  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  for (const [zone, n] of entries) {
    const chip = document.createElement('span');
    chip.className = 'zone-chip';
    const name = document.createElement('span');
    name.className = 'zone-name';
    name.textContent = zone;
    const sep = document.createTextNode(' · ');
    const countEl = document.createElement('span');
    countEl.className = 'zone-count';
    countEl.textContent = `${n} service${n === 1 ? '' : 's'}`;
    chip.appendChild(name);
    chip.appendChild(sep);
    chip.appendChild(countEl);
    chip.title = `${zone} · ${n} service${n === 1 ? '' : 's'}`;
    strip.appendChild(chip);
  }
}

function renderRoster(msg) {
  const roster = document.getElementById('roster');
  if (!roster) return;
  roster.innerHTML = '';

  // Group personas by canonical role, preserving first-seen order so the
  // visual order matches the user's manifest intent.
  const groups = new Map();  // canonRole → { label, items: [] }
  for (const p of (msg.personas || [])) {
    const canon = canonicalRole(p.role);
    if (!groups.has(canon)) {
      groups.set(canon, { label: labelForRole(canon), items: [] });
    }
    groups.get(canon).items.push(p);
  }

  let idx = 0;
  for (const [canon, { label, items }] of groups.entries()) {
    const details = document.createElement('details');
    details.className = 'roster-group';
    // First two role sections expand by default; remainder collapsed so
    // scenes with 8+ roles don't overwhelm the meta column.
    if (idx < 2) details.open = true;
    idx += 1;

    const summary = document.createElement('summary');
    summary.innerHTML = `
      <span class="roster-label">${escapeHtml(label)}</span>
      <span class="roster-count">${items.length}</span>`;
    details.appendChild(summary);

    const container = document.createElement('div');
    container.className = 'roster-items';
    for (const p of items) {
      const div = document.createElement('div');
      div.className = 'person';
      div.dataset.personaId = p.persona_id;
      const roleTag = (p.role || '').replace('_', ' ');
      div.innerHTML = `
        <span class="sprite ${p.role}"></span>
        <div class="person-body">
          <div class="person-name"><span class="speak-dot"></span>${escapeHtml(p.display_name)}</div>
          <div class="person-title">${escapeHtml(p.title || roleTag)}</div>
        </div>
        <div class="person-loc">${escapeHtml(roleTag)}</div>`;
      const tooltipLines = [
        p.display_name,
        p.tone ? '"' + p.tone + '"' : '',
        p.peers && p.peers.length ? 'peers: ' + p.peers.join(', ') : '',
      ].filter(Boolean);
      div.title = tooltipLines.join('\n');
      container.appendChild(div);
    }
    details.appendChild(container);
    roster.appendChild(details);
  }
}

// ---------- scene_init handler ----------
function handleSceneInit(msg) {
  // purge previous
  for (const n of npcs.values()) { scene.remove(n.group); n.desk && scene.remove(n.desk); n.nameTag?.remove(); }
  npcs.clear();
  for (const s of services.values()) { scene.remove(s.group); s.group.userData.label?.remove(); }
  services.clear();
  for (const role of ['red','blue']) { if (agents[role]) { scene.remove(agents[role]); agents[role] = null; } agents[role + 'Tag']?.remove?.(); }
  // remove any stray nametags from previous runs
  labelLayer.querySelectorAll('.nametag').forEach(el => el.remove());

  // Episode horizon scales with server cadence — ~30 visible tick marks no
  // matter whether tick_interval_s is 0.5s or 5s. Clamped to 30 minimum so
  // tiny manifests still show motion on the timeline.
  const tickIntervalS = Number.isFinite(+msg.tick_interval_s) ? +msg.tick_interval_s : 2.0;
  EP_HORIZON_TICKS = Math.max(30, Math.round(tickIntervalS * 30));

  // compute room plan from persona role counts (fully dynamic, scales with N).
  // Canonicalize so engineer + engineering + r_and_d share one room.
  const roleCounts = {};
  for (const p of msg.personas) {
    const r = canonicalRole(p.role);
    roleCounts[r] = (roleCounts[r] || 0) + 1;
  }
  ROOM_PLAN = computeRoomPlan(roleCounts);
  fitCameraToPlan(ROOM_PLAN);
  const anchors = computeSceneAnchors(ROOM_PLAN);
  const activeRoles = Object.keys(ROOM_PLAN);
  buildOfficeShell(activeRoles);
  buildRooms(activeRoles);

  // seat NPCs locally using computed ROOM_PLAN — ignores server coords so
  // layout is always self-consistent with the rendered rooms. Seat index
  // taken from persona order within role (deterministic).
  const seatIdx = {};
  const seatCounter = {};
  for (const p of msg.personas) {
    const r = canonicalRole(p.role);
    seatCounter[r] = (seatCounter[r] || 0) - 1;
  }
  for (const p of msg.personas) {
    const canonR = canonicalRole(p.role);
    const color = paletteForRole(p.role).desk;
    const ch = makeCharacter(color, p.persona_id);
    // Prefer server-computed coords; only fall back to local seat packer
    let dx = Number.isFinite(+p.desk_x) ? +p.desk_x : null;
    let dz = Number.isFinite(+p.desk_y) ? +p.desk_y : null;
    if (dx === null || dz === null) {
      [dx, dz] = seatFor(canonR, seatIdx[canonR] = (seatIdx[canonR] || 0) + 1, Math.abs(seatCounter[canonR]));
    }
    ch.position.set(dx, 0, dz);
    ch.rotation.y = Math.PI;  // face monitor
    ch.userData.baseY = 0;
    ch.userData.persona = p;
    scene.add(ch);

    const desk = makeDesk(color);
    desk.position.set(dx, 0.05, dz + 0.8);
    scene.add(desk);

    const tag = document.createElement('div');
    tag.className = 'nametag ' + p.role;
    tag.textContent = p.display_name;
    tag.dataset.worldX = dx;
    tag.dataset.worldY = 1.9;
    tag.dataset.worldZ = dz;
    labelLayer.appendChild(tag);

    // Each NPC has a daily itinerary — desk ↔ coffee ↔ chat ↔ cooler ↔ desk
    // Destinations are derived from ROOM_PLAN so layouts of any size work.
    const route = [
      { x: dx, z: dz, wait: 6 + Math.random() * 3 },
      { x: anchors.coffeeHub.x, z: anchors.coffeeHub.z, wait: 1.5 },
      { x: dx, z: dz, wait: 5 },
      { x: anchors.chatHub.x,   z: anchors.chatHub.z,   wait: 2 },
      { x: dx, z: dz, wait: 7 },
      { x: anchors.coolerHub.x, z: anchors.coolerHub.z, wait: 1.2 },
      { x: dx, z: dz, wait: 8 },
    ];
    let _h = 2166136261;
    const _s = p.persona_id + 'r';
    for (let _i = 0; _i < _s.length; _i++) { _h ^= _s.charCodeAt(_i); _h = Math.imul(_h, 16777619); }
    const startIdx = Math.abs(_h) % route.length;
    npcs.set(p.persona_id, {
      group: ch, desk, nameTag: tag, color,
      phase: Math.random() * Math.PI * 2,
      deskHome: [dx, dz],
      route,
      routeIdx: startIdx,
      routeT: 0,
      routeSpeed: 0.35 + 0.12 * Math.random(),
      dwellRemaining: route[startIdx].wait,
      speaking: false,
    });
  }
  // Log any personas whose role has no room (defense-in-depth)
  const misplaced = msg.personas.filter(p => !ROOM_PLAN[canonicalRole(p.role)]);
  if (misplaced.length) {
    console.warn('[openrange] no room for roles:', misplaced.map(p => `${p.display_name}(${p.role})`));
  }
  diag('scene_init rendered: ' + npcs.size + ' NPCs, ' + services.size + ' services');

  // services placed by kind → room, multi-instance safe
  const placements = computeServicePlacements(msg.services);
  for (const s of msg.services) {
    const placement = placements[s.id] || { x: 0, z: 0, label: serviceLabel(s.id) };
    const color = paletteForRole(s.kind).desk;
    const g = makeKiosk(s.kind, color);
    g.position.set(placement.x, 0, placement.z);
    // health ring
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x5aa148, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.1, 32), ringMat);
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02;
    g.add(ring);
    g.userData = { ring, ringMat, pulseStart: 0, pulseColor: null };
    scene.add(g);

    const el = document.createElement('div');
    el.className = 'nametag service';
    el.textContent = placement.label;
    el.dataset.worldX = placement.x;
    el.dataset.worldY = 2.1;
    el.dataset.worldZ = placement.z;
    labelLayer.appendChild(el);
    g.userData.label = el;

    services.set(s.id, { group: g });
  }

  // agents: red out on street, blue in SOC room — coords derived from ROOM_PLAN
  agents.red = makeCharacter(PAL.red);
  agents.red.position.set(anchors.redSpawn.x, 0, anchors.redSpawn.z);
  scene.add(agents.red);
  const redTag = document.createElement('div');
  redTag.className = 'nametag red';
  redTag.textContent = 'red · grok-4.1-fast';
  redTag.dataset.worldX = anchors.redSpawn.x; redTag.dataset.worldY = 2.0; redTag.dataset.worldZ = anchors.redSpawn.z;
  labelLayer.appendChild(redTag);
  agents.red.userData.tag = redTag;

  agents.blue = makeCharacter(PAL.blue);
  agents.blue.position.set(anchors.blueSpawn.x, 0, anchors.blueSpawn.z);
  scene.add(agents.blue);
  const blueTag = document.createElement('div');
  blueTag.className = 'nametag blue';
  blueTag.textContent = 'blue · grok-4.1-fast';
  blueTag.dataset.worldX = anchors.blueSpawn.x; blueTag.dataset.worldY = 2.0; blueTag.dataset.worldZ = anchors.blueSpawn.z;
  labelLayer.appendChild(blueTag);
  agents.blue.userData.tag = blueTag;

  // UI chrome updates (Vecna-style)
  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setText('hud-npcs', msg.personas.length);
  setText('pill-personas-count', msg.personas.length);
  setText('pill-services-count', msg.services.length);
  setText('pill-snapshot', (msg.snapshot_id || '').slice(-14) || '—');
  setText('meta-model', msg.llm_model || 'grok-4.1-fast');
  setText('k-world', (msg.snapshot_id || '').slice(-10) || '—');
  document.getElementById('pill-status').textContent = 'Active';
  const metaRuntime = document.getElementById('meta-runtime');
  const metaDocker = document.getElementById('meta-docker');
  const footerDocker = document.getElementById('footer-docker');
  if (msg.docker && msg.docker.present) {
    if (metaRuntime) metaRuntime.textContent = 'kind · live';
    if (metaDocker) metaDocker.textContent = msg.docker.short_id;
    if (footerDocker) footerDocker.textContent = `kind · ${msg.docker.short_id}`;
  } else {
    if (metaRuntime) metaRuntime.textContent = msg.live_backend || 'offline';
    if (footerDocker) footerDocker.textContent = 'offline';
  }

  // Metric strip — data-driven from scene_init. Cards adjust to whatever
  // context we actually received (3 or 5 cards; auto-fit grid handles width).
  renderMetricStrip(msg);

  // Zone chip strip — derived from services. Pure display, no interaction.
  renderZoneStrip(msg);

  // Roster: role-grouped with collapsible sections. First two roles open by
  // default; deeper ones collapse so 30-persona scenes stay readable.
  renderRoster(msg);

  // Service vitality — compact horizontal chip row (saves vertical space)
  const svcPanel = document.getElementById('services');
  svcPanel.innerHTML = '';
  for (const s of msg.services) {
    const row = document.createElement('div');
    row.className = 'svc-chip';
    row.innerHTML = `
      <span class="svc-name">${escapeHtml(serviceLabel(s.id))}</span>
      <span class="svc-health" id="svc-h-${s.id}">1.00</span>
      <div class="bar-track" style="grid-column:1/3;margin-top:3px;"><div id="svc-bar-${s.id}" class="bar-fill" style="width:100%"></div></div>`;
    svcPanel.appendChild(row);
  }
}

// ---------- attack lines & packets ----------
function drawAttackLine(fromRole, toId, color) {
  const a = agents[fromRole];
  const s = services.get(toId);
  if (!a || !s) return;
  const pA = a.position, pB = s.group.position;
  const mid = new THREE.Vector3((pA.x + pB.x) / 2, 3.8, (pA.z + pB.z) / 2);
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(pA.x, 0.8, pA.z),
    mid,
    new THREE.Vector3(pB.x, 1.2, pB.z),
  ]);
  const pts = curve.getPoints(64);
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  // Double-line for visibility against warm floor: thick dark outer + bright inner
  const outerMat = new THREE.LineBasicMaterial({ color: 0x1d1a14, transparent: true, opacity: 0.25 });
  const outerLine = new THREE.Line(geo.clone(), outerMat);
  outerLine.userData.createdAt = performance.now();
  scene.add(outerLine); attackLines.push(outerLine);
  // Inner bright line on top
  const brightColor = fromRole === 'red' ? 0xff5555 : 0x66b3ff;
  const mat = new THREE.LineBasicMaterial({ color: brightColor, transparent: true, opacity: 0.95 });
  const line = new THREE.Line(geo, mat);
  line.userData.createdAt = performance.now();
  scene.add(line); attackLines.push(line);

  for (let i = 0; i < 4; i++) {
    const mm = new THREE.MeshStandardMaterial({ color: brightColor, emissive: brightColor, emissiveIntensity: 1.4, flatShading: true });
    const pkt = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 9), mm);
    const tm = new THREE.MeshBasicMaterial({ color: brightColor, transparent: true, opacity: 0.45 });
    const trail = new THREE.Mesh(new THREE.SphereGeometry(0.34, 10, 8), tm);
    pkt.add(trail); scene.add(pkt);
    attackPackets.push({ mesh: pkt, mat: mm, trailMat: tm, curve, startedAt: performance.now() + i * 150, duration: 1050 });
  }
  s.group.userData.pulseStart = performance.now() + 1050;
  s.group.userData.pulseColor = new THREE.Color(brightColor);
}

function spawnActivityPulse(serviceId) {
  const s = services.get(serviceId);
  if (!s) return;
  const mat = new THREE.MeshBasicMaterial({ color: 0xf4ecd4, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.02, 32), mat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.copy(s.group.position); ring.position.y = 0.03;
  scene.add(ring);
  activityPulses.push({ mesh: ring, mat, t: 0, duration: 1300 });
}

function pulseAgent(role) { const g = agents[role]; if (g) g.userData.pulse = 1.0; }
function pulseService(id, color) {
  const s = services.get(id); if (!s) return;
  s.group.userData.pulseColor = new THREE.Color(color);
  s.group.userData.pulseStart = performance.now();
}

// ---------- speech callouts ----------
function pulseRosterRow(personaId) {
  const row = document.querySelector(`.person[data-persona-id="${personaId}"]`);
  if (!row) return;
  row.classList.add('speaking');
  clearTimeout(row._speakTimer);
  row._speakTimer = setTimeout(() => row.classList.remove('speaking'), 5200);
}

function showCallout({ personaId, displayName, activity, channel, text }) {
  pulseRosterRow(personaId);
  const n = npcs.get(personaId); if (!n) return;
  let entry = speechBubbles.get(personaId);
  let el;
  if (entry) { el = entry.el; el.className = 'callout ' + (channel || 'chat'); }
  else { el = document.createElement('div'); el.className = 'callout ' + (channel || 'chat'); labelLayer.appendChild(el); }
  el.innerHTML = `
    <div class="title-bar">${escapeHtml(activity || 'Activity')}</div>
    <div class="body"><span class="who">${escapeHtml(displayName)}:</span> ${escapeHtml(text)}</div>
    <div class="tail"></div>`;
  const wp = n.group.position;
  el.dataset.worldX = wp.x; el.dataset.worldY = 2.4; el.dataset.worldZ = wp.z;
  speechBubbles.set(personaId, { el, expiresAt: performance.now() + 6000 });
}

// ---------- ticker + thoughts ----------
const thoughtList = document.getElementById('thought-list');
const ticker = document.getElementById('ticker');
let eventsCount = 0;
let callCount = 0;

function tickerRow(cls, time, mark, text) {
  const row = document.createElement('div');
  row.className = 'tr ' + cls;
  row.innerHTML = `<span class="t">t=${Number(time||0).toFixed(2)}</span><span class="mark">${mark}</span><span>${escapeHtml(text)}</span>`;
  ticker.prepend(row);
  while (ticker.childElementCount > 36) ticker.lastElementChild?.remove();
}

// ---------- ws ----------
let ws;
function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/stream`);
  ws.addEventListener('open', () => {
    document.getElementById('pill-live').textContent = '● CONNECTED';
  });
  ws.addEventListener('close', () => {
    document.getElementById('pill-live').textContent = '● DISCONNECTED';
    document.getElementById('pill-live').style.color = 'var(--rose-dark)';
    setTimeout(connect, 1500);
  });
  ws.addEventListener('message', (evt) => {
    let m; try { m = JSON.parse(evt.data); } catch { return; }
    handleMsg(m);
  });
}
connect();

function handleMsg(msg) {
  if (msg.type === 'hello') {
    if (msg.model) document.getElementById('pill-model').textContent = msg.model;
    if (!msg.active && !window.__orAutoStarted) {
      window.__orAutoStarted = true;
      setTimeout(() => {
        try {
          fetch('/session/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_turns: 32 }),
          }).catch(() => {});
        } catch {}
      }, 600);
    }
  } else if (msg.type === 'scene_init') {
    handleSceneInit(msg);
    fetchAndRenderTraces();
  } else if (msg.type === 'turn') {
    advanceSimTime(msg.sim_time || 0);
    document.getElementById('k-sim').textContent    = msg.sim_time.toFixed(2);
    document.getElementById('k-turn').textContent   = msg.turn;
    document.getElementById('k-actor').textContent  = msg.actor;
    for (const h of (msg.service_health || [])) {
      const el = document.getElementById('svc-h-' + h.service_id);
      if (el) el.textContent = h.health.toFixed(2);
      const bar = document.getElementById('svc-bar-' + h.service_id);
      if (bar) bar.style.width = Math.max(0, h.health * 100).toFixed(0) + '%';
      const svc = services.get(h.service_id);
      if (svc?.group?.userData?.ringMat) {
        svc.group.userData.ringMat.color.setHex(
          h.health > 0.85 ? 0x5aa148 : h.health > 0.5 ? 0xe8a55c : 0xd95757
        );
      }
    }
  } else if (msg.type === 'agent_action') {
    pulseAgent(msg.actor);
    callCount += 1;
    document.getElementById('pill-tokens').textContent = `${callCount} / 0`;
    const t = document.createElement('div');
    t.className = 'thought ' + msg.actor;
    const act = `${escapeHtml(msg.kind)} → ${escapeHtml(msg.payload?.target || msg.payload?.path || msg.payload?.event_type || '—')}`;
    t.innerHTML = `<span class="actor">${msg.actor.toUpperCase()}</span><span class="act">${act}</span><div class="body">${escapeHtml(msg.thought || '')}</div>`;
    thoughtList.prepend(t);
    while (thoughtList.childElementCount > 6) thoughtList.lastElementChild?.remove();
    // mirror into the popover version for reading history
    const pop = document.getElementById('thought-list-pop');
    if (pop) {
      pop.prepend(t.cloneNode(true));
      while (pop.childElementCount > 40) pop.lastElementChild?.remove();
    }
    const target = msg.payload?.target;
    if (target) drawAttackLine(msg.actor, target, msg.actor === 'red' ? 0xd95757 : 0x4fa0d1);
    tickerRow(msg.actor, msg.sim_time || 0, msg.actor === 'red' ? '▲' : '◆', `${msg.kind} → ${target || '—'}${msg.thought ? ' · ' + msg.thought : ''}`);
    addTimelineMark(msg.actor, msg.sim_time || simTime);
  } else if (msg.type === 'event') {
    eventsCount++;
    document.getElementById('hud-events').textContent = eventsCount;
    pulseService(msg.target, msg.malicious ? 0xd95757 : 0x5aa148);
    tickerRow('event', msg.time || 0, '★', `${msg.event_type} · ${msg.target}${msg.malicious ? ' (mal)' : ''}`);
    addTimelineMark('event', msg.time || simTime);
  } else if (msg.type === 'speech') {
    showCallout({
      personaId: msg.persona_id, displayName: msg.display_name,
      activity: msg.activity || (msg.channel === 'email' ? 'Email' : 'Chat'),
      channel: msg.channel, text: msg.text,
    });
    tickerRow('speech', msg.time || 0, '❞', `${msg.display_name}: ${msg.text}`);
    addTimelineMark('speech', msg.time || simTime);
  } else if (msg.type === 'reward') {
    const red = fmtSigned(msg.red), blue = fmtSigned(msg.blue);
    const setT = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setT('k-red-r', red);
    setT('k-red-r2', red);
    setT('k-blue-r', blue);
    setT('k-blue-r2', blue);
    setT('k-cont', msg.continuity.toFixed(2));
    setT('k-cont2', msg.continuity.toFixed(2));
    const setW = (id, pct) => { const el = document.getElementById(id); if (el) el.style.width = pct; };
    setW('bar-red',  clampPct(msg.red));
    setW('bar-blue', clampPct(msg.blue));
    setW('bar-cont', (Math.max(0, msg.continuity) * 100).toFixed(1) + '%');
    if (msg.winner) setT('k-winner', msg.winner);
    // Feed live RL chart from real episode rewards
    _rl.red.push(msg.red);
    _rl.blue.push(msg.blue);
    if (_rl.red.length > 120) { _rl.red.shift(); _rl.blue.shift(); }
    drawLiveRl();
    setT('rl-red', red);
    setT('rl-blue', blue);
  } else if (msg.type === 'done') {
    const setT = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setT('k-winner', msg.winner || '—');
    setT('k-reason', msg.terminal_reason || '—');
    setT('pill-status', msg.winner ? (msg.winner.toUpperCase() + ' wins') : 'Done');
    tickerRow('event', 0, '▣', `episode ended · winner=${msg.winner || '—'} · ${msg.terminal_reason || ''}`);
  } else if (msg.type === 'docker') {
    // Runtime presence heartbeat — flip the badges if Kind/Docker drops.
    const metaRuntime = document.getElementById('meta-runtime');
    const metaDocker = document.getElementById('meta-docker');
    const footerDocker = document.getElementById('footer-docker');
    if (msg.present) {
      if (metaRuntime) { metaRuntime.textContent = 'kind · live'; metaRuntime.classList.remove('offline'); }
      if (metaDocker) metaDocker.textContent = msg.short_id || '—';
      if (footerDocker) footerDocker.textContent = `kind · ${msg.short_id || 'live'}`;
    } else {
      if (metaRuntime) { metaRuntime.textContent = 'offline'; metaRuntime.classList.add('offline'); }
      if (metaDocker) metaDocker.textContent = '—';
      if (footerDocker) footerDocker.textContent = 'offline';
    }
  } else if (msg.type === 'lineage') {
    renderLineage(msg.nodes || []);
  } else if (msg.type === 'mutation_pending') {
    tickerRow('event', 0, '↯', `Mutation: scoring parent · red=${msg.red_win_rate?.toFixed(2)} blue=${msg.blue_win_rate?.toFixed(2)}`);
  } else if (msg.type === 'mutation_admitted') {
    tickerRow('event', 0, '✓', `Mutation admitted · gen=${msg.generation} · ops=${(msg.ops || []).join(', ') || 'none'}`);
  } else if (msg.type === 'mutation_rejected') {
    tickerRow('event', 0, '✕', `Mutation rejected · ${msg.reason}`);
  } else if (msg.type === 'mutation_stages') {
    if (msg.world_id) lastMutationStages.set(msg.world_id, msg);
    renderAdmissionReport(msg);
  } else if (msg.type === 'episode_final_score') {
    renderEpisodeFinal(msg);
  } else if (msg.type === 'audit') {
    renderAuditHud(msg);
  } else if (msg.type === 'error') {
    tickerRow('event', 0, '⚠', msg.message);
  }
}

function renderLineage(nodes) {
  const el = document.getElementById('lineage-tree');
  const setT = (id, v) => { const x = document.getElementById(id); if (x) x.textContent = v; };
  setT('pill-lineage-count', nodes.length);
  setT('count-lineage', nodes.length);
  const latest = nodes[nodes.length - 1];
  if (latest) setT('k-gen', 'g' + latest.generation);
  if (!el) return;
  el.innerHTML = '';
  if (!nodes.length) return;

  // Group nodes by generation → columns (left → right).
  const gens = new Map();  // generation → node list
  for (const n of nodes) {
    const g = n.generation ?? 0;
    if (!gens.has(g)) gens.set(g, []);
    gens.get(g).push(n);
  }
  const generations = [...gens.keys()].sort((a, b) => a - b);

  // Build lookup for parent→child edges. Match parent_world_id to a node's
  // world_id (or its suffix — snapshot_id vs world_id ambiguity).
  const byId = new Map();
  for (const n of nodes) {
    if (n.world_id) byId.set(n.world_id, n);
    if (n.snapshot_id) byId.set(n.snapshot_id, n);
  }
  function resolveParent(p) {
    if (!p) return null;
    if (byId.has(p)) return byId.get(p);
    // Fallback: suffix match (world_id ends with parent string or vice versa)
    for (const n of nodes) {
      const keys = [n.world_id, n.snapshot_id].filter(Boolean);
      for (const k of keys) {
        if (k.endsWith(p) || p.endsWith(k)) return n;
      }
    }
    return null;
  }

  // Layout: x by generation index, y by position within generation.
  const W = 280, H = 260;
  const padX = 18, padY = 24;
  const colW = generations.length > 1
    ? (W - padX * 2) / (generations.length - 1)
    : 0;
  const pos = new Map();  // node → {x, y, n}
  generations.forEach((g, gi) => {
    const list = gens.get(g);
    const n = list.length;
    const rowH = n > 1 ? (H - padY * 2) / (n - 1) : 0;
    const x = generations.length > 1 ? padX + gi * colW : W / 2;
    list.forEach((node, i) => {
      const y = n > 1 ? padY + i * rowH : H / 2;
      pos.set(node, { x, y, node });
    });
  });

  // Draw edges, then nodes.
  const edges = [];
  for (const n of nodes) {
    const parent = resolveParent(n.parent_world_id);
    if (parent && pos.has(parent) && pos.has(n)) {
      const a = pos.get(parent), b = pos.get(n);
      edges.push(`<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${((a.x+b.x)/2).toFixed(1)} ${a.y.toFixed(1)}, ${((a.x+b.x)/2).toFixed(1)} ${b.y.toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}" fill="none" stroke="#9c9379" stroke-width="1.1" opacity="0.7"/>`);
    }
  }

  const nodeSvg = [];
  for (const [node, p] of pos.entries()) {
    const isRoot = (node.generation ?? 0) === 0 && !node.parent_world_id;
    const status = node.status || 'admitted';
    let fill = '#4d8548';        // admitted → green
    let stroke = '#1d1a14';
    if (status === 'rejected') fill = '#a64040';
    else if (status === 'proposed') fill = '#c79a18';
    if (isRoot) fill = '#c79a18';   // root seed → gold
    const short = node.short_id || (node.world_id || '').slice(-6) || 'seed';
    const title = [
      `g${node.generation ?? 0} · ${short}`,
      `status: ${status}`,
      node.mutation_ops?.length ? 'ops: ' + node.mutation_ops.join(', ') : '',
      (status === 'admitted' || status === 'proposed') && Number.isFinite(node.red_win_rate)
        ? `r=${node.red_win_rate.toFixed(2)} · b=${node.blue_win_rate.toFixed(2)} · eps=${node.episodes ?? 0}`
        : '',
      node.reason ? node.reason.slice(0, 140) : '',
    ].filter(Boolean).join('\n');
    nodeSvg.push(`
      <g>
        <title>${escapeHtml(title)}</title>
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="7" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>
        <text x="${p.x.toFixed(1)}" y="${(p.y + 19).toFixed(1)}" text-anchor="middle" font-family="JetBrains Mono, ui-monospace, monospace" font-size="9" fill="#3d3a30">${escapeHtml(short)}</text>
      </g>`);
  }

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" width="100%" height="100%">
    ${edges.join('')}
    ${nodeSvg.join('')}
  </svg>`;
}

// ---------- Admission report panel (consumes `mutation_stages` events) ----------
//
// Iterates the server payload exactly — never hardcodes stage names, gate keys,
// or score keys, so new validators added to OpenRange surface without UI changes.
function renderAdmissionReport(msg) {
  const body = document.getElementById('admission-body');
  const sub = document.getElementById('admission-sub');
  if (!body) return;
  if (!msg || typeof msg !== 'object') return;

  const worldId = msg.world_id || '';
  const shortId = worldId ? worldId.slice(-8) : '—';
  const status = msg.status === 'rejected' ? 'rejected' : (msg.status === 'admitted' ? 'admitted' : 'unknown');
  const stages = Array.isArray(msg.stages) ? msg.stages : [];
  const gates = (msg.gates && typeof msg.gates === 'object') ? msg.gates : {};
  const scores = (msg.scores && typeof msg.scores === 'object') ? msg.scores : {};
  const rejectionReasons = Array.isArray(msg.rejection_reasons) ? msg.rejection_reasons : [];

  // Aggregate pass/fail per stage name. Any non-bool `passed` is treated as unknown.
  const stageAgg = new Map();
  for (const s of stages) {
    const name = s?.stage || 'stage';
    let cur = stageAgg.get(name);
    if (!cur) { cur = { pass: 0, fail: 0, unknown: 0, failReasons: [] }; stageAgg.set(name, cur); }
    if (s?.passed === true) cur.pass++;
    else if (s?.passed === false) { cur.fail++; if (s?.reason) cur.failReasons.push(`${s.check || 'check'}: ${s.reason}`); }
    else cur.unknown++;
  }

  const stagePills = [];
  for (const [name, agg] of stageAgg.entries()) {
    let cls = 'unknown';
    if (agg.fail > 0) cls = 'fail';
    else if (agg.pass > 0 && agg.unknown === 0) cls = 'pass';
    const titleParts = [`${name} — ${agg.pass} pass / ${agg.fail} fail`];
    if (agg.failReasons.length) titleParts.push(...agg.failReasons);
    const title = titleParts.join('\n');
    stagePills.push(`<span class="stage-pill ${cls}" title="${escapeHtml(title)}">${escapeHtml(name)}</span>`);
  }

  const gateChips = [];
  for (const [key, val] of Object.entries(gates)) {
    let cls = '', glyph = '•';
    if (val === true) { cls = 'pass'; glyph = '✓'; }
    else if (val === false) { cls = 'fail'; glyph = '✕'; }
    gateChips.push(`<span class="gate-chip ${cls}" title="${escapeHtml(key)} = ${escapeHtml(String(val))}">${escapeHtml(key)} ${glyph}</span>`);
  }

  const scoreChips = [];
  for (const [key, val] of Object.entries(scores)) {
    let disp = '—';
    if (typeof val === 'number' && Number.isFinite(val)) {
      disp = (Math.abs(val) >= 100 || Number.isInteger(val)) ? String(val) : val.toFixed(3);
    } else if (val !== null && val !== undefined) {
      disp = String(val);
    }
    scoreChips.push(`<span class="gate-chip" title="${escapeHtml(key)}">${escapeHtml(key)}: ${escapeHtml(disp)}</span>`);
  }

  const reasonsHtml = (status === 'rejected' && rejectionReasons.length)
    ? `<ul class="rejection-reasons">${rejectionReasons.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>`
    : '';

  body.innerHTML = `
    <div class="admission-head">
      <span class="wid">${escapeHtml(shortId)}</span>
      <span class="status-chip ${status}">${escapeHtml(status)}</span>
    </div>
    ${stagePills.length ? `<div class="admission-section-label">stages</div><div>${stagePills.join('')}</div>` : ''}
    ${gateChips.length ? `<div class="admission-section-label">gates</div><div class="gate-row">${gateChips.join('')}</div>` : ''}
    ${scoreChips.length ? `<div class="admission-section-label">scores</div><div class="score-row">${scoreChips.join('')}</div>` : ''}
    ${reasonsHtml}
  `;
  if (sub) sub.textContent = `${shortId} · ${status}`;
}

// ---------- Episode outcome receipt (consumes `episode_final_score`) ----------
//
// Objective IDs are iterated from the server payload arrays; never hardcoded.
function renderEpisodeFinal(msg) {
  const body = document.getElementById('episode-final-body');
  const sub = document.getElementById('episode-final-sub');
  if (!body) return;
  if (!msg || typeof msg !== 'object') return;

  const winnerRaw = (msg.winner || '').toString().toLowerCase();
  let winnerCls = '';
  if (winnerRaw === 'red') winnerCls = 'red-side';
  else if (winnerRaw === 'blue') winnerCls = 'blue-side';
  else if (winnerRaw === 'timeout') winnerCls = 'timeout';
  else if (winnerRaw === 'failure') winnerCls = 'failure';
  const winnerDisp = winnerRaw || '—';

  const terminal = msg.terminal_reason || '—';
  const redObjs = Array.isArray(msg.red_objectives_satisfied) ? msg.red_objectives_satisfied : [];
  const blueObjs = Array.isArray(msg.blue_objectives_satisfied) ? msg.blue_objectives_satisfied : [];

  const objChips = (arr) => arr.length
    ? arr.map(o => `<span class="obj-chip">${escapeHtml(String(o))}</span>`).join('')
    : `<span class="obj-chip none">(none)</span>`;

  const events = (msg.event_count === undefined || msg.event_count === null) ? '—' : msg.event_count;
  const sim = (typeof msg.sim_time === 'number' && Number.isFinite(msg.sim_time)) ? msg.sim_time.toFixed(1) : '—';
  const cont = (typeof msg.continuity === 'number' && Number.isFinite(msg.continuity)) ? msg.continuity.toFixed(2) : '—';

  // Optional audit subsection — null means "no audit info for this episode".
  let auditHtml = '';
  const audit = msg.audit;
  if (audit && typeof audit === 'object') {
    const roles = Array.isArray(audit.role_diversity) ? audit.role_diversity : [];
    const byActor = {};
    for (const r of roles) { if (r?.actor) byActor[String(r.actor).toLowerCase()] = r; }
    const fmtRole = (actor) => {
      const r = byActor[actor];
      if (!r) return `${actor} —`;
      const uq = (r.unique_fingerprints === undefined || r.unique_fingerprints === null) ? '—' : r.unique_fingerprints;
      const tot = (audit.action_count === undefined || audit.action_count === null) ? '—' : audit.action_count;
      return `${actor} ${uq}/${tot} fp`;
    };
    auditHtml += `<div class="episode-audit-row">diversity: ${escapeHtml(fmtRole('red'))} · ${escapeHtml(fmtRole('blue'))}</div>`;
    if (audit.collapse_warning) {
      auditHtml += `<div class="episode-audit-warn">⚠ action collapse detected</div>`;
    }
    const bi = audit.binary_integrity;
    if (bi && Array.isArray(bi.deltas) && bi.deltas.length > 0) {
      auditHtml += `<div class="episode-audit-row">integrity: ${bi.deltas.length} binary changes</div>`;
    }
  }

  body.innerHTML = `
    <div class="episode-final-head">
      <span class="status-chip ${winnerCls}">${escapeHtml(winnerDisp)}</span>
      <span class="episode-final-reason">${escapeHtml(terminal)}</span>
    </div>
    <div class="episode-obj-cols">
      <div class="episode-obj-col red">
        <div class="episode-obj-label">Red achieved</div>
        <div>${objChips(redObjs)}</div>
      </div>
      <div class="episode-obj-col blue">
        <div class="episode-obj-label">Blue achieved</div>
        <div>${objChips(blueObjs)}</div>
      </div>
    </div>
    <div class="episode-meta-row">
      events: <strong>${escapeHtml(String(events))}</strong>
      <span class="sep">·</span> sim: <strong>${escapeHtml(sim)}s</strong>
      <span class="sep">·</span> continuity: <strong>${escapeHtml(cont)}</strong>
    </div>
    ${auditHtml}
  `;
  if (sub) sub.textContent = `${winnerDisp}`;
}

// ---------- Per-turn diversity HUD (consumes every `audit` event) ----------
//
// Cosmetic only — renders into an existing element inside the Live RL panel.
// Safe to call repeatedly with the same msg (idempotent writes).
const _SPARK_GLYPHS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
function _spark(score) {
  if (typeof score !== 'number' || !Number.isFinite(score)) return '—';
  const s = Math.max(0, Math.min(1, score));
  // Fill up to ~6 cells scaled by score; 8-step gradient feels richer than 6.
  const cells = 6;
  const filled = Math.max(1, Math.round(s * cells));
  let out = '';
  for (let i = 0; i < cells; i++) {
    if (i < filled) {
      // Gradient within filled region so sparklines aren't flat
      const gIdx = Math.min(_SPARK_GLYPHS.length - 1, Math.round((i + 1) / cells * (_SPARK_GLYPHS.length - 1)));
      out += _SPARK_GLYPHS[gIdx];
    } else {
      out += _SPARK_GLYPHS[0];
    }
  }
  return out;
}
function renderAuditHud(msg) {
  if (!msg || typeof msg !== 'object') return;
  const roles = Array.isArray(msg.role_diversity) ? msg.role_diversity : [];
  const byActor = {};
  for (const r of roles) { if (r?.actor) byActor[String(r.actor).toLowerCase()] = r; }
  const totalFp = (msg.action_count === undefined || msg.action_count === null) ? '—' : msg.action_count;

  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  const setClass = (id, cls) => { const el = document.getElementById(id); if (el) el.className = cls; };

  for (const actor of ['red', 'blue']) {
    const r = byActor[actor];
    const sparkEl = document.getElementById('audit-spark-' + actor);
    const fracEl  = document.getElementById('audit-frac-' + actor);
    if (r) {
      if (sparkEl) sparkEl.textContent = _spark(r.diversity_score);
      const uq = (r.unique_fingerprints === undefined || r.unique_fingerprints === null) ? '—' : r.unique_fingerprints;
      if (fracEl) fracEl.textContent = `(${uq}/${totalFp})`;
    } else {
      if (sparkEl) sparkEl.textContent = '—';
      if (fracEl)  fracEl.textContent = '(—/—)';
    }
  }

  // Warn state: promote the first actor flagged collapse_warning.
  let warnActor = null;
  for (const r of roles) {
    if (r?.collapse_warning) { warnActor = String(r.actor || '').toLowerCase(); break; }
  }
  // Top-level collapse_warning fallback if per-actor didn't fire.
  if (!warnActor && msg.collapse_warning) warnActor = 'any';
  const warnEl = document.getElementById('audit-warn');
  if (warnEl) {
    if (warnActor === 'red') { warnEl.textContent = 'RED'; warnEl.className = 'warn red'; }
    else if (warnActor === 'blue') { warnEl.textContent = 'BLUE'; warnEl.className = 'warn blue'; }
    else if (warnActor === 'any') { warnEl.textContent = 'COLLAPSE'; warnEl.className = 'warn red'; }
    else { warnEl.textContent = 'none'; warnEl.className = 'warn none'; }
  }
}

function fmtSigned(n) { return (n >= 0 ? '+' : '') + n.toFixed(3); }
function clampPct(n) { const v = Math.max(-1, Math.min(1, n)); return ((v + 1) / 2 * 100).toFixed(1) + '%'; }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

// ---------- mutation buttons ----------
async function triggerMutate(btn) {
  if (!btn) return;
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = 'mutating…';
  try {
    const res = await fetch('/mutation/propose', { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    setTimeout(() => { btn.disabled = false; btn.textContent = orig; }, 2500);
  } catch (e) {
    btn.textContent = 'retry';
    btn.disabled = false;
  }
}
document.getElementById('btn-mutate')?.addEventListener('click', (ev) => { ev.stopPropagation(); triggerMutate(ev.currentTarget); });
document.getElementById('btn-mutate-meta')?.addEventListener('click', (ev) => { ev.stopPropagation(); triggerMutate(ev.currentTarget); });

// Live RL reward chart: fills as episode turns emit real reward events
const _rl = { red: [], blue: [] };
function drawLiveRl() {
  const el = document.getElementById('chart-live-rl');
  if (!el) return;
  const w = el.clientWidth, h = el.clientHeight;
  const pad = 6;
  if (_rl.red.length === 0) { el.innerHTML = ''; return; }
  const all = [..._rl.red, ..._rl.blue, 0];
  const minY = Math.min(...all), maxY = Math.max(...all);
  const range = Math.max(0.02, maxY - minY);
  const yN = v => h - pad - ((v - minY) / range) * (h - pad * 2);
  const xN = i => pad + (_rl.red.length > 1 ? (i / (_rl.red.length - 1)) * (w - pad * 2) : 0);
  const zero = yN(0);
  const redPts = _rl.red.map((v, i) => `${xN(i).toFixed(1)},${yN(v).toFixed(1)}`).join(' ');
  const bluePts = _rl.blue.map((v, i) => `${xN(i).toFixed(1)},${yN(v).toFixed(1)}`).join(' ');
  el.innerHTML = `<svg width="100%" height="100%">
    <line x1="${pad}" x2="${w - pad}" y1="${zero}" y2="${zero}" stroke="#9c9379" stroke-width="0.5" stroke-dasharray="2 3"/>
    <polyline points="${redPts}" fill="none" stroke="#a64040" stroke-width="1.8" stroke-linejoin="round"/>
    <polyline points="${bluePts}" fill="none" stroke="#3d7aad" stroke-width="1.8" stroke-linejoin="round"/>
  </svg>`;
}
document.getElementById('btn-promote')?.addEventListener('click', async (ev) => {
  ev.stopPropagation();
  const btn = ev.currentTarget;
  btn.disabled = true;
  try {
    const res = await fetch('/mutation/promote', { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    btn.textContent = '● running child';
  } catch (e) {
    btn.textContent = '✕ no child yet';
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Promote latest'; }, 1500);
  }
});

// ---------- start button ----------
document.getElementById('start').addEventListener('click', async (ev) => {
  const btn = ev.currentTarget;
  btn.disabled = true;
  const origText = btn.innerHTML;
  btn.innerHTML = '<span class="dot" style="background:#e8b45a"></span><span>Starting…</span>';
  try {
    const res = await fetch('/session/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_turns: 32 }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    btn.innerHTML = '<span class="dot" style="background:#5fc97a"></span><span>Running</span>';
    setTimeout(() => { btn.disabled = false; btn.innerHTML = '<span class="dot" style="background:#e05e5e"></span><span>Restart</span>'; }, 40000);
  } catch (e) {
    console.error(e); btn.innerHTML = origText; btn.disabled = false;
  }
});

// ---------- render loop ----------
function resize() {
  const r = sceneEl.getBoundingClientRect();
  const w = Math.max(100, r.width | 0);
  const h = Math.max(100, r.height | 0);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
requestAnimationFrame(resize);
setTimeout(resize, 120);
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(sceneEl);

const _v = new THREE.Vector3();
function worldToScreen(x, y, z) {
  _v.set(x, y, z).project(camera);
  const r = sceneEl.getBoundingClientRect();
  return { x: (_v.x * 0.5 + 0.5) * r.width, y: (1 - (_v.y * 0.5 + 0.5)) * r.height };
}

// Label collision avoidance: after positioning, detect overlaps by bbox and
// stagger overlapping NPC nametags vertically (not service/zone labels).
function updateLabels() {
  const placements = [];
  for (const el of labelLayer.children) {
    const wx = parseFloat(el.dataset.worldX);
    const wy = parseFloat(el.dataset.worldY);
    const wz = parseFloat(el.dataset.worldZ);
    if (!Number.isFinite(wx)) continue;
    const s = worldToScreen(wx, wy, wz);
    el.style.left = s.x + 'px';
    el.style.top  = s.y + 'px';
    placements.push({ el, x: s.x, y: s.y });
  }
  // cluster detection: sort NPC tags by y, stagger if vertical gap < 14px
  const npcTags = placements.filter(p => p.el.matches('.nametag:not(.zone):not(.service)'));
  npcTags.sort((a, b) => a.y - b.y);
  for (let i = 1; i < npcTags.length; i++) {
    const prev = npcTags[i - 1], cur = npcTags[i];
    const dx = Math.abs(cur.x - prev.x);
    if (dx < 110 && (cur.y - prev.y) < 15) {
      const shift = 15 - (cur.y - prev.y);
      cur.y += shift;
      cur.el.style.top = cur.y + 'px';
    }
  }
}

let lastFrameT = performance.now();
function tick(now) {
  const dtMs = now - lastFrameT; lastFrameT = now;
  const dt = Math.min(0.08, dtMs / 1000);

  // NPC itineraries — walk between desk and hubs
  for (const n of npcs.values()) {
    if (!n.route) { n.group.position.y = Math.sin(now * 0.001 + n.phase) * 0.03; continue; }
    const wp = n.route[n.routeIdx];
    if (n.dwellRemaining > 0) {
      n.dwellRemaining -= dt;
      // idle bob at desk/hub
      n.group.position.y = Math.sin(now * 0.0015 + n.phase) * 0.04;
      continue;
    }
    const nextIdx = (n.routeIdx + 1) % n.route.length;
    const target = n.route[nextIdx];
    n.routeT = Math.min(1, n.routeT + dt * n.routeSpeed);
    const sx = wp.x, sz = wp.z, tx = target.x, tz = target.z;
    // ease-in-out for more human motion
    const k = n.routeT * n.routeT * (3 - 2 * n.routeT);
    n.group.position.x = sx + (tx - sx) * k;
    n.group.position.z = sz + (tz - sz) * k;
    // face travel direction
    const dxs = tx - sx, dzs = tz - sz;
    if (Math.abs(dxs) + Math.abs(dzs) > 0.01) {
      n.group.rotation.y = Math.atan2(dxs, dzs);
    }
    // walking bob — stronger
    n.group.position.y = Math.abs(Math.sin(now * 0.013)) * 0.12;
    if (n.routeT >= 1) {
      n.routeIdx = nextIdx;
      n.routeT = 0;
      n.dwellRemaining = target.wait;
    }
  }
  for (const role of ['red','blue']) {
    const g = agents[role]; if (!g) continue;
    const u = g.userData;
    if (u.pulse > 0) {
      u.pulse = Math.max(0, u.pulse - 0.03);
      const s = 1 + u.pulse * 0.25;
      g.scale.set(s, 1 + u.pulse * 0.12, s);
    } else { g.scale.set(1, 1, 1); }
    g.position.y = Math.sin(now * 0.002 + (role === 'red' ? 0 : 2)) * 0.05;
  }
  for (let i = attackPackets.length - 1; i >= 0; i--) {
    const p = attackPackets[i];
    const e = now - p.startedAt;
    if (e < 0) continue;
    const u = e / p.duration;
    if (u >= 1) {
      scene.remove(p.mesh); p.mat.dispose(); p.trailMat.dispose(); p.mesh.geometry.dispose();
      attackPackets.splice(i, 1);
      continue;
    }
    const pos = p.curve.getPointAt(u);
    p.mesh.position.copy(pos);
    p.mesh.scale.setScalar(0.85 + 0.25 * Math.sin(now * 0.02 + i));
  }
  for (let i = activityPulses.length - 1; i >= 0; i--) {
    const p = activityPulses[i];
    p.t += dtMs;
    const u = p.t / p.duration;
    if (u >= 1) {
      scene.remove(p.mesh); p.mat.dispose(); p.mesh.geometry.dispose();
      activityPulses.splice(i, 1);
      continue;
    }
    p.mesh.scale.setScalar(1 + u * 1.1);
    p.mat.opacity = 0.5 * (1 - u);
  }
  for (let i = attackLines.length - 1; i >= 0; i--) {
    const l = attackLines[i];
    const age = (now - l.userData.createdAt) / 1600;
    if (age >= 1) { scene.remove(l); attackLines.splice(i, 1); }
    else l.material.opacity = 0.55 * (1 - age);
  }
  for (const s of services.values()) {
    const u = s.group.userData;
    if (u.pulseStart) {
      const dt2 = (now - u.pulseStart) / 700;
      if (dt2 > 1) { u.pulseStart = 0; u.ring.scale.set(1,1,1); u.ringMat.opacity = 0.55; }
      else { u.ring.scale.setScalar(1 + dt2 * 0.8); u.ringMat.opacity = 0.55 * (1 - dt2); if (u.pulseColor) u.ringMat.color.lerp(u.pulseColor, 0.15); }
    }
  }
  for (const [pid, entry] of speechBubbles.entries()) {
    if (now > entry.expiresAt) { entry.el.remove(); speechBubbles.delete(pid); }
    else {
      const n = npcs.get(pid);
      if (n) {
        const s = worldToScreen(n.group.position.x, 2.5 + n.group.position.y, n.group.position.z);
        entry.el.style.left = s.x + 'px';
        entry.el.style.top  = s.y + 'px';
      }
    }
  }

  updateDayCycle(simTime);
  controls.update();
  updateLabels();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- timeline + clock ----------
const timelineMarks = document.getElementById('timeline-marks');
const timelineCursor = document.getElementById('timeline-cursor');
const clockTimeEl = document.getElementById('clock-time');
const clockPhaseEl = document.getElementById('clock-phase');
const clockTurnEl = document.getElementById('clock-turn');

function advanceSimTime(next) {
  simTime = Math.max(simTime, next);
  const tod = timeOfDay(simTime);
  if (clockTimeEl) clockTimeEl.textContent = `${String(tod.hh).padStart(2,'0')}:${String(tod.mm).padStart(2,'0')}`;
  if (clockPhaseEl) clockPhaseEl.textContent = tod.phase;
  if (clockTurnEl) clockTurnEl.textContent = `t=${simTime.toFixed(1)}`;
  if (timelineCursor) timelineCursor.style.left = (tod.t * 100).toFixed(2) + '%';
}
function addTimelineMark(cls, sim) {
  if (!timelineMarks) return;
  const t = Math.max(0, Math.min(1, sim / EP_HORIZON_TICKS));
  const el = document.createElement('div');
  el.className = 'timeline-mark ' + cls;
  el.style.left = (t * 100).toFixed(2) + '%';
  timelineMarks.appendChild(el);
  while (timelineMarks.childElementCount > 80) timelineMarks.firstChild?.remove();
}

// Populate the LLM traces panel immediately on first load so it's not empty
// while waiting for the first scene_init over WebSocket.
fetchAndRenderTraces();


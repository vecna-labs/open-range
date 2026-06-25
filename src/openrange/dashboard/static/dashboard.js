// OpenRange dashboard — full-bleed scene with collapsible inspector rail.
//
// Architecture:
//   * scene state (`sim`)            — three.js renderer, npcs, services, agents
//   * data feed (`refresh` + SSE)    — polls /api/{briefing,topology,lineage,state,...}
//   * chrome (`render*` functions)   — topbar, footbar, rail panels, toasts, banner
//
// The 3D rendering primitives (room layout, kiosks, characters, callouts) are
// ported from the living-office demo's three.js scene; chrome is editorial-light
// (warm parchment palette) so the panels disappear into the office.

const model = {
  briefing: { snapshot_id: null, title: "", goal: "", entrypoints: [], missions: [] },
  topology: {
    snapshot_id: null, world: {}, tasks: [], artifact_paths: [],
    services: [], edges: [], zones: [], users: [], green_personas: [],
  },
  lineage: { snapshot_id: null, admission: null, nodes: [] },
  state: {
    running: false, status: "waiting_for_snapshot",
    health: { uptime: 100, defense: 100, integrity: 100 },
    events: [],
  },
  actors: [],
  narration: { narration: "" },
};

const runState = {
  activeRun: null,
  runs: [],
  events: null,
  narration: null,
  followLatest: true,
  lastRefreshAt: null,
};

// Inspector rail state — which tab is open + per-tab sub-state.
const rail = {
  open: false,
  active: "build",          // build | world | lineage | activity | actor
  worldSubtab: "services",  // services | topology | network | cast
  selectedLineageId: null,
  selectedActorId: "",
};

// Build banner state — surfaces only when a NEW lineage node lands.
const banner = {
  knownLineageIds: new Set(),
  primed: false,
  current: null,            // { id, parentId, summary, ops, kind }
  dismissTimer: null,
};

// Toast stack — short-lived persona/agent highlights. We rate-limit to
// 4 visible at a time so the screen doesn't fill up with bubbles.
const toasts = {
  queue: [],
  maxVisible: 4,
  toastLifetimeMs: 4500,
  recentSpeakIds: new Set(),
};

// =============================================================
// Helpers
// =============================================================

function withRun(path) {
  if (!runState.activeRun) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}run=${encodeURIComponent(runState.activeRun)}`;
}

async function json(path, options) {
  const response = await fetch(withRun(path), options);
  return response.json();
}

function text(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function escapeHtml(value) {
  return text(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function shortText(value, max = 80) {
  const rendered = text(value);
  if (rendered.length <= max) return rendered;
  return `${rendered.slice(0, Math.max(0, max - 1))}…`;
}

function eventData(event) {
  return event.data && typeof event.data === "object" ? event.data : {};
}

function simulationRole(value) {
  const kind = typeof value === "string"
    ? value
    : eventData(value).actor_kind || value.actor || "event";
  if (kind === "agent" || kind === "red") return "agent";
  if (kind === "npc" || kind === "green") return "npc";
  if (kind === "system" || kind === "blue") return "system";
  return "event";
}

// =============================================================
// Palette / role canonicalization
// =============================================================

// Scene materials — pulled toward an actual office, not a cream legal pad.
// Floors are medium walnut, walls are flat off-white, trim is dark walnut.
// The chrome (CSS) sits on top of this and supplies the warm paper accent.
const PAL = {
  // Real-office hardwood: medium-dark walnut. Two plank tones drop the
  // saturation well below the previous tan so the floor reads brown,
  // not yellow.
  floor:    0x735542,   // medium walnut
  floorAlt: 0x55402c,   // darker plank
  wall:     0xeae6dc,   // flat off-white, the slightest warm undertone
  wallTrim: 0x4a341f,   // dark walnut baseboard
  red:      0xa64040,
  blue:     0x3d7aad,
  ink:      0x1d1a14,
};

const ROLE_ALIASES = {
  engineering: "engineer", it: "it_admin", soc: "it_admin",
  people: "hr", r_and_d: "engineer",
};

function canonicalRole(role) {
  const key = String(role || "").toLowerCase();
  return ROLE_ALIASES[key] || key;
}

// Carpet tones de-saturated and pulled toward gray so they read like real
// commercial floor inlays, not Saturday-cartoon mats. Desk emissive colors
// stay punchy — they're the role-identity cue on the monitor.
const ROLE_OVERRIDES = {
  engineer:  { desk: 0x3d7aad, carpet: 0xb8c2cb },   // cool gray-blue
  sales:     { desk: 0xc9881c, carpet: 0xc9bd97 },   // muted gold
  finance:   { desk: 0xa64040, carpet: 0xc4afa8 },   // dusty rose
  it_admin:  { desk: 0x7a5bae, carpet: 0xb8aec4 },   // lavender-gray
  hr:        { desk: 0x4d8548, carpet: 0xb0bba6 },   // sage-gray
  legal:     { desk: 0x8d7648, carpet: 0xc1b9a2 },   // taupe
  ops:       { desk: 0x5e7c3d, carpet: 0xb5bea6 },   // olive-gray
  external:  { desk: 0x8f6a3a, carpet: 0xb8a886 },   // weathered tan (street)
  dmz:       { desk: 0xc79a18, carpet: 0xc7b893 },   // muted gold
  data:      { desk: 0x4f6d88, carpet: 0xa6b3c1 },   // steel blue
  management:{ desk: 0x7a5bae, carpet: 0xb1a3bf },   // muted plum
};

function _hashString(s) {
  let h = 2166136261;
  const str = String(s || "");
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}

function _hslHex(h, s, l) {
  const c = new THREE.Color();
  c.setHSL(h, s, l);
  return (Math.round(c.r * 255) << 16) | (Math.round(c.g * 255) << 8) | Math.round(c.b * 255);
}

function paletteForRole(role) {
  const key = canonicalRole(role);
  if (ROLE_OVERRIDES[key]) return ROLE_OVERRIDES[key];
  const warmHues = [0.05, 0.10, 0.13, 0.27, 0.33, 0.94, 0.75, 0.08];
  const hue = warmHues[_hashString(key) % warmHues.length];
  return { desk: _hslHex(hue, 0.45, 0.48), carpet: _hslHex(hue, 0.28, 0.82) };
}

const KIOSK_ALIASES = {
  web: "web_app", http: "web_app", website: "web_app",
  mail: "email", smtp: "email", pop3: "email", imap: "email",
  database: "db", mysql: "db", postgres: "db", postgresql: "db",
  mssql: "db", mongo: "db", mongodb: "db",
  files: "fileshare", smb: "fileshare", nfs: "fileshare", share: "fileshare",
  ldap: "idp", ad: "idp", activedirectory: "idp", directory: "idp",
  monitoring: "siem", observability: "siem", logs: "siem",
  redis: "cache", memcached: "cache", memcache: "cache",
  queue: "cache", rabbitmq: "cache", kafka: "cache",
};

function canonicalKind(kind) {
  const key = String(kind || "").toLowerCase();
  return KIOSK_ALIASES[key] || key;
}

const ROLE_LABEL_OVERRIDES = {
  engineer: "ENGINEERING", it_admin: "IT · SOC",
  qa: "QA", devops: "DEVOPS", executive: "EXECUTIVE",
};

function labelForRole(role) {
  const key = canonicalRole(role);
  return ROLE_LABEL_OVERRIDES[key]
    || String(role || "").replace(/_/g, " ").toUpperCase();
}

function serviceLabel(svcId) {
  return String(svcId || "").replace(/^svc-/, "");
}

// =============================================================
// Simulation state
// =============================================================

const sim = {
  initialized: false,
  fallback: false,
  fingerprint: "",
  seenEvents: new Set(),
  scene: null, camera: null, renderer: null, controls: null, clock: null,
  worldGroup: null, dynamicNodes: [],
  npcs: new Map(),       // persona_id → { group, deskHome, route, ... }
  services: new Map(),   // service_id → { group, ringMat }
  agents: { red: null, blue: null },
  speechBubbles: new Map(),
  attackPackets: [], activityPulses: [], attackLines: [],
  roomPlan: {}, anchors: null,
  hemi: null, sun: null, labelLayer: null,
};

// =============================================================
// Scene primitives
// =============================================================

function makeFloor(w, d) {
  const g = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.25, d),
    new THREE.MeshStandardMaterial({ color: PAL.floor, roughness: 1.0, flatShading: true }),
  );
  base.position.y = 0.12;
  base.receiveShadow = true;
  g.add(base);
  const plankW = 1.1;
  const planks = Math.floor(w / plankW);
  for (let i = 0; i < planks; i += 1) {
    const p = new THREE.Mesh(
      new THREE.BoxGeometry(plankW * 0.95, 0.01, d * 0.995),
      new THREE.MeshStandardMaterial({
        color: i % 2 ? PAL.floorAlt : PAL.floor,
        roughness: 1.0, flatShading: true,
      }),
    );
    p.position.set(-w / 2 + plankW * (i + 0.5), 0.251, 0);
    p.receiveShadow = true;
    g.add(p);
  }
  return g;
}

function makeExteriorWalls(w, d, h = 1.4) {
  const g = new THREE.Group();
  const wallMat = new THREE.MeshStandardMaterial({
    color: PAL.wall, roughness: 0.9, flatShading: true,
  });
  const t = 0.22;
  const placements = [
    { size: [w, h, t], pos: [0, h / 2 + 0.24, -d / 2] },
    { size: [w, h, t], pos: [0, h / 2 + 0.24, d / 2] },
    { size: [t, h, d], pos: [w / 2, h / 2 + 0.24, 0] },
    { size: [t, h, d], pos: [-w / 2, h / 2 + 0.24, 0] },
  ];
  placements.forEach(({ size, pos }) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(...size), wallMat);
    m.position.set(...pos);
    m.castShadow = true;
    g.add(m);
  });
  return g;
}

function makeRoomCarpet(cx, cz, w, d, color) {
  const rug = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.02, d),
    new THREE.MeshStandardMaterial({ color, roughness: 1.0, flatShading: true }),
  );
  rug.position.set(cx, 0.26, cz);
  rug.receiveShadow = true;
  return rug;
}

function makeRoomWalls(cx, cz, w, d, h = 1.0) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    // Same off-white as exterior walls; interior partitions shouldn't
    // be a different colour family or the office reads jaundiced.
    color: 0xe2dccf, roughness: 0.92, flatShading: true,
  });
  const t = 0.15;
  const north = new THREE.Mesh(new THREE.BoxGeometry(w, h, t), mat);
  north.position.set(cx, h / 2 + 0.24, cz - d / 2);
  north.castShadow = true; north.receiveShadow = true;
  g.add(north);
  for (const sign of [-1, 1]) {
    const ew = new THREE.Mesh(new THREE.BoxGeometry(t, h, d * 0.55), mat);
    ew.position.set(cx + sign * (w / 2), h / 2 + 0.24, cz - d * 0.22);
    ew.castShadow = true; ew.receiveShadow = true;
    g.add(ew);
  }
  const stubW = Math.min(1.4, w / 3);
  for (const sign of [-1, 1]) {
    const st = new THREE.Mesh(new THREE.BoxGeometry(stubW, h, t), mat);
    st.position.set(cx + sign * (w / 2 - stubW / 2), h / 2 + 0.24, cz + d / 2);
    st.castShadow = true; st.receiveShadow = true;
    g.add(st);
  }
  return g;
}

function makeDesk(color) {
  const g = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.08, 0.55),
    new THREE.MeshStandardMaterial({ color: 0xa87345, roughness: 0.55, flatShading: true }),
  );
  top.position.y = 0.54; top.castShadow = true; top.receiveShadow = true;
  g.add(top);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x4a3323, flatShading: true });
  for (const [dx, dz] of [[0.42, 0.22], [-0.42, 0.22], [0.42, -0.22], [-0.42, -0.22]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.5, 0.07), legMat);
    leg.position.set(dx, 0.27, dz);
    leg.castShadow = true;
    g.add(leg);
  }
  const mon = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.32, 0.04),
    new THREE.MeshStandardMaterial({
      color: 0x0e1118, emissive: color, emissiveIntensity: 0.35, flatShading: true,
    }),
  );
  mon.position.set(0, 0.78, -0.18);
  g.add(mon);
  const kb = new THREE.Mesh(
    new THREE.BoxGeometry(0.48, 0.02, 0.14),
    new THREE.MeshStandardMaterial({ color: 0x2a2824, flatShading: true }),
  );
  kb.position.set(0, 0.59, 0.12);
  g.add(kb);
  const chair = new THREE.Mesh(
    new THREE.BoxGeometry(0.38, 0.06, 0.38),
    new THREE.MeshStandardMaterial({ color: 0x2e2a26, flatShading: true }),
  );
  chair.position.set(0, 0.32, 0.48); chair.castShadow = true;
  g.add(chair);
  const backrest = new THREE.Mesh(
    new THREE.BoxGeometry(0.38, 0.42, 0.05),
    new THREE.MeshStandardMaterial({ color: 0x2e2a26, flatShading: true }),
  );
  backrest.position.set(0, 0.54, 0.65); backrest.castShadow = true;
  g.add(backrest);
  return g;
}

const SKIN_TONES = [0xf0d4b0, 0xe8c9a8, 0xd9ae85, 0xb98860, 0x8a5e3c];
const HAIR_TONES = [0x2a1a10, 0x3d2818, 0x5a3a20, 0x7a5230, 0xa87345, 0xd4b078];

function makeCharacter(color, seed) {
  const group = new THREE.Group();
  const skin = seed != null
    ? SKIN_TONES[_hashString(seed + "skin") % SKIN_TONES.length]
    : 0xe8c9a8;
  const hairColor = seed != null
    ? HAIR_TONES[_hashString(seed + "hair") % HAIR_TONES.length]
    : 0x3d2818;
  const bodyGeo = new THREE.BoxGeometry(0.7, 1.05, 0.48);
  const body = new THREE.Mesh(
    bodyGeo,
    new THREE.MeshStandardMaterial({ color, roughness: 0.68, flatShading: true }),
  );
  body.position.y = 0.58; body.castShadow = true; body.receiveShadow = true;
  group.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 14, 10),
    new THREE.MeshStandardMaterial({ color: skin, roughness: 0.6, flatShading: true }),
  );
  head.position.y = 1.30; head.castShadow = true;
  group.add(head);
  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.35, 14, 8, 0, Math.PI * 2, 0, Math.PI * 0.55),
    new THREE.MeshStandardMaterial({ color: hairColor, flatShading: true }),
  );
  hair.position.y = 1.30; hair.castShadow = true;
  group.add(hair);
  const ringMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.62, side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.48, 0.62, 28), ringMat);
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.29;
  group.add(ring);
  group.userData = { body, head, hair, ringMat, pulse: 0 };
  return group;
}

const KIOSK_WOOD = 0x8a6a4a;
const KIOSK_METAL = 0x5a5046;
const KIOSK_DARK = 0x1d1a14;

function _kioskBase(w = 1.1, d = 0.9) {
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(w, 0.08, d),
    new THREE.MeshStandardMaterial({ color: KIOSK_WOOD, roughness: 0.7, flatShading: true }),
  );
  base.position.y = 0.3; base.castShadow = true;
  return base;
}

function _kioskScreen(w, h, color, emissive = 0.6) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.04),
    new THREE.MeshStandardMaterial({
      color: KIOSK_DARK, emissive: color, emissiveIntensity: emissive, flatShading: true,
    }),
  );
}

function _kioskGlow(w, h, color) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.02),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.95 }),
  );
}

function _makeWebKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.7));
  const stand = new THREE.Mesh(
    new THREE.BoxGeometry(0.14, 0.55, 0.14),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.6, flatShading: true }),
  );
  stand.position.y = 0.62; stand.castShadow = true; g.add(stand);
  const screen = _kioskScreen(0.86, 0.58, color, 0.7);
  screen.position.set(0, 1.05, 0); screen.castShadow = true; g.add(screen);
  return g;
}

function _makeDbKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.8));
  const tower = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 1.0, 0.55),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true }),
  );
  tower.position.y = 0.85; tower.castShadow = true; g.add(tower);
  for (let i = 0; i < 3; i += 1) {
    const led = _kioskGlow(0.55, 0.06, color);
    led.position.set(0, 0.58 + i * 0.21, 0.29);
    g.add(led);
  }
  return g;
}

function _makeMailKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.8));
  const stand = new THREE.Mesh(
    new THREE.BoxGeometry(0.1, 0.45, 0.1),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.6, flatShading: true }),
  );
  stand.position.y = 0.57; stand.castShadow = true; g.add(stand);
  const env = new THREE.Mesh(
    new THREE.BoxGeometry(0.78, 0.5, 0.06),
    new THREE.MeshStandardMaterial({ color: 0xfaf3dd, roughness: 0.9, flatShading: true }),
  );
  env.position.set(0, 0.95, 0); env.castShadow = true; g.add(env);
  const flap = _kioskGlow(0.1, 0.1, color);
  flap.position.set(0, 0.95, 0.03); g.add(flap);
  return g;
}

function _makeFilesKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.9));
  const cabinet = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 1.0, 0.7),
    new THREE.MeshStandardMaterial({ color: 0x6a4a2a, roughness: 0.6, flatShading: true }),
  );
  cabinet.position.y = 0.85; cabinet.castShadow = true; g.add(cabinet);
  for (let i = 0; i < 3; i += 1) {
    const drawer = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.28, 0.04),
      new THREE.MeshStandardMaterial({ color, roughness: 0.5, flatShading: true }),
    );
    drawer.position.set(0, 0.5 + i * 0.32, 0.36);
    g.add(drawer);
  }
  return g;
}

function _makeDirectoryKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.8));
  const post = new THREE.Mesh(
    new THREE.BoxGeometry(0.14, 1.2, 0.14),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true }),
  );
  post.position.y = 0.95; post.castShadow = true; g.add(post);
  for (let i = 0; i < 3; i += 1) {
    const sign = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.18, 0.04),
      new THREE.MeshStandardMaterial({ color, roughness: 0.6, flatShading: true }),
    );
    sign.position.set(i % 2 === 0 ? 0.38 : -0.38, 0.7 + i * 0.32, 0);
    g.add(sign);
  }
  return g;
}

function _makeSiemKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(1.0, 0.9));
  const screen = _kioskScreen(0.85, 0.65, color, 0.8);
  screen.position.set(0, 0.95, 0); screen.castShadow = true; g.add(screen);
  for (let i = 0; i < 4; i += 1) {
    const tick = _kioskGlow(0.16, 0.025, color);
    tick.position.set(-0.32 + i * 0.21, 0.9, 0.025);
    g.add(tick);
  }
  return g;
}

function _makeCacheKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.85, 0.7));
  const drum = new THREE.Mesh(
    new THREE.CylinderGeometry(0.32, 0.32, 0.85, 18),
    new THREE.MeshStandardMaterial({ color: KIOSK_METAL, roughness: 0.55, flatShading: true }),
  );
  drum.position.y = 0.78; drum.castShadow = true; g.add(drum);
  const led = _kioskGlow(0.1, 0.1, color);
  led.position.set(0, 1.18, 0.32); g.add(led);
  return g;
}

function _makeGenericKiosk(color) {
  const g = new THREE.Group();
  g.add(_kioskBase(0.9, 0.9));
  const block = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 0.85, 0.55),
    new THREE.MeshStandardMaterial({ color, roughness: 0.6, flatShading: true }),
  );
  block.position.y = 0.78; block.castShadow = true; g.add(block);
  return g;
}

function makeKiosk(kind, color) {
  const k = canonicalKind(kind);
  if (k === "web_app") return _makeWebKiosk(color);
  if (k === "db") return _makeDbKiosk(color);
  if (k === "email") return _makeMailKiosk(color);
  if (k === "fileshare") return _makeFilesKiosk(color);
  if (k === "idp") return _makeDirectoryKiosk(color);
  if (k === "siem") return _makeSiemKiosk(color);
  if (k === "cache") return _makeCacheKiosk(color);
  return _makeGenericKiosk(color);
}

function makePlant() {
  const g = new THREE.Group();
  const pot = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.28, 0.35, 12),
    new THREE.MeshStandardMaterial({ color: 0x8a5a3a, roughness: 0.8, flatShading: true }),
  );
  pot.position.y = 0.17; pot.castShadow = true; g.add(pot);
  const leaves = new THREE.Mesh(
    new THREE.SphereGeometry(0.38, 10, 7),
    new THREE.MeshStandardMaterial({ color: 0x4e7e3e, roughness: 0.8, flatShading: true }),
  );
  leaves.position.y = 0.6; leaves.scale.set(1.1, 1.2, 1.1); leaves.castShadow = true;
  g.add(leaves);
  return g;
}

// =============================================================
// Room layout (per-topology rebuild)
// =============================================================

function seatFor(role, seatIndex, totalSeats) {
  const plan = sim.roomPlan[canonicalRole(role)];
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
  const roles = Object.keys(roleCounts)
    .filter((r) => roleCounts[r] > 0)
    .sort((a, b) => roleCounts[b] - roleCounts[a]);
  const plan = {};
  const cols = 2;
  const gap = 0.8;
  let rowStartZ = -3.0;
  for (let i = 0; i < roles.length; i += cols) {
    const pair = roles.slice(i, i + cols);
    const rowMaxN = Math.max(...pair.map((r) => roleCounts[r]));
    const w = Math.max(5.5, Math.min(10, 4.5 + rowMaxN * 0.8));
    const d = Math.max(4.0, Math.min(6.5, 3.5 + rowMaxN * 0.35));
    pair.forEach((role, idx) => {
      const sign = idx === 0 ? -1 : 1;
      const cx = sign * (w / 2 + gap / 2);
      plan[role] = { cx, cz: rowStartZ + d / 2, w, d, label: labelForRole(role) };
    });
    rowStartZ += d + gap;
  }
  plan.external = { cx: 0, cz: -13.0, w: 24.0, d: 2.5, label: "EXTERNAL" };
  return plan;
}

const SERVICE_ROOM_OVERRIDES = {
  web_app: "external", email: "external",
  fileshare: "finance", db: "finance",
  idp: "it_admin", siem: "it_admin", cache: "it_admin",
};

function roomForService(svc) {
  const k = canonicalKind(svc.kind);
  const roomKeys = Object.keys(sim.roomPlan).filter((r) => r !== "corp");
  if (SERVICE_ROOM_OVERRIDES[k] && sim.roomPlan[SERVICE_ROOM_OVERRIDES[k]]) {
    return SERVICE_ROOM_OVERRIDES[k];
  }
  if (svc.zone && sim.roomPlan[svc.zone]) return svc.zone;
  const canonZone = canonicalRole(svc.zone);
  if (canonZone && sim.roomPlan[canonZone]) return canonZone;
  const interior = roomKeys.filter((r) => r !== "external");
  if (interior.length) {
    const idx = _hashString(k || svc.id || "") % interior.length;
    return interior[idx];
  }
  return roomKeys[0] || "external";
}

function computeServicePlacements(servicesArr) {
  const byRoom = new Map();
  for (const s of servicesArr) {
    const room = roomForService(s);
    if (!byRoom.has(room)) byRoom.set(room, []);
    byRoom.get(room).push(s);
  }
  const placements = {};
  for (const [roomKey, list] of byRoom.entries()) {
    const plan = sim.roomPlan[roomKey] || { cx: 0, cz: -12, w: 20, d: 4 };
    const n = list.length;
    const span = Math.max(3.0, plan.w - 2.0);
    for (let i = 0; i < n; i += 1) {
      const t = n === 1 ? 0 : (i / (n - 1) - 0.5);
      const x = roomKey === "external"
        ? plan.cx + t * Math.max(18, plan.w * 1.1)
        : plan.cx + t * span * 0.8;
      const z = roomKey === "external"
        ? plan.cz + ((i % 2 === 0) ? -0.2 : 0.6)
        : plan.cz - plan.d / 2 + 0.7 + ((i % 2 === 0) ? 0 : 0.9);
      placements[list[i].id] = { x, z, label: serviceLabel(list[i].id) };
    }
  }
  return placements;
}

function computeSceneAnchors() {
  const rooms = Object.entries(sim.roomPlan).filter(
    ([k, v]) => k !== "external" && k !== "corp" && v,
  );
  if (!rooms.length) {
    return {
      redSpawn: { x: -8, z: -12 }, blueSpawn: { x: 4, z: 2 },
      coffeeHub: { x: 0, z: 0 }, chatHub: { x: 0, z: 4 }, coolerHub: { x: -9, z: 0 },
    };
  }
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  let cxSum = 0, czSum = 0, biggest = null, biggestArea = -1;
  for (const [, p] of rooms) {
    minX = Math.min(minX, p.cx - p.w / 2);
    maxX = Math.max(maxX, p.cx + p.w / 2);
    minZ = Math.min(minZ, p.cz - p.d / 2);
    maxZ = Math.max(maxZ, p.cz + p.d / 2);
    cxSum += p.cx; czSum += p.cz;
    const area = p.w * p.d;
    if (area > biggestArea) { biggestArea = area; biggest = p; }
  }
  const avgCx = cxSum / rooms.length;
  const avgCz = czSum / rooms.length;
  const itRoom = sim.roomPlan.it_admin || sim.roomPlan.it || biggest;
  return {
    redSpawn: { x: minX + (maxX - minX) * 0.12, z: maxZ + 3.5 },
    blueSpawn: { x: itRoom.cx, z: itRoom.cz },
    coffeeHub: { x: avgCx, z: (minZ + maxZ) / 2 - 1.8 },
    chatHub: (sim.roomPlan.sales || sim.roomPlan.marketing)
      ? { x: (sim.roomPlan.sales || sim.roomPlan.marketing).cx,
          z: (sim.roomPlan.sales || sim.roomPlan.marketing).cz }
      : { x: avgCx, z: (minZ + maxZ) / 2 + 1.8 },
    coolerHub: { x: minX - 1, z: avgCz },
  };
}

function fitCameraToPlan() {
  const rects = Object.values(sim.roomPlan).filter((r) => r && Number.isFinite(r.cx));
  if (!rects.length || !sim.camera) return;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const r of rects) {
    minX = Math.min(minX, r.cx - r.w / 2);
    maxX = Math.max(maxX, r.cx + r.w / 2);
    minZ = Math.min(minZ, r.cz - r.d / 2);
    maxZ = Math.max(maxZ, r.cz + r.d / 2);
  }
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;
  const dx = maxX - minX, dz = maxZ - minZ;
  const diag = Math.max(14, Math.sqrt(dx * dx + dz * dz));
  if (sim.camera.isOrthographicCamera) {
    const span = Math.max(diag * 0.7, 16);
    sim.camera.left = -span; sim.camera.right = span;
    sim.camera.top = span; sim.camera.bottom = -span;
    sim.camera.position.set(cx + diag * 0.7, diag * 0.9, cz + diag * 0.9);
  } else {
    sim.camera.position.set(cx + diag * 0.45, Math.max(14, diag * 0.85), cz + diag * 0.75);
  }
  const lookZ = cz + 2;
  sim.camera.lookAt(cx, 0, lookZ);
  sim.camera.updateProjectionMatrix();
  if (sim.controls) {
    sim.controls.target.set(cx, 0, lookZ);
    sim.controls.minDistance = Math.max(10, diag * 0.35);
    sim.controls.maxDistance = Math.max(30, diag * 1.6);
    sim.controls.update();
  }
}

function disposeObject(object) {
  if (!object || !object.traverse) return;
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((m) => {
        if (m.map) m.map.dispose();
        m.dispose();
      });
    }
  });
}

function clearDynamic() {
  for (const n of sim.dynamicNodes) {
    sim.scene.remove(n);
    disposeObject(n);
  }
  sim.dynamicNodes = [];
  sim.npcs.forEach((entry) => {
    sim.scene.remove(entry.group);
    if (entry.desk) sim.scene.remove(entry.desk);
    if (entry.nameTag) entry.nameTag.remove();
  });
  sim.npcs.clear();
  sim.services.forEach((entry) => {
    sim.scene.remove(entry.group);
    if (entry.label) entry.label.remove();
  });
  sim.services.clear();
  for (const role of ["red", "blue"]) {
    if (sim.agents[role]) {
      sim.scene.remove(sim.agents[role]);
      if (sim.agents[role].userData?.tag) sim.agents[role].userData.tag.remove();
      sim.agents[role] = null;
    }
  }
  sim.speechBubbles.forEach((entry) => entry.el.remove());
  sim.speechBubbles.clear();
  if (sim.labelLayer) {
    sim.labelLayer.querySelectorAll(".dash-zone-label").forEach((el) => el.remove());
  }
}

function buildOfficeShell() {
  const interiorRooms = Object.entries(sim.roomPlan)
    .filter(([k, v]) => k !== "external" && v)
    .map(([, v]) => v);
  let minX = -8, maxX = 8, minZ = -5, maxZ = 5;
  for (const p of interiorRooms) {
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
  sim.scene.add(floor); sim.dynamicNodes.push(floor);
  const outer = makeExteriorWalls(floorW, floorD, 1.35);
  outer.position.set(floorCX, 0, floorCZ);
  sim.scene.add(outer); sim.dynamicNodes.push(outer);
  const street = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(24, floorW + 4), 0.14, 4.2),
    // Cool sidewalk grey — separates from the warm wood interior.
    new THREE.MeshStandardMaterial({ color: 0xafa89a, roughness: 1.0, flatShading: true }),
  );
  street.position.set(floorCX, 0.07, minZ - padZ - 3.8);
  street.receiveShadow = true;
  sim.scene.add(street); sim.dynamicNodes.push(street);
}

function buildRooms() {
  for (const role of Object.keys(sim.roomPlan)) {
    const plan = sim.roomPlan[role];
    if (!plan || role === "external" || role === "corp") continue;
    // No room carpet — the hardwood runs continuous through every
    // office. Walls + nametags + decor identify each room. (A small
    // throw-rug per role would be a future polish item, but a wall-to-
    // wall mat broke the floor and made the office look like patchwork.)
    const walls = makeRoomWalls(plan.cx, plan.cz, plan.w, plan.d, 0.7);
    sim.scene.add(walls); sim.dynamicNodes.push(walls);
    addRoomLabel(plan.label, plan.cx, plan.cz - plan.d / 2 + 0.3);
    const plant = makePlant();
    plant.position.set(plan.cx - plan.w / 2 + 0.5, 0, plan.cz - plan.d / 2 + 0.5);
    sim.scene.add(plant); sim.dynamicNodes.push(plant);
  }
}

function addRoomLabel(text, x, z) {
  if (!sim.labelLayer) return;
  const el = document.createElement("div");
  el.className = "dash-zone-label";
  el.textContent = text;
  el.dataset.worldX = x; el.dataset.worldY = 1.0; el.dataset.worldZ = z;
  sim.labelLayer.appendChild(el);
}

function initSimulation() {
  if (sim.initialized) return;
  sim.initialized = true;
  const canvas = document.getElementById("sim-canvas");
  if (!canvas || !window.THREE) {
    sim.fallback = true;
    return;
  }
  sim.labelLayer = document.getElementById("sim-label-layer");

  sim.scene = new THREE.Scene();
  sim.scene.background = new THREE.Color(0xeae6dc);  // matches PAL.wall
  sim.clock = new THREE.Clock();

  sim.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  sim.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  sim.renderer.shadowMap.enabled = true;
  sim.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  sim.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 220);
  sim.camera.position.set(15, 23, 20);
  sim.camera.lookAt(0, 0, 4);

  if (THREE.OrbitControls) {
    sim.controls = new THREE.OrbitControls(sim.camera, canvas);
    sim.controls.enableDamping = true;
    sim.controls.dampingFactor = 0.09;
    sim.controls.minDistance = 12;
    sim.controls.maxDistance = 46;
    sim.controls.maxPolarAngle = Math.PI * 0.46;
    sim.controls.target.set(0, 0, 4);
  }

  // Neutral daylight so the off-white walls actually read white. The
  // previous gold-cast lights were the second source of the "yellow
  // office" look — even with white materials they tinted everything.
  sim.hemi = new THREE.HemisphereLight(0xfaf6ee, 0x9a8e76, 0.95);
  sim.scene.add(sim.hemi);
  sim.sun = new THREE.DirectionalLight(0xfff7ec, 1.15);
  sim.sun.position.set(16, 26, 12);
  sim.sun.castShadow = true;
  sim.sun.shadow.mapSize.set(2048, 2048);
  sim.sun.shadow.camera.left = -24;
  sim.sun.shadow.camera.right = 24;
  sim.sun.shadow.camera.top = 24;
  sim.sun.shadow.camera.bottom = -14;
  sim.sun.shadow.bias = -0.00025;
  sim.scene.add(sim.sun);
  // Cool window-side bounce keeps the off-white walls from looking flat.
  const fill = new THREE.DirectionalLight(0xd0dcec, 0.42);
  fill.position.set(-10, 14, -10);
  sim.scene.add(fill);

  resizeSimulation();
  window.addEventListener("resize", resizeSimulation);
  installActorSelection(canvas);
  animateSimulation();
}

function simulationFingerprint() {
  const personas = (model.topology.green_personas || []).slice();
  const personaPart = personas
    .map((p) => `${p.id || p.display_name || ""}:${p.role || ""}`)
    .sort().join("|");
  const services = (model.topology.services || []).slice();
  const servicePart = services.map((s) => s.id).sort().join("|");
  return `${model.topology.snapshot_id || "empty"}::${personaPart}::${servicePart}`;
}

function rebuildSimulationWorld() {
  if (sim.fallback || !sim.scene) return;
  const next = simulationFingerprint();
  if (sim.fingerprint === next) return;
  sim.fingerprint = next;
  clearDynamic();
  sim.seenEvents.clear();

  const personas = model.topology.green_personas || [];
  const services = model.topology.services || [];

  const roleCounts = {};
  if (personas.length) {
    for (const p of personas) {
      const r = canonicalRole(p.role || "engineer");
      roleCounts[r] = (roleCounts[r] || 0) + 1;
    }
  } else {
    roleCounts.engineer = 1;
    roleCounts.it_admin = 1;
  }
  sim.roomPlan = computeRoomPlan(roleCounts);
  sim.anchors = computeSceneAnchors();
  fitCameraToPlan();
  buildOfficeShell();
  buildRooms();

  const seatCounter = {};
  for (const p of personas) {
    const r = canonicalRole(p.role || "engineer");
    seatCounter[r] = (seatCounter[r] || 0) - 1;
  }
  const seatIdx = {};
  for (const p of personas) {
    const id = p.id || p.display_name || `persona-${sim.npcs.size}`;
    const role = p.role || "engineer";
    const canonR = canonicalRole(role);
    seatIdx[canonR] = (seatIdx[canonR] || 0) + 1;
    const [dx, dz] = seatFor(role, seatIdx[canonR], Math.abs(seatCounter[canonR]));
    const color = paletteForRole(role).desk;
    const ch = makeCharacter(color, id);
    ch.position.set(dx, 0, dz);
    ch.rotation.y = Math.PI;
    ch.userData.actorId = id;
    ch.userData.persona = p;
    ch.traverse((child) => { child.userData.actorId = id; });
    sim.scene.add(ch);
    const desk = makeDesk(color);
    desk.position.set(dx, 0.05, dz + 0.8);
    sim.scene.add(desk);
    const tag = document.createElement("div");
    tag.className = `dash-nametag ${canonR}`;
    tag.textContent = p.display_name || id;
    tag.dataset.worldX = dx; tag.dataset.worldY = 1.95; tag.dataset.worldZ = dz;
    if (sim.labelLayer) sim.labelLayer.appendChild(tag);
    const route = sim.anchors ? [
      { x: dx, z: dz, wait: 6 + Math.random() * 3 },
      { x: sim.anchors.coffeeHub.x, z: sim.anchors.coffeeHub.z, wait: 1.5 },
      { x: dx, z: dz, wait: 5 },
      { x: sim.anchors.chatHub.x, z: sim.anchors.chatHub.z, wait: 2 },
      { x: dx, z: dz, wait: 7 },
      { x: sim.anchors.coolerHub.x, z: sim.anchors.coolerHub.z, wait: 1.2 },
      { x: dx, z: dz, wait: 8 },
    ] : null;
    sim.npcs.set(id, {
      group: ch, desk, nameTag: tag, color,
      phase: Math.random() * Math.PI * 2,
      deskHome: [dx, dz],
      route,
      routeIdx: route ? (_hashString(id) % route.length) : 0,
      routeT: 0, routeSpeed: 0.35 + 0.12 * Math.random(),
      dwellRemaining: route ? route[0].wait : 0,
      visitTarget: null,
    });
  }

  const placements = computeServicePlacements(services);
  for (const s of services) {
    const placement = placements[s.id] || { x: 0, z: 0, label: serviceLabel(s.id) };
    const color = paletteForRole(s.kind).desk;
    const g = makeKiosk(s.kind, color);
    g.position.set(placement.x, 0, placement.z);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x5aa148, transparent: true, opacity: 0.55, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.1, 32), ringMat);
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02;
    g.add(ring);
    g.userData = { ring, ringMat, pulseStart: 0, pulseColor: null, serviceId: s.id };
    g.traverse((child) => { child.userData.serviceId = s.id; });
    sim.scene.add(g);
    const label = document.createElement("div");
    label.className = "dash-nametag service";
    label.textContent = placement.label;
    label.dataset.worldX = placement.x; label.dataset.worldY = 2.1; label.dataset.worldZ = placement.z;
    if (sim.labelLayer) sim.labelLayer.appendChild(label);
    sim.services.set(s.id, { group: g, ringMat, label });
  }

  if (sim.anchors) {
    sim.agents.red = makeCharacter(PAL.red, "red-agent");
    sim.agents.red.position.set(sim.anchors.redSpawn.x, 0, sim.anchors.redSpawn.z);
    sim.agents.red.userData.actorId = "agent";
    sim.agents.red.traverse((c) => { c.userData.actorId = "agent"; });
    sim.scene.add(sim.agents.red);
    const redTag = document.createElement("div");
    redTag.className = "dash-nametag red";
    redTag.textContent = "agent";
    redTag.dataset.worldX = sim.anchors.redSpawn.x;
    redTag.dataset.worldY = 2.0;
    redTag.dataset.worldZ = sim.anchors.redSpawn.z;
    if (sim.labelLayer) sim.labelLayer.appendChild(redTag);
    sim.agents.red.userData.tag = redTag;

    sim.agents.blue = makeCharacter(PAL.blue, "blue-agent");
    sim.agents.blue.position.set(sim.anchors.blueSpawn.x, 0, sim.anchors.blueSpawn.z);
    sim.agents.blue.userData.actorId = "system";
    sim.agents.blue.traverse((c) => { c.userData.actorId = "system"; });
    sim.scene.add(sim.agents.blue);
    const blueTag = document.createElement("div");
    blueTag.className = "dash-nametag blue";
    blueTag.textContent = "runtime";
    blueTag.dataset.worldX = sim.anchors.blueSpawn.x;
    blueTag.dataset.worldY = 2.0;
    blueTag.dataset.worldZ = sim.anchors.blueSpawn.z;
    if (sim.labelLayer) sim.labelLayer.appendChild(blueTag);
    sim.agents.blue.userData.tag = blueTag;
  }
}

// =============================================================
// Event → scene wiring + toast spawn
// =============================================================

function updateSimulationFromEvents() {
  if (sim.fallback) return;
  const events = (model.state.events || []);
  for (const event of events) {
    if (sim.seenEvents.has(event.id)) continue;
    sim.seenEvents.add(event.id);
    applySimulationEvent(event);
  }
}

function applySimulationEvent(event) {
  const role = simulationRole(event);
  const data = eventData(event);
  const action = (data.action && typeof data.action === "object") ? data.action : {};
  const actorId = data.actor_id || event.actor || role;

  if (role === "agent") {
    const targetId = data.target || event.target;
    if (sim.services.has(targetId)) {
      pulseService(targetId, 0xff5555);
      drawAttackLine("red", targetId, 0xff5555);
    }
    if (sim.agents.red) sim.agents.red.userData.pulse = 1.0;
    return;
  }
  if (role === "system") {
    const targetId = data.target || event.target;
    if (sim.services.has(targetId)) {
      pulseService(targetId, 0x66b3ff);
      drawAttackLine("blue", targetId, 0x66b3ff);
    }
    if (sim.agents.blue) sim.agents.blue.userData.pulse = 1.0;
    return;
  }
  if (role !== "npc") return;
  const persona = sim.npcs.get(actorId);
  if (!persona) {
    spawnLateNpc(actorId, action);
    return;
  }
  if (action.present) return;
  if (typeof action.speak === "string" && action.speak) {
    showCallout({
      personaId: actorId,
      displayName: action.display_name || actorId,
      activity: action.activity || (action.move ? "walking" : "speaking"),
      channel: action.channel || (action.kind === "mail" ? "email" : "chat"),
      text: action.speak,
    });
    pushToast({
      kind: "npc",
      who: action.display_name || actorId,
      text: action.speak,
      key: event.id,
    });
  }
  if (action.move) {
    const target = action.target_name && sim.npcs.get(action.target_name);
    if (target?.deskHome) {
      const [tx, tz] = target.deskHome;
      persona.visitTarget = { x: tx + 1.4, z: tz + 0.2 };
      persona.dwellRemaining = 0;
    }
  }
  if (action.visit && sim.services.size) {
    const svc = sim.services.get(data.target) || sim.services.get(action.visit);
    if (svc) pulseService(svc.group.userData.serviceId, persona.color);
  }
}

function spawnLateNpc(actorId, action) {
  if (!sim.scene) return;
  const role = action.role || "engineer";
  const color = paletteForRole(role).desk;
  const ch = makeCharacter(color, actorId);
  const x = sim.anchors ? sim.anchors.chatHub.x : 0;
  const z = sim.anchors ? sim.anchors.chatHub.z : 0;
  ch.position.set(x, 0, z);
  ch.userData.actorId = actorId;
  ch.traverse((c) => { c.userData.actorId = actorId; });
  sim.scene.add(ch);
  const tag = document.createElement("div");
  tag.className = `dash-nametag ${canonicalRole(role)}`;
  tag.textContent = action.display_name || actorId;
  tag.dataset.worldX = x; tag.dataset.worldY = 1.95; tag.dataset.worldZ = z;
  if (sim.labelLayer) sim.labelLayer.appendChild(tag);
  sim.npcs.set(actorId, {
    group: ch, desk: null, nameTag: tag, color,
    phase: Math.random() * Math.PI * 2, deskHome: [x, z],
    route: null, routeIdx: 0, routeT: 0, routeSpeed: 0,
    dwellRemaining: 0, visitTarget: null,
  });
}

function pulseService(serviceId, color) {
  const s = sim.services.get(serviceId);
  if (!s) return;
  s.group.userData.pulseStart = performance.now();
  s.group.userData.pulseColor = new THREE.Color(color);
  spawnActivityPulse(serviceId);
}

function spawnActivityPulse(serviceId) {
  const s = sim.services.get(serviceId);
  if (!s) return;
  const mat = new THREE.MeshBasicMaterial({
    color: 0xf4ecd4, transparent: true, opacity: 0.5, side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.02, 32), mat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.copy(s.group.position);
  ring.position.y = 0.03;
  sim.scene.add(ring);
  sim.activityPulses.push({ mesh: ring, mat, t: 0, duration: 1300 });
}

function drawAttackLine(fromRole, toId, brightColor) {
  const a = sim.agents[fromRole];
  const s = sim.services.get(toId);
  if (!a || !s) return;
  const pA = a.position;
  const pB = s.group.position;
  const mid = new THREE.Vector3((pA.x + pB.x) / 2, 3.8, (pA.z + pB.z) / 2);
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(pA.x, 0.8, pA.z),
    mid,
    new THREE.Vector3(pB.x, 1.2, pB.z),
  ]);
  const pts = curve.getPoints(48);
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({
    color: brightColor, transparent: true, opacity: 0.9,
  });
  const line = new THREE.Line(geo, mat);
  line.userData.createdAt = performance.now();
  sim.scene.add(line);
  sim.attackLines.push(line);
  for (let i = 0; i < 3; i += 1) {
    const mm = new THREE.MeshStandardMaterial({
      color: brightColor, emissive: brightColor, emissiveIntensity: 1.2, flatShading: true,
    });
    const pkt = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 9), mm);
    sim.scene.add(pkt);
    sim.attackPackets.push({
      mesh: pkt, mat: mm, curve,
      startedAt: performance.now() + i * 150, duration: 950,
    });
  }
}

// One callout per persona; cap total visible so a busy moment doesn't
// fill the screen with stacked bubbles. The toast stack on the right
// captures the same speech as a persistent ticker, so the on-scene
// callouts only need to fire briefly.
const MAX_VISIBLE_CALLOUTS = 3;
const CALLOUT_LIFETIME_MS = 3800;

function showCallout({ personaId, displayName, activity, channel, text }) {
  if (!sim.labelLayer) return;
  const persona = sim.npcs.get(personaId);
  if (!persona) return;
  let entry = sim.speechBubbles.get(personaId);
  let el;
  if (entry) {
    el = entry.el;
    el.className = `dash-callout ${channel || "chat"}`;
  } else {
    // Keep the visible-callout count bounded by evicting the oldest
    // (lowest expiresAt) when we'd otherwise exceed the cap.
    if (sim.speechBubbles.size >= MAX_VISIBLE_CALLOUTS) {
      let oldestKey = null;
      let oldestExpiry = Infinity;
      sim.speechBubbles.forEach((v, k) => {
        if (v.expiresAt < oldestExpiry) {
          oldestExpiry = v.expiresAt;
          oldestKey = k;
        }
      });
      if (oldestKey) {
        const stale = sim.speechBubbles.get(oldestKey);
        stale.el.remove();
        sim.speechBubbles.delete(oldestKey);
      }
    }
    el = document.createElement("div");
    el.className = `dash-callout ${channel || "chat"}`;
    sim.labelLayer.appendChild(el);
  }
  el.innerHTML = `
    <div class="title-bar">${escapeHtml(activity || "Activity")}</div>
    <div class="body"><span class="who">${escapeHtml(displayName)}:</span> ${escapeHtml(text)}</div>
    <div class="tail"></div>`;
  const wp = persona.group.position;
  el.dataset.worldX = wp.x;
  // Offset Y by the index of this persona among visible bubbles so
  // simultaneous callouts at adjacent desks don't pancake on top of
  // each other.
  el.dataset.worldY = 2.4 + (sim.speechBubbles.size * 0.45);
  el.dataset.worldZ = wp.z;
  sim.speechBubbles.set(personaId, { el, expiresAt: performance.now() + CALLOUT_LIFETIME_MS });
}

// =============================================================
// Animation loop + label projection
// =============================================================

const _v = new THREE.Vector3();
function worldToScreen(x, y, z) {
  const canvas = document.getElementById("sim-canvas");
  if (!canvas || !sim.camera) return { x: 0, y: 0 };
  _v.set(x, y, z).project(sim.camera);
  const r = canvas.getBoundingClientRect();
  return {
    x: (_v.x * 0.5 + 0.5) * r.width,
    y: (1 - (_v.y * 0.5 + 0.5)) * r.height,
  };
}

function updateLabels() {
  if (!sim.labelLayer) return;
  // First pass: project all anchored elements to screen.
  const placements = [];
  for (const el of sim.labelLayer.children) {
    const wx = parseFloat(el.dataset.worldX);
    const wy = parseFloat(el.dataset.worldY);
    const wz = parseFloat(el.dataset.worldZ);
    if (!Number.isFinite(wx)) continue;
    const s = worldToScreen(wx, wy, wz);
    el.style.left = `${s.x}px`;
    el.style.top = `${s.y}px`;
    placements.push({ el, x: s.x, y: s.y });
  }
  // Second pass: stagger overlapping nametags vertically. Service tags
  // and zone labels are excluded — they're scenery anchors, not floating
  // identities. Speech callouts are excluded too; they handle stacking
  // via Y world-offset at spawn time.
  const npcTags = placements.filter(({ el }) =>
    el.classList.contains("dash-nametag") &&
    !el.classList.contains("service")
  ).sort((a, b) => a.y - b.y);
  for (let i = 1; i < npcTags.length; i += 1) {
    const prev = npcTags[i - 1];
    const cur = npcTags[i];
    if (Math.abs(cur.x - prev.x) < 110 && (cur.y - prev.y) < 16) {
      const shift = 16 - (cur.y - prev.y);
      cur.y += shift;
      cur.el.style.top = `${cur.y}px`;
    }
  }
}

let _lastFrameT = performance.now();
function animateSimulation() {
  if (!sim.renderer || !sim.scene || !sim.camera) return;
  requestAnimationFrame(animateSimulation);
  const now = performance.now();
  const dtMs = now - _lastFrameT;
  _lastFrameT = now;
  const dt = Math.min(0.08, dtMs / 1000);

  sim.npcs.forEach((n) => {
    if (n.visitTarget) {
      const dx = n.visitTarget.x - n.group.position.x;
      const dz = n.visitTarget.z - n.group.position.z;
      const d = Math.sqrt(dx * dx + dz * dz);
      if (d > 0.18) {
        const speed = 2.6;
        n.group.position.x += (dx / d) * dt * speed;
        n.group.position.z += (dz / d) * dt * speed;
        n.group.rotation.y = Math.atan2(dx, dz);
        n.group.position.y = Math.abs(Math.sin(now * 0.013)) * 0.10;
      } else {
        n.visitTarget = null;
        n.dwellRemaining = 4 + Math.random() * 2;
      }
      return;
    }
    if (!n.route) {
      n.group.position.y = Math.sin(now * 0.001 + n.phase) * 0.03;
      return;
    }
    const wp = n.route[n.routeIdx];
    if (n.dwellRemaining > 0) {
      n.dwellRemaining -= dt;
      n.group.position.y = Math.sin(now * 0.0015 + n.phase) * 0.04;
      return;
    }
    const nextIdx = (n.routeIdx + 1) % n.route.length;
    const target = n.route[nextIdx];
    n.routeT = Math.min(1, n.routeT + dt * n.routeSpeed);
    const k = n.routeT * n.routeT * (3 - 2 * n.routeT);
    n.group.position.x = wp.x + (target.x - wp.x) * k;
    n.group.position.z = wp.z + (target.z - wp.z) * k;
    const dxs = target.x - wp.x;
    const dzs = target.z - wp.z;
    if (Math.abs(dxs) + Math.abs(dzs) > 0.01) {
      n.group.rotation.y = Math.atan2(dxs, dzs);
    }
    n.group.position.y = Math.abs(Math.sin(now * 0.013)) * 0.12;
    if (n.routeT >= 1) {
      n.routeIdx = nextIdx;
      n.routeT = 0;
      n.dwellRemaining = target.wait;
    }
  });

  for (const role of ["red", "blue"]) {
    const g = sim.agents[role];
    if (!g) continue;
    const u = g.userData;
    if (u.pulse > 0) {
      u.pulse = Math.max(0, u.pulse - 0.03);
      const s = 1 + u.pulse * 0.25;
      g.scale.set(s, 1 + u.pulse * 0.12, s);
    } else {
      g.scale.set(1, 1, 1);
    }
    g.position.y = Math.sin(now * 0.002 + (role === "red" ? 0 : 2)) * 0.05;
  }

  for (let i = sim.attackPackets.length - 1; i >= 0; i -= 1) {
    const p = sim.attackPackets[i];
    const e = now - p.startedAt;
    if (e < 0) continue;
    const u = e / p.duration;
    if (u >= 1) {
      sim.scene.remove(p.mesh);
      p.mat.dispose();
      p.mesh.geometry.dispose();
      sim.attackPackets.splice(i, 1);
      continue;
    }
    const pos = p.curve.getPointAt(u);
    p.mesh.position.copy(pos);
    p.mesh.scale.setScalar(0.85 + 0.25 * Math.sin(now * 0.02 + i));
  }

  for (let i = sim.activityPulses.length - 1; i >= 0; i -= 1) {
    const p = sim.activityPulses[i];
    p.t += dtMs;
    const u = p.t / p.duration;
    if (u >= 1) {
      sim.scene.remove(p.mesh);
      p.mat.dispose();
      p.mesh.geometry.dispose();
      sim.activityPulses.splice(i, 1);
      continue;
    }
    p.mesh.scale.setScalar(1 + u * 1.1);
    p.mat.opacity = 0.5 * (1 - u);
  }

  for (let i = sim.attackLines.length - 1; i >= 0; i -= 1) {
    const l = sim.attackLines[i];
    const age = (now - l.userData.createdAt) / 1600;
    if (age >= 1) {
      sim.scene.remove(l);
      l.geometry.dispose();
      l.material.dispose();
      sim.attackLines.splice(i, 1);
    } else {
      l.material.opacity = 0.55 * (1 - age);
    }
  }

  sim.services.forEach((s) => {
    const u = s.group.userData;
    if (u.pulseStart) {
      const dt2 = (now - u.pulseStart) / 700;
      if (dt2 > 1) {
        u.pulseStart = 0;
        u.ring.scale.set(1, 1, 1);
        s.ringMat.opacity = 0.55;
      } else {
        u.ring.scale.setScalar(1 + dt2 * 0.8);
        s.ringMat.opacity = 0.55 * (1 - dt2);
        if (u.pulseColor) s.ringMat.color.lerp(u.pulseColor, 0.15);
      }
    }
  });

  sim.speechBubbles.forEach((entry, pid) => {
    if (now > entry.expiresAt) {
      entry.el.remove();
      sim.speechBubbles.delete(pid);
    }
  });

  if (sim.controls) sim.controls.update();
  updateLabels();
  sim.renderer.render(sim.scene, sim.camera);
}

function resizeSimulation() {
  const canvas = document.getElementById("sim-canvas");
  if (!canvas) return;
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;
  if (sim.renderer && sim.camera) {
    sim.renderer.setSize(width, height, false);
    if (sim.camera.isPerspectiveCamera) {
      sim.camera.aspect = width / Math.max(1, height);
    }
    sim.camera.updateProjectionMatrix();
  }
}

function installActorSelection(canvas) {
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  canvas.addEventListener("click", (event) => {
    if (!sim.camera) return;
    const bounds = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * 2 - 1;
    pointer.y = -(((event.clientY - bounds.top) / Math.max(1, bounds.height)) * 2 - 1);
    raycaster.setFromCamera(pointer, sim.camera);
    const targets = [];
    sim.npcs.forEach((p) => targets.push(p.group));
    if (sim.agents.red) targets.push(sim.agents.red);
    if (sim.agents.blue) targets.push(sim.agents.blue);
    sim.services.forEach((s) => targets.push(s.group));
    const hits = raycaster.intersectObjects(targets, true);
    for (const hit of hits) {
      const actorId = hit.object.userData.actorId || hit.object.userData.serviceId;
      if (actorId) {
        rail.selectedActorId = actorId;
        showRailTab("actor");
        return;
      }
    }
  });
}

// =============================================================
// Toast stack
// =============================================================

function pushToast({ kind, who, text, key }) {
  if (toasts.recentSpeakIds.has(key)) return;
  toasts.recentSpeakIds.add(key);
  if (toasts.recentSpeakIds.size > 200) {
    // bound the dedup set
    toasts.recentSpeakIds = new Set(Array.from(toasts.recentSpeakIds).slice(-100));
  }
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast ${kind || ""}`;
  el.innerHTML = `<span class="who">${escapeHtml(who)}:</span>${escapeHtml(text)}`;
  stack.appendChild(el);
  // Cap visible.
  while (stack.children.length > toasts.maxVisible) {
    stack.firstChild.remove();
  }
  setTimeout(() => {
    el.classList.add("fading");
    setTimeout(() => el.remove(), 400);
  }, toasts.toastLifetimeMs);
}

// =============================================================
// Build banner — surfaces only when a NEW lineage node arrives.
// =============================================================

function checkForNewLineage() {
  const nodes = (model.lineage.nodes || []);
  const ids = nodes.map((n) => n.id).filter(Boolean);
  if (!banner.primed) {
    // First-load: seed knownLineageIds without showing the banner —
    // an existing lineage isn't "new" to this session.
    banner.primed = true;
    ids.forEach((id) => banner.knownLineageIds.add(id));
    return;
  }
  for (const node of nodes) {
    if (!node.id || banner.knownLineageIds.has(node.id)) continue;
    banner.knownLineageIds.add(node.id);
    showBuildBanner(node);
  }
}

function showBuildBanner(node) {
  const bannerEl = document.getElementById("build-banner");
  if (!bannerEl) return;
  const isFirst = (model.lineage.nodes || []).indexOf(node) === 0;
  banner.current = node;
  document.getElementById("build-banner-eyebrow").textContent =
    isFirst ? "Initial build" : "Evolved build";
  const title = node.builder_summary
    || (isFirst ? `Built world ${(node.id || "").slice(-10)}` : `Mutated → ${(node.id || "").slice(-10)}`);
  document.getElementById("build-banner-title").textContent = shortText(title, 90);
  const direction = evoDirectionWord(node.evolve?.direction);
  const sub = node.parent_id
    ? `${direction ? `${direction} · ` : ""}from …${(node.parent_id || "").slice(-10)}`
    : "starting world";
  document.getElementById("build-banner-sub").textContent = sub;
  bannerEl.hidden = false;
  // Mark the build tab so the user notices it.
  const buildTab = document.querySelector('.rail-tab[data-tab="build"]');
  if (buildTab) buildTab.classList.add("has-update");
  if (banner.dismissTimer) clearTimeout(banner.dismissTimer);
  banner.dismissTimer = setTimeout(dismissBuildBanner, 12000);
}

function dismissBuildBanner() {
  const bannerEl = document.getElementById("build-banner");
  if (!bannerEl) return;
  bannerEl.hidden = true;
  if (banner.dismissTimer) {
    clearTimeout(banner.dismissTimer);
    banner.dismissTimer = null;
  }
}

// =============================================================
// Topbar / footbar / rail rendering
// =============================================================

function renderTopbar() {
  const snapId = model.topology.snapshot_id;
  const status = (model.state.status || "waiting_for_snapshot").replaceAll("_", " ");
  document.getElementById("snapshot-pill-value").textContent =
    snapId ? `…${snapId.slice(-8)}` : "—";
  const statusEl = document.getElementById("topbar-status");
  const statusKey = (model.state.status || "waiting").includes("playing") ? "playing"
    : (model.state.status || "").includes("paused") ? "paused"
    : (model.state.status || "").includes("ready") ? "ready" : "waiting";
  statusEl.dataset.status = statusKey;
  document.getElementById("topbar-status-text").textContent = status;
  const eventCount = model.state.event_count || (model.state.events || []).length;
  document.getElementById("topbar-clock").textContent = String(eventCount).padStart(2, "0");
  // Run picker label.
  const runEl = document.getElementById("run-picker-current");
  if (runEl) runEl.textContent = runState.activeRun || "—";
}

function renderFootbar() {
  const health = model.state.health || {};
  setPip("pip-uptime", health.uptime);
  setPip("pip-defense", health.defense);
  setPip("pip-integrity", health.integrity);
  // Narrator: prefer the latest meaningful event over the "X agent_step Y" join.
  const narrator = document.getElementById("footbar-narrator");
  const events = model.state.events || [];
  const latest = events[events.length - 1];
  if (latest) {
    const data = eventData(latest);
    const action = data.action || {};
    let line;
    if (action.speak) line = `${latest.actor}: ${shortText(action.speak, 90)}`;
    else if (action.method) line = `${latest.actor} → ${latest.target} ${action.method} ${shortText(action.path || "", 50)}`;
    else line = `${latest.actor} → ${latest.target}: ${latest.type}`;
    narrator.textContent = line;
  } else {
    narrator.textContent = "Waiting for episode activity…";
  }
  const meta = document.getElementById("footbar-meta");
  if (meta) {
    const taskCount = (model.topology.tasks || []).length;
    const svcCount = (model.topology.services || []).length;
    const personaCount = (model.topology.green_personas || []).length;
    meta.textContent = model.topology.snapshot_id
      ? `${plural(taskCount, "task")} · ${plural(svcCount, "service")} · ${plural(personaCount, "persona")}`
      : "";
  }
}

function setPip(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("green", "amber", "red");
  if (typeof value !== "number") {
    el.classList.add("green");
    return;
  }
  if (value > 66) el.classList.add("green");
  else if (value > 33) el.classList.add("amber");
  else el.classList.add("red");
}

function renderEmptyState() {
  const el = document.getElementById("empty-state");
  if (!el) return;
  el.hidden = !!model.topology.snapshot_id;
}

// =============================================================
// Rail tab management + per-panel render
// =============================================================

function showRailTab(name) {
  rail.active = name;
  rail.open = true;
  const railEl = document.getElementById("rail");
  if (railEl) railEl.dataset.open = "true";
  document.querySelectorAll(".rail-tab").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.tab === name);
    if (btn.dataset.tab === name) btn.classList.remove("has-update");
  });
  document.querySelectorAll(".rail-panel").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== name;
  });
  // Actor tab toggles visibility from "no actor" to "showing actor".
  const actorTab = document.querySelector('.rail-tab[data-tab="actor"]');
  if (actorTab) actorTab.hidden = !rail.selectedActorId;
  renderRailPanel(name);
}

function closeRail() {
  rail.open = false;
  const railEl = document.getElementById("rail");
  if (railEl) railEl.dataset.open = "false";
  document.querySelectorAll(".rail-tab").forEach((btn) => {
    btn.classList.remove("is-active");
  });
}

function renderRailPanel(name) {
  if (name === "build") renderBuildPanel();
  if (name === "world") renderWorldPanel();
  if (name === "lineage") renderLineagePanel();
  if (name === "activity") renderActivityPanel();
  if (name === "actor") renderActorPanel();
}

function renderBuildPanel() {
  const nodes = model.lineage.nodes || [];
  const target = rail.selectedLineageId
    ? nodes.find((n) => n.id === rail.selectedLineageId)
    : nodes[nodes.length - 1];
  document.getElementById("build-snapshot-id").textContent =
    model.topology.snapshot_id ? `snapshot ${model.topology.snapshot_id}` : "—";
  const root = document.getElementById("build-content");
  if (!target) {
    root.innerHTML = emptyCard("No build to inspect yet.");
    return;
  }
  const isFirst = nodes.indexOf(target) === 0;
  const parent = target.parent_id
    ? nodes.find((n) => n.id === target.parent_id)
    : null;
  const briefing = model.briefing;
  const html = [];

  // Eyebrow + goal.
  html.push(`
    <div class="callout-card">
      <div class="eyebrow">${isFirst ? "Initial build" : "Evolved build"}</div>
      <div><strong>${escapeHtml(briefing.title || target.manifest?.world?.title || "Untitled world")}</strong>
        — ${escapeHtml(briefing.goal || target.manifest?.world?.goal || target.builder_summary || "")}</div>
    </div>`);

  // The biggest diff between an initial build and an evolved build is the
  // CURRICULUM the mutator applied. Surface that first and prominently —
  // it's the "what was different this time" question the operator asks.
  if (parent) {
    const ops = curriculumChanges(parent.curriculum || {}, target.curriculum || {});
    html.push(`<div class="section-title">What changed from parent</div>`);
    if (ops.length) {
      html.push(`<div class="diff-block">${
        ops.map(o => `<span class="diff-line ${o.kind}">${escapeHtml(o.text)}</span>`).join("")
      }</div>`);
    } else {
      html.push(`<div class="empty-card">No curriculum delta — same inputs, fresh sample seed.</div>`);
    }
    // Manifest delta: usually empty across auto_evolve, but worth surfacing
    // when the operator manually re-builds with a different manifest.
    const mdiff = diffMappings(parent.manifest || {}, target.manifest || {});
    const realChanges = mdiff.filter(d => d.kind !== "unchanged");
    if (realChanges.length) {
      html.push(`<div class="section-title">Manifest delta</div>`);
      html.push(`<div class="diff-block">${
        realChanges.map(d => `<span class="diff-line ${d.kind}">${escapeHtml(d.text)}</span>`).join("")
      }</div>`);
    }
    if (target.builder_summary) {
      html.push(`<div class="callout-card"><div class="eyebrow">Why</div>${escapeHtml(target.builder_summary)}</div>`);
    }
  } else if (target.builder_summary) {
    html.push(`<div class="section-title">Builder summary</div>
      <div class="callout-card">${escapeHtml(target.builder_summary)}</div>`);
  }

  // Manifest snapshot — what the operator passed in.
  html.push(`
    <div class="section-title">Manifest</div>
    <dl class="kv-grid">
      <dt>Pack</dt><dd>${escapeHtml(target.pack?.id || "—")}</dd>
      <dt>Builder</dt><dd>${escapeHtml(target.manifest?.builder || "default")}</dd>
      <dt>Difficulty</dt><dd>${target.world_difficulty != null ? Number(target.world_difficulty).toFixed(1) : "—"}</dd>
      <dt>NPCs</dt><dd>${(target.manifest?.npc || []).map((n) => escapeHtml(n.type)).join(", ") || "—"}</dd>
      <dt>Mode</dt><dd>${escapeHtml(target.manifest?.mode || "simulation")}</dd>
    </dl>`);

  // Curriculum — the active set on this build (not just the delta).
  if (target.curriculum && Object.keys(target.curriculum).length) {
    html.push(`
      <div class="section-title">Curriculum on this build</div>
      <dl class="kv-grid">${
        Object.entries(target.curriculum).map(([k, v]) => (
          `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(JSON.stringify(v))}</dd>`
        )).join("")
      }</dl>`);
  }

  // Realized world summary — quick snapshot of what got built.
  const services = model.topology.services || [];
  const personas = model.topology.green_personas || [];
  const tasks = model.topology.tasks || [];
  if (services.length || personas.length || tasks.length) {
    html.push(`
      <div class="section-title">Realized world</div>
      <dl class="kv-grid">
        <dt>Services</dt><dd>${services.length} (${[...new Set(services.map(s => s.kind || "?"))].join(", ")})</dd>
        <dt>Zones</dt><dd>${(model.topology.zones || []).join(", ") || "—"}</dd>
        <dt>Tasks</dt><dd>${tasks.length}</dd>
        <dt>Personas</dt><dd>${personas.length}</dd>
      </dl>
      <div style="margin-top:8px;">
        <a class="rail-link" data-jump="world">See full topology →</a>
      </div>`);
  }

  // Touched files (collapsed when long).
  if ((target.touched_files || []).length) {
    const files = target.touched_files;
    const showCount = Math.min(files.length, 12);
    html.push(`
      <div class="section-title">Touched files (${files.length})</div>
      <div class="diff-block">${
        files.slice(0, showCount).map(f => `<span class="diff-line unchanged">${escapeHtml(f)}</span>`).join("")
      }${files.length > showCount ? `<span class="diff-line unchanged" style="color:var(--ink-3)">… +${files.length - showCount} more</span>` : ""}</div>`);
  }

  // Parent breadcrumb.
  if (parent) {
    html.push(`
      <div class="section-title">Parent</div>
      <div class="lineage-row" data-jump-lineage="${escapeHtml(parent.id)}" style="cursor:pointer;">
        <span style="font-family:var(--f-mono);font-size:11.5px;">${escapeHtml(parent.id)}</span>
        <span class="zone-pill">prior build</span>
      </div>`);
  }

  root.innerHTML = html.join("");
  // Wire jump links.
  root.querySelectorAll('[data-jump="world"]').forEach((el) => {
    el.style.cursor = "pointer";
    el.style.color = "var(--accent)";
    el.style.fontWeight = "600";
    el.addEventListener("click", () => showRailTab("world"));
  });
  root.querySelectorAll("[data-jump-lineage]").forEach((el) => {
    el.addEventListener("click", () => {
      rail.selectedLineageId = el.dataset.jumpLineage;
      renderBuildPanel();
    });
  });
}

// Curriculum diff: produce ordered + lines for each curriculum op (add/remove
// or arbitrary keys) so the operator sees exactly what the mutator did.
function curriculumChanges(beforeC, afterC) {
  const out = [];
  const keys = new Set([...Object.keys(beforeC || {}), ...Object.keys(afterC || {})]);
  for (const key of [...keys].sort()) {
    const b = beforeC?.[key];
    const a = afterC?.[key];
    if (JSON.stringify(b) === JSON.stringify(a)) continue;
    if (Array.isArray(a) || Array.isArray(b)) {
      const before = new Set(b || []);
      const after = new Set(a || []);
      for (const item of after) if (!before.has(item)) out.push({ kind: "added", text: `+ curriculum.${key}: ${JSON.stringify(item)}` });
      for (const item of before) if (!after.has(item)) out.push({ kind: "removed", text: `- curriculum.${key}: ${JSON.stringify(item)}` });
    } else if (b === undefined) {
      out.push({ kind: "added", text: `+ curriculum.${key}: ${JSON.stringify(a)}` });
    } else if (a === undefined) {
      out.push({ kind: "removed", text: `- curriculum.${key}: ${JSON.stringify(b)}` });
    } else {
      out.push({ kind: "removed", text: `- curriculum.${key}: ${JSON.stringify(b)}` });
      out.push({ kind: "added", text: `+ curriculum.${key}: ${JSON.stringify(a)}` });
    }
  }
  return out;
}

function diffMappings(before, after, prefix = "") {
  // Two-pass tiny diff: for each key in either map, classify
  // {added | removed | unchanged | changed} and emit human-readable
  // lines. Recurses one level into objects so manifest.world / .pack
  // get inline diffs without dumping the whole subtree.
  const out = [];
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  for (const key of [...keys].sort()) {
    const k = prefix ? `${prefix}.${key}` : key;
    const b = before[key];
    const a = after[key];
    const bj = JSON.stringify(b);
    const aj = JSON.stringify(a);
    if (bj === aj) {
      // Don't dump unchanged scalars; only show the key as a header
      // when it's an object so the user can see the field exists.
      if (b !== null && typeof b === "object" && !Array.isArray(b) && Object.keys(b || {}).length) {
        out.push({ kind: "unchanged", text: `  ${k}: …` });
      }
      continue;
    }
    if (bj === undefined || b === undefined) {
      out.push({ kind: "added", text: `+ ${k}: ${aj}` });
    } else if (aj === undefined || a === undefined) {
      out.push({ kind: "removed", text: `- ${k}: ${bj}` });
    } else if (typeof a === "object" && typeof b === "object" && !Array.isArray(a) && !Array.isArray(b)) {
      const sub = diffMappings(b, a, k);
      out.push(...sub);
    } else {
      out.push({ kind: "removed", text: `- ${k}: ${bj}` });
      out.push({ kind: "added", text: `+ ${k}: ${aj}` });
    }
  }
  return out;
}

function renderWorldPanel() {
  document.querySelectorAll(".rail-subtab").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.worldTab === rail.worldSubtab);
  });
  const root = document.getElementById("world-content");
  if (rail.worldSubtab === "services") return renderWorldServices(root);
  if (rail.worldSubtab === "topology") return renderWorldTopology(root);
  if (rail.worldSubtab === "network") return renderWorldNetwork(root);
  return renderWorldServices(root);
}

function renderCastRail() {
  const root = document.getElementById("cast-content");
  if (root) renderWorldCast(root);
  const sub = document.getElementById("cast-rail-sub");
  if (sub) sub.textContent = plural((model.topology.green_personas || []).length, "persona");
}

function renderWorldServices(root) {
  const services = model.topology.services || [];
  if (!services.length) return root.innerHTML = emptyCard("No services in topology.");
  const zones = (model.topology.zones || []);
  const orderedZones = zones.length
    ? zones
    : Array.from(new Set(services.map((s) => s.zone || "—")));
  root.innerHTML = orderedZones.map((zone) => {
    const members = services.filter((s) => (s.zone || "—") === zone);
    if (!members.length) return "";
    const rows = members.map((s) => {
      const ports = (s.ports || []).join(", ");
      const vulns = (s.vulns || []).map(v => `<span class="vuln-tag">${escapeHtml(v)}</span>`).join("");
      return `<div class="svc-row">
        <span class="name">${escapeHtml(s.id)}</span>
        <span class="kind">${escapeHtml(s.kind || "service")}${ports ? ` :${escapeHtml(ports)}` : ""}</span>
        ${vulns ? `<div class="vulns">${vulns}</div>` : ""}
      </div>`;
    }).join("");
    return `<div style="margin-bottom:12px;">
      <div class="zone-pill" style="margin-bottom:6px;display:inline-block;">${escapeHtml(zone)}</div>
      ${rows}
    </div>`;
  }).join("");
}

function renderWorldTopology(root) {
  const tasks = model.topology.tasks || [];
  const services = model.topology.services || [];
  if (!tasks.length && !services.length) return root.innerHTML = emptyCard("No topology yet.");
  const html = [];
  if (tasks.length) {
    html.push(`<div class="section-title">Tasks (${tasks.length})</div>`);
    html.push(tasks.map(t => {
      const eps = (t.entrypoints || []).map(e => `<code>${escapeHtml(e.kind)}:${escapeHtml(e.target)}</code>`).join(" ");
      return `<div style="margin-bottom:10px;">
        <div style="font-family:var(--f-mono);font-size:12px;color:var(--ink-0);">${escapeHtml(t.id)}</div>
        <div style="font-size:12px;color:var(--ink-1);margin:3px 0;">${escapeHtml(t.instruction)}</div>
        ${eps ? `<div style="font-size:10.5px;color:var(--ink-3);">${eps}</div>` : ""}
      </div>`;
    }).join(""));
  }
  if ((model.topology.artifact_paths || []).length) {
    html.push(`<div class="section-title">Artifacts (${model.topology.artifact_paths.length})</div>`);
    html.push(`<div class="diff-block">${
      model.topology.artifact_paths.slice(0, 80).map(p => `<span class="diff-line unchanged">${escapeHtml(p)}</span>`).join("")
    }</div>`);
  }
  root.innerHTML = html.join("");
}

function renderWorldNetwork(root) {
  // SVG-rendered service graph: nodes per service, edges per `backed_by`.
  const services = model.topology.services || [];
  const edges = model.topology.edges || [];
  if (!services.length) return root.innerHTML = emptyCard("No services to chart.");
  const W = 360, H = 280;
  // Force-free layout: zone-grouped column flow.
  const byZone = new Map();
  services.forEach((s) => {
    const z = s.zone || "—";
    if (!byZone.has(z)) byZone.set(z, []);
    byZone.get(z).push(s);
  });
  const positions = new Map();
  const zoneList = [...byZone.keys()];
  const colW = W / Math.max(1, zoneList.length);
  zoneList.forEach((zone, zi) => {
    const members = byZone.get(zone) || [];
    members.forEach((s, mi) => {
      const x = colW * (zi + 0.5);
      const y = 36 + (H - 60) * ((mi + 0.5) / Math.max(1, members.length));
      positions.set(s.id, { x, y, zone });
    });
  });
  const nodes = services.map((s) => {
    const p = positions.get(s.id);
    if (!p) return "";
    const label = serviceLabel(s.id);
    return `
      <circle cx="${p.x}" cy="${p.y}" r="14" fill="var(--bg-3)" stroke="var(--ink-0)" stroke-width="1.5"/>
      <text x="${p.x}" y="${p.y + 28}" text-anchor="middle" fill="var(--ink-1)"
            style="font-family:var(--f-mono);font-size:10.5px">${escapeHtml(label)}</text>
      <text x="${p.x}" y="${p.y + 4}" text-anchor="middle" fill="var(--ink-0)"
            style="font-family:var(--f-mono);font-size:9px;">${escapeHtml((s.kind || "?").slice(0, 4))}</text>`;
  }).join("");
  const edgesSvg = edges.map((e) => {
    const a = positions.get(e.source);
    // Try both raw target id and svc-prefixed target id
    const b = positions.get(e.target) || positions.get(`svc-${e.target}`);
    if (!a || !b) return "";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
      stroke="var(--ink-3)" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.6"/>`;
  }).join("");
  const zoneLabels = zoneList.map((zone, zi) => `
    <text x="${colW * (zi + 0.5)}" y="14" text-anchor="middle"
      style="font-family:var(--f-display);font-style:italic;font-size:11px;fill:var(--ink-3);">
      ${escapeHtml(zone)}
    </text>`).join("");
  root.innerHTML = `
    <div style="background:var(--bg-2);border-radius:6px;padding:8px;">
      <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;">
        ${zoneLabels}
        ${edgesSvg}
        ${nodes}
      </svg>
    </div>
    <div class="section-title">Edges</div>
    <div style="font-size:12px;color:var(--ink-2);">
      ${edges.length ? plural(edges.length, "backed_by edge") + " visible above." : "No edges declared by the pack."}
    </div>`;
}

function renderWorldCast(root) {
  const personas = model.topology.green_personas || [];
  if (!personas.length) return root.innerHTML = emptyCard("No personas seated. Add cyber.office_persona NPCs to the manifest.");
  root.innerHTML = personas.map(p => `
    <div class="persona-row">
      <span class="name">${escapeHtml(p.display_name || p.id)}</span>
      <span class="role">${escapeHtml(p.role || "")}${p.title ? ` · ${escapeHtml(p.title)}` : ""}</span>
    </div>`).join("");
}

function renderLineagePanel() {
  const nodes = model.lineage.nodes || [];
  document.getElementById("lineage-count-sub").textContent = plural(nodes.length, "node");
  const root = document.getElementById("lineage-content");
  if (!nodes.length) return root.innerHTML = emptyCard("No lineage steps recorded.");
  root.innerHTML = `<ul class="lineage-list">${
    nodes.map((node, idx) => {
      const isActive = node.id === rail.selectedLineageId
        || (rail.selectedLineageId == null && idx === nodes.length - 1);
      const summary = evoSummaryText(node) || node.prompt || "";
      const stepLabel = idx === 0 ? "Initial world" : `Evolution ${idx}`;
      const direction = evoDirectionWord(node.evolve?.direction);
      return `<li class="lineage-node ${isActive ? "is-active" : ""}" data-node-id="${escapeHtml(node.id)}">
        <h4>${escapeHtml(stepLabel)} · …${escapeHtml((node.id || "").slice(-10))}</h4>
        ${summary ? `<p class="summary">${escapeHtml(shortText(summary, 200))}</p>` : ""}
        <p class="meta">${direction ? `<span class="zone-pill">${escapeHtml(direction)}</span> ` : ""}${node.parent_id ? `from …${escapeHtml((node.parent_id || "").slice(-10))}` : "root"}</p>
      </li>`;
    }).join("")
  }</ul>`;
  root.querySelectorAll(".lineage-node").forEach((el) => {
    el.addEventListener("click", () => {
      rail.selectedLineageId = el.dataset.nodeId;
      showRailTab("build");
    });
  });
}

function renderActivityPanel() {
  const events = (model.state.events || []).slice(-60).reverse();
  document.getElementById("activity-count").textContent = plural(events.length, "event");
  const list = document.getElementById("activity-list");
  if (!events.length) {
    list.innerHTML = `<li><span class="dot"></span><span>No activity yet.</span></li>`;
    return;
  }
  list.innerHTML = events.map((e) => {
    const role = simulationRole(e);
    const data = eventData(e);
    const action = data.action || {};
    let head = `${escapeHtml(e.actor || "?")} → ${escapeHtml(e.target || "?")}`;
    let body = "";
    if (action.speak) body = `“${escapeHtml(shortText(action.speak, 100))}”`;
    else if (action.method) body = `${escapeHtml(action.method)} ${escapeHtml(shortText(action.path || "", 60))}`;
    else if (action.visit) body = `visit ${escapeHtml(shortText(action.visit, 60))}`;
    else body = escapeHtml(e.type || "");
    return `<li>
      <span class="dot ${role}"></span>
      <span>
        <strong>${head}</strong><br>
        ${body}
        <span class="meta">t=${(e.time || 0).toFixed(1)}</span>
      </span>
    </li>`;
  }).join("");
}

function renderActorPanel() {
  const actorId = rail.selectedActorId;
  if (!actorId) {
    document.getElementById("actor-name").textContent = "Actor";
    document.getElementById("actor-role").textContent = "no actor selected";
    document.getElementById("actor-content").innerHTML = emptyCard("Click an actor in the scene to inspect.");
    return;
  }
  const persona = (model.topology.green_personas || []).find(p => p.id === actorId || p.display_name === actorId);
  const summary = (model.actors || []).find(a => a.actor_id === actorId);
  const service = (model.topology.services || []).find(s => s.id === actorId);
  document.getElementById("actor-name").textContent = persona?.display_name || actorId;
  document.getElementById("actor-role").textContent =
    persona ? `${persona.role || ""}${persona.title ? ` · ${persona.title}` : ""}`
    : service ? `service · ${service.kind || ""}`
    : (summary?.actor_kind || "event");
  const root = document.getElementById("actor-content");
  const html = [];
  if (persona) {
    html.push(`<dl class="kv-grid">
      <dt>Role</dt><dd>${escapeHtml(persona.role || "—")}</dd>
      ${persona.title ? `<dt>Title</dt><dd>${escapeHtml(persona.title)}</dd>` : ""}
      ${persona.tone ? `<dt>Tone</dt><dd>${escapeHtml(persona.tone)}</dd>` : ""}
      ${(persona.colleagues || []).length ? `<dt>Colleagues</dt><dd>${(persona.colleagues || []).map(escapeHtml).join(", ")}</dd>` : ""}
    </dl>`);
  } else if (service) {
    html.push(`<dl class="kv-grid">
      <dt>Kind</dt><dd>${escapeHtml(service.kind || "—")}</dd>
      <dt>Zone</dt><dd>${escapeHtml(service.zone || "—")}</dd>
      ${(service.ports || []).length ? `<dt>Ports</dt><dd>${(service.ports || []).join(", ")}</dd>` : ""}
      ${(service.vulns || []).length ? `<dt>Vulns</dt><dd>${(service.vulns || []).map(escapeHtml).join(", ")}</dd>` : ""}
    </dl>`);
  }
  // Recent activity for this actor.
  const events = (model.state.events || []).filter((e) => {
    const data = eventData(e);
    return e.actor === actorId || data.actor_id === actorId
      || e.target === actorId || data.target === actorId;
  }).slice(-12).reverse();
  if (events.length) {
    html.push(`<div class="section-title">Recent activity</div>`);
    html.push(`<ul class="activity-list" style="padding:0;">${events.map((e) => {
      const data = eventData(e);
      const action = data.action || {};
      let body = action.speak ? `“${shortText(action.speak, 90)}”`
        : action.method ? `${action.method} ${shortText(action.path || "", 50)}`
        : action.visit ? `visit ${shortText(action.visit, 50)}`
        : e.type;
      return `<li>
        <span class="dot ${simulationRole(e)}"></span>
        <span>${escapeHtml(body)}<span class="meta">t=${(e.time || 0).toFixed(1)}</span></span>
      </li>`;
    }).join("")}</ul>`);
  } else {
    html.push(`<div class="section-title">Recent activity</div>${emptyCard("Nothing yet.")}`);
  }
  root.innerHTML = html.join("");
}

function emptyCard(message) {
  return `<div class="empty-card">${escapeHtml(message)}</div>`;
}

// One layout over the union of all worlds, so a node added by an evolution
// pops in at a fixed spot instead of reshuffling the whole graph.

const SVGNS = "http://www.w3.org/2000/svg";

const EVO_FILL = {
  network: "#6e5e44", host: "#9e8e72",
  service: "#3d6a8a", endpoint: "#6f9bb5",
  data_store: "#5e7c3d", record: "#7c9456",
  secret: "#b88521", account: "#6e5a85", credential: "#8a7aa0",
  vulnerability: "#8d3f3a",
};
const EVO_R = {
  network: 12, host: 9, service: 12, endpoint: 6.5, data_store: 10,
  record: 6, secret: 10, account: 8, credential: 6.5, vulnerability: 7.5,
};
const EVO_BAND = {
  network: 0, host: 0, service: 1, endpoint: 2,
  vulnerability: 3, data_store: 4, record: 4, secret: 4, account: 4, credential: 4,
};
const EVO_LABELED = new Set([
  "network", "host", "service", "endpoint", "data_store", "secret",
  "account", "credential", "vulnerability",
]);
const EVO_LEGEND = [
  ["infrastructure", "#9e8e72"], ["service", "#3d6a8a"], ["endpoint", "#6f9bb5"],
  ["data / records", "#5e7c3d"], ["secret", "#b88521"], ["account / cred", "#6e5a85"],
  ["vulnerability", "#8d3f3a"], ["added this step", "ring"], ["changed this step", "ring-amber"],
];

const evo = {
  active: false,
  step: 0,
  chainKey: "",
  mounted: false,
  W: 1280, H: 760,
  nodes: [], edges: [], steps: [],
  stepNodeSets: [], stepEdgeSets: [], stepAttrs: [],
  nodeEls: new Map(), edgeEls: new Map(),
  view: { x: 0, y: 0, k: 1 },
};

function evoNodeLabel(n) {
  const L = (n.label || "").trim();
  if (L) return L;
  const id = n.id || "";
  if (n.kind === "network") return "network";
  if (n.kind === "host") return "host";
  if (n.kind === "secret") return "flag";
  if (n.kind === "service") return id.replace(/^svc_/, "");
  if (n.kind === "vulnerability") return id.replace(/^vuln_/, "").replace(/_\d+$/, "");
  if (n.kind === "data_store") return id.replace(/^ds_/, "");
  if (n.kind === "account") return "account";
  if (n.kind === "credential") return "cred";
  return "";
}

function setEvoView(mode) {
  evo.active = (mode === "evolution");
  document.querySelectorAll(".view-toggle-btn").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.view === mode));
  const stage = document.getElementById("evo-stage");
  const scrub = document.getElementById("evo-scrubber");
  if (stage) stage.hidden = !evo.active;
  const canvas = document.getElementById("sim-canvas");
  const labels = document.getElementById("sim-label-layer");
  if (canvas) canvas.style.visibility = evo.active ? "hidden" : "visible";
  if (labels) labels.style.visibility = evo.active ? "hidden" : "visible";
  if (evo.active) {
    evo.chainKey = "";        // force a rebuild from current model
    renderEvoView();
    if (scrub) scrub.hidden = (evo.steps.length === 0);
  } else if (scrub) {
    scrub.hidden = true;
  }
}

function buildEvo() {
  const steps = ((model.lineage || {}).nodes || []).filter((n) => n && n.graph);
  const key = steps.map((s) => s.id).join("|");
  if (key === evo.chainKey) return false;
  evo.chainKey = key;
  evo.steps = steps;
  const unodes = new Map(), uedges = new Map();
  steps.forEach((s) => {
    (s.graph.nodes || []).forEach((n) => { if (!unodes.has(n.id)) unodes.set(n.id, { ...n }); });
    (s.graph.edges || []).forEach((e) => { if (!uedges.has(e.id)) uedges.set(e.id, { ...e }); });
  });
  evo.nodes = [...unodes.values()];
  evo.edges = [...uedges.values()];
  evo.stepNodeSets = steps.map((s) => new Set((s.graph.nodes || []).map((n) => n.id)));
  evo.stepEdgeSets = steps.map((s) => new Set((s.graph.edges || []).map((e) => e.id)));
  evo.stepAttrs = steps.map((s) => {
    const m = new Map();
    (s.graph.nodes || []).forEach((n) => m.set(n.id, n.attrs || {}));
    return m;
  });
  if (evo.step > steps.length - 1) evo.step = Math.max(0, steps.length - 1);
  layoutEvo();
  evo.mounted = false;
  return true;
}

function layoutEvo() {
  const padX = 110, padY = 60, W = evo.W, H = evo.H;
  const bands = 5;
  const colX = (b) => padX + b * ((W - 2 * padX) / (bands - 1));
  const byId = new Map(evo.nodes.map((n) => [n.id, n]));
  const nbrs = new Map(evo.nodes.map((n) => [n.id, []]));
  evo.edges.forEach((e) => {
    if (byId.has(e.src) && byId.has(e.dst)) {
      nbrs.get(e.src).push(e.dst);
      nbrs.get(e.dst).push(e.src);
    }
  });
  const cols = Array.from({ length: bands }, () => []);
  evo.nodes.slice().sort((a, b) => (a.id < b.id ? -1 : 1)).forEach((n) => {
    n.band = EVO_BAND[n.kind] ?? 4;
    cols[n.band].push(n);
  });
  const assignY = () => cols.forEach((arr, b) => {
    const n = arr.length;
    arr.forEach((node, i) => {
      node.x = colX(b);
      node.y = n <= 1 ? H / 2 : padY + i * ((H - 2 * padY) / (n - 1));
    });
  });
  assignY();
  for (let it = 0; it < 12; it++) {
    cols.forEach((arr) => {
      arr.forEach((node) => {
        const ns = nbrs.get(node.id).map((id) => byId.get(id)).filter(Boolean);
        node._k = ns.length ? ns.reduce((s, m) => s + m.y, 0) / ns.length : node.y;
      });
      arr.sort((a, b) => a._k - b._k);
    });
    assignY();
  }
}

function mountEvoGraph() {
  const svg = document.getElementById("evo-graph");
  if (!svg) return;
  svg.setAttribute("viewBox", `0 0 ${evo.W} ${evo.H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.innerHTML = "";
  evo.nodeEls.clear();
  evo.edgeEls.clear();

  const vp = document.createElementNS(SVGNS, "g");
  vp.setAttribute("id", "evo-viewport");
  svg.appendChild(vp);
  const gEdges = document.createElementNS(SVGNS, "g");
  const gNodes = document.createElementNS(SVGNS, "g");
  vp.appendChild(gEdges); vp.appendChild(gNodes);
  evo._vp = vp;

  const byId = new Map(evo.nodes.map((n) => [n.id, n]));
  evo.edges.forEach((e) => {
    const s = byId.get(e.src), t = byId.get(e.dst);
    if (!s || !t) return;
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", s.x); ln.setAttribute("y1", s.y);
    ln.setAttribute("x2", t.x); ln.setAttribute("y2", t.y);
    ln.setAttribute("class", "evo-edge" + (e.kind === "affects" ? " affects" : ""));
    gEdges.appendChild(ln);
    evo.edgeEls.set(e.id, ln);
  });

  evo.nodes.forEach((n) => {
    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", "evo-node");
    g.setAttribute("transform", `translate(${n.x},${n.y})`);
    const r = EVO_R[n.kind] || 7;
    const halo = document.createElementNS(SVGNS, "circle");
    halo.setAttribute("r", r + 5); halo.setAttribute("class", "halo");
    halo.setAttribute("fill", "none"); halo.setAttribute("stroke", "#9e7232");
    halo.setAttribute("stroke-width", "2.6");
    g.appendChild(halo);
    const mod = document.createElementNS(SVGNS, "circle");
    mod.setAttribute("r", r + 5); mod.setAttribute("class", "mod");
    mod.setAttribute("fill", "none"); mod.setAttribute("stroke", "#b88521");
    mod.setAttribute("stroke-width", "2.6");
    g.appendChild(mod);
    const c = document.createElementNS(SVGNS, "circle");
    c.setAttribute("r", r); c.setAttribute("class", "main");
    c.setAttribute("fill", EVO_FILL[n.kind] || "#9e8e72");
    c.setAttribute("stroke", n.public ? "#4f3a1f" : "#faf7ee");
    c.setAttribute("stroke-width", n.public ? "2.6" : "1.4");
    g.appendChild(c);
    const title = document.createElementNS(SVGNS, "title");
    title.textContent = `${n.id}  ·  ${n.kind}${n.zone ? "  ·  " + n.zone : ""}${n.public ? "  ·  public" : ""}`;
    g.appendChild(title);
    const lab = evoNodeLabel(n);
    if (lab && (EVO_LABELED.has(n.kind) || n.public)) {
      const side = (n.kind === "endpoint" || n.kind === "host");
      const t = document.createElementNS(SVGNS, "text");
      if (side) {
        t.setAttribute("x", r + 5); t.setAttribute("y", 3.2); t.setAttribute("text-anchor", "start");
        t.setAttribute("class", "nsub");
      } else {
        t.setAttribute("x", 0); t.setAttribute("y", r + 12); t.setAttribute("text-anchor", "middle");
        t.setAttribute("class", "nlab" + (n.kind === "service" || n.kind === "secret" || n.kind === "vulnerability" ? " strong" : ""));
      }
      t.textContent = lab;
      g.appendChild(t);
      let sub = "";
      if (n.kind === "service" && n.zone) sub = n.public ? n.zone + " · public" : n.zone;
      else if (n.kind === "network") sub = "network";
      if (sub) {
        const z = document.createElementNS(SVGNS, "text");
        z.setAttribute("x", 0); z.setAttribute("y", r + 23); z.setAttribute("text-anchor", "middle");
        z.setAttribute("class", "nsub");
        z.textContent = sub;
        g.appendChild(z);
      }
    }
    const pill = document.createElementNS(SVGNS, "text");
    pill.setAttribute("x", r + 4); pill.setAttribute("y", -r - 2);
    pill.setAttribute("class", "npill"); pill.textContent = "new";
    g.appendChild(pill);
    gNodes.appendChild(g);
    evo.nodeEls.set(n.id, g);
  });

  mountEvoLegend();
  mountEvoDots();
  wireEvoPanZoom(svg);
  fitEvoView();
  evo.mounted = true;
}

function mountEvoLegend() {
  const el = document.getElementById("evo-legend");
  if (!el) return;
  el.innerHTML = EVO_LEGEND.map(([name, color]) => {
    const isRing = color === "ring" || color === "ring-amber";
    const cls = color === "ring-amber" ? " ring amber" : color === "ring" ? " ring" : "";
    const style = isRing ? "" : ` style="background:${color}"`;
    return `<div class="row"><span class="sw${cls}"${style}></span>${escapeHtml(name)}</div>`;
  }).join("");
}

function mountEvoDots() {
  const wrap = document.getElementById("evo-dots");
  if (!wrap) return;
  wrap.innerHTML = "";
  evo.steps.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "evo-dot";
    b.setAttribute("aria-label", i === 0 ? "Initial world" : "Evolution " + i);
    b.addEventListener("click", () => { evo.step = i; renderEvoStep(); });
    wrap.appendChild(b);
  });
}

// Fit into the band between the top bar and scrubber (both float over the SVG).
function fitEvoView() {
  const svg = document.getElementById("evo-graph");
  if (!svg || !evo.nodes.length) return;
  const xs = evo.nodes.map((n) => n.x), ys = evo.nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const tx0 = 70, ty0 = 86, tx1 = evo.W - 70, ty1 = evo.H - 152;
  const bw = Math.max(1, maxX - minX), bh = Math.max(1, maxY - minY);
  const k = Math.min((tx1 - tx0) / bw, (ty1 - ty0) / bh, 1.5);
  evo.view.k = k;
  evo.view.x = tx0 + ((tx1 - tx0) - k * bw) / 2 - k * minX;
  evo.view.y = ty0 + ((ty1 - ty0) - k * bh) / 2 - k * minY;
  applyEvoTransform();
}

function applyEvoTransform() {
  if (evo._vp) {
    evo._vp.setAttribute("transform",
      `translate(${evo.view.x},${evo.view.y}) scale(${evo.view.k})`);
  }
}

function wireEvoPanZoom(svg) {
  let dragging = false, lx = 0, ly = 0;
  svg.onpointerdown = (e) => { dragging = true; lx = e.clientX; ly = e.clientY; svg.classList.add("grabbing"); svg.setPointerCapture(e.pointerId); };
  svg.onpointermove = (e) => {
    if (!dragging) return;
    evo.view.x += e.clientX - lx; evo.view.y += e.clientY - ly;
    lx = e.clientX; ly = e.clientY; applyEvoTransform();
  };
  svg.onpointerup = (e) => { dragging = false; svg.classList.remove("grabbing"); try { svg.releasePointerCapture(e.pointerId); } catch (_) {} };
  svg.onwheel = (e) => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const scale = evo.W / rect.width;                  // client px → viewBox units
    const mx = (e.clientX - rect.left) * scale, my = (e.clientY - rect.top) * scale;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const nk = Math.max(0.4, Math.min(4, evo.view.k * factor));
    evo.view.x = mx - (mx - evo.view.x) * (nk / evo.view.k);
    evo.view.y = my - (my - evo.view.y) * (nk / evo.view.k);
    evo.view.k = nk; applyEvoTransform();
  };
}

function evoShortName(vid) {
  return (vid || "").replace(/^vuln_/, "").replace(/_\d+$/, "");
}

// "harden"/"soften" collide with security-hardening (the opposite); show the
// agent-difficulty sense instead.
const EVO_DIRECTION_WORD = { harden: "harder", soften: "easier", diversify: "varied" };
function evoDirectionWord(d) { return EVO_DIRECTION_WORD[d] || d || ""; }

function evoSummaryText(node) {
  const ev = node.evolve;
  if (!ev) return node.builder_summary || "";   // initial world
  const word = evoDirectionWord(ev.direction);
  const fam = ev.family ? ` · ${ev.family}` : "";
  const note = ev.note ? ` — ${ev.note}` : "";
  return `${word}${fam}${note}`;
}

function renderEvoStep() {
  if (!evo.steps.length) return;
  const i = Math.max(0, Math.min(evo.step, evo.steps.length - 1));
  evo.step = i;
  const ns = evo.stepNodeSets[i], es = evo.stepEdgeSets[i];
  const prevN = i > 0 ? evo.stepNodeSets[i - 1] : null;
  const prevE = i > 0 ? evo.stepEdgeSets[i - 1] : null;

  const attrsNow = evo.stepAttrs[i], attrsPrev = i > 0 ? evo.stepAttrs[i - 1] : null;
  const modified = [];
  if (attrsPrev) {
    ns.forEach((id) => {
      if (!prevN || !prevN.has(id)) return;
      const a = attrsNow.get(id) || {}, b = attrsPrev.get(id) || {};
      for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
        if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) {
          modified.push({ id, key: k, from: b[k], to: a[k] });
          break;
        }
      }
    });
  }
  const modIds = new Set(modified.map((m) => m.id));

  evo.nodeEls.forEach((el, id) => {
    const on = ns.has(id);
    // Linger one step as a ghost before going fully hidden.
    const justRemoved = !on && prevN != null && prevN.has(id);
    el.classList.toggle("removed", justRemoved);
    el.classList.toggle("gone", !on && !justRemoved);
    el.classList.toggle("added", on && prevN != null && !prevN.has(id));
    el.classList.toggle("modified", on && modIds.has(id));
  });
  evo.edgeEls.forEach((el, id) => {
    const on = es.has(id);
    const justRemoved = !on && prevE != null && prevE.has(id);
    el.classList.toggle("removed", justRemoved);
    el.classList.toggle("gone", !on && !justRemoved);
    el.classList.toggle("added", on && prevE != null && !prevE.has(id));
  });

  const step = evo.steps[i];
  document.getElementById("evo-step-label").textContent = i === 0 ? "Initial world" : "Evolution " + i;
  document.getElementById("evo-step-snap").textContent = "…" + (step.id || "").slice(-8);
  document.getElementById("evo-step-summary").textContent = evoSummaryText(step);

  const addedN = prevN ? [...ns].filter((x) => !prevN.has(x)) : [];
  const addedE = prevE ? [...es].filter((x) => !prevE.has(x)) : [];
  const removedN = prevN ? [...prevN].filter((x) => !ns.has(x)) : [];
  const ch = document.getElementById("evo-step-changes");
  const tally = `<span class="neutral"> · ${ns.size} nodes / ${es.size} edges</span>`;
  const byId = new Map(evo.nodes.map((n) => [n.id, n]));
  const vulnNamesOf = (ids) => ids.map((x) => byId.get(x)).filter((n) => n && n.kind === "vulnerability").map((n) => evoShortName(n.id));
  if (addedN.length || addedE.length || removedN.length) {
    const addV = vulnNamesOf(addedN), remV = vulnNamesOf(removedN);
    const parts = [];
    if (addedN.length) parts.push(`<span class="add">+${addedN.length} node${addedN.length > 1 ? "s" : ""}${addV.length ? ` (${addV.join(", ")})` : ""}</span>`);
    if (addedE.length) parts.push(`<span class="add">+${addedE.length} edge${addedE.length > 1 ? "s" : ""}</span>`);
    if (removedN.length) parts.push(`<span class="rm">−${removedN.length} node${removedN.length > 1 ? "s" : ""}${remV.length ? ` (${remV.join(", ")})` : ""}</span>`);
    ch.innerHTML = parts.join(" ") + tally;
  } else if (modified.length) {
    const m = modified[0];
    const nm = evoNodeLabel(byId.get(m.id)) || m.id;
    const delta = (m.from === undefined || m.from === null)
      ? `${escapeHtml(m.key)} → ${escapeHtml(String(m.to))}`
      : `${escapeHtml(m.key)} ${escapeHtml(String(m.from))}→${escapeHtml(String(m.to))}`;
    ch.innerHTML = `<span class="mod">${delta}</span> <span class="neutral">on ${escapeHtml(nm)}${tally}</span>`;
  } else if (i === 0) {
    ch.innerHTML = `<span class="neutral">starting world · ${ns.size} nodes / ${es.size} edges</span>`;
  } else {
    ch.innerHTML = `<span class="neutral">no change · ${ns.size} nodes / ${es.size} edges</span>`;
  }

  document.getElementById("evo-prev").disabled = (i === 0);
  document.getElementById("evo-next").disabled = (i === evo.steps.length - 1);
  const dots = document.getElementById("evo-dots").children;
  for (let d = 0; d < dots.length; d++) dots[d].classList.toggle("on", d === i);
}

function renderEvoView() {
  if (!evo.active) return;
  buildEvo();
  const empty = document.getElementById("evo-emptymsg");
  const scrub = document.getElementById("evo-scrubber");
  if (!evo.steps.length) {
    if (empty) empty.hidden = false;
    if (scrub) scrub.hidden = true;
    const svg = document.getElementById("evo-graph");
    if (svg) svg.innerHTML = "";
    evo.mounted = false;
    return;
  }
  if (empty) empty.hidden = true;
  if (scrub) scrub.hidden = false;
  if (!evo.mounted) mountEvoGraph();
  renderEvoStep();
}

// =============================================================
// Top-level render coordinator
// =============================================================

function render() {
  initSimulation();
  renderEmptyState();
  renderTopbar();
  renderFootbar();
  if (!sim.fallback) {
    rebuildSimulationWorld();
    updateSimulationFromEvents();
  }
  // Re-render the active rail panel so it reflects fresh data.
  if (rail.open) renderRailPanel(rail.active);
  renderCastRail();
  renderEvoView();
  // Surface a build banner when a new lineage node lands.
  checkForNewLineage();
}

// =============================================================
// Data feed
// =============================================================

async function refresh() {
  const [briefing, actors, topology, lineage, state, narration] = await Promise.all([
    json("/api/briefing"),
    json("/api/actors"),
    json("/api/topology"),
    json("/api/lineage"),
    json("/api/state"),
    json("/api/narrate"),
  ]);
  model.briefing = briefing;
  model.actors = actors;
  model.topology = topology;
  model.lineage = lineage;
  model.state = state;
  model.narration = narration;
  render();
}

function closeStreams() {
  if (runState.events) { runState.events.close(); runState.events = null; }
  if (runState.narration) { runState.narration.close(); runState.narration = null; }
}

let _refreshScheduled = false;
function scheduleRefresh() {
  if (_refreshScheduled) return;
  _refreshScheduled = true;
  setTimeout(() => { _refreshScheduled = false; refresh(); }, 150);
}

function openStreams() {
  closeStreams();
  if (!runState.activeRun) return;
  runState.events = new EventSource(withRun("/api/events/stream"));
  for (const eventType of ["agent_step", "env_turn", "note", "builder_step"]) {
    runState.events.addEventListener(eventType, scheduleRefresh);
  }
  runState.narration = new EventSource(withRun("/api/narrate/stream"));
  runState.narration.addEventListener("narration", scheduleRefresh);
}

function applyRunsToPicker(runs, defaultId) {
  runState.runs = runs;
  const list = document.getElementById("run-picker-list");
  if (!list) return;
  if (!runs.length) {
    list.innerHTML = `<li class="empty">No runs found</li>`;
    runState.activeRun = null;
    closeStreams();
    return;
  }
  const newest = defaultId || runs[0].id;
  const previous = runState.activeRun;
  let target;
  if (runState.followLatest) target = newest;
  else if (previous && runs.some((r) => r.id === previous)) target = previous;
  else target = newest;
  list.innerHTML = runs.map(run => {
    const ts = run.modified ? new Date(run.modified * 1000).toLocaleString() : "";
    return `<li class="${run.id === target ? "is-active" : ""}" data-run-id="${escapeHtml(run.id)}">
      <span class="run-id">${escapeHtml(run.id)}</span>
      <span class="run-meta">${escapeHtml(ts)}</span>
    </li>`;
  }).join("");
  list.querySelectorAll("li[data-run-id]").forEach((li) => {
    li.addEventListener("click", () => selectRun(li.dataset.runId, true));
  });
  if (target !== runState.activeRun) {
    runState.activeRun = target;
    openStreams();
    refresh();
  }
}

function selectRun(runId, fromUser) {
  if (!runId || runId === runState.activeRun) return;
  if (fromUser) {
    const followToggle = document.getElementById("run-picker-follow");
    if (followToggle && followToggle.checked) {
      followToggle.checked = false;
      runState.followLatest = false;
    }
  }
  runState.activeRun = runId;
  openStreams();
  refresh();
}

async function refreshRuns() {
  try {
    const payload = await fetch("/api/runs").then(r => r.json());
    applyRunsToPicker(payload.runs || [], payload.default || null);
  } catch (err) {
    console.warn("failed to list runs", err);
  }
}

async function safeRefresh() {
  try { await refresh(); runState.lastRefreshAt = Date.now(); }
  catch (err) { console.warn("refresh failed", err); }
}

async function safeRefreshRuns() {
  try { await refreshRuns(); }
  catch (err) { console.warn("refreshRuns failed", err); }
}

// =============================================================
// Bootstrap event wiring + main loop
// =============================================================

function wireUI() {
  // Episode controls.
  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await json(`/api/episode/${btn.dataset.action}`, { method: "POST" });
      await refresh();
    });
  });
  // Snapshot pill opens the build tab.
  document.getElementById("snapshot-pill")?.addEventListener("click", () => {
    rail.selectedLineageId = null;
    showRailTab("build");
  });
  // Run picker dropdown.
  const runBtn = document.getElementById("run-picker-btn");
  const runMenu = document.getElementById("run-picker-menu");
  runBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = runMenu.hidden;
    runMenu.hidden = !open;
    runBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", (e) => {
    if (runMenu && !runMenu.hidden && !runMenu.contains(e.target) && e.target !== runBtn) {
      runMenu.hidden = true;
      runBtn?.setAttribute("aria-expanded", "false");
    }
  });
  document.getElementById("run-picker-follow")?.addEventListener("change", async (e) => {
    runState.followLatest = e.target.checked;
    if (runState.followLatest) await refreshRuns();
  });

  // Rail tabs.
  document.querySelectorAll(".rail-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (rail.open && rail.active === btn.dataset.tab) {
        closeRail();
      } else {
        showRailTab(btn.dataset.tab);
      }
    });
  });
  // World subtabs.
  document.querySelectorAll(".rail-subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      rail.worldSubtab = btn.dataset.worldTab;
      renderRailPanel("world");
    });
  });
  // Rail close button.
  document.getElementById("rail-close")?.addEventListener("click", closeRail);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (runMenu && !runMenu.hidden) {
        runMenu.hidden = true;
        runBtn?.setAttribute("aria-expanded", "false");
        return;
      }
      if (rail.open) closeRail();
    }
  });
  document.querySelectorAll(".view-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => setEvoView(btn.dataset.view));
  });
  document.getElementById("evo-prev")?.addEventListener("click", () => {
    if (evo.step > 0) { evo.step--; renderEvoStep(); }
  });
  document.getElementById("evo-next")?.addEventListener("click", () => {
    if (evo.step < evo.steps.length - 1) { evo.step++; renderEvoStep(); }
  });
  document.addEventListener("keydown", (e) => {
    if (!evo.active) return;
    if (e.key === "ArrowLeft" && evo.step > 0) { evo.step--; renderEvoStep(); }
    if (e.key === "ArrowRight" && evo.step < evo.steps.length - 1) { evo.step++; renderEvoStep(); }
  });

  // Build banner.
  document.getElementById("build-banner-cta")?.addEventListener("click", () => {
    if (banner.current) rail.selectedLineageId = banner.current.id;
    showRailTab("build");
    dismissBuildBanner();
  });
  document.getElementById("build-banner-dismiss")?.addEventListener("click", dismissBuildBanner);
}

(async () => {
  wireUI();
  await safeRefreshRuns();
  await safeRefresh();
  showRailTab("world");
  setInterval(safeRefreshRuns, 5000);
  setInterval(() => {
    if (runState.activeRun) safeRefresh();
  }, 1000);
})();

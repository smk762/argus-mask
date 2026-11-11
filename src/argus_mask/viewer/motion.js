// What a face does on top of its frames: shared by the viewer and the editor's previews.
//
// - The lip budget. Audio2Face's weights are tuned for its own character; on another mesh, the shapes
//   that close the lips (mouthClose, mouthShrugLower, the rolls, pucker, press) can push the lower lip up
//   through the upper one. Each mouth shape's effect on the gap between the lips is measured once, from the
//   face's own morph targets, and on every frame the closing shapes are scaled down so they never close
//   the lips further than the opening shapes (jawOpen, funnel, ...) leave room for, plus a millimetre for
//   lips that press together. Lips still meet for m, b and p; they no longer pass through each other.
// - The rest set: a named baseline expression (warm, calm, ...) the face holds at rest and speaks from.
// - Idle life, while no stream plays: blinks, glances with the head following, breathing, and small
//   expression shifts that come and go. Cosmetic: never recorded, and absent from takes.
//
// Everything here is plain arithmetic on weights by name; nothing reads the network.

// The 52 ARKit names, in the order every frame's weights use (argus_mask/arkit.py).
export const ARKIT = [
  'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft',
  'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight',
  'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
  'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft',
  'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
  'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft',
  'mouthUpperUpRight', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
  'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
];

// Tuned on ginger-3 against nine Audio2Face takes (three held out): lips overlapping by over 1 mm fell from
// 24% of frames to 3%, by over 3 mm from 12% to none, and the lower lip no longer rises 2 mm above rest.
export const LIP_MARGIN_MM = 1.0;
export const LIP_GAINS = { mouthShrugLower: 0.7, mouthRollLower: 0.7, mouthRollUpper: 0.7 };
const SENSITIVE_MM = 0.3;   // a shape that moves the lip gap less than this per unit weight is left alone

function percentile(values, p) {
  const s = Float64Array.from(values).sort();
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] : 0;
}

// The lip budget of a loaded face, or null when it has no mouth shapes to measure.
// Returns { closers: {name: mm per unit}, openers: {name: mm per unit} }.
export function lipBudget(root) {
  let mesh = null;
  root.traverse((node) => {
    const dict = node.isMesh && node.morphTargetDictionary;
    if (!dict || !('jawOpen' in dict) || !('mouthFunnel' in dict) || !('mouthPucker' in dict)) return;
    const n = node.geometry.attributes.position.count;
    if (!mesh || n > mesh.geometry.attributes.position.count) mesh = node;
  });
  if (!mesh) return null;
  const geo = mesh.geometry;
  const base = geo.attributes.position;
  const morphs = geo.morphAttributes.position || [];
  const relative = geo.morphTargetsRelative !== false;
  const dict = mesh.morphTargetDictionary;
  const n = base.count;
  const scale = mesh.getWorldScale(mesh.position.clone()).y || 1;
  const delta = (name) => {
    const attr = morphs[dict[name]];
    if (!attr) return null;
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      for (let k = 0; k < 3; k += 1) out[i * 3 + k] = attr.getComponent(i, k) - (relative ? 0 : base.getComponent(i, k));
    }
    return out;
  };
  const len = (d, i) => Math.hypot(d[i * 3], d[i * 3 + 1], d[i * 3 + 2]);
  const funnel = delta('mouthFunnel');
  const pucker = delta('mouthPucker');
  const jaw = delta('jawOpen');
  const moved = new Float32Array(n);
  for (let i = 0; i < n; i += 1) moved[i] = len(funnel, i) + len(pucker, i);
  const cut = percentile(moved, 0.975);
  const mouth = [];
  for (let i = 0; i < n; i += 1) if (moved[i] > cut) mouth.push(i);
  if (mouth.length < 8) return null;
  const jawMax = Math.max(...mouth.map((i) => Math.abs(jaw[i * 3 + 1])));
  const lower = mouth.filter((i) => jaw[i * 3 + 1] < -0.4 * jawMax);   // go down with the jaw
  const upper = mouth.filter((i) => Math.abs(jaw[i * 3 + 1]) < 0.08 * jawMax);
  if (!lower.length || !upper.length) return null;
  const xs = mouth.map((i) => base.getX(i));
  const cx = percentile(xs, 0.5);
  const half = 0.3 * (Math.max(...xs) - Math.min(...xs));
  const band = 0.2 * half;
  const columns = Array.from({ length: 9 }, (_, c) => cx - half + (2 * half * c) / 8);

  // Per column across the mouth: the lowest upper-lip point less the highest lower-lip point, in mm.
  const gaps = (d) => columns.map((c) => {
    let top = Infinity; let bottom = -Infinity;
    for (const i of upper) {
      const x = base.getX(i) + (d ? d[i * 3] : 0);
      if (Math.abs(x - c) < band) top = Math.min(top, base.getY(i) + (d ? d[i * 3 + 1] : 0));
    }
    for (const i of lower) {
      const x = base.getX(i) + (d ? d[i * 3] : 0);
      if (Math.abs(x - c) < band) bottom = Math.max(bottom, base.getY(i) + (d ? d[i * 3 + 1] : 0));
    }
    return Number.isFinite(top) && Number.isFinite(bottom) ? (top - bottom) * 1000 * scale : NaN;
  });
  const rest = gaps(null);
  const closers = {};
  const openers = {};
  for (const name of Object.keys(dict)) {
    if (!/^(mouth|jaw)/.test(name)) continue;
    const d = delta(name);
    if (!d) continue;
    const g = gaps(d);
    let overlap = -Infinity;
    for (let c = 0; c < g.length; c += 1) if (!Number.isNaN(g[c]) && !Number.isNaN(rest[c])) overlap = Math.max(overlap, rest[c] - g[c]);
    if (!Number.isFinite(overlap)) continue;
    if (overlap > SENSITIVE_MM) closers[name] = overlap;
    else if (overlap < -SENSITIVE_MM) openers[name] = -overlap;
  }
  return Object.keys(closers).length ? { closers, openers } : null;
}

// A function that shapes a Float32Array of weights in *names* order, in place: the gains, then the budget.
export function lipShaper(budget, names, { margin = LIP_MARGIN_MM, gains = LIP_GAINS } = {}) {
  const at = (name) => names.indexOf(name);
  const gain = Object.entries(gains).map(([k, g]) => [at(k), g]).filter(([i]) => i >= 0);
  if (!budget) {
    // No budget measured: at least never close the lips past the jaw (ARKit's own rule for mouthClose).
    const jaw = at('jawOpen');
    const close = at('mouthClose');
    return (w) => {
      for (const [i, g] of gain) w[i] *= g;
      if (jaw >= 0 && close >= 0 && w[close] > w[jaw]) w[close] = w[jaw];
    };
  }
  const closers = Object.entries(budget.closers).map(([k, a]) => [at(k), a]).filter(([i]) => i >= 0);
  const openers = Object.entries(budget.openers).map(([k, a]) => [at(k), a]).filter(([i]) => i >= 0);
  return (w) => {
    for (const [i, g] of gain) w[i] *= g;
    let close = 0;
    for (const [i, a] of closers) close += w[i] * a;
    let room = margin;
    for (const [i, a] of openers) room += w[i] * a;
    if (close > room && close > 0) {
      const k = Math.max(0, room) / close;
      for (const [i] of closers) w[i] *= k;
    }
  };
}

// ---- idle life -------------------------------------------------------------------------------------------

// A small seeded generator, so a page's idle is the same each time it is opened.
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
const between = (r, lo, hi) => lo + (hi - lo) * r();
const smooth = (x) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };

// Small expressions that come and go at rest: an interested brow, a passing smile, a thoughtful press.
const SHIFTS = [
  { browInnerUp: 0.18, browOuterUpLeft: 0.08, browOuterUpRight: 0.08 },
  { mouthSmileLeft: 0.18, mouthSmileRight: 0.18, cheekSquintLeft: 0.08, cheekSquintRight: 0.08, eyeSquintLeft: 0.06, eyeSquintRight: 0.06 },
  { mouthPressLeft: 0.10, mouthPressRight: 0.10, browDownLeft: 0.06, browDownRight: 0.06 },
  { eyeSquintLeft: 0.10, eyeSquintRight: 0.10, mouthSmileLeft: 0.08, mouthSmileRight: 0.08 },
  { browInnerUp: 0.10, mouthPucker: 0.06 },
];

export class Idle {
  // life: 0 still, 1 the default amount of movement, up to 2.
  constructor({ seed = 7, life = 1 } = {}) {
    this.r = rng(seed);
    this.life = life;
    this.t = 0;
    this.nextBlink = 1.2;
    this.blinkAt = -10;
    this.secondBlink = null;
    this.gaze = { x: 0, y: 0 };
    this.gazeFrom = { x: 0, y: 0 };
    this.gazeTo = { x: 0, y: 0 };
    this.gazeAt = -10;
    this.nextGaze = 0.8;
    this.headGaze = { x: 0, y: 0 };
    this.shift = null;
    this.nextShift = between(this.r, 3, 6);
  }

  // The idle weights (by name) and head angles (degrees) at *t* seconds, given *dt* since the last call.
  sample(t, dt) {
    const r = this.r;
    const life = this.life;
    const w = {};
    // Blinks: 2 to 5 s apart, one in six doubled; a big glance often brings one.
    if (t >= this.nextBlink) {
      this.blinkAt = t;
      this.secondBlink = r() < 0.16 ? t + 0.32 : null;
      this.nextBlink = t + between(r, 2.0, 5.0) / Math.max(0.5, Math.min(1.5, life));
    }
    if (this.secondBlink !== null && t >= this.secondBlink) { this.blinkAt = t; this.secondBlink = null; }
    const b = t - this.blinkAt;
    const blink = b < 0.07 ? b / 0.07 : b < 0.19 ? 1 - (b - 0.07) / 0.12 : 0;
    w.eyeBlinkLeft = blink;
    w.eyeBlinkRight = blink;

    // Glances: hold 0.8 to 3.5 s, then a quick move; now and then a longer look away, the head following.
    if (t >= this.nextGaze) {
      const far = r() < 0.22;
      const reach = (far ? 0.38 : 0.16) * Math.min(1.5, life);
      this.gazeFrom = { ...this.gaze };
      this.gazeTo = r() < 0.35 && !far ? { x: 0, y: 0 } : { x: between(r, -reach, reach), y: between(r, -0.5 * reach, 0.6 * reach) };
      this.gazeAt = t;
      this.nextGaze = t + (far ? between(r, 0.9, 2.0) : between(r, 0.8, 3.5));
      if (far && t - this.blinkAt > 0.6 && r() < 0.6) { this.blinkAt = t; this.nextBlink = Math.max(this.nextBlink, t + 1.5); }
    }
    const k = smooth((t - this.gazeAt) / 0.07);
    this.gaze = { x: this.gazeFrom.x + (this.gazeTo.x - this.gazeFrom.x) * k, y: this.gazeFrom.y + (this.gazeTo.y - this.gazeFrom.y) * k };
    const { x, y } = this.gaze;
    // x > 0: the face's own left; y > 0: up (as argus_mask/idle.py).
    w.eyeLookOutLeft = Math.max(0, x); w.eyeLookInRight = Math.max(0, x);
    w.eyeLookInLeft = Math.max(0, -x); w.eyeLookOutRight = Math.max(0, -x);
    w.eyeLookUpLeft = w.eyeLookUpRight = Math.max(0, y);
    w.eyeLookDownLeft = w.eyeLookDownRight = Math.max(0, -y);

    // The head follows the eyes part of the way, slowly; under it, a slow sway and breathing.
    const follow = 1 - Math.exp(-dt / 0.6);
    this.headGaze.x += (x - this.headGaze.x) * follow;
    this.headGaze.y += (y - this.headGaze.y) * follow;
    const head = {
      yaw: life * (1.4 * Math.sin(t / 7.3 * 2 * Math.PI)) + 22 * this.headGaze.x,
      pitch: life * (0.7 * Math.sin(t / 5.9 * 2 * Math.PI) + 0.45 * Math.sin(t / 4.1 * 2 * Math.PI)) - 14 * this.headGaze.y,
      roll: life * 0.6 * Math.sin(t / 9.7 * 2 * Math.PI) - 5 * this.headGaze.x,
    };

    // Expression shifts: every 6 to 14 s, one of SHIFTS eases in, holds 1.5 to 4 s and eases out.
    if (!this.shift && t >= this.nextShift && life > 0) {
      const hold = between(r, 1.5, 4.0);
      this.shift = { w: SHIFTS[Math.floor(r() * SHIFTS.length)], at: t, hold };
    }
    if (this.shift) {
      const s = t - this.shift.at;
      const ease = 0.7;
      const amount = s < ease ? smooth(s / ease) : s < ease + this.shift.hold ? 1 : 1 - smooth((s - ease - this.shift.hold) / ease);
      if (s > 2 * ease + this.shift.hold) { this.shift = null; this.nextShift = t + between(r, 6, 14) / Math.max(0.5, life); }
      else for (const [name, v] of Object.entries(this.shift.w)) w[name] = (w[name] || 0) + v * amount * Math.min(1.5, life);
    }
    return { weights: w, head };
  }
}

// ---- putting it together ---------------------------------------------------------------------------------

// How much of the rest set holds while speaking: the brows, eyes and cheeks fully, the mouth half,
// so a face speaks from its mood but the mouth stays the driver's.
const UNDER_SPEECH = (name) => (/^mouth|^jaw/.test(name) ? 0.5 : 1.0);

// Composes what a face shows: frame (or relaxing) weights, the rest set, idle life when not live, then the
// lip budget. Call `present` once per drawn frame with a Float32Array in *names* order.
export class Motion {
  constructor({ budget = null, rest = {}, seed = 7, life = 1 } = {}) {
    this.budget = budget;
    this.rest = rest;
    this.idle = new Idle({ seed, life });
    this.names = null;
    this.shape = null;
    this.out = null;
  }

  setRest(rest) { this.rest = rest || {}; }

  _bind(names) {
    this.names = names;
    this.index = new Map(names.map((n, i) => [n, i]));
    this.shape = lipShaper(this.budget, names);
    this.out = new Float32Array(names.length);
  }

  // Returns the weights to show (a Float32Array in names order) and, when idle, the head to show.
  present(weights, names, { live, t, dt }) {
    if (this.names !== names || !this.out || this.out.length !== names.length) this._bind(names);
    const out = this.out;
    out.set(weights);
    for (const [name, v] of Object.entries(this.rest)) {
      const i = this.index.get(name);
      if (i !== undefined) out[i] += v * (live ? UNDER_SPEECH(name) : 1);
    }
    let head = null;
    if (!live) {
      const idle = this.idle.sample(t, dt);
      for (const [name, v] of Object.entries(idle.weights)) {
        const i = this.index.get(name);
        if (i === undefined) continue;
        out[i] = name.startsWith('eyeBlink') ? Math.max(out[i], v) : out[i] + v;
      }
      head = idle.head;
    }
    this.shape(out);
    for (let i = 0; i < out.length; i += 1) out[i] = Math.min(1, Math.max(0, out[i]));
    return { weights: out, head };
  }
}

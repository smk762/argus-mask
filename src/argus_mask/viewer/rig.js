// A rigged head's turn (#62): the head turns through its neck and head bones, each taking a share of the turn
// about world axes, so the shoulders stay put. Shared by the viewer and the editor's previews.
//
// The heads the editor builds carry MPFB's game_engine rig (neck_01, head, ...). A model without those bones
// (an older head, or one from elsewhere) is not rigged: the caller turns it whole, as before.
//
// Each bone's local rotation becomes rest * (W^-1 D W): W its world rotation at rest, D its share of the turn
// (a slerp of the whole turn). The shares are slerps of one rotation, so down the chain they compose to exactly
// the turn asked for. The splat head does the same with its own bones (splat.js).

export const NECK_CHAIN = [['neck_01', 0.4], ['head', 0.6]];

// three.js's loader makes names unique, so the bone "head" and the mesh "head" may load as "head" and "head_1";
// the name in the file is kept in userData.name.
const nameOf = (node) => (node.userData && node.userData.name) || node.name;

export class HeadTurn {
  constructor(THREE, root) {
    this.THREE = THREE;
    const bones = new Map();
    root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
    const chain = NECK_CHAIN.filter(([name]) => bones.has(name));
    const total = chain.reduce((sum, [, part]) => sum + part, 0);
    root.updateMatrixWorld(true);
    this.turners = chain.map(([name, part]) => {
      const bone = bones.get(name);
      const world = bone.getWorldQuaternion(new THREE.Quaternion());
      return { bone, rest: bone.quaternion.clone(), share: part / total, world, inverse: world.clone().invert() };
    });
    this.euler = new THREE.Euler();
    this.turn = new THREE.Quaternion();
    this.part = new THREE.Quaternion();
    this.local = new THREE.Quaternion();
  }

  // Whether the model has the bones: if not, turn it whole.
  get rigged() { return this.turners.length > 0; }

  // The head turned by pitch, yaw and roll in degrees (as a mesh head turns whole: Euler YXZ, about world axes).
  apply(pitch = 0, yaw = 0, roll = 0) {
    const rad = Math.PI / 180;
    this.euler.set(pitch * rad, yaw * rad, roll * rad, 'YXZ');
    this.turn.setFromEuler(this.euler);
    for (const t of this.turners) {
      this.part.identity().slerp(this.turn, t.share);
      this.local.copy(t.inverse).multiply(this.part).multiply(t.world);
      t.bone.quaternion.copy(t.rest).multiply(this.local);
    }
  }
}

// A rest pose for a framing that shows the arms: MakeHuman builds in an A-pose (arms out at about 45 degrees),
// and a waist-up figure stands relaxed, arms down. Each bone takes any of:
//   world  [x, y, z] degrees about world axes (+x pitches forward, z swings a limb in the picture plane);
//   twist  degrees about the bone's own length (a forearm turning the palm in);
//   bend   degrees about the bone's own x axis (a finger curling).
// Applied once, after loading, to the bones the model has; a plain [x, y, z] is a world turn.
const FINGERS = ['index', 'middle', 'ring', 'pinky'];
const curl = (side, deg) => Object.fromEntries(FINGERS.flatMap((f) => [1, 2, 3].map((j) => [`${f}_0${j}_${side}`, { bend: deg * (j === 1 ? 0.6 : 1) }])));
export const ARMS_DOWN = {
  upperarm_l: { world: [4, 0, -40] },
  upperarm_r: { world: [4, 0, 40] },
  lowerarm_l: { world: [-6, 0, 0], twist: 25 },
  lowerarm_r: { world: [-6, 0, 0], twist: -25 },
  ...curl('l', 10),
  ...curl('r', 10),
};

// A full-body model stands as well: MakeHuman's legs are built apart, so the thighs come in a little and the feet
// turn back by as much, soles flat on the floor.
export const STANDING = {
  ...ARMS_DOWN,
  thigh_l: { world: [0, 0, -3.5] },
  thigh_r: { world: [0, 0, 3.5] },
  foot_l: { world: [0, 0, 3.5] },
  foot_r: { world: [0, 0, -3.5] },
};

// Set a pose (see ARMS_DOWN), parents first (the order of *pose*). Each bone's world rotation at rest is read
// before any is moved, so a child's turn is about the axes at rest, carried along by its parent's.
export function setPose(THREE, root, pose) {
  const bones = new Map();
  root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
  root.updateMatrixWorld(true);
  const rad = Math.PI / 180;
  const axisOf = (bone, local) => local.clone().applyQuaternion(bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
  const plan = Object.entries(pose).filter(([name]) => bones.has(name)).map(([name, spec]) => {
    const bone = bones.get(name);
    const { world: [x, y, z] = [0, 0, 0], twist = 0, bend = 0 } = Array.isArray(spec) ? { world: spec } : spec;
    const W = bone.getWorldQuaternion(new THREE.Quaternion());
    const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(x * rad, y * rad, z * rad, 'YXZ'));
    if (twist) turn.multiply(new THREE.Quaternion().setFromAxisAngle(axisOf(bone, new THREE.Vector3(0, 1, 0)), twist * rad));
    if (bend) turn.multiply(new THREE.Quaternion().setFromAxisAngle(axisOf(bone, new THREE.Vector3(1, 0, 0)), bend * rad));
    return { bone, local: W.clone().invert().multiply(turn).multiply(W) };
  });
  for (const { bone, local } of plan) bone.quaternion.multiply(local);
  root.updateMatrixWorld(true);
  return plan.length;
}

// Life below the neck for a framing that shows the body (#62): breathing, a slow shift of weight, and the chest
// following a nod a little. Like the head's idle life, a function of the face's seed and the time (the posed
// frame's in a recording), never the clock or a random draw: the same take records the same body.
// Degrees about world axes from the posed rest: each bone's local rotation becomes rest * (W^-1 D W), as above.
const BREATH_S = 4.4;                 // one breath at rest
const SHIFT_S = [11.3, 7.1];          // two slow, unrelated periods: the sway never quite repeats
export const BODY = {
  breath: { spine_02: [-0.35, 0, 0], spine_03: [-0.7, 0, 0], clavicle_l: [0, 0, 1.4], clavicle_r: [0, 0, -1.4] },
  shift: { pelvis: [0, 0.4, 1.0], spine_01: [0, 0, -0.55], spine_03: [0, 0, -0.3] },
  nod: { spine_03: 0.12 },            // the share of the head's pitch the chest follows
  // A full-body model's thighs take back the pelvis's turn, so the hips sway over feet that stay where they stand.
  plant: ['thigh_l', 'thigh_r'],
  // Each arm drifts on its own: a degree or so at the shoulder and elbow, on periods that share no rhythm, so the
  // arms never move in step and never quite repeat.
  arms: {
    upperarm_l: { amp: [0.9, 0, 1.1], period: 6.7 },
    upperarm_r: { amp: [0.9, 0, -1.1], period: 8.3 },
    lowerarm_l: { amp: [-1.6, 0, 0], period: 5.9 },
    lowerarm_r: { amp: [-1.6, 0, 0], period: 7.7 },
  },
};

export class BodyLife {
  constructor(THREE, root, seed = 0) {
    this.THREE = THREE;
    const bones = new Map();
    root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
    root.updateMatrixWorld(true);
    const names = new Set([...Object.keys(BODY.breath), ...Object.keys(BODY.shift), ...Object.keys(BODY.nod), ...Object.keys(BODY.arms), ...BODY.plant]);
    this.bones = new Map([...names].filter((n) => bones.has(n)).map((n) => {
      const bone = bones.get(n);
      const world = bone.getWorldQuaternion(new THREE.Quaternion());
      return [n, { bone, rest: bone.quaternion.clone(), world, inverse: world.clone().invert(), angle: [0, 0, 0] }];
    }));
    // Each face breathes and sways in its own phase, from its seed.
    const phase = (k) => ((Math.imul(seed ^ Math.imul(k, 0x9e3779b9), 2654435761) >>> 0) % 6283) / 1000;
    this.phases = [phase(1), phase(2), phase(3)];
    this.armPhases = Object.keys(BODY.arms).map((_, i) => phase(10 + i));
    this.euler = new THREE.Euler();
    this.turn = new THREE.Quaternion();
    this.local = new THREE.Quaternion();
  }

  get alive() { return this.bones.size > 0; }

  // The body at time *t* (seconds), the head pitched by *pitch* degrees (the chest follows a little).
  apply(t, pitch = 0) {
    const tau = 2 * Math.PI;
    const breath = 0.5 - 0.5 * Math.cos(tau * t / BREATH_S + this.phases[0]);   // 0 out, 1 in
    const shift = 0.7 * Math.sin(tau * t / SHIFT_S[0] + this.phases[1]) + 0.3 * Math.sin(tau * t / SHIFT_S[1] + this.phases[2]);
    for (const b of this.bones.values()) b.angle.fill(0);
    const add = (table, k) => {
      for (const [name, [x, y, z]] of Object.entries(table)) {
        const b = this.bones.get(name);
        if (b) { b.angle[0] += x * k; b.angle[1] += y * k; b.angle[2] += z * k; }
      }
    };
    add(BODY.breath, breath);
    add(BODY.shift, shift);
    for (const [name, share] of Object.entries(BODY.nod)) { const b = this.bones.get(name); if (b) b.angle[0] += share * pitch; }
    Object.entries(BODY.arms).forEach(([name, { amp, period }], i) => {
      const b = this.bones.get(name);
      if (!b) return;
      const k = Math.sin(tau * t / period + this.armPhases[i]);
      b.angle[0] += amp[0] * k; b.angle[1] += amp[1] * k; b.angle[2] += amp[2] * k;
    });
    const rad = Math.PI / 180;
    const pelvis = this.bones.get('pelvis');
    const planted = new Set(BODY.plant);
    for (const [name, b] of this.bones) {
      if (planted.has(name)) {
        if (!pelvis) continue;
        // The pelvis's own turn, undone: in world terms the thigh ends where it was at rest.
        const [x, y, z] = pelvis.angle;
        this.turn.setFromEuler(this.euler.set(x * rad, y * rad, z * rad, 'YXZ')).invert();
      } else {
        this.turn.setFromEuler(this.euler.set(b.angle[0] * rad, b.angle[1] * rad, b.angle[2] * rad, 'YXZ'));
      }
      this.local.copy(b.inverse).multiply(this.turn).multiply(b.world);
      b.bone.quaternion.copy(b.rest).multiply(this.local);
    }
  }
}

// Where a full-body model meets the floor: a soft shadow under it, broad around the stance and dark under each
// foot, so it stands on the floor (plain or a backdrop's) rather than floating in front of it. No shadow map: it
// is a flat, see-through card at the soles' height, the same on every profile, and a function of the pose alone.
// Call after setPose. Returns the card (added to *parent*), or null if the model has no feet.
export function contactShadow(THREE, root, parent, { strength = 0.8 } = {}) {
  const bones = new Map();
  root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
  if (!bones.has('foot_l') || !bones.has('foot_r')) return null;
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true);
  const feet = ['foot_l', 'foot_r'].map((n) => bones.get(n).getWorldPosition(new THREE.Vector3()));
  const floor = box.min.y + 0.002;   // just above the soles: never under them, never z-fighting
  const centre = new THREE.Vector3((feet[0].x + feet[1].x) / 2, floor, (feet[0].z + feet[1].z) / 2 + 0.03);
  const width = Math.max(0.6, Math.abs(feet[0].x - feet[1].x) + 0.5);
  const depth = 0.55;
  // The card lies flat (turned -90 degrees about x), so its v runs toward -z.
  const local = (p) => new THREE.Vector2((p.x - centre.x) / width + 0.5, (centre.z - p.z) / depth + 0.5);
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    toneMapped: false,
    uniforms: { uFeet: { value: feet.map((f) => local(new THREE.Vector3(f.x, 0, f.z + 0.05))) }, uStrength: { value: strength },
      uAspect: { value: width / depth } },
    vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `
      varying vec2 vUv;
      uniform vec2 uFeet[2];
      uniform float uStrength;
      uniform float uAspect;
      float blob(vec2 p, vec2 at, vec2 size) { vec2 d = (p - at) / size; return exp(-dot(d, d)); }
      void main() {
        vec2 p = vUv;
        float broad = blob(p, vec2(0.5), vec2(0.36, 0.36)) * 0.6;
        float soles = 0.0;
        for (int i = 0; i < 2; i++) soles += blob(p, uFeet[i], vec2(0.13, 0.13 * uAspect));
        float edge = 1.0 - smoothstep(0.42, 0.5, max(abs(p.x - 0.5), abs(p.y - 0.5)));   // nothing at the card's edge
        gl_FragColor = vec4(0.0, 0.0, 0.0, clamp(broad + 0.8 * soles, 0.0, 1.0) * uStrength * edge);
      }`,
  });
  const card = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), material);
  card.rotation.x = -Math.PI / 2;
  card.position.copy(centre);
  card.renderOrder = -1;   // before the model, so its blend sits under the shoes
  card.name = 'contact-shadow';
  parent.add(card);
  return card;
}

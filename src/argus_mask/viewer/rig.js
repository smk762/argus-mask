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
// and a waist-up figure stands with its arms down. Degrees about world axes, as (x, y, z): +x pitches forward,
// z swings a limb in the picture plane. Applied once, after loading, to the bones the model has.
export const ARMS_DOWN = {
  upperarm_l: [0, 0, -42],
  upperarm_r: [0, 0, 42],
  lowerarm_l: [-14, 0, 0],
  lowerarm_r: [-14, 0, 0],
};

// Set a pose: each named bone turned by its (x, y, z) degrees about world axes, from its rest, parents first
// (the order of *pose*). Each bone's world rotation at rest is read before any is moved, so a child's turn is
// about the world's axes at rest, carried along by its parent's.
export function setPose(THREE, root, pose) {
  const bones = new Map();
  root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
  root.updateMatrixWorld(true);
  const rad = Math.PI / 180;
  const plan = Object.entries(pose).filter(([name]) => bones.has(name)).map(([name, [x, y, z]]) => {
    const bone = bones.get(name);
    const world = bone.getWorldQuaternion(new THREE.Quaternion());
    const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(x * rad, y * rad, z * rad, 'YXZ'));
    return { bone, local: world.clone().invert().multiply(turn).multiply(world) };
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
};

export class BodyLife {
  constructor(THREE, root, seed = 0) {
    this.THREE = THREE;
    const bones = new Map();
    root.traverse((node) => { if (node.isBone) bones.set(nameOf(node), node); });
    root.updateMatrixWorld(true);
    const names = new Set([...Object.keys(BODY.breath), ...Object.keys(BODY.shift), ...Object.keys(BODY.nod)]);
    this.bones = new Map([...names].filter((n) => bones.has(n)).map((n) => {
      const bone = bones.get(n);
      const world = bone.getWorldQuaternion(new THREE.Quaternion());
      return [n, { bone, rest: bone.quaternion.clone(), world, inverse: world.clone().invert(), angle: [0, 0, 0] }];
    }));
    // Each face breathes and sways in its own phase, from its seed.
    const phase = (k) => ((Math.imul(seed ^ Math.imul(k, 0x9e3779b9), 2654435761) >>> 0) % 6283) / 1000;
    this.phases = [phase(1), phase(2), phase(3)];
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
    const rad = Math.PI / 180;
    for (const b of this.bones.values()) {
      this.turn.setFromEuler(this.euler.set(b.angle[0] * rad, b.angle[1] * rad, b.angle[2] * rad, 'YXZ'));
      this.local.copy(b.inverse).multiply(this.turn).multiply(b.world);
      b.bone.quaternion.copy(b.rest).multiply(this.local);
    }
  }
}

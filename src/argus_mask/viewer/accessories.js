// Runtime accessories as committed Three.js factories (#67 spike).
//
// Mesh faces: parent the group to a named bone. Splat faces: the vendored renderer draws
// threeScene *before* the Gaussians and hides the avatar mesh, so a bone child would sit
// behind the head or stay invisible. Instead we keep a visible overlay scene and copy each
// bone's world matrix every frame, then draw after the splat pass (same camera, depth on).

import * as THREE from 'three';
import { createEarpiece } from './accessories/earpiece.js';

export const FACTORIES = {
  earpiece: createEarpiece,
};

/** Parent *group* under *bone* with a local TRS. For mesh heads. */
export function attach(bone, group, { offset = [0, 0, 0], rotation = [0, 0, 0], scale = 1 } = {}) {
  bone.add(group);
  group.position.fromArray(offset);
  group.rotation.set(rotation[0], rotation[1], rotation[2], 'YXZ');
  const s = Array.isArray(scale) ? scale : [scale, scale, scale];
  group.scale.set(s[0], s[1], s[2]);
  return group;
}

/** Find a bone by file name (userData.name or .name). */
export function findBone(root, name) {
  let found = null;
  root.traverse((node) => {
    if (found || !node.isBone) return;
    const n = (node.userData && node.userData.name) || node.name;
    if (n === name) found = node;
  });
  return found;
}

/**
 * Overlay that follows bones for a splat head (or any skeleton drawn separately).
 * Call sync() after the skeleton's world matrices are up to date, then draw(renderer, camera).
 */
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();

export class BoneOverlay {
  constructor() {
    this.scene = new THREE.Scene();
    this.bindings = [];   // { bone, holder }
    this.light = new THREE.DirectionalLight(0xffffff, 1.1);
    this.light.position.set(0.4, 1.2, 0.8);
    this.scene.add(this.light);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  }

  /**
   * Add a factory-built group that tracks *bone*. Local TRS is relative to the bone's
   * world pose with unit scale (LAM's rig scale is stripped so offsets stay in metres).
   * Returns the holder (an Object3D in this.scene).
   */
  follow(bone, group, { offset = [0, 0, 0], rotation = [0, 0, 0], scale = 1 } = {}) {
    const holder = new THREE.Object3D();
    holder.name = `follow:${bone.name || 'bone'}`;
    holder.add(group);
    group.position.fromArray(offset);
    group.rotation.set(rotation[0], rotation[1], rotation[2], 'YXZ');
    const s = Array.isArray(scale) ? scale : [scale, scale, scale];
    group.scale.set(s[0], s[1], s[2]);
    this.scene.add(holder);
    this.bindings.push({ bone, holder });
    return holder;
  }

  sync() {
    for (const { bone, holder } of this.bindings) {
      bone.updateWorldMatrix(true, false);
      bone.matrixWorld.decompose(_p, _q, _s);
      holder.position.copy(_p);
      holder.quaternion.copy(_q);
      holder.scale.set(1, 1, 1);   // metres; ignore the skin.glb's ~0.01/100 bone scale
    }
  }

  draw(renderer, camera) {
    if (!this.bindings.length) return;
    this.sync();
    const auto = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(this.scene, camera);
    renderer.autoClear = auto;
  }

  dispose() {
    for (const { holder } of this.bindings) {
      holder.traverse((n) => {
        if (n.geometry) n.geometry.dispose();
        if (n.material) {
          if (Array.isArray(n.material)) n.material.forEach((m) => m.dispose());
          else n.material.dispose();
        }
      });
      this.scene.remove(holder);
    }
    this.bindings = [];
  }
}

/** Default local TRS for the spike earpiece on a LAM head bone. Metres, YXZ radians.
 *  Tuned on splat-01: wearer's right ear (viewer's left). */
export const EARPIECE_ON_HEAD = {
  offset: [-0.085, 0.0, 0.02],
  rotation: [0.15, 0.55, -0.25],
  scale: 1,
};

/**
 * Build the named accessory. Returns null if the name is unknown.
 * @param {string} name
 */
export function create(name) {
  const factory = FACTORIES[name];
  return factory ? factory() : null;
}

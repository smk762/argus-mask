// Comms earpiece as a code-only THREE.Group (#67 accessories spike).
//
// Authored to the img2threejs output contract (a reviewed factory, not an agent at runtime):
// primitives + materials, no network textures, no unseeded RNG or wall clock. Reference:
// docs/media/earpiece-ref.png (synthetic, repo-owned). Tool evaluated: img2threejs (Apache-2.0).

import * as THREE from 'three';

const BODY = 0x3a4048;
const ACCENT = 0x6aa0e0;
const STEM = 0x4a5060;

/** @returns {THREE.Group} */
export function createEarpiece() {
  const root = new THREE.Group();
  root.name = 'earpiece';

  const mat = (color, opts = {}) => new THREE.MeshStandardMaterial({
    color, metalness: 0.35, roughness: 0.45, flatShading: false, ...opts,
  });
  const body = mat(BODY);
  const accent = mat(ACCENT, { metalness: 0.2, roughness: 0.35 });
  const stem = mat(STEM);

  // Hook that sits over the ear (torus arc).
  const hook = new THREE.Mesh(new THREE.TorusGeometry(0.028, 0.005, 10, 24, Math.PI * 1.35), body);
  hook.rotation.set(0.15, 0.4, Math.PI * 0.55);
  hook.position.set(0.002, 0.012, -0.004);
  root.add(hook);

  // Bud against the ear canal.
  const bud = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), body);
  bud.position.set(0.006, 0, 0.008);
  root.add(bud);

  const window = new THREE.Mesh(new THREE.CircleGeometry(0.006, 16), accent);
  window.position.set(0.017, 0.001, 0.01);
  window.rotation.y = Math.PI / 2;
  root.add(window);

  // Short boom toward the mouth.
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.0022, 0.0028, 0.055, 8), stem);
  boom.position.set(0.004, -0.028, 0.018);
  boom.rotation.set(0.9, 0, 0.35);
  root.add(boom);

  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.0045, 10, 8), body);
  tip.position.set(0.01, -0.052, 0.038);
  root.add(tip);

  // Socket other props / animation can find (img2threejs-style hierarchy hint).
  const socket = new THREE.Object3D();
  socket.name = 'socket_attach';
  socket.position.set(0, 0, 0);
  root.add(socket);

  root.traverse((n) => {
    if (n.isMesh) {
      n.castShadow = false;
      n.receiveShadow = false;
      n.frustumCulled = false;
    }
  });
  return root;
}

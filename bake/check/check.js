// Route A viewer check (#126): trained splats bound to every part's triangles, driven by ARKit weights, in Spark.
// ?run=<dir under trained/>  ?mode=splat|mesh (mesh: the head as the demo draws it, look.js high)  ?splats=<file in the run>.
// window.__bake.view(i) draws camera i of cameras.json at its expression, exactly (sorted, then presented).
// Served from a work directory that holds this page, spark.module.js, viewer/ (the repo's), head.glb,
// cameras.json, expressions.json and trained/<run>/: see bake/README.md.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SparkRenderer, SplatMesh, dyno } from './spark.module.js';
import { partName, Look, lookOptions } from './viewer/look.js';

const P = new URLSearchParams(location.search);
const RUN = `trained/${P.get('run') || 'a'}`;
const MODE = P.get('mode') || 'splat';
const state = (window.__bake = { ready: false });
const SIZE = 768;

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(SIZE, SIZE);
document.body.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color().setRGB(16 / 255, 18 / 255, 22 / 255, THREE.SRGBColorSpace);
const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
camera.position.set(0, -0.03, 0.62);
camera.lookAt(0, -0.06, 0);
const spark = new SparkRenderer({ renderer, minSortIntervalMs: 0 });
scene.add(spark);

const json = async (p) => (await fetch(p)).json();
const [info, ex, cams] = await Promise.all([json(`${RUN}/head.json`), json('expressions.json'), json('cameras.json')]);
const binding = new Float32Array(await (await fetch(`${RUN}/binding.bin`)).arrayBuffer());
const gltf = await new GLTFLoader().loadAsync('./head.glb');
const root = gltf.scene;
scene.add(root);
const byName = new Map();
root.traverse((n) => { if (n.isMesh) byName.set(partName(n), n); });
const parts = info.parts.map((p) => ({ ...p, mesh: byName.get(p.name) }));
root.updateMatrixWorld(true);

function setWeights(weights) {
  for (const { mesh } of parts) {
    if (!mesh.morphTargetDictionary) continue;
    ex.blendshapes.forEach((name, i) => {
      const k = mesh.morphTargetDictionary[name];
      if (k !== undefined) mesh.morphTargetInfluences[k] = weights[i];
    });
  }
}

const W = 4096;
const tex = (n, data) => {
  const t = new THREE.DataTexture(data, W, Math.ceil(n / W), THREE.RGBAFormat, THREE.FloatType);
  t.internalFormat = 'RGBA32F';
  t.needsUpdate = true;
  return t;
};
const pad = (n) => new Float32Array(W * Math.ceil(n / W) * 4);
const nV = info.vertices;
const restData = pad(nV), curData = pad(nV);
const v = new THREE.Vector3();
function vertices(out) {
  root.updateMatrixWorld(true);
  for (const { mesh, offset, count } of parts) {
    for (let i = 0; i < count; i++) {
      mesh.getVertexPosition(i, v);
      out.set([v.x, v.y, v.z], (offset + i) * 4);
    }
  }
}
setWeights(new Array(ex.blendshapes.length).fill(0));
vertices(restData);
curData.set(restData);
const restTex = tex(nV, restData), curTex = tex(nV, curData);
const triData = pad(info.triangles);
let t = 0;
for (const { mesh, offset } of parts) {
  const idx = mesh.geometry.index.array;
  for (let i = 0; i < idx.length; i += 3, t++) triData.set([idx[i] + offset, idx[i + 1] + offset, idx[i + 2] + offset], t * 4);
}
const triTex = tex(info.triangles, triData);
const bindData = pad(info.splats);
for (let i = 0; i < info.splats; i++) bindData.set(binding.subarray(i * 3, i * 3 + 3), i * 4);
const bindTex = tex(info.splats, bindData);

const RIDE = await (await fetch('./ride.glsl')).text();
const uBind = dyno.dynoSampler2D(bindTex), uTris = dyno.dynoSampler2D(triTex);
const uRest = dyno.dynoSampler2D(restTex), uCur = dyno.dynoSampler2D(curTex);
const ride = (gsplat) => new dyno.Dyno({
  inTypes: { gsplat: dyno.Gsplat, bind: 'sampler2D', tris: 'sampler2D', rest: 'sampler2D', cur: 'sampler2D' },
  outTypes: { gsplat: dyno.Gsplat },
  globals: () => [RIDE],
  inputs: { gsplat, bind: uBind, tris: uTris, rest: uRest, cur: uCur },
  statements: ({ inputs, outputs }) => [
    `${outputs.gsplat} = ${inputs.gsplat};`,
    `if (isGsplatActive(${outputs.gsplat}.flags)) rideTriangle(${inputs.bind}, ${inputs.tris}, ${inputs.rest}, ${inputs.cur}, ${outputs.gsplat}.index, ${outputs.gsplat}.center, ${outputs.gsplat}.quaternion);`,
  ],
}).outputs.gsplat;
const splats = new SplatMesh({ url: `${RUN}/${P.get('splats') || 'splats.ply'}` });
await splats.initialized;
splats.objectModifier = dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => ({ gsplat: ride(gsplat) }));
splats.updateGenerator();
scene.add(splats);  // vertices are in the scene's space (identity nodes)
if (MODE === 'splat') for (const { mesh } of parts) mesh.material.visible = false;
else splats.visible = false;
// The mesh head as the demo shows it today: look.js's high profile (GTAO, DoF, bloom, grain).
let look = null;
if (MODE === 'mesh') {
  look = new Look(renderer, scene, camera, lookOptions(new URLSearchParams('quality=high')));
  look.apply(root);
  look.setSize(SIZE, SIZE);
}
const draw = () => (look ? look.render(0) : renderer.render(scene, camera));

function pose(weights) {
  setWeights(weights);
  vertices(curData);
  curTex.needsUpdate = true;
  splats.updateVersion();
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(r));
async function exact() {
  await spark.update({ scene, camera });
  while (spark.sorting || spark.sortDirty) {
    await new Promise((r) => setTimeout(r, 1));
    if (spark.sortDirty && !spark.sorting) await spark.update({ scene, camera });
  }
  await nextFrame();
  draw();
  await nextFrame();
  await nextFrame();
}
state.cameras = cams;
// Camera i of cameras.json, with its own set's expression: the render it was made for is cams[i].file.
state.view = async (i) => {
  const c = cams[i];
  const m = new THREE.Matrix4().set(...c.c2w.flat());
  camera.matrixAutoUpdate = false;
  camera.matrix.copy(m);
  camera.fov = c.fov_y;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  if (look) look.focus(root);
  pose(ex[c.set][c.expression]);
  await exact();
  return c.file;
};
state.weights = async (w) => { pose(w); await exact(); };
state.splats = info.splats;
state.ready = true;

// How a face is lit and finished: lights, environment, tone mapping, shadows, skin, eyes, hair edges
// and the post-processing chain, in three quality profiles. Shared by the viewer and the editor's
// previews, so a face looks the same in both.
//
// It works on any face by name, the names the head recipe gives the parts: the mesh "head" is skin,
// "eyes" holds MakeHuman's eyeball and, outside it, a cornea shell, and "hair" is cut-out hair cards.
// A face without those names just gets the lighting.
//
// URL switches (the viewer passes its query string): quality=low|medium|high picks a profile, and
// each effect can be set on its own over it:
//   ibl, shadows, skin, cornea, ao, smaa, dof, grain, bloom   =1 or =0
//   hair=a2c|hash|cutout    tone=neutral|agx|aces|none    exposure=<number>

import * as THREE from 'three';
import { RoomEnvironment } from './vendor/environments/RoomEnvironment.js';
import { EffectComposer } from './vendor/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/postprocessing/RenderPass.js';
import { GTAOPass } from './vendor/postprocessing/GTAOPass.js';
import { SMAAPass } from './vendor/postprocessing/SMAAPass.js';
import { BokehPass } from './vendor/postprocessing/BokehPass.js';
import { UnrealBloomPass } from './vendor/postprocessing/UnrealBloomPass.js';
import { FilmPass } from './vendor/postprocessing/FilmPass.js';
import { OutputPass } from './vendor/postprocessing/OutputPass.js';

// low: anything that renders, a Raspberry Pi included. medium: the default; shadows and skin, no
// post. high: everything; for `record`, which renders offline and need not keep up.
export const PROFILES = {
  low: { ibl: true, shadows: false, skin: false, cornea: true, hair: 'a2c', ao: false, smaa: false, dof: false, grain: false, bloom: false },
  medium: { ibl: true, shadows: true, skin: true, cornea: true, hair: 'a2c', ao: false, smaa: false, dof: false, grain: false, bloom: false },
  high: { ibl: true, shadows: true, skin: true, cornea: true, hair: 'a2c', ao: true, smaa: false, dof: true, grain: true, bloom: true },
};
const SWITCHES = ['ibl', 'shadows', 'skin', 'cornea', 'ao', 'smaa', 'dof', 'grain', 'bloom'];
const TONES = { agx: THREE.AgXToneMapping, aces: THREE.ACESFilmicToneMapping, neutral: THREE.NeutralToneMapping, none: THREE.NoToneMapping };

// The options a page asked for: a profile, then any single switches over it.
export function lookOptions(params, fallback = 'medium') {
  const quality = PROFILES[params.get('quality')] ? params.get('quality') : fallback;
  // Khronos PBR Neutral: keeps the skin's and the clothes' own colours (AgX greys them, ACES pushes them).
  const o = { quality, ...PROFILES[quality], tone: 'neutral', exposure: 1.0 };
  for (const key of SWITCHES) {
    if (params.get(key) === '1') o[key] = true;
    if (params.get(key) === '0') o[key] = false;
  }
  if (['a2c', 'hash', 'cutout'].includes(params.get('hair'))) o.hair = params.get('hair');
  if (TONES[params.get('tone')] !== undefined) o.tone = params.get('tone');
  const exposure = Number(params.get('exposure'));
  if (params.has('exposure') && Number.isFinite(exposure) && exposure > 0) o.exposure = Math.min(4, exposure);
  return o;
}

// ---- skin: wrap lighting, the light past the terminator reddened as it is when it scatters under skin

const WRAP = 0.35;
function skinShader(shader) {
  const chunk = THREE.ShaderChunk.lights_physical_pars_fragment;
  const line = 'reflectedLight.directDiffuse += irradiance * diffuseBRDF * ( 1.0 - F );';
  if (!chunk.includes(line)) return;  // a three.js that wrote this differently: plain physical skin
  shader.fragmentShader = shader.fragmentShader.replace('#include <lights_physical_pars_fragment>', chunk.replace(line, `
    float wrapNL = saturate( ( dot( geometryNormal, directLight.direction ) + ${WRAP.toFixed(2)} ) / ${(1 + WRAP).toFixed(2)} );
    vec3 scatter = vec3( 0.55, 0.28, 0.20 );
    vec3 wrapped = directLight.color * ( wrapNL * scatter + dotNL * ( vec3( 1.0 ) - scatter ) );
    reflectedLight.directDiffuse += wrapped * diffuseBRDF * ( 1.0 - F );`));
}

// Pore-scale detail, made here: MakeHuman's skins ship a colour map only. Soft bumps on a tiling
// height field, as a normal map, and a roughness map that varies with them.
let detail = null;
function skinDetail() {
  if (detail) return detail;
  const n = 256;
  const height = new Float32Array(n * n);
  let seed = 7;
  const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let k = 0; k < 900; k += 1) {  // pores and bumps, wrapping at the edges so the tile repeats
    const cx = random() * n, cy = random() * n, r = 1.2 + random() * 2.2, depth = random() < 0.7 ? -1 : 0.5;
    for (let y = Math.floor(cy - 3 * r); y <= cy + 3 * r; y += 1) {
      for (let x = Math.floor(cx - 3 * r); x <= cx + 3 * r; x += 1) {
        const d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (r * r);
        height[((y + n) % n) * n + ((x + n) % n)] += depth * Math.exp(-d2);
      }
    }
  }
  const normal = new Uint8Array(n * n * 4), rough = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const h = (dx, dy) => height[((y + dy + n) % n) * n + ((x + dx + n) % n)];
      const nx = (h(-1, 0) - h(1, 0)) * 0.9, ny = (h(0, -1) - h(0, 1)) * 0.9, len = Math.hypot(nx, ny, 1);
      const i = (y * n + x) * 4;
      normal.set([(nx / len) * 127.5 + 127.5, (ny / len) * 127.5 + 127.5, (1 / len) * 127.5 + 127.5, 255], i);
      const r = Math.max(0, Math.min(255, 200 - height[y * n + x] * 60));  // pores rougher, bumps a touch shinier
      rough.set([r, r, r, 255], i);
    }
  }
  const tex = (data, colour) => {
    const t = new THREE.DataTexture(data, n, n);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(28, 28);
    t.colorSpace = colour;
    t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    t.userData.shared = true;  // one copy for every face and stage: a page disposing a face must leave it
    return t;
  };
  detail = { normal: tex(normal, THREE.NoColorSpace), rough: tex(rough, THREE.NoColorSpace) };
  return detail;
}

function skin(mesh, anisotropy) {
  const source = mesh.material;
  const m = new THREE.MeshPhysicalMaterial();
  THREE.MeshStandardMaterial.prototype.copy.call(m, source);  // the glTF's own maps and colours, then skin's
  m.defines = { STANDARD: '', PHYSICAL: '' };  // the standard copy resets these to STANDARD alone: no sheen
  const { normal, rough } = skinDetail();
  normal.anisotropy = rough.anisotropy = anisotropy;
  if (!m.normalMap) { m.normalMap = normal; m.normalScale.set(0.35, 0.35); }
  if (!m.roughnessMap) m.roughnessMap = rough;
  m.roughness = 0.62;
  m.metalness = 0;
  m.specularIntensity = 0.55;
  m.sheen = 0.15;  // a little velvet at grazing angles; more whitens the skin
  m.sheenRoughness = 0.6;
  m.sheenColor = new THREE.Color(1.0, 0.86, 0.8);
  m.onBeforeCompile = skinShader;
  m.customProgramCacheKey = () => 'argus-skin';
  mesh.material = m;
}

// ---- eyes: the cornea shell drawn again as a reflection only, for catch-lights that move with gaze

// MakeHuman's eye mesh is, per eye, an eyeball and a slightly larger shell (the cornea), whose texture
// is almost all transparent, so the cut-out material hides it. Find each eye's outer shell.
function shellTriangles(geometry) {
  const index = geometry.index;
  const pos = geometry.attributes.position;
  if (!index || !pos) return null;
  const parent = new Int32Array(pos.count).map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let t = 0; t < index.count; t += 3) {
    const a = find(index.getX(t)), b = find(index.getX(t + 1)), c = find(index.getX(t + 2));
    parent[b] = a; parent[find(c)] = a;
  }
  const parts = new Map();
  for (let i = 0; i < pos.count; i += 1) {
    const r = find(i);
    if (!parts.has(r)) parts.set(r, []);
    parts.get(r).push(i);
  }
  const v = new THREE.Vector3();
  const shells = new Map();  // side (-1 or 1) -> { root, radius }
  const pieces = new Map();  // side -> how many parts lie on it
  for (const [root, verts] of parts) {
    const c = new THREE.Vector3();
    for (const i of verts) c.add(v.fromBufferAttribute(pos, i));
    c.divideScalar(verts.length);
    const side = Math.sign(c.x) || 1;
    pieces.set(side, (pieces.get(side) || 0) + 1);
    if (verts.length < 64) continue;  // stray bits, not a sphere
    let radius = 0;
    for (const i of verts) radius += v.fromBufferAttribute(pos, i).distanceTo(c);
    radius /= verts.length;
    if (!shells.has(side) || shells.get(side).radius < radius) shells.set(side, { root, radius });
  }
  // Only a real pair of nested spheres: each side must have had an eyeball inside its shell, or the
  // "shell" found is an eyeball itself.
  if (shells.size !== 2 || [...shells.keys()].some((side) => pieces.get(side) < 2)) return null;
  const roots = new Set([...shells.values()].map((s) => s.root));
  const out = [];
  for (let t = 0; t < index.count; t += 3) if (roots.has(find(index.getX(t)))) out.push(index.getX(t), index.getX(t + 1), index.getX(t + 2));
  return out.length ? out : null;
}

function cornea(mesh) {
  const triangles = shellTriangles(mesh.geometry);
  if (!triangles) return;
  const g = new THREE.BufferGeometry();
  for (const [name, attribute] of Object.entries(mesh.geometry.attributes)) g.setAttribute(name, attribute);
  g.morphAttributes = mesh.geometry.morphAttributes;  // the same eyeLook* shapes: the shell turns with the eye
  g.morphTargetsRelative = mesh.geometry.morphTargetsRelative;
  g.setIndex(triangles);
  // Black, and added on top: only reflections show. The lights make the catch-lights; the room is kept
  // faint, or its whole reflection lies over the iris like a cataract.
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x000000, roughness: 0.04, metalness: 0, ior: 1.376,
    specularIntensity: 1, envMapIntensity: 0.15, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  });
  // A face with a head bone skins its eyes: the shell must ride the same skeleton, or it stays behind.
  const shell = mesh.isSkinnedMesh ? new THREE.SkinnedMesh(g, m) : new THREE.Mesh(g, m);
  if (mesh.isSkinnedMesh) { shell.bindMode = mesh.bindMode; shell.bind(mesh.skeleton, mesh.bindMatrix); }
  shell.name = 'cornea';
  shell.morphTargetInfluences = mesh.morphTargetInfluences;  // one array: weights set on the eyes reach it
  shell.morphTargetDictionary = mesh.morphTargetDictionary;
  shell.renderOrder = (mesh.renderOrder || 0) + 1;
  shell.position.copy(mesh.position);
  shell.quaternion.copy(mesh.quaternion);
  shell.scale.copy(mesh.scale);
  mesh.parent.add(shell);
  if (mesh.material.roughness !== undefined) mesh.material.roughness = Math.min(mesh.material.roughness, 0.3);  // wet whites
}

// ---- the look ------------------------------------------------------------------------------------

const WET = new Set(['head', 'eyes', 'teeth', 'tongue']);  // parts that keep the glTF's own roughness

export class Look {
  constructor(renderer, scene, camera, options = lookOptions(new URLSearchParams())) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.o = { ...options };
    this.composer = null;
    this.transparent = !scene.background;  // post-processing does not keep the alpha a transparent page needs
    renderer.toneMapping = TONES[this.o.tone];
    renderer.toneMappingExposure = this.o.exposure;
    renderer.shadowMap.enabled = this.o.shadows;
    renderer.shadowMap.type = THREE.PCFShadowMap;  // soft: r187's PCF takes five rotated taps within shadow.radius

    if (this.o.ibl) {  // image-based light from a lit room: soft, all round, and something for the eyes to reflect
      const pmrem = new THREE.PMREMGenerator(renderer);
      const room = new RoomEnvironment();
      scene.environment = pmrem.fromScene(room, 0.04).texture;
      scene.environmentIntensity = 0.3;  // fill and reflections; the lights do the modelling
      room.dispose();
      pmrem.dispose();
    }

    // Three-point rig in the head's own frame (the face looks down +Z, the eyes at the origin).
    this.rig = new THREE.Group();
    this.rig.name = 'look-rig';
    const key = new THREE.DirectionalLight(new THREE.Color(1.0, 0.93, 0.84), this.o.ibl ? 2.7 : 2.9);
    key.position.set(0.55, 0.45, 1.0);  // warm, high and to one side: it models the face
    const fill = new THREE.DirectionalLight(new THREE.Color(0.86, 0.91, 1.0), this.o.ibl ? 0.55 : 1.1);
    fill.position.set(-0.8, 0.05, 0.75);  // soft, from the other side, keeps the shadow side from going black
    const rimL = new THREE.DirectionalLight(new THREE.Color(0.75, 0.85, 1.0), 1.6);
    rimL.position.set(-0.55, 0.5, -0.9);  // cool, from behind: separates the head from a dark background
    const rimR = rimL.clone();
    rimR.position.set(0.6, 0.35, -0.9);
    rimR.intensity = 1.0;
    for (const light of [key, fill, rimL, rimR]) this.rig.add(light, light.target);
    if (!this.o.ibl) this.rig.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 0.9));
    if (this.o.shadows) {
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      key.shadow.radius = 3;
      key.shadow.bias = -0.0004;
      key.shadow.normalBias = 0.0015;
    }
    this.key = key;
    scene.add(this.rig);
  }

  // Dress a loaded face: shadows, skin, cornea, hair edges. Call once per model, then focus() once the
  // camera is placed.
  apply(root) {
    const anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const meshes = [];
    root.traverse((node) => { if (node.isMesh) meshes.push(node); });
    for (const mesh of meshes) {
      const m = mesh.material;
      if (this.o.shadows) {
        mesh.receiveShadow = true;
        mesh.castShadow = !m.transparent;  // blended brows and lashes would shadow as solid cards
      }
      if (mesh.name === 'head' && this.o.skin) skin(mesh, anisotropy);
      if (mesh.name === 'eyes' && this.o.cornea) cornea(mesh);
      // Hair and cloth are matte: exported smooth, they mirror the room and the rim lights like plastic.
      // Not skin, eyes or the wet mouth.
      if (!WET.has(mesh.name) && m.roughness !== undefined) m.roughness = Math.max(m.roughness, 0.6);
      // Hair cards only: other cut-outs (the eyes' cornea layer, the teeth) are not edges to soften, and
      // partial coverage would draw the eyes' all-but-transparent cornea as a skin-coloured film.
      if (mesh.name === 'hair' && m.alphaTest > 0 && !m.transparent) {
        if (this.o.hair === 'a2c') m.alphaToCoverage = true;  // soft edges from MSAA's samples
        if (this.o.hair === 'hash') { m.alphaHash = true; m.alphaTest = 0; }  // dithered, smoothed by grain or AA
        m.needsUpdate = true;
      }
    }
    if (this.o.shadows) this.fitShadow(root);
  }

  fitShadow(root) {  // the key light's shadow camera, just around the head: every texel spent on the face
    const box = new THREE.Box3().setFromObject(root);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const cam = this.key.shadow.camera;
    const r = sphere.radius * 1.05;
    cam.left = cam.bottom = -r;
    cam.right = cam.top = r;
    cam.near = 0.01;
    cam.far = r * 6;
    this.key.position.copy(sphere.center).addScaledVector(new THREE.Vector3(0.55, 0.45, 1.0).normalize(), r * 3);
    this.key.target.position.copy(sphere.center);
    cam.updateProjectionMatrix();
  }

  // The post chain, for profiles with any post effect. Not for a transparent page.
  setSize(width, height) {
    const wants = this.o.ao || this.o.smaa || this.o.dof || this.o.grain || this.o.bloom;
    if (!wants || this.transparent) { this.composer = null; return; }
    const ratio = this.renderer.getPixelRatio();
    if (!this.composer) {
      // Half float, and multisampled unless SMAA is on: alpha-to-coverage needs the samples.
      const target = new THREE.WebGLRenderTarget(width * ratio, height * ratio, { type: THREE.HalfFloatType, samples: this.o.smaa ? 0 : 4 });
      const c = new EffectComposer(this.renderer, target);
      c.addPass(new RenderPass(this.scene, this.camera));
      if (this.o.ao) {  // contact shadows: eyes, nostrils, lips, under the jaw. Radii in metres, a head's scale.
        const ao = new GTAOPass(this.scene, this.camera, width, height);
        ao.updateGtaoMaterial({ radius: 0.025, distanceExponent: 1.4, thickness: 0.02, scale: 1.0, samples: 16 });
        ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16 });
        ao.blendIntensity = 0.85;
        c.addPass(ao);
      }
      if (this.o.dof) {  // a little depth: the face sharp, the shoulders softening
        this.bokeh = new BokehPass(this.scene, this.camera, { focus: this.camera.position.length(), aperture: 0.0012, maxblur: 0.004 });
        c.addPass(this.bokeh);
      }
      if (this.o.bloom) c.addPass(new UnrealBloomPass(new THREE.Vector2(width, height), 0.12, 0.35, 0.92));  // the catch-lights only
      c.addPass(new OutputPass());  // tone mapping and sRGB, from the renderer's settings
      if (this.o.smaa) c.addPass(new SMAAPass());
      if (this.o.grain) c.addPass(new FilmPass(0.05, false));  // just enough to break up banding and flat skin
      this.composer = c;
    }
    this.composer.setPixelRatio(ratio);
    this.composer.setSize(width, height);
  }

  focus(root) {
    if (!this.bokeh) return;
    const centre = new THREE.Box3().setFromObject(root).getCenter(new THREE.Vector3());
    this.bokeh.uniforms.focus.value = this.camera.position.distanceTo(centre) - 0.06;  // the front of the face
  }

  render() {
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  }

  // What is on, for window.__mask and for anyone asking which look a recording used.
  get summary() {
    const o = this.o;
    const post = Boolean(this.composer);
    return { quality: o.quality, tone: o.tone, exposure: o.exposure, ibl: o.ibl, shadows: o.shadows, skin: o.skin,
      cornea: o.cornea, hair: o.hair, post, ao: o.ao && post, smaa: o.smaa && post, dof: o.dof && post,
      grain: o.grain && post, bloom: o.bloom && post };
  }
}

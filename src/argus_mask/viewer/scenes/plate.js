// Plates: a still image behind the face, brought to life by a few built-in effects (#55). The image comes
// from argus-darkroom, with a depth map darkroom derived from it; `argus-mask plate generate` writes the
// directory and docs/backgrounds.md says how. A built-in plate lives in scenes/<name>/ next to this module.
//
// background.json (models.py: PlateManifest) names the images and how much each effect moves them:
//   drift    the camera drifts slowly across the plate; with a depth map near things move more (parallax)
//   sway     leaves move in a breeze, where sway_mask is light (or by depth: near more than far)
//   dapple   light through leaves drifts over the scene
//   pulse    lights breathe, where light_mask is light
//   soft     how far out of focus the plate is (its mipmaps, so it costs nothing per pixel)
//   exposure the plate's brightness, so it sits behind the face
//
// Like every scene, the picture is a pure function of uTime and uRes (and the images): no clock, no
// randomness, so a recording of the same take comes out the same. Every motion is a sum of sines and smooth
// noise of time, so it never jumps. Output is linear light: the plate is decoded from sRGB on sampling.
//
// Plates of your own live in <assets>/backgrounds/<name>/ and are served by the server (/scene/<name>/).

const FILE = /^[a-z0-9][a-z0-9_-]{0,63}\.png$/;   // models.py: PLATE_FILE_PATTERN
const SHOWN = new Set(['personal', 'stock', 'synthetic']);   // a plate shows no one: never consented; curated is refused

const fragmentShader = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform vec2 uRes;
  uniform sampler2D uPlate;
  uniform sampler2D uDepth;
  uniform sampler2D uSway;
  uniform sampler2D uLight;
  uniform float uHasDepth;
  uniform float uAspect;     // the plate's width / height
  uniform float uSpeed;
  uniform float uDrift;
  uniform float uSwayAmount;
  uniform float uDapple;
  uniform float uPulse;
  uniform float uSoft;       // mipmap bias: 0 sharp, 3 well out of focus
  uniform float uExposure;
  varying vec2 vUv;

  const float TAU = 6.2831853;
  const float MARGIN = 0.05;   // of the plate kept off screen on each side, for the drift to move into

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
  }

  // The plate covers the screen, cropped to its shape, with MARGIN to spare.
  vec2 cover(vec2 uv) {
    float screen = uRes.x / max(uRes.y, 1.0);
    vec2 scale = screen > uAspect ? vec2(1.0, uAspect / screen) : vec2(screen / uAspect, 1.0);
    return (uv - 0.5) * scale * (1.0 - 2.0 * MARGIN) + 0.5;
  }

  void main() {
    float t = uTime * uSpeed;
    vec2 uv = cover(vUv);

    // Drift: a slow Lissajous path (47 s by 61 s). With depth, each point moves by its nearness, found by a
    // short fixed-point search so edges stay put; without, the whole plate pans a little.
    vec2 drift = vec2(sin(TAU * t / 47.0), sin(TAU * t / 61.0 + 1.3)) * MARGIN * 0.8 * uDrift;
    vec2 p = uv + drift * 0.5;
    float near = 0.5;
    if (uHasDepth > 0.5) {
      p = uv;
      for (int i = 0; i < 3; i++) { near = texture2D(uDepth, p).r; p = uv + drift * (near - 0.35); }
    }

    // Sway: a gusting breeze, where the mask allows, near leaves more than far ones.
    vec2 q = p * vec2(uAspect, 1.0);
    float weight = texture2D(uSway, p).r * (uHasDepth > 0.5 ? smoothstep(0.2, 0.85, near) : 1.0);
    vec2 wind = vec2(sin(q.y * 18.0 + t * 1.3 + 2.0 * noise(q * 3.0 + t * 0.15)),
                     cos(q.x * 15.0 + t * 1.1 + 2.0 * noise(q * 3.0 - t * 0.12)));
    float gust = 0.6 + 0.4 * sin(TAU * t / 9.0 + q.x * 2.0);
    p += wind * gust * 0.0025 * uSwayAmount * weight;

    vec3 colour = texture2D(uPlate, p, uSoft).rgb;

    // Dapple: soft patches of light drifting across what is already lit.
    float n = 0.6 * noise(q * 4.0 + vec2(t * 0.07, t * 0.04)) + 0.4 * noise(q * 9.0 - vec2(t * 0.05, -t * 0.08));
    float lum = dot(colour, vec3(0.2126, 0.7152, 0.0722));
    colour *= 1.0 + uDapple * 0.5 * (n - 0.5) * smoothstep(0.02, 0.3, lum);

    // Pulse: lights in the light mask breathe on a 5.3 s cycle.
    colour *= 1.0 + uPulse * 0.35 * texture2D(uLight, p).r * (0.5 + 0.5 * sin(TAU * t / 5.3));

    // Darker toward the edges, so the eye stays on the face.
    float r = length((vUv - 0.5) * vec2(uRes.x / max(uRes.y, 1.0), 1.0));
    colour *= uExposure * (1.0 - 0.35 * smoothstep(0.35, 0.95, r));
    gl_FragColor = vec4(colour, 1.0);
  }
`;

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// The plate named *name*, whose files are under *base* (a URL ending in /): its manifest, refused unless it
// says where it came from. A built-in plate's base is next to this module; one of your own is the server's.
export async function load(name, base) {
  const reply = await fetch(new URL('background.json', base));
  if (reply.status === 404) throw new Error(`Unknown scene "${name}": no such plate, or the server refused it (argus-mask plate check).`);
  if (!reply.ok) throw new Error(`The plate "${name}" could not be read (${reply.status}).`);
  const manifest = await reply.json();
  const provenance = manifest.provenance || {};
  if (manifest.kind !== 'plate' || !SHOWN.has(provenance.kind) || !String(provenance.source || '').trim()) {
    throw new Error(`The plate "${name}" does not say where it came from: it is not shown.`);
  }
  for (const role of ['plate', 'depth', 'sway_mask', 'light_mask']) {
    if (manifest[role] && !FILE.test(manifest[role].file)) throw new Error(`The plate "${name}" names a bad file.`);
  }
  if (!manifest.plate) throw new Error(`The plate "${name}" has no image.`);
  return {
    meta: { title: manifest.title, tint: manifest.tint || '#ffffff' },
    material: (THREE, options) => material(THREE, base, manifest, options),
  };
}

async function material(THREE, base, manifest, options = {}) {
  const loader = new THREE.TextureLoader();
  const image = async (entry, colour) => {
    const texture = await loader.loadAsync(new URL(entry.file, base).href);
    texture.colorSpace = colour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = colour;   // the plate is softened through its mipmaps; masks are sampled as they are
    texture.minFilter = colour ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
    return texture;
  };
  const flat = (value) => {   // 1x1 stand-in for a mask the plate does not have
    const texture = new THREE.DataTexture(new Uint8Array([value, value, value, 255]), 1, 1);
    texture.needsUpdate = true;
    return texture;
  };
  const [plate, depth, sway, light] = await Promise.all([
    image(manifest.plate, true),
    manifest.depth ? image(manifest.depth, false) : flat(128),
    manifest.sway_mask ? image(manifest.sway_mask, false) : flat(255),
    manifest.light_mask ? image(manifest.light_mask, false) : flat(0),
  ]);
  const e = manifest.effects || {};
  const unit = (v, fallback) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : fallback);
  const speed = Number.isFinite(options.speed) ? Math.min(4, Math.max(0.1, options.speed)) : 1;
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uRes: { value: new THREE.Vector2(1, 1) },
      uPlate: { value: plate },
      uDepth: { value: depth },
      uSway: { value: sway },
      uLight: { value: light },
      uHasDepth: { value: manifest.depth ? 1 : 0 },
      uAspect: { value: manifest.plate.width / manifest.plate.height },
      uSpeed: { value: speed },
      uDrift: { value: unit(e.drift, 0.5) },
      uSwayAmount: { value: unit(e.sway, 0) },
      uDapple: { value: unit(e.dapple, 0) },
      uPulse: { value: unit(e.pulse, 0) },
      uSoft: { value: 3 * unit(e.soft, 0.4) },
      uExposure: { value: Number.isFinite(e.exposure) ? Math.min(2, Math.max(0.1, e.exposure)) : 0.8 },
    },
    depthTest: false,
    depthWrite: false,
  });
}

// ftl: stars streaking past a window at faster-than-light speed. Procedural, so no asset to license: our own
// code, under the package's licence.
//
// A scene module exports `meta` (title, and the light it suggests) and `material(THREE, options)`, which
// returns a ShaderMaterial for a full-screen quad. The host (backdrop.js) sets two uniforms before each draw:
// `uTime`, seconds on the scene's clock (the take's own timeline in a recording), and `uRes`, the size it
// draws at. The picture must be a pure function of those two: no clock, no randomness, so a recording of the
// same take comes out the same. Output linear RGB: the viewer's tone mapping and colour space apply after.

export const meta = {
  title: 'FTL starstreaks',
  tint: '#9fc3ff',   // cool blue from the window, for the rim light
};

const fragmentShader = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform vec2 uRes;
  uniform float uSpeed;
  uniform float uDensity;
  uniform vec3 uTint;
  varying vec2 vUv;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

  void main() {
    // Centred, square units: y from -0.5 to 0.5. The streaks leave a point a little above the middle,
    // behind the head, so they seem to come from beyond it.
    vec2 p = (vUv - 0.5) * vec2(uRes.x / max(uRes.y, 1.0), 1.0) - vec2(0.0, 0.06);
    float r = length(p);
    float turn = atan(p.y, p.x) / 6.2831853 + 0.5;   // 0..1 around the point

    // Deep space: a dim blue that lifts a little away from the centre.
    vec3 colour = vec3(0.004, 0.006, 0.014) + vec3(0.006, 0.010, 0.024) * smoothstep(0.0, 0.9, r);

    for (int layer = 0; layer < 3; layer++) {
      float fl = float(layer);
      float bins = floor((70.0 + 60.0 * fl) * uDensity) + 1.0;
      float sector = floor(turn * bins);
      float seed = hash(vec2(sector, 17.0 + fl * 13.0));
      if (seed > 0.25 + 0.6 * uDensity) continue;   // not every sector has a star
      float offset = hash(vec2(sector, 3.0 + fl * 7.0));
      float centre = (sector + 0.2 + 0.6 * offset) / bins;
      float across = abs(turn - centre) * 6.2831853 * r;   // distance from the streak's line

      // Each star comes toward the window from far away and leaves the frame: its depth z cycles 0..1.
      float z = fract(seed * 7.13 + uTime * uSpeed * (0.10 + 0.05 * fl));
      float head = 0.03 + 0.9 * z * z;
      float tail = head - (0.01 + 0.35 * z * z) * uSpeed;
      float along = smoothstep(tail, head, r) * (1.0 - smoothstep(head, head + 0.004, r));
      float width = 0.0008 + 0.0025 * z;
      float line = exp(-pow(across / width, 2.0));
      float light = along * line * (0.15 + 1.0 * z) * smoothstep(0.02, 0.12, r);
      colour += light * mix(uTint, vec3(1.0), 0.35 + 0.3 * offset);
    }

    // Darker toward the edges, so the eye stays on the face.
    colour *= 1.0 - 0.45 * smoothstep(0.35, 0.95, r);
    gl_FragColor = vec4(colour, 1.0);
  }
`;

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// options: speed (default 1), density (0.2-1, default 0.6), tint (a CSS colour, default meta.tint).
export function material(THREE, options = {}) {
  const tint = new THREE.Color();
  try { tint.set(options.tint || meta.tint); } catch { tint.set(meta.tint); }
  const clamp = (v, lo, hi, fallback) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback);
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uRes: { value: new THREE.Vector2(1, 1) },
      uSpeed: { value: clamp(options.speed, 0.1, 4, 1) },
      uDensity: { value: clamp(options.density, 0.2, 1, 0.6) },
      uTint: { value: new THREE.Vector3(tint.r, tint.g, tint.b) },   // THREE.Color is linear already
    },
    depthTest: false,
    depthWrite: false,
  });
}

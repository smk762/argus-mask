# Backgrounds

A screen can show a backdrop behind the face instead of a plain colour: `?scene=ftl` on the viewer, or
`argus-mask record --scene ftl`. Backdrops are cosmetic and belong to the viewer, like idle life at rest. They
are never in a take, so the engine's determinism is untouched.

Built in today:

| Scene | What it is | Parameters |
|---|---|---|
| `ftl` | Stars streaking past a window at faster-than-light speed, from a point behind the head. Procedural. | `scene.speed` (0.1 to 4, default 1), `scene.density` (0.2 to 1, default 0.6), `scene.tint` (a CSS colour) |

Where a backdrop does not apply: `display=hld` keeps its subject on black, `bg=transparent` shows what is behind
the page, and a splat face draws its own background, so it refuses a scene rather than ignore it.

## How a backdrop is drawn

`viewer/backdrop.js` renders the scene into a render target and the viewer uses that as the three.js scene's
background, so the look's post-processing (depth of field on `high`) softens it like any far background. The
size and rate are capped by the look's profile: `low` draws at most 480 px on the long side and 30 times a second,
`medium` 960 px at up to 60, and `high` (and `record`) full size every frame. A backdrop also leans the rim
lights toward its own light (`meta.tint`), so the head's edges seem lit from the scene. A hidden tab draws
nothing.

**Time.** Live, a backdrop runs on the page's clock. In a recording, its clock is the posed frame's `t_ms`,
so two recordings of the same take show the same backdrop. The `high` profile's film grain is not yet
reproducible, so compare two recordings with `grain=0`.

## Adding a procedural scene

A scene is one module under `src/argus_mask/viewer/scenes/<name>.js`, listed in `SCENES` in `backdrop.js` (the
viewer only imports what that table names, so `?scene=` never names a file) and in `SCENES` in `record.py` (a
test holds the two together). Copy `scenes/ftl.js` and change the shader. A module exports:

- `meta`: `{ title, tint }`. `tint` is a CSS colour, the light the scene suggests for the rim lights.
- `material(THREE, options)`: a `THREE.ShaderMaterial` for a full-screen quad (the vertex shader passes
  `position.xy` straight to clip space). `options` holds the `scene.*` URL parameters, already read: clamp
  them and fall back to defaults.

The host sets two uniforms before each draw: `uTime`, seconds on the scene's clock, and `uRes`, the size it
draws at. The rules:

- **A pure function of `uTime` and `uRes`.** No `Date`, no `performance.now()`, no `Math.random()` in the
  shader's inputs. Use a hash of position and index for randomness, so the same time gives the same picture.
- **Time-continuous, or periodic.** Either motion that never jumps (a star that leaves comes back as another),
  or a loop whose period divides into every animated parameter.
- **Linear light out.** Write linear RGB; the viewer's tone mapping and colour space apply after. The render
  target is half float, so dark gradients do not band.
- **Soft and dark.** The face is the subject: keep the middle quiet and the edges darker, and nothing bright
  behind the eyes.
- **Cheap.** It must not drop the viewer's frame rate on `low`. Check the HUD's fps at `?quality=low` with and
  without the scene.
- **Ours, or CC0.** A procedural scene is our code. No text, logos, or real places.

Then check it by eye: run the server, open `/?scene=<name>` on each `quality`, and `argus-mask push` a WAV.

Not built yet (#55): video loops and still plates (from argus-darkroom) served from the assets directory with
provenance, a face's default backdrop chosen on Saved faces, and backdrops on the editor's previews.

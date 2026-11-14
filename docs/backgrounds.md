# Backgrounds

A screen can show a backdrop behind the face instead of a plain colour: `?scene=ftl` on the viewer, or
`argus-mask record --scene ftl`. A backdrop is either a procedural scene (a shader of our own) or a plate (a
still image from argus-darkroom, brought to life by built-in effects). Backdrops are cosmetic and belong to the viewer, like idle life at rest. They
are never in a take, so the engine's determinism is untouched.

Built in today:

| Scene | What it is | Parameters |
|---|---|---|
| `ftl` | Stars streaking past a window at faster-than-light speed, from a point behind the head. Procedural. | `scene.speed` (0.1 to 4, default 1), `scene.density` (0.2 to 1, default 0.6), `scene.tint` (a CSS colour) |
| `garden` | An overgrown glass conservatory: the camera drifts, fronds sway, light dapples. A plate from argus-darkroom (FLUX.1 schnell, Apache-2.0). | `scene.speed`, `scene.tint` |

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
so two recordings of the same take show the same backdrop, film grain included (a test records each scene
twice and compares the frames).

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

## Adding a plate

A plate is a still picture with a little motion added by the viewer: the camera drifts across it (with
parallax, given a depth map), leaves sway, light dapples, lights pulse. It needs no shader of your own: the
picture comes from argus-darkroom, which checks the prompt and the image and keeps a record of both, and
`viewer/scenes/plate.js` animates it.

**1. Generate it.** With darkroom running (`--darkroom`, or `ARGUS_MASK_DARKROOM_URL`, default
`http://localhost:8106`):

```bash
argus-mask plate generate src/argus_mask/viewer/scenes/<name> \
  --title "Conservatory garden" \
  --prompt "inside an overgrown glass conservatory, lush ferns and palm fronds in the foreground at the edges, \
soft diffuse daylight through the glass roof, quiet calm open space in the middle, shallow depth of field" \
  --tint "#f3e6c8" --drift 0.6 --sway 0.5 --dapple 0.4 --soft 0.5 --exposure 0.65
```

The directory's name is the plate's (`?scene=<name>`), and an existing plate is never written over. Mask
asks darkroom for the picture under its SFW `default` policy, with no prompt expansion, and adds "no people,
no text, no logos" to the prompt (`--as-is` leaves it off). It then asks for darkroom's depth map of the
same record (`POST /depth/{id}`: Depth Anything V2 Small, Apache-2.0), checks every file against darkroom's
sha256, and writes:

- `plate.png`: darkroom's image, byte for byte, so anyone can match it to darkroom's record.
- `depth.png`: 8-bit greyscale, near is light. `--no-depth` leaves it out; `argus-mask plate depth <dir>`
  adds it later.
- `background.json` (`PlateManifest` in `models.py`): the title, the light it suggests (`tint`), the
  effects, provenance (`synthetic`, the checkpoint's licence, and the model, seed and prompt as `generator`),
  and the darkroom record id with the render settings.

The default checkpoint is FLUX.1 schnell, whose Apache-2.0 licence lets its output ship. For any other
checkpoint, `--license` states its output licence: check the model's terms first, since this is your claim.

**2. Tune the effects** in `background.json` (each 0 to 1), and check by eye as for a procedural scene:

| Effect | What it does | Notes |
|---|---|---|
| `drift` | The camera drifts slowly across the plate, on a 47 s by 61 s path. With a depth map, near things move more. | The plate is cropped 5% on each side so the drift never shows an edge. |
| `sway` | A gusting breeze moves the plate locally. | Where `sway_mask` is light; without a mask, by depth (near more than far). Without either, everywhere: then keep it low. |
| `dapple` | Soft patches of light drift over what is already lit. | |
| `pulse` | Lights breathe on a 5.3 s cycle. | Only where `light_mask` is light. |
| `soft` | Out of focus, through the plate's mipmaps. | Default 0.4. The face should stand clear of it. |
| `exposure` | Brightness, before the vignette. | Default 0.8. Below 1 sets it back behind the face. |

Masks are optional greyscale PNGs the size of the plate, painted by hand (light means on). Add them to
`background.json` as `sway_mask` and `light_mask` (`file`, `sha256`, `width`, `height`).

**3. Check it and list it.** `argus-mask plate check <dir>` checks the manifest, each file's sha256 and size,
and provenance. Then add the name to `SCENES` in `backdrop.js` (`<name>: plate('<name>')`) and in
`record.py`.

**Rules.**
- **Cleared only in the repo.** A shipped plate is `synthetic` from a model whose licence allows it, and
  names its darkroom record (a test checks every plate under `scenes/`).
- **No one in it.** No people, real places, brands, artists' names or text in the prompt; the face is the
  only person on screen. A plate cannot be `consented`, and `curated` is refused.
- **Small.** At most 4 MB per image and 6 MB per plate. A 1344x768 plate is about 2 MB.
- **Soft and dark in the middle**, as for any scene: the face is the subject.

## Plates of your own

A plate need not ship with mask. Generate it into the server's assets instead:

```bash
argus-mask plate generate "$ARGUS_MASK_ASSETS_DIR/backgrounds/rain" --title "Rain on a window" --prompt "…"
```

The server serves every plate in `<assets>/backgrounds/<name>/` that passes `plate check`, and refuses the
rest (no provenance, a changed or linked file, a size over the limits) as it refuses a face. Any screen can
then show it as `?scene=rain`, and `argus-mask record --scene rain` records it. `GET /scenes` lists what a
screen can show: the built-in scenes first, then the plates. A plate cannot take a built-in scene's name,
and `backgrounds` is not a face id. A plate of your own may be `personal` (the user's own screens only); only
a cleared plate may ship, be demoed or be published.

## A face's own backdrop

A face can have a backdrop of its own, shown on every screen that does not ask for another. Choose it in the
editor's Saved faces: **Review** a face, then **Backdrop**. The stage previews it behind the face, and for an
exported face the choice is kept in `<assets>/scenes.json` (`{"faces": {"ginger-3": "garden"}}`), which the
server reads like `expressions.json`. It follows the face when the face is renamed and goes when the face is
deleted.

- A screen's own `?scene=` wins; `?scene=none` shows a plain background whatever the face's choice.
- `GET /face/{id}/scene` says which backdrop a face has (`{"scene": null}` for none).
- `argus-mask record` without `--scene` records the face's own backdrop, as a screen would show it;
  `--scene none` leaves it out.
- A choice whose plate has gone (removed, or refused by `plate check`) is ignored: the face shows a plain
  background rather than failing to load.
- A splat face draws its own background, so it ignores the choice.

Not built yet (#55): video loops, and a third scene.

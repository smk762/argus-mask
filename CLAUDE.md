# CLAUDE.md — argus-mask

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

A talking face for the Argus suite: audio in, facial animation frames out, rendered in a browser on any screen on the LAN

Shared code — the target taxonomy, wire-schema versioning, and the local/remote model contract — lives in **argus-cortex**; import it rather than re-implementing.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Layout

`src/argus_mask/` (the engine is pydantic, numpy, soxr and cortex; the CLI and the server are extras, imported lazily):

- `arkit.py`: the 52 ARKit blendshape names, in the order every frame's weights use.
- `models.py`: the wire contract (`StreamHeader`, `Frame`, `StreamEnd`, `SourceHello`, `FaceManifest`, `Health`) and `render_wire_schema()`. **This is the API surface**: change it, then run `make schema` and commit the result. `ModelFile.format` is `gltf` (a `.glb`, the default) or `lam-splat` (a LAM Gaussian-splat head's `.zip`); file suffix and format must agree.
- `audio.py`: `PcmDecoder` (bytes to mono float32, across chunk boundaries), `Resampler` (streaming, to 16 kHz), `read_wav`.
- `drivers/`: the seam. `Driver` / `DriverSession` protocols, `energy.py` (CPU, deterministic), `a2f.py` (WebSocket client of the Audio2Face server; its protocol is in the module docstring).
- `prosody.py`: speech cues per frame from the 16 kHz stream (loudness, pitch; stresses, phrase starts and ends) and the nods, settles and brow raises they add. Causal: a frame's cue depends only on audio up to that frame's end.
- `idle.py`: blinks (timed to pauses), saccades and glances, micro-motion and the head (sway, cues, a share of each glance); a function of the seed, the frame number and the cue.
- `expressions.py`: rest sets, the named baseline expressions a face holds at rest (built in, plus `expressions.json` in the assets root, written by the editor and read by the server like `aliases.json`), and each face's choice. Served as `GET /face/{id}/rest`.
- `performance.py`: `Performer` joins resampler, driver, prosody and idle layer and stamps frames; `perform()` runs a whole clip.
- `speech.py`: what `say` does to text and audio by default (`--as-is` turns it off): numbers as words (`speakable`), short sentences joined (`join_short`), fades and levelling (`tidy_audio`, `Edges` for streams). Pure.
- `voice.py`: checks a reference clip for a cloned voice (`analyse`) and a set of them (`assess`), each check with a verdict and advice. Pure numpy.
- `take.py`: take files (`.ndjson`) and `line()`, the one place a message becomes JSON.
- `faces.py`: reads a `.glb` with the standard library, checks it against the contract, and `FaceStore` decides which faces may be served. `inspect_splat` checks a LAM splat archive in memory (zipfile, never extracted): one folder, known members only, counts/sizes/ratio capped, no path tricks or links, `skin.glb`'s morph targets ARKit only; the store caches the verdict per file version. `Aliases` keeps stable names for faces (`aliases.json` in the assets root, written by the editor, read by the server; sources and viewers meet on the name they use).
- `config.py`: `Settings.from_env()` (`ARGUS_MASK_*`).
- `client.py`: `push` and `say`, clients of the source socket.
- `server/app.py`: `create_app()`. Routes, the two sockets, the `Stage` per face. `perform()` is the one source path: the source socket, `POST /push` (a WAV or raw PCM, paced at the speed of speech) and `POST /speak` (text through `ARGUS_MASK_TTS_URL`) all call it.
- `viewer/`: the static viewer (`index.html`, `viewer.js`), `splat.js` (everything splat-specific: `SplatHead`, `readSkin`, `jawCap`, `splatFraming`; viewer.js loads it, by dynamic import, only for a `lam-splat` face, and the editor's `stage.js` reuses it), `look.js` (lights, materials, post-processing and the `low`/`medium`/`high` profiles; the editor's previews import it too), `motion.js` (playback between frames, the lip budget measured from each face's mesh, the rest set, and idle life at rest; shared with the editor too), `backdrop.js` and `scenes/` (`?scene=`: a procedural backdrop drawn into a capped render target used as the scene's background; its clock is the posed frame's `t_ms` in `record`, never the wall clock; `docs/backgrounds.md` has the module interface) and vendored three.js r187 with the example modules it needs (`environments/`, `postprocessing/`, `shaders/`, `math/`, `controls/OrbitControls.js`, copied unchanged from the npm release), Myned AI's `gsplat-flame-avatar-renderer.esm.min.js` 1.4.0 and `jszip.min.js` 3.10.1 (unchanged, with their licences), and `jszip.js`, our ES-module shim over JSZip's UMD build. The import maps (viewer and editor pages) map `three`, `three/examples/jsm/` and `jszip` for the renderer. No build step.
- `editor/`: the face editor, a separate optional process (`argus-mask editor`). `preset.py` (the preset model), `library.py` (installed MPFB assets and their licences), `builder.py` (runs Blender into `assets/.preview/<id>`; export copies a preview to `assets/<id>`), `app.py`, `page/` (`index.html` the editor, `faces.html` saved faces, `stage.js` the shared 3D preview, which loops `talk.json`, 12 s of Audio2Face weights from a generated voice, so a face is judged speaking). Rename and delete touch only faces with a `preset.json`; delete moves to `assets/.trash/`. Builds are also cached in `assets/.preview/.cache/<key>` (key: the preset without id/name, plus the recipe and Blender), least recently used out first. `modifiers.json` is the allow-list of shape sliders, generated by `scripts/make_modifiers.py` and read by both `preset.py` and `build_head.py`. `hair_colour.py` is the hair colour model (melanin to RGB, and the recolour that keeps strand variation): pure numpy with no package imports, because `build_head.py` loads it by path; the editor's swatch in `editor.js` mirrors its constants. `voices.py` is the Voices page's state (clips staged by hash under `voices/.staged`, a `voice.json` record with provenance per voice, synthetic candidates as jobs); the page is `page/voices.html`. `broadcast.py` runs the Broadcast page's `say` and `push` jobs with the CLI's own client (the text is never logged). `build_head.py` is the Blender script: it imports `bpy` and is never imported, only handed to Blender by path. `packs.py` is the Asset packs page's state (`page/packs.html`): MakeHuman community packs from `asset_catalogue.json` (generated by `scripts/make_asset_catalogue.py`, which scrapes the community's static pack pages, since the old JSON index is gone; a refresh is a reviewed commit), installed by catalogue id only, from the catalogued URL on `HOSTS`, checked against the pinned sha256, fenced under MPFB's data dir and never overwriting an installed file; the pack's `packs/<id>.json` goes in last, so the library offers nothing half-installed.
- `cli.py`: Typer app; entry point `argus-mask`.

`a2f/` is the Audio2Face server: a separate Docker image, not part of the Python package, not linted or tested by CI. `server.py` (Python, the network) supervises `worker/main.cpp` (C++, the SDK). Its contract is the protocol in `drivers/a2f.py`'s docstring; `a2f/check.py` checks a running container against it. Read `a2f/README.md` before building: the build needs about 25 GB of free Docker disk.

`scripts/` holds the stock head presets and lock files (`stock_heads/`) and `fetch_stock_head.py`. The recipe itself is `editor/build_head.py`.

`tests/helpers.py` builds every fixture in code: a one-triangle `.glb`, tone WAVs, a fake Audio2Face server, a fake TTS service, and `live_server()` for clients that need a real port.

## Commands

```bash
make install   # uv venv + editable install with the "dev,server,cli" extras
make test      # pytest --tb=short -q
make lint      # ruff check + format --check (pinned ruff 0.15.16)
make format    # ruff format + --fix
make schema    # regenerate schema/mask-wire.schema.json after changing models.py
```

Run one test: `uv run --no-sync pytest tests/test_x.py::test_name -q`.

## Conventions & gotchas

- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `src/argus_mask/_version.py` is generated (gitignored). Tag `vX.Y.Z` to release.
- **`src/` layout, `py.typed`, Pydantic v2, `structlog`.** Ruff line-length 120; keep it clean (`make lint`) before committing.
- **Any endpoint that does something is a boundary.** There's no auth by default — treat reaching the port as host access. Gate state-changing routes behind the shared read-only / origin guards, and fence any caller-supplied path under a configured root.
- Wire schemas are checked in and verified in CI — regenerate and commit when the types change, don't let the committed copy drift.
- **Determinism is a feature.** The same audio, driver and seed must give a byte-identical take (`test_a_clip_gives_the_same_take_every_time`). Nothing in the engine may read the clock or an unseeded RNG. `drive` and `push` feed the same block size (`performance.BLOCK_MS`) so offline and live agree.
- **Frame boundaries use integer arithmetic** (`frame * 16000 // fps`). Floating-point frame lengths drift at rates that do not divide the sample rate.
- **Mask never falls back from `a2f` to `energy`.** A down driver refuses the source with the reason. A silent downgrade would look like a bad model.
- **The server's own process uses no GPU**, so it does not take cortex's GPU lease. The Audio2Face server is a tenant outside the lease, like ComfyUI.
- **WebSockets are outside `WriteGuard` and CORS.** Both sockets call the same `cross_site_refuse` predicate by hand (`cross_site()` in `server/app.py`). A new socket route must do the same.
- **A viewer may speak on its socket, once: its state.** `ViewerStatus` (`audio`, `display`, `kiosk`), at most 1 KB, anything else ignored; the server keeps the latest per viewer as `Health.screens`. Never media, never a command. A socket client that never reports (`record`) is not a screen.
- **Audio is held until frames cover it** (`held` / `release()` in the source handler). Drivers answer late; sending audio on arrival would put the sound ahead of the mouth by the driver's lag.
- **Free the face before telling the source it is done** (`stage.claimed = False`, then send `done`), or a source that reconnects at once is told the face is taken.
- **The viewer handler uses an anyio task group, not bare asyncio tasks.** Bare tasks made the test client raise `CancelledError` on exit about one run in three.
- **The editor only replaces faces with a `preset.json`, and only offers assets a pack licenses.** A preset's asset references are looked up in the installed library before anything reaches Blender; never pass a request's path to Blender's command line.
- **Faces carry provenance or are not served.** Two tiers (README § Personal and cleared): `personal` is anything the user may lawfully use privately (a cartoon, a character LoRA), served but never `cleared`; `stock`, `synthetic` and `consented` (subject plus release sha256) are cleared. Voices mirror this (`personal` source in `voices.py`). Don't gate personal use on copyright or likeness, but the floor is the law: what argus-darkroom refuses is not a face here either. Keep `synthetic` honest: a face from a character LoRA is `personal`, not synthetic. `curated` is refused until spec.md's 0.3 work defines it. Only cleared material goes in the repo, tests, demos or anything commercial.
- **Splat heads (`viewer/splat.js`) have traps.** (1) LAM's `skin.glb` keeps a 0.01 scale on its node while its vertices are metres: `lipBudget` multiplies by world scale, so `readSkin` measures with every node's scale reset to 1, or the budget is null. (2) The renderer's skeleton is a full body rig (`neckLower`, `neckUpper`, `head`), not FLAME's `neck`: `getNeckPose({neck})` moves nothing. `SplatHead.pose()` (the `getNeckPose` callback) restores the rest pose from the file and turns those bones itself, about world axes, then calls `boneRoot.updateMatrixWorld`, because the renderer reads the skeleton before updating world matrices (a pose would show a frame late). (3) LAM's clips must not move anything: no `getChatState`, the mixer's actions stopped. (4) The renderer's rAF loop is cancelled after every `render()`; the page drives it (`step()` live, `exact()` in record mode). (5) `exact()` waits for any depth sort in flight (a worker), runs one for this pose and camera, waits for it, then draws in a rAF: only then may `record` screenshot. (6) The renderer's ResizeObserver is disconnected (`fit()` sizes the canvas before each draw): its callback runs after rAF callbacks, and resizing clears the canvas, so a frame drawn that tick was presented blank. (7) `create()` resolves after the first section of a progressive load; wait for `!isLoadingOrUnloading() && splatMesh.finalBuild` or the first recorded frames race the final build. (8) Record mode sets the reveal to Instant: the fade-in reads the clock. (9) The renderer wants a URL ending `.zip` (it parses a name from it): hence `/face/{id}/model/{file}`. (10) It builds no OrbitControls (its `setupControls` is commented out); if a later version does, never let them `listenToKeyEvents(window)` (W A S D are lost to text boxes). It needs no COOP/COEP: it rendered on a non-secure, non-isolated origin in headless Chromium.
- **The viewer has no automated test** beyond `node --check`. After touching `viewer/`, run the server and `argus-mask push` a WAV, and watch a browser. `window.__mask` exposes its state (`.timeline`, `.weights` and `.head` as drawn), `window.__mask.look` the profile, `.lips` the measured lip budget, `.rest` the rest set. Lip shaping and idle life live in the viewer, never in takes: the engine's determinism is untouched. The look recognises parts by name (`head`, `eyes`, `hair`, `wetline`): a recipe change that renames them silently loses skin, catch-lights or soft hair.
- Template-managed files (`Makefile`, `Dockerfile`, workflows) have local edits; `copier update` will ask about them.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never open a PR against `main` or push work to it directly: `main` must never be ahead of `dev`. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected and needs a PR, merge the release PR, then immediately merge `main` back into `dev` so the merge commit is not left on `main` alone.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow. Release publishes to PyPI (OIDC trusted publishing) + GHCR on `v*` tags. Run `copier update` to pull template (CI/release/tooling) changes.

# argus-mask — build doc

A talking face for the Argus suite. Audio goes in; a stream of facial animation frames comes out; a browser page renders a rigged head from those frames, in sync with the audio, on any screen on the LAN. The look is a stylised photo: a believable human head, not a cartoon and not a claim of photorealism. The face is an asset the user picks, so other styles work too.

Mask sits off the main line, like argus-darkroom and argus-atlas. It uses what the line produces (later: a curated face set) and feeds nothing back. This doc is the build plan; the repo does not exist yet.

```
audio source ──▶ argus-mask :8109 ──▶ viewer (browser, any LAN device)
(file, agent,     drive + sync         flat panel │ HLD │ Looking Glass Go (0.2)
 TTS output)          │
                      ▼
               Audio2Face service (external, by URL)
```

## Decisions already made

- **Display baseline is an ordinary 2D panel.** A Looking Glass HLD takes plain 2D video, so it is the same single view with different framing. The Looking Glass Go (multi-view) is a later tier, not a requirement.
- **The renderer is a browser page.** Mask sends frames, not pixels. This is what makes LAN serving cheap: a frame is about 60 numbers.
- **Faces are stock or synthetic for now.** Faces built from argus-curator output come later (see [Faces from curator](#faces-from-curator-03)).
- **The look leans stylised photo**, and the face asset is swappable, so a user who wants another style brings another head.

## What already exists

Do not rebuild these.

- **NVIDIA Audio2Face-3D.** The models (regression v2.3, diffusion v3.0) are under the NVIDIA Open Model License; the SDK is MIT. Audio is resampled to 16 kHz. The SDK is C++/CUDA on TensorRT with no Python bindings, and recommends 4 GB+ of VRAM. Mask reaches it as an external service by URL, the way darkroom reaches ComfyUI. Mask never links TensorRT.
- **Miso TTS** (a separate TTS server, `:8034` where this was developed), optional. `POST /synthesise/stream` takes text and streams raw 24 kHz mono float32 PCM, with the format in `X-Sample-Rate` / `X-Audio-Format` / `X-Channels` headers. It is one way to get speech to push into mask, never a dependency: only the `say` command talks to it.
- **argus-cortex** (`>=0.4.6,<0.5`):
  - `wire.make_versioned_base` and `render_schema` for the wire schema.
  - `server.WriteGuard` / `cross_site_refuse` / `env_flag`.
  - `gpu.free_vram_mb` for the `/health` report. Mask does not take the GPU lease (see [GPU](#gpu)).
- **three.js** renders glTF morph targets in the browser. **`@lookingglass/webxr`** (with Looking Glass Bridge on the display host) adds the Go in 0.2.
- **MPFB** (the MakeHuman plugin for Blender) builds the stock head. Its core assets (base mesh, targets, skins) are CC0, and since 2.0.15 its `faceunits01` pack adds the ARKit face shapes under their standard names (`browDownLeft`, `eyeBlinkLeft`). It is an authoring tool, run once per head. Nothing in the package imports it.
- **ICT-FaceKit Light** (MIT): 53 ARKit-style expression shapes and 100 identity shape modes, with eyes, teeth, gums and tongue, but no hair and no skin texture. It is the candidate template for fitting a head to photos in 0.3, not the stock head.
- **A TLS proxy on the LAN** (Caddy, on the operator's own domain). Mask speaks plain HTTP and WebSocket and leaves TLS to the proxy.
- **argus-pkg-template** scaffolds the package, CI and release.

## Flow

1. **Connect.** A source opens the audio socket for a face and states its sample rate. One source per face at a time; a second is refused.
2. **Drive.** Mask resamples to 16 kHz and hands the audio to the driver, which returns 52 ARKit blendshape weights per animation frame (30 fps by default).
3. **Layer.** Mask adds the idle layer: blinks, small eye movement and a slow head sway, generated from a seed so a take replays identically. The result is one frame: weights plus a head rotation.
4. **Stamp.** Every frame carries its time on the audio timeline, in milliseconds from the start of the session.
5. **Send.** Each viewer of that face receives the audio and the frames on one socket.
6. **Render.** The viewer plays the audio through WebAudio and applies each frame when the audio clock reaches its timestamp, interpolating between frames up to the screen's refresh rate. Lip sync is the viewer's audio clock, not arrival time, so network jitter costs latency and never sync.

## Drivers

A driver turns 16 kHz audio into ARKit weights. It is the seam: everything else is tested without a GPU.

| Driver | Needs | Purpose |
|---|---|---|
| `energy` | nothing (CPU, numpy) | Jaw and mouth opening from loudness. Deterministic. The default in tests, and a fallback on a host with no NVIDIA GPU. It is visibly crude and the docs say so. |
| `a2f` | an Audio2Face service at `ARGUS_MASK_A2F_URL` | The real driver. |

With `ARGUS_MASK_DRIVER=a2f` and no service answering, the server starts, `/health` says the driver is down, and a source is refused with the reason. Mask does not fall back to `energy` on its own: a silent downgrade would look like a bad model.

### The Audio2Face server

Mask builds its own small server from the MIT SDK instead of using NVIDIA's NIM container. No NGC key, and no licence terms beyond the SDK's and the models'.

- It lives in this repo as `a2f/` (a Dockerfile and the server), built by compose and not published to PyPI. This is how argus-cull keeps its NudeNet and SingGuard sidecars.
- The image is a CUDA base with TensorRT and the SDK built from source (CUDA 12.8+, TensorRT 10.13+). Models download from Hugging Face on first start. TensorRT engines are built on the host's own card and cached on a volume, so the first start is slow and later ones are not.
- `WS /drive`: the first message is JSON (model, fps); then 16 kHz mono float32 PCM in, and one message out per animation frame (`t_ms` and the weights).
- `GET /health`: model state, whether the engines are built, VRAM in use.
- One stream at a time. Mask already allows one source per face, and 0.1 runs one face at a time against it.

The one risk this choice carries: the SDK's README describes vertex positions as its output and does not mention ARKit weights. If the SDK has no blendshape solve of its own, this server has to do it, fitting the 52 weights to each frame's vertices against the SDK character's blendshapes. Step 1 settles that before anything else is built on it.

## Faces

A face is a directory under the assets root: a `.glb` and a `manifest.json`.

- **Model contract.** glTF 2.0 binary with morph targets named for the 52 ARKit blendshapes. `argus-mask inspect` lists which of the 52 are present, which are missing, and any extras. A face missing shapes still loads; the missing ones are reported and ignored.
- **Manifest** (versioned, wire schema committed): `id`, `name`, `model` (file name and sha256), `blendshapes` (the names found), `framing` (camera distance and height for the head), and `provenance`.
- **Provenance is required.** `kind` is `stock`, `synthetic` or `curated`; with it go `source`, `license` and `notes`. The server refuses to list a face whose manifest has no provenance. `curated` is refused until 0.3 defines what a curated face must prove.
- **No real person's likeness in 0.1 or 0.2.** The repo, its fixtures and its examples hold stock or synthetic heads only.

### The stock head

One head is the project's mock data, built with MPFB. The repo carries the recipe and a lock file, not the model: `scripts/fetch_stock_head.py` downloads the built `.glb` from the repo's `stock-heads` release and checks it against the lock's size and sha256. It was chosen over the other candidates because it is the only one that is a whole head under a licence a public repo can carry:

| Candidate | ARKit shapes | Hair and skin | Licence | Verdict |
|---|---|---|---|---|
| MPFB / MakeHuman | yes, standard names, from the `faceunits01` pack | yes | core assets CC0 | **the stock head** |
| ICT-FaceKit Light | yes, but split and renamed (`_L`/`_R`), so a name map is needed | neither | MIT | kept for 0.3 fitting |
| three.js `facecap.glb` | yes, loads in three.js as is | skin only | none stated | spike only, never committed |

- **Recipe, not a hand-made file.** `scripts/build_stock_head.py`, run in headless Blender, reads a preset (`scripts/stock_heads/<id>.json`: MakeHuman parameters, which skin, hair and eye assets, the bust cut, the budgets), builds the character, crops it to head and shoulders and exports the `.glb`. The preset and the tool versions go into the manifest's provenance, so the head can be rebuilt. More synthetic heads are more presets. `scripts/README.md` has the versions and the commands.
- **Adult only.** The script refuses a MakeHuman age below 0.5 (about 25 years). Below that MakeHuman blends in its child target, so the line is drawn there, not at 18.
- **Shapes on every part.** Eyelashes, eyebrows, teeth and tongue are separate meshes, and glTF cannot drive one mesh from another. MPFB 2.0.18 transfers the face shapes onto them itself; the script calls it. `inspect` reports any mesh under the head that has no morph targets.
- **Budget for weak screens.** A Raspberry Pi and a base Apple-silicon Mac both render it, and the Go renders it 66 times a frame: at most 30k triangles and 2k textures for the whole head. Textures are PNG or JPEG, with no Draco, meshopt or KTX2, because the viewer loads no decoder.
- **Licences are settled.** MPFB's own pack metadata and the MakeHuman asset-pack pages state CC0 for the base mesh and targets, for the `faceunits01` face shapes, and for every skin, hair, eye, brow, lash, teeth and tongue asset used. MPFB's code is GPL, but none of it is in the exported model.

The first head, `stock-01`, is 23,184 triangles and 5 MB, with all 52 shapes. Known rough edges, all in the recipe's reach: the tongue rides up and fills the mouth when the jaw opens wide; eyebrows, lashes and the hairline have hard cut-out edges; the lower teeth poke through slightly on `mouthFunnel`.

## Display tiers

The viewer takes the tier from its URL, so one server feeds different screens at once.

| Tier | Screen | View |
|---|---|---|
| `flat` (default) | any monitor, laptop, phone, OBS browser source | one view; landscape or portrait; background is a colour, an image or transparent |
| `hld` | Looking Glass HLD | one view, 1080×1920 portrait, black background, head framed for the panel |
| `go` (0.2) | Looking Glass Go | 66 views (11×6 quilt, 4092×4092) through `@lookingglass/webxr` and Bridge |

### Screens and hosts

The GPU server is headless and never drives a screen. A screen hangs off a small client that opens the viewer in a browser.

| Client | Tiers | Notes |
|---|---|---|
| Raspberry Pi (Chromium kiosk) | `flat`, `hld` | No Looking Glass Bridge for ARM Linux was found, so not the Go. |
| Mac mini (Apple silicon) | `flat`, `hld`, `go` | Bridge supports Apple silicon on macOS 12+. It does not work with Safari, so the viewer runs in Chrome. |
| Any laptop or phone on the LAN | `flat` | |

Viewers open the page through the TLS proxy (`https://…`), which makes it a secure context on every client. That is what WebXR needs for the Go, and what a browser microphone or webcam would need later. For this to work:

- `serve --host` binds an address the proxy can reach, and the proxy forwards WebSocket upgrades (Caddy does by default);
- `ARGUS_MASK_CORS_ORIGINS` names the proxy's `https://` origin, or the write guard refuses the source socket;
- the viewer builds its socket URLs from the page's own origin (`wss://` under `https://`), never from a configured host and port.

## GPU

Mask's own process uses no GPU: the driver runs in the Audio2Face service and rendering runs in the viewer's browser. The Audio2Face service is a tenant outside the lease, like ComfyUI and Ollama, and the lease's VRAM wait already accounts for such tenants.

Mask does not take the lease per utterance. A live face that stalls for minutes behind a captioning job is worse than no face. `/health` reports free VRAM so the operator can see the contention.

## What it does not do

- **Speech recognition, language models or text to speech.** Mask takes audio. Whatever produced the audio owns what was said, and guards it. The `say` command is a client: it asks a TTS service for audio and pushes that audio in. The engine and the server never see text.
- **Send pixels.** No server-side rendering and no video stream in 0.1. A recorded clip is a later feature (a headless browser, or the take file replayed in a viewer).
- **Body, hands or full-body rigs.** Face blendshapes and head rotation only.
- **Build a face from photos** before 0.3.
- **Public hosting.** The server binds to localhost by default. Serving the LAN is an explicit `--host`.

## Shape

The package is scaffolded from `argus-pkg-template` with the CLI and server options on. Three layers: an engine (pydantic, numpy, cortex), an optional Typer CLI and an optional FastAPI server that also serves the viewer.

**CLI:**

- `argus-mask inspect <face>`: check a face against the model contract and print its manifest. `--write-manifest` writes the manifest beside a `.glb`, with its sha256; it needs the provenance options, so a face cannot be added without one.
- `argus-mask drive <audio.wav> --out take.ndjson`: audio to a take file, offline.
- `argus-mask push <audio.wav> --face <id>`: play a file into a running server in real time. The demo and the smoke test.
- `argus-mask say "<text>" --face <id>`: the same, with the audio streamed from the TTS service at `ARGUS_MASK_TTS_URL`. It accepts any service that streams PCM with those three headers.
- `argus-mask serve --assets-dir … --port 8109`
- `argus-mask schema [--check]`

**Server:**

- `GET /health`: version, the driver and whether it answers, free VRAM, `read_only`, which faces have a live source.
- `GET /faces`, `GET /face/{id}` (the manifest), `GET /face/{id}/model` (the `.glb`). Ids only, no path input.
- `WS /session/{face_id}/source`: first message is JSON (`sample_rate`, `channels`, `format`, optional `seed`), then binary PCM, then `{"type": "end"}`. The server answers `ready` once the stream is open and `done` when every frame is out. A refusal is an `error` message with the reason, then a close. A second source for the same face is refused. A source sends no faster than real time plus about a second: a viewer that falls 2048 messages behind is dropped.
- `WS /session/{face_id}/view`: first message is the stream header, then binary audio chunks and JSON frames, then an `end`. A viewer that joins mid-stream gets the header and then whatever comes next.
- Both sockets apply the write guard's origin rule by hand, because neither CORS nor `WriteGuard` covers a WebSocket: no `Origin`, the server's own host, or a configured origin.
- `GET /`: the viewer. Static files in the package (plain ES modules, three.js vendored, no build step), the pattern argus-survey uses for its bundled viewer.
- `WriteGuard` and `cross_site_refuse` from cortex. Opening a source socket is a write.

**Wire** (`schema/mask-wire.schema.json`, from `argus-mask schema`):

```
StreamHeader (mask_version "1.0")
  face_id, session_id, fps, seed
  blendshapes: [name]          # the order of every frame's weights
  audio: {sample_rate, channels, format}
  driver: {name, model}

Frame
  seq, t_ms                    # t_ms on the audio timeline
  weights: [float]             # 0..1, same order as the header
  head: {yaw, pitch, roll}     # degrees

StreamEnd
  frames, duration_ms

Audio chunk (binary, viewer socket only)
  8 bytes: little-endian float64, the chunk's start on the audio timeline in ms
  then:    mono f32le PCM at the header's sample rate, whatever the source sent

Take file (.ndjson): one StreamHeader line, one Frame per line, one StreamEnd line.
```

Every JSON message carries a `type` (`header`, `frame`, `end`), so a socket and a take file read the same way.

The version lives on the header, not on every frame. A take file plus its audio replays in the viewer with no server-side model.

**Config** (`ARGUS_MASK_*`): `ASSETS_DIR`, `DRIVER` (`energy` or `a2f`), `A2F_URL`, `TTS_URL` (used by `say` only; empty turns the command off), `FPS` (default 30), `CORS_ORIGINS`, `READ_ONLY`. Flags are read with cortex's `env_flag`. `ARGUS_GPU_DEVICE` (cortex's own name) selects the card `/health` probes.

## Faces from curator (0.3)

A later version builds a face from a curated image set: fit a template head to the images, then project a texture. Mask will need curator to export a **face reference set** for one identity. The request is filed as [argus-curator#130](https://github.com/smk762/argus-curator/issues/130). In short:

- one face cluster, with pose coverage: frontal and three-quarter to each side;
- images picked for a large, sharp, evenly lit, unoccluded face with a neutral expression;
- per-face data in the manifest: bbox, landmarks, yaw, pitch and roll. Curator computes landmarks today and drops them;
- the set's age verdict and quarry's provenance carried through, because mask will refuse a set that is not adult and sourced.

The demo UI for that preset belongs in argus-cull, once curator's side has a shape. Nothing in 0.1 or 0.2 depends on any of this.

## Testing

No real person's face goes in the repo or the tests.

- A fake Audio2Face service, in the pattern of cortex's `_Fake`.
- The `energy` driver is deterministic: one WAV gives the same take file sha256 on every run.
- The idle layer is deterministic for a seed.
- Session tests over the WebSocket test client: the header arrives first, frame `t_ms` never goes backwards, a second source is refused, a viewer that joins mid-session gets a header and then live frames.
- Face tests with a tiny generated `.glb`: all 52 shapes, missing shapes, no provenance (refused), `curated` (refused).
- `a2f` configured but down: the server starts, `/health` reports it, a source is refused with the reason, and no `energy` frames are sent.
- `say` against a fake TTS server that streams PCM: the audio reaches the source socket at the stated sample rate, and a TTS error ends the session cleanly.
- `schema --check` in CI.
- The viewer is checked by hand against `argus-mask push` with a fixture WAV. Not in CI.

## Open questions

Settled: the Audio2Face service is our own server on the MIT SDK ([The Audio2Face server](#the-audio2face-server)). The stock head is built with MPFB ([The stock head](#the-stock-head)). The Go runs from the Mac mini through the TLS proxy ([Screens and hosts](#screens-and-hosts)).

The first two below are answered by the spike in step 1. They block step 6, not the engine.

- **Does the SDK return ARKit weights?** If it returns only vertices, the server needs its own solve. The spike also measures frame rate, latency and VRAM on the 3090.
- **Whether Miso and Audio2Face fit on one card.** Miso is an 8B model loaded in bfloat16, so roughly 16 GB of weights on a 24 GB 3090 before Audio2Face's 4 GB+, and before ComfyUI. The spike measures both loaded together: VRAM, whether Miso still generates faster than real time while Audio2Face runs, and the cold-load delay after Miso's idle unload.
- **How the stock head looks under real speech.** Its blink closes and its shapes all work, but the tongue and the hard edges on hair, brows and lashes are rough (see [The stock head](#the-stock-head)). Whether that matters depends on how it looks driven by Audio2Face, which rarely opens the jaw fully. Fixes go in the recipe.
- **Weak clients (0.2 for the Go).** Whether a base Apple-silicon Mac holds 66 views at a steady frame rate, and whether a Raspberry Pi holds one. The triangle and texture budget is a guess until measured.
- **Emotion.** Audio2Face accepts emotion input. Not in 0.1.

## Build order

1. **Spike, thrown away afterwards.** Build the SDK and its sample on the 3090; a WAV in, and see what comes out (weights or vertices). Drive `facecap.glb` in three.js from that output; that file has no stated licence, so it stays out of the repo. Load Miso alongside and measure. Answers the first two open questions.
2. **Engine.** Wire models and schema, the `energy` driver, the idle layer, `drive` to a take file.
3. **Faces.** The model contract, the manifest with provenance, `inspect`. The stock head recipe, and the head it builds.
4. **Server.** Sessions, the two sockets, `/health`, `push`.
5. **Viewer.** `flat`, then `hld`. Audio-clock sync and interpolation.
6. **The Audio2Face server** (`a2f/`), then the `a2f` driver against the fake and against it.
7. **Schema, README, CLAUDE.md, compose.** Compose runs mask and builds the Audio2Face server. Miso and the TLS proxy are reached by URL.
8. **0.2:** the `go` tier. **0.3:** faces from curator.

Repo conventions from day one: `main` for releases, `dev` for work, PRs into `dev`. Lint, tests and `schema --check` run locally before every PR. Versions come from git tags (`hatch-vcs`). Release notes are the version's section of `CHANGELOG.md`.

## Done when (0.1)

- `argus-mask push hello.wav` moves the stock head's mouth in a browser on another LAN device, and the lips stay with the sound for a two-minute clip.
- The same page opened with the `hld` tier shows a portrait head on black at 1080×1920.
- Two viewers of one face show the same motion.
- `argus-mask drive` on the same WAV twice gives the same take file.
- With the Audio2Face service stopped, `/health` says so and no source is accepted under `DRIVER=a2f`.
- A face without provenance does not appear in `/faces`.

## Scaffold

```bash
uvx copier copy gh:smk762/argus-pkg-template ../argus-mask   # has_cli, has_server: true
```

Package `argus-mask`, module `argus_mask`, port 8109, MIT, Python >=3.11. Extras: `[cli]`, `[server]`, `[a2f]` (the service client), `[all]`.

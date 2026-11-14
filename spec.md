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
- **Miso TTS** (a separate TTS server, `:8034` where this was developed), optional. `POST /synthesise/stream` takes text and streams raw 24 kHz mono float32 PCM, with the format in `X-Sample-Rate` / `X-Audio-Format` / `X-Channels` headers. It is one way to get speech to push into mask, never a dependency: only the `say` command talks to it. `say` also speaks Wyoming (`wyoming://host:port`), the protocol of Home Assistant's TTS servers and of agent-composer's XTTS and Miso bridges, so a TTS on another machine serves too. It is slower than real time on a 3090 (see [Open questions](#open-questions)).
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
3. **Layer.** Mask reads speech cues from the same audio (stresses, phrase starts and ends) and adds the idle layer: nods and brow raises on the cues, blinks timed to pauses, saccades, micro-motion and a slow head sway, from the audio and a seed so a take replays identically. The result is one frame: weights plus a head rotation.
4. **Stamp.** Every frame carries its time on the audio timeline, in milliseconds from the start of the session.
5. **Send.** Each viewer of that face receives the audio and the frames on one socket. A driver answers late, so an audio chunk is sent only once frames reach its end: the mouth for a sound always arrives before the sound.
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

What the spike found, on the RTX 3090:

- **The SDK returns ARKit weights itself.** The models emit vertices, and the SDK ships a blendshape solver and executors on top (`IBlendshapeExecutor`). No solve of our own is needed. Each frame is 68 weights: the 52 ARKit shapes, in ARKit's order and under ARKit's names, then 16 tongue shapes. Mask keeps the 52 by name.
- **It does not blink or look around.** The eight `eyeLook*` shapes and `tongueOut` are switched off in the model's config, and `eyeBlink*` barely moves. The idle layer owns blinks and gaze.
- **Streaming is built in.** Audio goes in as 16 kHz float chunks and frames come out as they are ready.
- **Two models.** Regression v2.3 (one model per character: Mark, Claire, James) runs at any frame rate, about 10 times faster than real time at 30 fps, in about 0.6 GB of VRAM with an engine built for one stream. Diffusion v3.0 is fixed at 60 fps, needs about 1.3 GB, and delivers frames in half-second bursts. Mask uses regression.
- **The model looks ahead.** Regression needs about a quarter second of audio that has not been spoken yet, so its frames trail the audio by 0.3 to 0.45 s. Mask's server holds each audio chunk until frames reach its end (see [Flow](#flow)), so viewers never get sound without the mouth for it.
- **Nothing is gated.** The Audio2Face models download from Hugging Face with no token, under the NVIDIA Open Model License. Audio2Emotion is gated and is left out; emotion input stays neutral.

The server:

- It lives in this repo as `a2f/` (a Dockerfile and the server), built by compose and not published to PyPI. This is how argus-cull keeps its NudeNet and SingGuard sidecars.
- The image is a CUDA 12.9 base with TensorRT 10.13 and the SDK built from source at a pinned commit. The model downloads on first start. Its TensorRT engine is built on the host's own card, for one stream, and cached on a volume, so the first start takes a minute or two and later ones do not.
- `WS /drive`: the first message is JSON (`fps`, `sample_rate`); the server answers with the names of its weights; then 16 kHz mono float32 PCM in, and one message out per animation frame (its number and the weights). `{"end": true}` flushes the tail.
- `GET /health`: whether it is ready, which model, and why not if not.
- One stream at a time. Mask already allows one source per face, and 0.1 runs one face at a time against it.
- Inside, a small C++ worker holds the model and is reset between streams; a Python front owns the network and the first-start work. A reset was checked to give bit-identical frames for the same audio. Port 8110.
- The default character model is `james`. On real speech `mark` holds the jaw to one side; `james` is symmetric and opens further. That rests on one clip.
- Measured on the RTX 3090: about 0.5 GB of VRAM, 0.28 s behind the audio, ready two seconds after a warm start. The image is 4.3 GB and is not published: it contains NVIDIA's CUDA and TensorRT runtime libraries, whose redistribution terms nobody has read yet.

## Faces

A face is a directory under the assets root: a `.glb` and a `manifest.json`.

- **Model contract.** glTF 2.0 binary with morph targets named for the 52 ARKit blendshapes. `argus-mask inspect` lists which of the 52 are present, which are missing, and any extras. A face missing shapes still loads; the missing ones are reported and ignored.
- **Manifest** (versioned, wire schema committed): `id`, `name`, `model` (file name and sha256), `blendshapes` (the names found), `framing` (camera distance and height for the head), and `provenance`.
- **Provenance is required.** `kind` is `personal`, `stock`, `synthetic`, `consented` or `curated`; with it go `source`, `license` (not needed for `personal`) and `notes`, and for `consented` the `subject` and the release's sha256; a generated face may add a `generator` (model, its sha256, seed, prompt). The server refuses to list a face whose manifest has no provenance, or a `consented` face whose release's consent record (`<assets>/.consent/`, docs/consent.md) does not cover it now; a consented face always carries the disclosure label. `curated` is refused until 0.3 defines what a curated face must prove.
- **Personal and cleared.** `personal` is the user's own use, anything lawful in private (a cartoon, a film character), and is never `cleared`; `stock`, `synthetic` and `consented` are. The floor is the law: what argus-darkroom refuses is not a face here either. Only cleared faces and voices go in the repo, its fixtures, its examples, demos or anything commercial.

### The stock head

One head is the project's mock data, built with MPFB. The repo carries the recipe and a lock file, not the model: `scripts/fetch_stock_head.py` downloads the built `.glb` from the repo's `stock-heads` release and checks it against the lock's size and sha256. It was chosen over the other candidates because it is the only one that is a whole head under a licence a public repo can carry:

| Candidate | ARKit shapes | Hair and skin | Licence | Verdict |
|---|---|---|---|---|
| MPFB / MakeHuman | yes, standard names, from the `faceunits01` pack | yes | core assets CC0 | **the stock head** |
| ICT-FaceKit Light | yes, but split and renamed (`_L`/`_R`), so a name map is needed | neither | MIT | kept for 0.3 fitting |
| three.js `facecap.glb` | yes, loads in three.js as is | skin only | none stated | spike only, never committed |

- **Recipe, not a hand-made file.** `src/argus_mask/editor/build_head.py` (until 0.2 `scripts/build_stock_head.py`), run in headless Blender, reads a preset (`scripts/stock_heads/<id>.json`: MakeHuman parameters, which skin, hair and eye assets, the bust cut, the budgets), builds the character, crops it to head and shoulders and exports the `.glb`. The preset and the tool versions go into the manifest's provenance, so the head can be rebuilt. More synthetic heads are more presets. `scripts/README.md` has the versions and the commands.
- **18 and over, and dressed.** The recipe refuses a MakeHuman age below 0.3542, which is 18 years on MakeHuman's scale, and a preset with no clothes. Every head wears at least an outfit.
- **Shapes on every part.** Eyelashes, eyebrows, teeth and tongue are separate meshes, and glTF cannot drive one mesh from another. MPFB 2.0.18 transfers the face shapes onto them itself; the script calls it. `inspect` reports any mesh under the head that has no morph targets.
- **Budget for weak screens.** A Raspberry Pi and a base Apple-silicon Mac both render it, and the Go renders it 66 times a frame: at most 45k triangles and 2k textures for the whole head (30k at first; 35k once heads wore clothes, a garment adding 1–5k beside MakeHuman's 7k teeth; 45k while the weakest panels are not targets). Textures are PNG or JPEG, with no Draco, meshopt or KTX2, because the viewer loads no decoder.
- **Licences are settled.** MPFB's own pack metadata and the MakeHuman asset-pack pages state CC0 for the base mesh and targets, for the `faceunits01` face shapes, and for every skin, hair, eye, brow, lash, teeth and tongue asset used. MPFB's code is GPL, but none of it is in the exported model.

The first head, `stock-01`, is 23,184 triangles and 5 MB, with all 52 shapes.

Its first build looked wrong under real Audio2Face weights: the tongue filled the open mouth. MakeHuman's tongue rests as a dome as tall as the palate, the upper teeth sit level with the lip's edge, and the mouth lining is lit like a cheek. The recipe now flattens the tongue's rest shape, lowers the upper teeth by 2.5 mm, darkens the mouth lining by depth, strengthens `mouthPucker` by half, whitens the teeth and blends the brows and lashes. All of these are preset fields. An open mouth now shows a dark cavity with teeth above and below.

Still rough: the cavity is a coarse dark grey, the tongue is hidden in ordinary speech from a level camera, the hairline is a hard cut-out, and the lower teeth stand clear of the lip when the mouth is wide open.

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

**Config** (`ARGUS_MASK_*`): `ASSETS_DIR`, `DRIVER` (`energy` or `a2f`), `A2F_URL`, `TTS_URL` (used by `say`, the server's `POST /speak` and the editor's Broadcast and Voices pages; empty turns them off), `TTS_VOICE`, `FPS` (default 30), `CORS_ORIGINS`, `READ_ONLY`. Flags are read with cortex's `env_flag`. `ARGUS_GPU_DEVICE` (cortex's own name) selects the card `/health` probes.

## Faces from curator (0.3)

A later version builds a face from a curated image set: fit a template head to the images, then project a texture. Mask will need curator to export a **face reference set** for one identity. The request is filed as [argus-curator#130](https://github.com/smk762/argus-curator/issues/130). In short:

- one face cluster, with pose coverage: frontal and three-quarter to each side;
- images picked for a large, sharp, evenly lit, unoccluded face with a neutral expression;
- per-face data in the manifest: bbox, landmarks, yaw, pitch and roll. Curator computes landmarks today and drops them;
- the set's age verdict and quarry's provenance carried through, because mask will refuse a set that is not adult and sourced.

The demo UI for that preset belongs in argus-cull, once curator's side has a shape. Nothing in 0.1 or 0.2 depends on any of this.

## Testing

Only cleared faces go in the repo or the tests: stock or synthetic, never a real person's.

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

Also settled by the spike: the SDK returns ARKit weights, so the server needs no solve of its own.

Measured since, with Miso and Audio2Face loaded together on the RTX 3090 and nothing else heavy: 22 GB of the 24 GB in use (Miso 18 GB, Audio2Face 0.5 GB), so ComfyUI cannot be resident at the same time. Miso loads cold in about 85 s and generates at about 0.27 times real time, so `say` fetches the whole utterance before playing it rather than streaming it; a ten-second line starts about 35 s after it is typed. The chain Miso → mask → Audio2Face → viewer was run end to end: 300 frames for 10 s of speech, the sound never ahead of the mouth.

- **How the stock head looks in motion.** The mouth was fixed from still frames of a real take (see [The stock head](#the-stock-head)). Nobody has yet watched it speak on a real screen with sound.
- **Weak clients (0.2 for the Go).** Whether a base Apple-silicon Mac holds 66 views at a steady frame rate, and whether a Raspberry Pi holds one. The triangle and texture budget is a guess until measured.
- **Emotion.** Audio2Face accepts emotion input. Not in 0.1.

## Build order

1. **Spike, thrown away afterwards.** Done: the SDK builds in Docker and returns ARKit weights; the findings are in [The Audio2Face server](#the-audio2face-server). The Miso measurement is in [Open questions](#open-questions).
2. **Engine.** Wire models and schema, the `energy` driver, the idle layer, `drive` to a take file.
3. **Faces.** The model contract, the manifest with provenance, `inspect`. The stock head recipe, and the head it builds.
4. **Server.** Sessions, the two sockets, `/health`, `push`.
5. **Viewer.** `flat`, then `hld`. Audio-clock sync and interpolation.
6. **The Audio2Face server** (`a2f/`), then the `a2f` driver against the fake and against it. Done.
7. **Schema, README, CLAUDE.md, compose.** Compose runs mask and builds the Audio2Face server. Miso and the TLS proxy are reached by URL.
8. **0.2:** the `go` tier. **0.3:** faces from curator.

Repo conventions from day one: `main` for releases, `dev` for work, PRs into `dev`. Lint, tests and `schema --check` run locally before every PR. Versions come from git tags (`hatch-vcs`). Release notes are the version's section of `CHANGELOG.md`.

## Done when (0.1)

- `argus-mask push hello.wav` moves the stock head's mouth in a browser on another LAN device, and the lips stay with the sound for a two-minute clip. **Measured, not yet watched:** a 20 s clip through the real Audio2Face server reached a viewer with the sound never more than 34 ms ahead of the mouth; nobody has yet watched a long clip on a real screen.
- The same page opened with the `hld` tier shows a portrait head on black at 1080×1920. **Done** (headless browser).
- Two viewers of one face show the same motion. **Done** (test).
- `argus-mask drive` on the same WAV twice gives the same take file. **Done** (test, and with the real Audio2Face server).
- With the Audio2Face service stopped, `/health` says so and no source is accepted under `DRIVER=a2f`. **Done** (test).
- A face without provenance does not appear in `/faces`. **Done** (test).

## Scaffold

```bash
uvx copier copy gh:smk762/argus-pkg-template ../argus-mask   # has_cli, has_server: true
```

Package `argus-mask`, module `argus_mask`, port 8109, MIT, Python >=3.11. Extras: `[cli]`, `[server]`, `[a2f]` (the service client), `[all]`.

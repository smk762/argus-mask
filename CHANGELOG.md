# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The engine.** Audio in, animation frames out: a streaming resampler to 16 kHz, a driver, a seeded idle layer
  (blinks, gaze, head sway) and frames stamped on the audio timeline. The same audio, driver and seed give the same
  take file.
- **Drivers.** `energy` moves the mouth from loudness on the CPU, for tests and hosts with no NVIDIA card. `a2f` is
  the client of the Audio2Face server.
- **Faces.** A `.glb` with ARKit-named morph targets and a `manifest.json` with required provenance. A face with no
  provenance, a `curated` face, or one whose model no longer matches its sha256 is not served.
- **Server** (`argus-mask serve`, port 8109): `/health`, `/faces`, `/face/{id}`, `/face/{id}/model`, a source socket
  (one per face) and a viewer socket, both under the same origin rule as cortex's write guard.
- **Viewer**, bundled and served at `/`: three.js, lip sync on the page's own audio clock, the `flat` and `hld`
  display tiers.
- **CLI:** `inspect` (with `--write-manifest`), `drive`, `push`, `say`, `serve`, `schema`.
- **The stock head.** `scripts/build_stock_head.py` builds a head with headless Blender and MPFB from a preset (CC0
  MakeHuman assets, adults only). The model is not in the repo: `scripts/fetch_stock_head.py` (`make head`) downloads
  it and checks it against `scripts/stock_heads/stock-01.lock.json`.
- Wire schema `schema/mask-wire.schema.json`.
- **The Audio2Face server** (`a2f/`): NVIDIA's Audio2Face-3D SDK behind the `a2f` driver's protocol, in its own
  Docker image. It downloads its model and builds a TensorRT engine for the host's card on first start, and holds
  about 0.5 GB of VRAM. `docker compose --profile a2f` runs it beside mask. The image is built locally, not published.

- The viewer's audio buffer is tunable (`?buffer=<ms>`, default 250).
- The stock head lock points at revision 2 of `stock-01`, with the mouth fixes.
- **Clothes.** A preset's `assets.clothes` lists garments; the body under them is removed and they are cut with the
  bust. `stock-01` revision 3 wears `thegreatengineer_galactic_warrior_uniform` from the MakeHuman community's
  suits02 pack (CC0): a high-collared crew tunic, since a bust in nothing looked undressed.

- **`say` speaks Wyoming** (`--tts-url wyoming://host:port`, `-p voice=…`), the protocol of Home Assistant's TTS
  servers and agent-composer's XTTS and Miso bridges, beside the HTTP PCM stream it already understood.
- `say` says it is waiting for the TTS service, and Ctrl-C stops the wait at once instead of hanging on the socket.
- **`say --voice`** and `ARGUS_MASK_TTS_VOICE`: the voice to ask the TTS service for, a stored clone's name or a
  speaker number, put in the field each kind of service expects.
- **`say --pause`**: the text goes to the TTS service one sentence at a time, with 300 ms of silence between
  sentences and 600 ms between paragraphs, so sentences no longer run into each other; `--pause 0` sends it whole.
- `say` pads the utterance with 150 ms of silence before and 300 ms after, so a waking output device does not eat the
  first syllable and the mouth closes before the stream ends.
- The viewer counts audio chunks that arrived while the browser had paused its audio clock (`__mask.audioDropped`) and
  asks for the clock back, so a clipped start can be told from a clipped reply.

### Fixed

- **`say` no longer dumps a slow TTS model's speech all at once.** It fetches the whole utterance before claiming the
  face and plays it at the speed of speech (`--stream` keeps the old behaviour for a service faster than real time),
  and a source's pacing clock now starts at its first chunk rather than at connect.

- **Audio no longer runs ahead of the mouth with a slow driver.** Audio2Face answers about 0.3 s behind the audio it
  is given. The server now holds each audio chunk until frames reach its end, and ends a stream whose driver falls
  more than 10 s behind.

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
- **The stock head.** `src/argus_mask/editor/build_head.py` (was `scripts/build_stock_head.py`) builds a head with headless Blender and MPFB from a preset (CC0
  MakeHuman assets, adults only). The model is not in the repo: `scripts/fetch_stock_head.py` (`make head`) downloads
  it and checks it against `scripts/stock_heads/stock-01.lock.json`.
- Wire schema `schema/mask-wire.schema.json`.
- **Face editor** (`argus-mask editor`, port 8111): a page that builds heads from presets with headless Blender and
  MPFB, about ten seconds a build, with a posed preview. Skin, hair, clothes and a hat are picked from thumbnails. It offers only assets an installed pack licenses, writes their
  licences into the face's provenance, replaces only faces it made, and is behind the write guards. Blender, MPFB and
  the asset packs are the user's to install; none ships.
- **A second stock preset**, `stock-02`: a female head, about 25, short brown hair. Not published yet.
- **The Audio2Face server** (`a2f/`): NVIDIA's Audio2Face-3D SDK behind the `a2f` driver's protocol, in its own
  Docker image. It downloads its model and builds a TensorRT engine for the host's card on first start, and holds
  about 0.5 GB of VRAM. `docker compose --profile a2f` runs it beside mask. The image is built locally, not published.

- The viewer's audio buffer is tunable (`?buffer=<ms>`, default 250).
- The stock head lock points at revision 2 of `stock-01`, with the mouth fixes.

### Fixed

- **`say` no longer dumps a slow TTS model's speech all at once.** It fetches the whole utterance before claiming the
  face and plays it at the speed of speech (`--stream` keeps the old behaviour for a service faster than real time),
  and a source's pacing clock now starts at its first chunk rather than at connect.

- **Audio no longer runs ahead of the mouth with a slow driver.** Audio2Face answers about 0.3 s behind the audio it
  is given. The server now holds each audio chunk until frames reach its end, and ends a stream whose driver falls
  more than 10 s behind.

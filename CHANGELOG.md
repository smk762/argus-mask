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

### Fixed

- **Audio no longer runs ahead of the mouth with a slow driver.** Audio2Face answers about 0.3 s behind the audio it
  is given. The server now holds each audio chunk until frames reach its end, and ends a stream whose driver falls
  more than 10 s behind.

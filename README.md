# argus-mask

A talking face for the Argus suite. Audio goes in; a stream of facial animation frames comes out; a browser page renders a rigged head from those frames, in step with the audio, on any screen on the LAN.

Part of the [Argus suite](https://github.com/smk762?tab=repositories&q=argus). Mask sits off the main line, like argus-darkroom and argus-atlas. The design and build plan are in [spec.md](spec.md).

```
audio source ──▶ argus-mask :8109 ──▶ viewer (browser, any LAN device)
(file, agent,     drive + sync         flat panel │ Looking Glass HLD
 TTS output)          │
                      ▼
               Audio2Face server (external, by URL)
```

## Status

Built and tested: the engine, both drivers, faces, the server, the viewer's `flat` and `hld` tiers, the CLI, and the Audio2Face server in [a2f/](a2f/README.md).

Not built yet: the Looking Glass Go tier (0.2), and faces built from argus-curator output (0.3).

Not yet seen by a person: the stock head speaking on a real screen with sound. Its mouth was tuned from still frames of real Audio2Face output.

## Quick start

```bash
make install
make head        # downloads the stock head into assets/stock-01 (5 MB, CC0)
uv run argus-mask serve --assets-dir assets
```

Open `http://127.0.0.1:8109/`, click once to allow sound, then from another terminal:

```bash
uv run argus-mask push hello.wav --face stock-01
```

That uses the `energy` driver, which moves the mouth from loudness and nothing more. For real lip sync, run the Audio2Face server (an NVIDIA GPU with about 1 GB of VRAM to spare) and point mask at it:

```bash
docker build -t argus-mask-a2f a2f/      # once; see a2f/README.md for what it needs
docker run -d --name a2f --gpus all -p 127.0.0.1:8110:8110 -v a2f-data:/data argus-mask-a2f
ARGUS_MASK_DRIVER=a2f ARGUS_MASK_A2F_URL=http://127.0.0.1:8110 uv run argus-mask serve --assets-dir assets
```

Or both in containers: `docker compose --profile a2f up --build`, with `DRIVER=a2f` and `A2F_URL=http://a2f:8110` in `.env`.

## Commands

| Command | What it does |
|---|---|
| `argus-mask inspect <face>` | Checks a face directory or a `.glb`: which of the 52 ARKit shapes it has, triangles, texture sizes. `--write-manifest` writes `manifest.json` beside a `.glb`; it needs `--kind`, `--source` and `--license`, and takes `--distance`, `--height` and `--fov` for the camera framing. |
| `argus-mask drive <audio.wav> --out take.ndjson` | Audio to a take file, offline. The same audio, driver and seed give the same file. |
| `argus-mask push <audio.wav> --face <id>` | Plays a WAV into a running server in real time. |
| `argus-mask say "<text>" --face <id>` | The same, with the audio streamed from a TTS service. |
| `argus-mask serve` | The server and the viewer. Binds `127.0.0.1` unless given `--host`. |
| `argus-mask schema [--check]` | Writes or checks `schema/mask-wire.schema.json`. |

## Faces

A face is a directory under the assets root holding a `.glb` and a `manifest.json`.

- **Model:** glTF 2.0 binary, with morph targets named for the 52 ARKit blendshapes (`jawOpen`, `eyeBlinkLeft`, …). Missing shapes are reported and ignored. Textures must be PNG or JPEG; the viewer loads no KTX2 or meshopt decoder.
- **Manifest:** id, name, the model's file name and sha256, the shapes found, camera framing, and provenance.
- **Provenance is required.** `kind` is `stock` or `synthetic`, with a `source` and a `license`. A face with no provenance is not served. `curated` is refused until faces from curator are designed.
- A face is also refused when its manifest id differs from its directory name, or its model no longer matches the recorded sha256. `inspect` says why.

No real person's likeness belongs in this repo, its fixtures or its examples.

### The stock head

The repo carries no model files. `make head` (or `python scripts/fetch_stock_head.py`) downloads `stock-01`, a generated adult head built from CC0 MakeHuman assets, and checks it against the size and sha256 in `scripts/stock_heads/stock-01.lock.json` before writing anything. To build it yourself with headless Blender and MPFB, or to make another head from a new preset, see [scripts/README.md](scripts/README.md).

### Your own head

```bash
mkdir -p assets/my-head && cp /path/to/head.glb assets/my-head/
uv run argus-mask inspect assets/my-head/head.glb --write-manifest \
    --kind stock --source "where it came from" --license "CC0-1.0" \
    --distance 0.9 --height -0.08
```

## Viewer

`GET /` serves the viewer. URL parameters:

| Parameter | Values |
|---|---|
| `face` | a face id; default: the first one the server lists |
| `display` | `flat` (default), or `hld` for a Looking Glass HLD: 1080×1920 portrait on black |
| `bg` | a CSS colour, or `transparent` (flat tier) |
| `hud` | `0` hides the status line |
| `buffer` | how far behind arrival the audio plays, in ms (default 250). Raise it on a jittery network; lower it if the face feels late. |

Lip sync runs on the page's audio clock: audio plays `buffer` ms behind arrival and each frame is shown when the clock reaches its timestamp. Network jitter costs latency, not sync.

Browsers start audio only after a click or key press. A kiosk can waive that; for Chromium, start it with `--autoplay-policy=no-user-gesture-required`.

## Serving the LAN

The server speaks plain HTTP and WebSocket, and binds localhost by default. To put the viewer on other machines:

1. Bind an address your TLS proxy can reach: `argus-mask serve --host <address>`.
2. Proxy to it with TLS. Caddy forwards WebSocket upgrades without extra configuration.
3. Name the proxy's origin in `ARGUS_MASK_CORS_ORIGINS`, for example `https://mask.example.org`.

Both sockets refuse a browser whose `Origin` is neither the server's own host nor in that list. Clients that send no `Origin`, such as the CLI, are allowed. There is no authentication: reaching the port is access.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARGUS_MASK_ASSETS_DIR` | `assets` | Root of the faces |
| `ARGUS_MASK_DRIVER` | `energy` | `energy` or `a2f` |
| `ARGUS_MASK_A2F_URL` | empty | The Audio2Face server, for the `a2f` driver |
| `ARGUS_MASK_TTS_URL` | empty | The TTS stream endpoint, used by `say` only |
| `ARGUS_MASK_TTS_VOICE` | empty | The TTS voice `say` asks for: a stored clone's name, or a speaker number |
| `ARGUS_MASK_FPS` | `30` | Animation frames per second |
| `ARGUS_MASK_CORS_ORIGINS` | empty | Browser origins allowed to open the sockets, comma-separated |
| `ARGUS_MASK_READ_ONLY` | off | Refuses every audio source; viewers still connect |

With `DRIVER=a2f` and no server answering, mask starts, `/health` reports the driver down, and a source is refused with the reason. It never falls back to `energy` on its own.

## Drivers

| Driver | Needs | What you get |
|---|---|---|
| `energy` | nothing | The jaw opens with loudness and the mouth widens on hissy sounds. No lip shapes. For tests and for hosts with no NVIDIA GPU. |
| `a2f` | the [Audio2Face server](a2f/README.md) | Real mouth shapes from NVIDIA's Audio2Face-3D, at 30 fps, about 0.3 s behind the audio it is given. |

Neither driver blinks or moves the eyes. Mask's idle layer adds blinks, glances and a slow head sway, from a seed, so a take replays identically.

## Speech from a TTS service

`say` asks a TTS service for the speech and plays it into the face at the speed of speech. The whole utterance is fetched first: Miso TTS 8B generates at about a quarter of real time on an RTX 3090, and streaming it live would stall the face mid-sentence. `--stream` forwards the audio as it is generated, for a service that is faster than real time. The face stays free for other sources while the speech is being generated, and Ctrl-C stops the wait at once. The text is sent to that service and nowhere else; mask's server only ever receives audio.

Two kinds of service, told apart by the URL:

- **`http(s)://…`**: an endpoint that takes `{"text": …}` and streams raw PCM, stating the format in `X-Sample-Rate`, `X-Audio-Format` (`pcm_f32le` or `pcm_s16le`) and `X-Channels` headers. Miso TTS's `/synthesise/stream` is one. Extra request fields go in with `-p key=value`.
- **`wyoming://host:port`**: a [Wyoming](https://github.com/OHF-Voice/wyoming) TTS server, the protocol Home Assistant's voice pipeline uses; agent-composer's XTTS and Miso bridges speak it. `-p language=…` and `-p speaker=…` pass through. Audio must be 16-bit.

`--voice` (default `ARGUS_MASK_TTS_VOICE`) picks the voice. A name is a stored clone (`voice_clone_id` on the HTTP contract, the voice's name on Wyoming, where agent-composer's `clone:` prefix is also understood); a number is one of the service's own speakers. Any `-p speaker=…`, `voice_clone_id=…` or `voice=…` you give wins over it.

```bash
export ARGUS_MASK_TTS_URL=http://127.0.0.1:8034/synthesise/stream
export ARGUS_MASK_TTS_VOICE=inara-nobility
argus-mask say "Hello from the other side of the room." --face stock-01

argus-mask say "Or through a Wyoming server." --face stock-01 --tts-url wyoming://192.168.1.20:10200 --voice "Claribel Dervla"
```

`say` sends the text one sentence at a time and leaves `--pause` ms of silence between sentences (default 300; twice that between paragraphs, which are blank lines in the text). TTS models run sentences into each other, and a short request wanders less than a long one. `--pause 0` sends the text whole. It also puts 150 ms of silence before the utterance and 300 ms after it: TTS models start speaking on their first sample, and an output device that has to wake up would otherwise eat the first syllable.

## Wire

`schema/mask-wire.schema.json` is the contract. A viewer socket carries:

- one `header` (JSON): session, fps, seed, the order of the blendshape weights, the audio format;
- `frame` messages (JSON): `seq`, `t_ms` on the audio timeline, `weights`, `head` rotation in degrees;
- audio chunks (binary): 8 bytes of little-endian float64 giving the chunk's start in ms, then mono `f32le` PCM. The server sends a chunk only once frames reach its end, so a driver that answers late delays the sound, never the mouth;
- one `end` (JSON).

A take file is the same header, frames and end, one JSON object per line.

A source socket takes one JSON hello (`sample_rate`, `channels`, `format`, optional `seed`), then binary PCM, then `{"type": "end"}`. Sources should send no faster than real time plus about a second; a viewer that falls 2048 messages behind is dropped.

## Develop

```bash
make install   # venv + editable install with every extra
make test
make lint
make schema    # after changing models.py
```

Tests use fake Audio2Face and TTS services and a one-triangle `.glb` built in code. The viewer is not covered by CI; check it by hand with `argus-mask push`. The real Audio2Face server has its own live check, `a2f/check.py`, run by hand against a running container.

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow, plus `argus-mask schema --check`.
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags.
- Versioning is derived from git tags via `hatch-vcs`: tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template). Run `copier update` to pull template changes.

The viewer vendors [three.js](https://threejs.org) (MIT); its licence is in `src/argus_mask/viewer/vendor/LICENSE-three`.

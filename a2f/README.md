# The Audio2Face server

Speech audio in, ARKit blendshape weights out. This is the service behind argus-mask's `a2f` driver: mask sends it 16 kHz audio over a WebSocket and gets back one set of weights per animation frame, which is what makes the lips match the words. Without it, mask's only driver is `energy`, which opens the jaw by loudness.

It runs [NVIDIA's Audio2Face-3D SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK) on one NVIDIA GPU. The SDK does the hard part, including the solve from mesh vertices to blendshape weights; this directory is the thin layer that keeps a model loaded and puts mask's protocol in front of it.

```
mask ──WS /drive──▶ server.py ──stdin/stdout──▶ a2f-worker ──▶ Audio2Face-3D SDK ──▶ GPU
      16 kHz PCM    (Python, the network,        (C++, the model,
      ◀── weights    first-start work)            reset between streams)
```

It lives in the mask repo and is built by Docker. It is not part of the `argus-mask` Python package and is not published to PyPI.

## Run it

```bash
docker build -t argus-mask-a2f a2f/
docker run -d --name a2f --gpus all -p 127.0.0.1:8110:8110 -v a2f-data:/data argus-mask-a2f
curl -s http://127.0.0.1:8110/health
```

Then point mask at it:

```bash
ARGUS_MASK_DRIVER=a2f ARGUS_MASK_A2F_URL=http://127.0.0.1:8110 argus-mask serve --assets-dir assets
```

**The first start is slow.** The server downloads the model from Hugging Face (220 to 320 MB depending on the model, no account or token needed) and builds a TensorRT engine for the GPU it finds (about a minute on an RTX 3090). `/health` answers throughout with `"ready": false` and what it is doing. Both are kept in the `/data` volume, so later starts are ready in about two seconds.

**Building the image is slower.** The build stage pulls about 5.5 GB of CUDA and TensorRT development packages and unpacks them to about 19 GB. The finished image is far smaller (see [Measured](#measured)), but the build needs that room. Have about 25 GB free in Docker's data directory before building or rebuilding: BuildKit can unpack that layer again even when it reports the step as cached, and a rebuild on a nearly full disk filled it completely during development.

The container needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host, a driver that supports CUDA 12.9, and a GPU TensorRT 10.13 supports. It runs as uid 10001; a bind mount in place of the named volume must be writable by that user.

## Configuration

| Variable | Default | |
|---|---|---|
| `A2F_MODEL` | `james` | `mark`, `claire` or `james`: which character's regression model to load. See [Which model](#which-model). |
| `A2F_GPU_SOLVER` | `0` | `1` runs the blendshape solve on the GPU. See [Measured](#measured). |
| `A2F_PORT` | `8110` | The port inside the container. |
| `A2F_HOST` | `0.0.0.0` | The address inside the container. Which host address it is published on is the `-p` flag's choice. |
| `A2F_FPS` | `30` | The frame rate the model is warmed for. A stream asking for another rate reloads it (about a second). |
| `A2F_DATA` | `/data` | Where models and engines are cached. |

There is no authentication. Publish the port on `127.0.0.1` or a private network only. Sockets opened by a web page (anything carrying an `Origin` header) are refused, so a site cannot drive the GPU through a visitor's browser.

## Protocol

`GET /health` always answers 200 while the server is up:

```json
{"ready": true, "model": "Audio2Face-3D-v2.3-Mark", "fps": 30, "gpu_solver": false, "busy": false, "streams": 3}
```

When `ready` is false there is a `reason`: downloading (with file and megabytes), building the engine, loading the model, or the failure it is retrying after.

`WS /drive` carries one stream:

| | Direction | Message |
|---|---|---|
| 1 | client | `{"fps": 30, "sample_rate": 16000}` |
| 2 | server | `{"blendshapes": ["eyeBlinkLeft", ...]}`: 68 names, the order of every frame's weights |
| 3 | client | binary: 16 kHz mono `f32le` PCM, in chunks of any size, as it arrives |
| 4 | server | `{"i": 0, "weights": [...]}`: one per finished frame, in order, while the audio is still coming |
| 5 | client | `{"end": true}` |
| 6 | server | the remaining frames, then `{"end": true}` |

- A failure at any point is `{"error": "..."}` and the socket closes.
- **One stream at a time.** A second gets `{"error": "busy: ..."}` and a close.
- The sample rate must be 16000; the client resamples. `fps` is 1 to 240.
- A stream of N samples gives `ceil(N * fps / 16000)` frames. Frame `i` is the face at `i / fps` seconds.
- A client that disconnects without `{"end": true}` ends its stream; the next one starts clean.

### What the weights are

The 68 names are the SDK's: ARKit's 52 in ARKit's order, then 16 tongue shapes (`tongueTipUp` … `tongueNarrow`) that ARKit does not have. Mask keeps the ones it knows by name.

Not every ARKit shape moves:

- **The eight `eyeLook*` shapes and `tongueOut` are always 0.** The model's own configuration switches them off.
- **There are no blinks.** `eyeBlink*` stays under 0.1. Blinks and gaze are the caller's job; mask adds them in its idle layer.
- The emotion input is fixed at neutral. Audio2Emotion, the model that would drive it, is not used and never downloaded.

## Which model

All three are NVIDIA's regression models (v2.3), one per character it was trained on. They differ in the face they were fitted to, so the same audio gives different weights:

- **`james`** (the default) is symmetric (`jawLeft` and `jawRight` both average 0.04 on a 20-second clip), opens the jaw further (0.70 against 0.54 for `mark` on the same clip) and closes the lips harder. It is the default on that evidence, which is one clip.
- **`mark`** holds the jaw to one side: on the same clip `jawRight` never dropped below 0.10 and averaged 0.21, against 0.05 for `jawLeft`.
- **`claire`** has not been run through this server.

The SDK also has a diffusion model (v3.0). This server does not offer it: it produces frames in bursts of thirty every half second, lags the audio by 0.6 s where regression lags 0.28 s, and only runs at 60 fps.

## Measured

On one RTX 3090 that other services were using at the same time, with NVIDIA's sample speech clip.

| | |
|---|---|
| VRAM held while loaded | 582 MiB (`mark`), 500 MiB (`james`) |
| Lag behind the audio, live | 0.28 s mean, 0.31 s worst, with 40 ms chunks sent at the speed of speech |
| Speed, CPU solver | about 10× real time at 30 fps, 2.8× at 60 fps |
| Speed, GPU solver | about 15× real time at 30 fps |
| First start | model download (321 MB for `mark`, 221 MB for `james`), then 50 to 65 s to build the engine |
| Later starts | ready 1.6 s after the container starts |
| Image | 4.3 GB unpacked, 2.7 GB compressed |

The lag is the model's own: each frame needs 0.26 s of audio from after its moment. This server adds nothing that shows in the measurement; the same model driven without it lags by the same 0.28 s.

The two solvers do not give identical weights. On the same clip `jawOpen` peaked at 0.54 with the CPU solver and 0.51 with the GPU one. The speed figures are from the worker driven directly, before the server was built around it.

The engine is built for one stream. The SDK's defaults build it for 128 at once, which holds 1.4 GB of VRAM for nothing here. Half of the image is one TensorRT library that is only needed to build the engine.

## How it is built

- **`worker/main.cpp`** is the only C++. It loads one model through the SDK's blendshape-solve executor, reads framed audio on stdin and writes framed weights on stdout. At the end of a stream it flushes the frames the model was holding back, then resets the SDK's executor and audio buffer.
- **One worker, reset between streams**, and not a fresh process for each. `a2f-worker <model.json> --selftest clip.f32` is the evidence that this is safe: the same clip gives bit-identical weights fed whole, fed in 40 ms chunks, and again after an unrelated stream and a reset. A reset costs nothing, where a fresh process costs a second and a second copy of the model in VRAM while a spare warms up.
- **`server.py`** owns the network and never touches the GPU. If the worker dies, mid-stream or idle, the stream in progress gets an error and a new worker is loaded; `/health` is not ready for the second that takes.
- **`check.py`** runs the protocol against a live server with real audio: `python a2f/check.py http://127.0.0.1:8110 speech.wav`. It needs the GPU, so it is run by hand and not in CI.
- **Engines are named for the GPU and TensorRT version** they were built with (`network-b1-trt10.13.3.9-nvidia-geforce-rtx-3090.trt`), because an engine runs on nothing else. A new card or a new image builds a new one beside the old.
- **Model revisions are pinned** in `server.py`, so a change to the upstream repositories cannot change what a deployment runs.

## Licences

- **This directory** is MIT, like the rest of argus-mask.
- **The Audio2Face-3D SDK** is MIT. It is fetched and compiled when the image is built (pinned to commit `1ca0f02`); its licence text is in the image at `/opt/a2f/licenses/`.
- **The models** are under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). They are downloaded from NVIDIA's Hugging Face repositories when the server first starts. They are not in this repository and not in the image, and nothing here redistributes them.
- **The image contains NVIDIA's CUDA and TensorRT runtime libraries**, which have their own licence agreements. Building and running it yourself is the intended use. Check those terms before publishing a built image to a registry; that has not been done.
- **Audio2Emotion** has a separate, gated licence. It is not used.

## Not verified

- Any GPU but an RTX 3090, any driver but 580, and any host but Linux x86-64.
- `claire`. `mark` and `james` both pass `check.py`.
- Speed and lag with the card to itself. Every number above was taken while another service was using the same GPU.
- Streams longer than 20 seconds. Used audio is dropped as it goes, which is the SDK's way of keeping long streams bounded, but an hour-long stream has not been run.
- Non-neutral emotion. The input exists in the SDK and is not exposed here.
- How the weights look on any head other than mask's stock one. On that head the tongue fills the open mouth; that is the head's fault, not the weights'.

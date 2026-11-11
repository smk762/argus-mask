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
make head        # downloads the stock head into assets/stock-01 (10 MB, CC0)
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
| `argus-mask record --face <id> --out clip.mp4` | Waits for the next session on a face and renders it to a video, offline and frame-exact, with the audio. |
| `argus-mask serve` | The server and the viewer. Binds `127.0.0.1` unless given `--host`. |
| `argus-mask editor` | The face editor (port 8111): build heads from presets with Blender. See [Face editor](#face-editor). |
| `argus-mask schema [--check]` | Writes or checks `schema/mask-wire.schema.json`. |

## Faces

A face is a directory under the assets root holding a `.glb` and a `manifest.json`.

- **Model:** glTF 2.0 binary, with morph targets named for the 52 ARKit blendshapes (`jawOpen`, `eyeBlinkLeft`, …). Missing shapes are reported and ignored. Textures must be PNG or JPEG; the viewer loads no KTX2 or meshopt decoder.
- **Manifest:** id, name, the model's file name and sha256, the shapes found, camera framing, and provenance.
- **Provenance is required.** `kind` is `stock` or `synthetic`, with a `source` and a `license`. A face with no provenance is not served. `curated` is refused until faces from curator are designed.
- A face is also refused when its manifest id differs from its directory name, or its model no longer matches the recorded sha256. `inspect` says why.

No real person's likeness belongs in this repo, its fixtures or its examples.

To make a synthetic source portrait (an SDXL image of a person who does not exist) for the planned photoreal face routes, see [docs/synthetic-faces.md](docs/synthetic-faces.md).

### The stock head

The repo carries no model files. `make head` (or `python scripts/fetch_stock_head.py`) downloads `stock-01`, a generated adult head built from CC0 MakeHuman assets, and checks it against the size and sha256 in `scripts/stock_heads/stock-01.lock.json` before writing anything. `python scripts/fetch_stock_head.py --id stock-02` fetches the second, a woman of about 25. Both wear the crew uniform and carry the wet line and baked skin maps (revision 4 of `stock-01`, the first of `stock-02`). To build it yourself with headless Blender and MPFB, or to make another head from a new preset, see [scripts/README.md](scripts/README.md).

### Your own head

```bash
mkdir -p assets/my-head && cp /path/to/head.glb assets/my-head/
uv run argus-mask inspect assets/my-head/head.glb --write-manifest \
    --kind stock --source "where it came from" --license "CC0-1.0" \
    --distance 0.9 --height -0.08
```

### Face aliases

An alias is a stable name for a face: `holly` for a persona's face, used by the agent that speaks and by the kiosk that shows it, while which face stands behind it is chosen in the editor. Mithrandir names `holly` once; changing her face is a click on **Saved faces**, not a config change.

- **Where:** `aliases.json` in the assets directory, `{"aliases": {"holly": "ginger-3"}}`. The editor writes it (Saved faces → Aliases: add, repoint, remove, open on a screen); the server only reads it, so its assets mount may stay read-only, and it reads it again whenever it changes: no restart.
- **Everywhere a face is named:** `/face/holly`, `/face/holly/model`, `/session/holly/view` and `/source`, `POST /push?face=holly`, `/speak`. `GET /aliases` and `/health`'s `aliases` list them; `/faces` lists faces only, as before.
- **Sources and screens meet on the name they use.** A source on `holly` reaches screens watching `holly`, and `/health`'s `live` and `screens` list `holly`. A push to `ginger-3` directly reaches screens watching `ginger-3`, not `holly`.
- **Repointing:** new connections get the new face at once. A screen already watching `holly` switches on the next thing said: each stream's header names the face itself, and a viewer that loaded another face reloads, with its URL, and joins the stream where it has got to.
- **Rules:** an alias is a face id that isn't a face's directory (it never hides a face), it points at a face the server serves, and never at another alias. An entry that breaks a rule is ignored (the editor shows why). Renaming a face in the editor carries its aliases along; deleting one an alias points at is refused until the alias points elsewhere.

A kiosk for a persona, with the sound on the house speakers (see [House audio](#house-audio-snapcast)):

```
http://192.168.1.39:8109/?face=holly&display=hld&kiosk=1&audio=external&delay=750
```

## Viewer

`GET /` serves the viewer. URL parameters:

| Parameter | Values |
|---|---|
| `face` | a face id; default: the first one the server lists |
| `display` | `flat` (default), or `hld` for a Looking Glass HLD: 1080×1920 portrait on black |
| `bg` | a CSS colour, or `transparent` (flat tier) |
| `hud` | `0` hides the status line |
| `kiosk` | `1` for a dedicated screen: no status line, no cursor, full screen on the first tap |
| `buffer` | how far behind arrival the audio plays, in ms (default 250). Raise it on a jittery network; lower it if the face feels late. |
| `audio` | `external`: the sound plays elsewhere (house speakers, Snapcast). The page plays none, needs no tap for sound, and moves the face on the session's timeline. `mute=1` is the same. |
| `delay` | with `audio=external`, show each frame this many ms later (0 to 5000, default 0), to meet speakers that play behind |
| `quality` | the look: `low`, `medium` (default) or `high`. See below. |
| `ibl`, `shadows`, `skin`, `cornea`, `ao`, `smaa`, `dof`, `grain`, `bloom` | `1` or `0`: one effect on or off over the profile |
| `hair` | `blend` (default: an opaque core, then the fringe blended over it, so wisps fade), `a2c`, `hash`, or `cutout` |
| `pores` | `0` drops the generated pore detail over a head's baked skin maps (for comparing) |
| `tone`, `exposure` | tone mapping: `neutral` (default), `agx`, `aces` or `none`; exposure, default `1.0` |

The look (`viewer/look.js`, shared with the face editor's previews) lights the head with a three-point rig (warm key, cool fill, cool rims that lift it off a dark background), image-based light from a neutral room, and PBR Neutral tone mapping. On the profiles:

| Profile | Adds | For |
|---|---|---|
| `low` | the rig, image-based light, the cornea's catch-lights, two-pass blended hair | weak screens: a Raspberry Pi |
| `medium` | soft shadows from the key light, physical skin (wrap lighting for subsurface scatter, sheen, pore-scale detail) | the default |
| `high` | post-processing: ambient occlusion (GTAO), a little depth of field, light bloom on highlights, fine grain | `record`, which renders offline |

It finds the parts by the names the head recipe gives them: `head` is skin (with the normal and roughness maps the recipe bakes from its colour map, and generated pores over them), `eyes` holds MakeHuman's eyeball and its cornea shell (drawn again as a reflection, for catch-lights that follow the gaze), `wetline` is the recipe's tear line along each lower lid, and `hair` is hair cards. A face without those names still gets the lighting. A transparent background turns the post-processing off, since it would lose the alpha. `window.__mask.look` says what is on.

Lip sync runs on the page's audio clock: audio plays `buffer` ms behind arrival and each frame is shown when the clock reaches its timestamp. Network jitter costs latency, not sync.

Browsers start audio only after a click or key press. A kiosk can waive that; for Chromium, start it with `--autoplay-policy=no-user-gesture-required`.

Until then the viewer says so: a full-screen **Tap to start**. One tap or key press turns the sound on and fills the screen, and a **● Ready** pill shows briefly once the screen is connected and can be heard. `kiosk=1` is for dedicated screens: no status line, no cursor, full screen on that first tap; started with the flag above, it needs no tap at all. The viewer tells the server its state on its socket (`audio`, `display`, `kiosk`, `sound`; never media), and `/health` lists it under `screens`, per face.

### House audio (Snapcast)

For Mithrandir's `face+house` mode the house speakers play the speech and the panel only moves. Synthesise once, send the same audio to Snapcast and to mask (`push`), and open the panel's viewer with `audio=external`:

```
http://<mask>:8109/?face=<id>&display=hld&kiosk=1&audio=external&delay=750
```

- **No double voice:** the page makes no AudioContext at all, so nothing reaches its own speakers, and it needs no tap to be ready (`kiosk=1` still goes full screen on a tap, if wanted).
- **The clock:** with no audio clock, the viewer keeps time with the page's own clock, anchored to the session's timeline exactly as the audio clock would be. Frames are shown at their timestamps, `buffer` + `delay` ms after arrival.
- **Choosing the delay:** Snapcast plays its buffer (1000 ms by default) behind what it is sent, and the viewer already plays `buffer` (250 ms) behind arrival, so start at the Snapcast buffer less 250 ms, then adjust by eye: raise it if the lips lead the sound, lower it if they trail. A driver's lag (Audio2Face holds the audio until its frames cover it) shifts both alike only if mask and Snapcast are fed at the same moment; if the lips trail by a constant amount, take it off the delay.
- **Ready:** the screen reports `sound: "external"`, and `/health` counts it in `screens` with `audio_ready: true` once it is connected: ready to speak in sync, not "can be heard". The editor shows it as "1 screen ready (house audio)".

## Serving the LAN

The server speaks plain HTTP and WebSocket, and binds localhost by default. To put the viewer on other machines:

1. Bind an address your TLS proxy can reach: `argus-mask serve --host <address>`.
2. Proxy to it with TLS. Caddy forwards WebSocket upgrades without extra configuration.
3. Name the proxy's origin in `ARGUS_MASK_CORS_ORIGINS`, for example `https://mask.example.org`.

Both sockets refuse a browser whose `Origin` is neither the server's own host nor in that list. Clients that send no `Origin`, such as the CLI, are allowed. There is no authentication: reaching the port is access.

### Face editor

`argus-mask editor` is a page for making heads. Pick a skin, hair, clothes, a hat, brows, lashes, eyes, teeth and tongue from thumbnails (hover for a large view). Set sex, age, build and ancestry, and shape the head, forehead, brow ridge, eyes, ears, cheeks, nose, lips, chin and neck with 117 sliders in collapsible panels (the filter box finds one by name). Press **Build** and a preview appears about ten seconds later (the preview fades behind a spinner and a timer while Blender works); click it for a large view that can rest, open, smile, talk or turn. Under it, a **Before** card shows the build before this one and what changed between them; the large view shows the two side by side, posed together. **Undo** and **Redo** (Ctrl+Z, Ctrl+Shift+Z) step back and forth through your builds. Every build is kept in a cache (`assets/.preview/.cache/`, the last `ARGUS_MASK_BUILD_CACHE` builds, 40 by default), so going back to one, or putting a slider back where it was, copies it in a fraction of a second instead of running Blender. With **Live update** on, it builds by itself a second after you stop changing things; changes made during a build get one more build, and typing a name or id never builds.

A build is a preview, kept in `assets/.preview/`, which the server never lists. **Export** copies the preview on show into `assets/<id>`, and it is on the server's `/faces` at once. **Download preset** saves the form as a `.json` preset, the same format as `scripts/stock_heads/`, and **Open preset…** loads one.

**Saved faces** (`/faces.html`, linked from the editor) shows every face in the assets root and every preview as a card with a thumbnail. **Review** opens it in 3D with its provenance; **Remix** opens its preset in the editor under a new id, so the original is untouched; **Rename** changes its name or id (the export and the preview together); **Delete** moves it to `assets/.trash/`, which the server skips, so moving it back restores it. Stock and hand-added faces can be reviewed, not changed.

**On screen…** (beside Export, and on each exported face's card) takes a face to a screen. The editor checks that the server is up and lists the face, and says plainly if it does not (the usual cause: the server and the editor were started on different assets directories). It builds the viewer link at the screens' origin, with display, quality, kiosk, status line and background, to open, copy, or show as a QR code for a tablet or panel to scan. It shows which screens are watching and whether they can be heard ("1 screen ready", "1 screen, no sound yet: tap it"), plays a test chime through the server's own source socket (refused, like any source, by a read-only server), and gives the `argus-mask say … --face <id>` command to speak from a terminal. The editor reaches the server at `ARGUS_MASK_SERVER_URL` (`--server-url`, default `http://127.0.0.1:8109`) and hands out links at `ARGUS_MASK_VIEWER_URL` (`--viewer-url`: the TLS proxy on a LAN; default, the server's).

**Broadcast** (`/broadcast.html`, linked from both pages) is where a face speaks. It lists the faces the server serves and whether each is ready (a screen watching that can be heard), already speaking, or has no one to hear it. Pick one, and either **Say** text through the TTS service, with every option `argus-mask say` takes (voice, from the service's own list when it is a Wyoming server, or typed; pause between sentences; temperature, 0.1 to 1.5 or the service's own; seed; stream; the TTS URL; extra `-p key=value` fields), or **Push** a WAV file, as `argus-mask push` does. The editor runs the CLI's own code against the server, so the server's read-only switch and one-source-per-face rule apply as from a terminal; the page shows each step (the session, every sentence, the speech's length, done or why not) and can stop it. The matching command line is shown, to copy. The text goes to the TTS service and is never logged; the page shows its first 80 characters while it plays. **Record to file** records the broadcast as `argus-mask record` does: the recorder listens on the face first, then, once the broadcast ends, renders every frame in a headless browser (frame size 1280×720, 1920×1080 or the HLD's 1080×1920; the viewer's `high` look by default) and muxes the MP4 with the audio, its WAV and take beside it, into `ARGUS_MASK_RECORDINGS_DIR` (`--recordings-dir`, default `./recordings`). The page shows the clip to play or download, and lists recent ones. It needs the `record` extra and `playwright install chromium-headless-shell` where the editor runs; without them the checkbox says so.

**Voices** (`/voices.html`) is where a voice is made. Upload one to three reference clips with their transcripts, or **generate** a synthetic candidate (the TTS service's own speaker, or a described voice on a service that designs them, reading a script you choose), and each is checked as a reference: speech length, sample rate, bandwidth, clipping, level, background noise, edges, pauses, delivery, and whether the transcript fits the audio, each with what to change. **Make the voice** sends the chosen clips and transcript to the TTS service (`/clone-voice`, Miso's contract) and keeps a record in `ARGUS_MASK_VOICES_DIR` (`--voices-dir`, default `./voices`): where the voice comes from (generated, your own, or a consenting speaker's under a release, by its sha256), its delivery, and the clips by hash, kept beside it. A clip with a bad check is not made into a voice. **Try a voice** speaks a line in the browser, no face involved. Voices made elsewhere show as having no record. See [docs/voices.md](docs/voices.md): what a good reference sounds like, recording your own, synthetic voices, and one voice per delivery (warm, brisk, …).

It needs what [scripts/README.md](scripts/README.md) sets up, on the machine that runs it: Blender 4.2 or later, the MPFB extension, and the MakeHuman asset packs (`makehuman_system_assets` and `faceunits01`). None of it ships with argus-mask; the server never needs it.

```bash
argus-mask editor --assets-dir assets \
    --blender $WORK/blender/blender --blender-user $WORK/user    # BLENDER_USER_RESOURCES
```

- **Only licensed assets.** The editor offers the assets that an installed pack lists with a licence, and writes each part's licence and author into the face's provenance (`kind: synthetic`). An asset from a CC-BY pack shows up in the licence line.
- **Adults only.** Age starts at 25 (MakeHuman's 0.5); the recipe refuses less.
- **It only replaces its own faces.** Each face it builds keeps its `preset.json` beside the model; a directory without one (a stock head, a face added by hand) is never overwritten.
- **Builds and exports go one at a time.** A failed one leaves nothing behind, and the face that was there stays.
- **Keep it local.** A build starts Blender. The editor binds `127.0.0.1`, and `/build` and `/export` are behind the same origin and read-only guards as the server's writes.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARGUS_MASK_ASSETS_DIR` | `assets` | Root of the faces |
| `ARGUS_MASK_DRIVER` | `energy` | `energy` or `a2f` |
| `ARGUS_MASK_A2F_URL` | empty | The Audio2Face server, for the `a2f` driver |
| `ARGUS_MASK_TTS_URL` | empty | The TTS stream endpoint, used by `say`, the editor and the server's `POST /speak` |
| `ARGUS_MASK_TTS_VOICE` | empty | The TTS voice `say` and `/speak` ask for: a stored clone's name, or a speaker number |
| `ARGUS_MASK_FPS` | `30` | Animation frames per second |
| `ARGUS_MASK_CORS_ORIGINS` | empty | Browser origins allowed to open the sockets, comma-separated |
| `ARGUS_MASK_READ_ONLY` | off | Refuses every audio source; viewers still connect. The editor refuses builds. |
| `ARGUS_MASK_BLENDER` | `blender` | The editor's Blender binary |
| `ARGUS_MASK_BLENDER_USER` | `$BLENDER_USER_RESOURCES` | Blender's user directory, holding MPFB and its asset packs |
| `ARGUS_MASK_MPFB_DATA` | under the user directory | MPFB's data directory, if it is somewhere else |
| `ARGUS_MASK_BUILD_TIMEOUT_S` | `300` | The editor stops a build that takes longer |
| `ARGUS_MASK_BUILD_CACHE` | `40` | Builds the editor keeps (about 6 MB each), so undo and repeats skip Blender; `0` keeps none |
| `ARGUS_MASK_SERVER_URL` | `http://127.0.0.1:8109` | How the editor reaches the server, to check a face and send a test |
| `ARGUS_MASK_VIEWER_URL` | the server URL | The origin screens open the viewer at (the TLS proxy, on a LAN); the editor's links point here |
| `ARGUS_MASK_RECORDINGS_DIR` | `recordings` | Where the Broadcast page's recordings go (an MP4, its WAV and take) |
| `ARGUS_MASK_VOICES_DIR` | `voices` | The Voices page's checked clips, and each voice's record and clips |

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

`--voice` (default `ARGUS_MASK_TTS_VOICE`) picks the voice; [docs/voices.md](docs/voices.md) covers making one. A name is a stored clone (`voice_clone_id` on the HTTP contract, the voice's name on Wyoming, where agent-composer's `clone:` prefix is also understood); a number is one of the service's own speakers. Any `-p speaker=…`, `voice_clone_id=…` or `voice=…` you give wins over it.

```bash
export ARGUS_MASK_TTS_URL=http://127.0.0.1:8034/synthesise/stream
export ARGUS_MASK_TTS_VOICE=inara-nobility
argus-mask say "Hello from the other side of the room." --face stock-01

argus-mask say "Or through a Wyoming server." --face stock-01 --tts-url wyoming://192.168.1.20:10200 --voice "Claribel Dervla"
```

`say` sends the text one sentence at a time and leaves `--pause` ms of silence between sentences (default 300; twice that between paragraphs, which are blank lines in the text). TTS models run sentences into each other, and a short request wanders less than a long one. `--pause 0` sends the text whole. By default `say` also tidies the speech (`argus_mask.speech`): numbers, times, dates and symbols go as words ("9:30am" as "nine thirty a m"), since a model guesses at digits; a sentence of one to three words goes with the next, since a lone "Greetings." comes back furthest from the rest in pitch and level; and each sentence is faded in and out (5 ms, 30 ms) and brought to the same speech level (-20 dBFS, within 9 dB), since a model often stops while the last sound is still at full level, and sentences generated apart differ in loudness. With `--stream` there are fades but no levelling. `--as-is` (the Broadcast page's **Tidy speech** checkbox) sends the text as written and plays the audio as it came. It also puts 150 ms of silence before the utterance and 300 ms after it: TTS models start speaking on their first sample, and an output device that has to wake up would otherwise eat the first syllable.

## Recording a clip

Screen capture fights the viewer for the machine and leaves the audio to the operating system. A take is deterministic, so a clip need not be captured at all:

```bash
uv sync --extra record && uv run playwright install chromium-headless-shell   # once
argus-mask record --face stock-01 --out clip.mp4 &
argus-mask say "Hello from the bridge." --face stock-01
```

`record` listens on the face's viewer socket, keeps the next session's header, frames and audio, renders every frame in a headless browser with the viewer's `high` look (`--quality low|medium|high`) at `--width`×`--height` (default 1280×720; `--display hld --width 1080 --height 1920` for the portrait panel), and muxes them with the audio using the `ffmpeg` on PATH or the one the extra brings. The session's take and WAV are written beside the clip. Rendering runs at a few frames a second in software; a 20 s clip takes a minute or two and never drops a frame.

## Wire

`schema/mask-wire.schema.json` is the contract. A viewer socket carries:

- one `header` (JSON): session, fps, seed, the order of the blendshape weights, the audio format;
- `frame` messages (JSON): `seq`, `t_ms` on the audio timeline, `weights`, `head` rotation in degrees;
- audio chunks (binary): 8 bytes of little-endian float64 giving the chunk's start in ms, then mono `f32le` PCM. The server sends a chunk only once frames reach its end, so a driver that answers late delays the sound, never the mouth;
- one `end` (JSON).

A take file is the same header, frames and end, one JSON object per line.

A source socket takes one JSON hello (`sample_rate`, `channels`, `format`, optional `seed`), then binary PCM, then `{"type": "end"}`. Sources should send no faster than real time plus about a second; a viewer that falls 2048 messages behind is dropped.

## Agent clients: HTTP push and speak

An agent that would rather not speak the socket protocol can drive a face over plain HTTP. Both routes are sources like any other: one source per face (`409` if the face is taken), refused by a read-only server (`403`) and by another site's page (the same origin guard as every write), refused with the reason when the driver is down (`503`; never a fall-back to `energy`). Each request lasts as long as the speech, played at its pace (real time, about a second ahead, as the CLI sends), and answers with the stream's closing numbers once the face is free again:

```json
{"session_id": "54a3d9b0…", "frames": 90, "duration_ms": 3000}
```

**`POST /push?face=<id>`** takes audio the agent already has. The server never sees any text. The body is a WAV file, or raw PCM described by the same headers a TTS stream carries (`X-Sample-Rate`, `X-Audio-Format`: `pcm_f32le` or `pcm_s16le`, `X-Channels`); at most 25 MB. Query: `seed` (the idle layer's, default 0), `require_screen` (`true`: `409` unless a screen watching the face can be heard, see `/health` `screens`).

```bash
curl -X POST "http://127.0.0.1:8109/push?face=stock-01&require_screen=true" \
     -H "Content-Type: audio/wav" --data-binary @briefing.wav

# raw PCM, as a TTS service streamed it (24 kHz mono float32 here)
curl -X POST "http://127.0.0.1:8109/push?face=stock-01" \
     -H "X-Sample-Rate: 24000" -H "X-Audio-Format: pcm_f32le" -H "X-Channels: 1" --data-binary @speech.f32
```

To play the same speech in the house (Snapcast) and on the face, synthesise once and send the same audio to both.

**`POST /speak`** takes text, as `argus-mask say` does, through the server's own TTS service (`ARGUS_MASK_TTS_URL`; `422` if none is set). The text goes to that service and is never logged. JSON body: `face`, `text`, and optionally `voice` (default `ARGUS_MASK_TTS_VOICE`), `temperature`, `seed`, `pause` (ms between sentences, default 300), `tidy` (default `true`: numbers as words, short sentences joined, even loudness, soft edges), `require_screen`. The speech is fetched whole before the face is claimed, so a slow TTS delays the start, never the mouth.

```bash
curl -X POST http://127.0.0.1:8109/speak -H "Content-Type: application/json" \
     -d '{"face": "stock-01", "text": "Good morning. Three things to look at today.", "temperature": 0.7}'
```

From Python, `argus_mask.client.push` and `say` drive the socket directly and report each step as it happens.

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

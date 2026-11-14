# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Consent records are argus-cortex's** (#83). `ConsentRecord`, `AgeCheck`, `Revocation`, `ConsentVerdict`,
  `consent_valid` and `consent_matches` now come from `argus_cortex.consent` (argus-cortex 0.5.0, now required).
  Mask keeps the store, `consent revoke` and the checks where it serves and speaks. Differences from mask's own record:
  - a record whose age check names nobody, or whose revocation is dated before the signing, is refused;
  - a face's subject and a voice's speaker must match the record's exactly, spaces at the ends aside (no longer
    case-insensitive);
  - record dates are read as UTC days, and `consent revoke` dates a revocation by the UTC day, so it stops use at
    once east of UTC too;
  - a record lists its uses in cortex's order (avatar, video, speech, training).

  `consent new` still refuses delivery names the Voices page wouldn't use (`Warm`) and overlong text.

### Added

- **Consented faces and voices** (#26). A real person can be the avatar under a release written for this use:
  `docs/likeness-release.md` (a template for a lawyer to review), `docs/consented-faces.md` (the face capture
  protocol and reference images), `docs/voices.md` § A consenting speaker (the voice session) and `docs/consent.md`.
  `argus-mask consent new` writes a consent record (`<assets>/.consent/<release_sha256>.json`: the release's, reference
  images' and reference recordings' hashes, never the documents, media or ID; the face, the voice and its deliveries;
  uses `avatar`, `video`, `speech`, `training`; dates; revocation terms), `consent list` shows them, and
  `consent revoke` ends one: nothing under the release is served or spoken from its effective day, its voices are
  deleted from the TTS services they were made on and moved to the trash, and its faces are moved to the trash. It can
  be re-run if a service was down. A `consented` face is served only while its record covers it (`avatar`; `record`
  also needs `video`, asked of the new `GET /face/{id}/consent`). The Voices page makes a consented voice only with
  a record that covers it and its delivery. `say`, `POST /speak`, the Broadcast page and **Try** refuse one the
  record no longer covers. The types (`ConsentRecord`, `consent_valid`, `consent_matches`) are the definition
  argus-cortex#31 moves into cortex.
- **The disclosure label.** A consented face's provenance carries `disclosure`, and a stream in a consenting
  person's voice carries it in its header (`SourceHello.disclose`, set by `say` and Broadcast). The viewer shows it on
  every screen, kiosks and `?hud=0` included, and in every recording.
- **Generated faces say how:** `Provenance.generator` (the model, its sha256, the seed and the prompt), for
  `synthetic` and `personal` faces only; `inspect --generator … --generator-sha256 … --generator-seed … --prompt …`.
- Wire contract 1.3 (additive): `Provenance.generator` and `disclosure`, `SourceHello.disclose`,
  `StreamHeader.disclosure` (left out of a take when unset, so takes are unchanged), `ConsentVerdict`.

- **Hair colour** (part of #59). A preset's `hair_colour` bakes a colour into the hair's texture at build time:
  `eumelanin` (brown to black) and `pheomelanin` (red to ginger), each 0 to 1, as in a physically based hair shader,
  plus an optional `dye` mixed over them by `dye_strength`. The texture keeps its own strand variation and its alpha.
  `parts` can add the brows and lashes. The editor has a Hair colour panel under the hair picker, with a swatch of the
  colour the build will bake. The model is `editor/hair_colour.py`, pure numpy: the recipe loads it by path and the
  tests import it, so both run the same code, and it is part of the build cache's key. Takes and the viewer are
  unchanged. Presets without `hair_colour` build as before.

## [0.1.0] - 2026-11-13

### Added

- **Splat heads**: a LAM Gaussian-splat avatar (a `.zip`: `skin.glb`, `offset.ply`, `animation.glb`,
  `vertex_order.json`) is a face like any other: served (`GET /face/{id}/model`, and `/face/{id}/model/{file}` by its
  own name), shown by the viewer and the editor's Saved faces previews, driven by mask's frames, rest sets, idle life
  and lip budget (measured from the archive's `skin.glb`, with a jaw cap for the weak splat mouth: `?jaw=`), and
  recorded frame-exact by `record` and the Broadcast page. The manifest's model gains `format` (`gltf`, the default,
  or `lam-splat`); the wire contract is 1.2, and older manifests read as before. The archive is checked in memory,
  never unpacked to disk: one folder of known members, size, count and compression-ratio caps, no path tricks or
  links, ARKit morph targets only. `inspect` takes a `.zip` (and `--write-manifest` for it, with the splat framing).
  The viewer draws one with Myned AI's gsplat-flame-avatar-renderer 1.4.0 (MIT) and JSZip 3.10.1 (MIT), vendored
  with three.js r187's `OrbitControls.js`; `window.__mask.renderer` says `splat` or `mesh`. LAM's weights (CC BY-NC
  4.0) and FLAME are non-commercial, so a LAM head is `personal`, never cleared. `display=hld` refuses a splat face
  for now.
- **Personal and cleared faces and voices** (#19). Provenance gains `personal` (your own use, anything lawful in
  private, such as a cartoon or a character LoRA; needs only a `source`) and `consented` (a real person or owned
  character under a release: `subject` and `release_sha256`). Every provenance carries a derived `cleared`: true for
  `stock`, `synthetic` and `consented`. `/faces`, the Saved faces page and the Broadcast page show it.
  `inspect --write-manifest` defaults to `personal`; `stock` and `synthetic` still need a licence. The Voices page has a
  **Personal use** source that records where the clips came from and is never cleared. The wire contract is 1.1.
- **The face follows the voice** (#14, #15). A new `prosody.py` reads each frame's loudness and pitch from the audio
  and marks stressed syllables, phrase ends and phrase starts. In takes:
  - **Nods and brows.** The head nods on stressed syllables (up to about 2.5 degrees), settles with a small tilt at
    each phrase end (alternating sides), and the brows rise on emphasis, about once a phrase, added to the driver's own.
  - **Blinks in pauses.** Most blinks come just as a phrase ends; none starts while a word is sounding unless the last
    was more than 8 s ago. About 18 a minute, sometimes doubled.
  - **Saccades.** The eyes jump and hold for 0.2-0.6 s with a slight drift. A phrase start or a pause sometimes takes
    them away (to the side, or up, as people look away to think), and the head turns a little with them.
  - **Micro-motion.** Slow, small changes of brow tension, and of mouth tension while not speaking.

  It is all a function of the audio and the seed, so takes stay byte-identical. On five synthetic voices: 1.8 to 2.0
  stresses a second of speech, brows raised in 7-20% of frames, and every blink starting in a pause or between words.
  Analysis costs about 0.3% of real time.
- **Rest sets and idle life**: a face at rest shows a named baseline expression, `warm` by default (also `neutral`,
  `calm`, `bright`, `attentive`, and your own in the assets' `expressions.json`), chosen per face on Saved faces and
  served as `GET /face/{id}/rest`; a screen may override it (`?rest=`). While speaking, the set's brows, eyes and cheeks
  hold and its mouth weights go at half. At rest the face blinks (some doubled), glances with the head following,
  breathes, and shifts expression now and then (`?life=0` to `2`). Viewer-side only: takes are unchanged.
- **Lip budget**: the viewer measures how each mouth shape moves the gap between the lips, from the face's own mesh,
  and scales the lip-closing shapes so the lips never pass through each other (1 mm allowed for pressing), with
  `mouthShrugLower` and the rolls at 0.7. On ginger-3, lips overlapping by more than 3 mm went from 12% of frames to
  none, and the lower lip no longer rides up over the upper. Screens, `record` and the editor's previews alike.
- **Smooth playback between frames** (#13): screens and the editor's `speak` preview draw a cubic through the
  frames on either side instead of a straight line between two, so 30 fps motion has no corners (the speed jump at
  each frame, up to 0.44 of a weight per frame on Audio2Face output, is now none). Weights use a monotone cubic, so a
  mouth never overshoots a closure; the head turns as Catmull-Rom. The curve passes through every frame at its own
  time: lip timing is unchanged, and `record` (which renders at the frame times) shows exactly the frames, as before.
- **The editor's previews speak**: the `speak` pose (now the default) loops 12 s of real Audio2Face motion, with the
  lip budget, rest set and idle life a screen shows, instead of a sine-wave mouth; a `rest:` picker previews each set.

- **Named TTS speakers and delivery instructions**: a voice written `speaker:<name>` (`--voice`, `ARGUS_MASK_TTS_VOICE`,
  `/speak`'s `voice`) is sent as the service's own speaker, not as a clone, for services that name their speakers, such
  as Qwen3-TTS's CustomVoice (`speaker:Vivian`). `POST /speak` takes an optional `instruction`, how to say it, passed
  to a service that reads one; like the text, it is never logged.
- **Face aliases** (#41): a stable name for a face, such as `holly` for a persona, that the agent speaking and the
  kiosk showing it use for good while the face behind it is chosen in the editor (Saved faces → Aliases: add,
  repoint, remove, open on a screen). Kept in the assets' `aliases.json`, written by the editor and read by the server
  whenever it changes, so its read-only mount and no restart. Every route that names a face takes an alias; sources
  and screens meet on the name they use; `GET /aliases` and `/health`'s `aliases` list them, `/faces` is unchanged. A
  screen watching an alias that is repointed loads the new face on the next stream, whose header names the face
  itself. An alias never hides a face, points only at a served face and never at another alias; renaming a face
  carries its aliases, deleting one an alias points at is refused.
- **Record a broadcast from the Broadcast page**: a Record to file switch (frame size, look) for Say and Push. The
  recorder listens on the face first, the broadcast goes, then the clip renders and muxes, as `argus-mask record` does,
  into `ARGUS_MASK_RECORDINGS_DIR`; the page plays it, offers it to download and lists recent clips. `record` now
  tells its caller it is waiting only once its socket is open, and muxes off the event loop.
- **Face voices** (#37): the editor's **Voices** page checks reference clips for a cloned voice (speech length, sample
  rate, bandwidth, clipping, level, noise, edges, pauses, delivery, whether the transcript fits), each check with a
  verdict and what to change (`argus_mask.voice`, pure numpy); generates synthetic candidates from the TTS service's
  own speaker or a described voice, reading a script per delivery; makes the voice on the service (`/clone-voice`)
  with a `voice.json` record of where it came from (generated, your own, or a consenting speaker's under a release,
  by its sha256) in `ARGUS_MASK_VOICES_DIR`; refuses unsuitable clips and ids the service already has; tries a line
  in the browser. `docs/voices.md`: the reference, recording your own, synthetic voices, one voice per delivery.
- **`say` tidies speech** (#36, on by default, `--as-is` and the Broadcast page's Tidy speech turn it off): numbers,
  times, dates and symbols go as words (`speech.speakable`), a sentence of one to three words goes with the next, and
  each sentence is faded in and out and brought to the same speech level. Recordings through Miso had invented
  words for digits, one-word sentences far off in pitch, clicks at sentence ends, and 9 dB between sentences.
- **`POST /push` and `POST /speak`** on the server (#38), for agents: a WAV or raw PCM body (or text through the
  configured TTS) plays into a face exactly as a source does, paced, one source per face, behind the same guards;
  `require_screen=1` refuses when no screen can hear it. The server never sees text with `/push`.
- **External sound for house speakers** (#39): `?audio=external` (or `mute=1`) makes a viewer animate on the session's
  timeline without playing anything and without a tap; `?delay=<ms>` lines the lips up with a speaker system's
  buffer (Snapcast). `/health.screens` gains `sound: local | external`; an external screen is ready once connected.
  The On screen dialog sets both.
- **`say --context`** (#40, opt-in): each sentence goes to an HTTP TTS with the one before it, text and levelled
  audio, as Miso's `context` (voluptas#4); a service that refuses the field is asked again without it. Off by default:
  with a cloned voice on Miso, chaining the audio made pitch drift up and loudness sink over a briefing, where
  sentences sent alone held steady. The Broadcast page has it as an experimental checkbox.
- **Recording presets** (#35): 1080p, 720p and 480p, and the HLD portrait panel; an info tooltip on temperature.
- **Temperature on the Broadcast page**: a slider (0.1 to 1.5, Miso's range), left to the service's own value unless
  set; it goes as the request's `temperature` (`-p temperature=…`). Over Wyoming, `-p` fields other than the voice's
  now ride beside the text in the `synthesize` event, for a bridge to pass on, instead of being dropped.
- **Voices from an HTTP TTS**: for a `…/synthesise[/stream]` URL (Miso, agent-composer's TTS servers) the editor lists
  `GET …/speakers` and `GET …/clones`, named as the Wyoming bridge names them (`0`, `clone:<id>`), so the Broadcast
  page's voice picker works talking to Miso directly.
- **Broadcast page** in the editor: the server's faces with their readiness, and a form with every `say` option
  (voice listed from a Wyoming service, pause, seed, stream, TTS URL, `-p` fields) or a WAV to `push`, run through
  the CLI's own client with live progress, Stop, and the matching command line. `client.tts_voices()` asks a Wyoming
  service for its voices.
- **Editor to screen** (#28). On screen… in the editor and on Saved faces: checks the server lists the exported
  face (and says plainly when it does not), builds the viewer link at `ARGUS_MASK_VIEWER_URL` with display, quality,
  kiosk and background, to open, copy or scan as a QR code, shows whether screens are watching and can be heard,
  plays a test chime through the server's source socket, and gives the `say` command. The viewer shows **Tap to
  start** until its sound is unlocked and a brief **Ready** after, takes `kiosk=1`, and reports its state on its
  socket (`ViewerStatus`, in the wire schema); `/health` lists `screens` per face. The viewer's reconnect backoff
  now grows as it was meant to.
- **Stock heads republished**: `stock-01` revision 4 and `stock-02`'s first lock, both built by the recipe with the wet
  line and baked skin maps (about 10 MB each). `fetch_stock_head.py --id stock-02` fetches the second.
- **Recipe: wet line and skin maps** (#11, #9). A tear line along each lower lid (`wetline`, found by casting
  rays at the eye, moving with the lid under every shape key), and normal and roughness maps baked from the skin's
  colour map into the `.glb`. The viewer gives the wet line a fading glossy film and lays its generated pores over
  the baked maps.
- **Two-pass hair** (#12): an opaque core, then the fringe blended over it. On a 10 s recording, 26% less edge
  shimmer and 59% fewer edge pops than the cut-out; alpha to coverage alone managed 6%. The default in every profile.
- **Triangle budget 45k** (was 35k), while the weakest panels are not targets.
- **Viewer look** (`viewer/look.js`, issue #20 tier 1): a three-point light rig, image-based light and PBR Neutral
  tone mapping (#8); soft shadows from the key light (#8); physical skin with wrap lighting for subsurface scatter,
  sheen and a generated pore-scale detail map (#9); catch-lights from the eyes' cornea shell, which follow the
  gaze (#11); alpha-to-coverage hair edges (#12); and post-processing (GTAO, depth of field, bloom, grain, optional
  SMAA) in `low`, `medium` and `high` profiles with a URL switch per effect (#10). `record` renders with `high`
  (`--quality`). The editor's previews use the same look.
- **Triangle budget 35k** (was 30k): a dressed head (a uniform is ~5k triangles, MakeHuman's teeth 7k) ran just over it.
- **Editor: compare, undo, build cache.** A Before card beside the preview shows the previous build and what
  changed; the large view shows both side by side, posed together. Undo and redo step through the builds. Each
  build is cached by what it was made from (the preset without id and name, the recipe, the Blender), so a repeat
  copies in well under a second instead of running Blender: `GET /cache/{key}` serves an earlier build directly.
  `ARGUS_MASK_BUILD_CACHE` sets how many are kept (40).
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
  MPFB, about ten seconds a build, with a posed preview. Ages 18 and up; every head wears an outfit. Skin, hair,
  clothes and a hat are picked from thumbnails;
  117 shape sliders (head, forehead, brow ridge, eyes, ears, cheeks, nose, mouth, chin, neck) over MakeHuman's CC0
  modelling targets, from `modifiers.json`, generated from MPFB's catalogue. Collapsible panels, a filter, a hover
  preview of each asset, a preview card that opens a large view, a spinner and timer while a build runs, and
  **Live update**, which builds a second after the last change. A build is a preview (`assets/.preview/`);
  **Export** puts it in `assets/<id>` and on `/faces`; **Download preset** saves the form and
  **Open preset…** loads one. A **Saved faces** page reviews, remixes, renames and deletes (to `assets/.trash/`) the
  faces the editor made, and reviews the others. It offers only assets an installed pack licenses, writes their
  licences into the face's provenance, replaces only faces it made, and is behind the write guards. Blender, MPFB and
  the asset packs are the user's to install; none ships.
- **A second stock preset**, `stock-02`: a female head, about 25, short brown hair, in the same crew uniform as
  `stock-01` (suits02 pack, CC0). Not published yet.
- **`argus-mask record`**: waits for the next session on a face and renders it to a video (MP4 or WebM), offline and
  frame-exact, in a headless browser, muxed with the session's audio; the take and WAV are written beside it. The
  viewer gets a `?record=1` mode posed from outside. Needs the `record` extra (playwright, imageio-ffmpeg).
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

- **A provenance says what it means** (#64). An unknown field is refused, not dropped in silence, and `licence` is
  read as `license` (a provenance with both, disagreeing, is refused). A `source` of only whitespace is refused like a
  missing one. A dumped record's `cleared` still reads back, and is recomputed.
- **A source that sends faster than real time is slowed, not cut off** (#2). The `a2f` driver lets up to 2 s of audio
  wait on frames, then holds the source until the model catches up; a burst of a 15 s clip used to end the stream with
  "fell more than 10s behind". A model that sends no frame for 10 s while the source is held still ends it.
- **The Audio2Face server reports a dead worker at once** (#4), not on the client's next message: a worker that crashed
  during a pause in speech left the face frozen with no error.
- **`POST /speak` in Docker**: `compose.yaml` passes `TTS_URL` and `TTS_VOICE` from `.env` to the server as
  `ARGUS_MASK_TTS_URL` and `ARGUS_MASK_TTS_VOICE`; without them `/speak` always answered `422`.
- A Broadcast recording that caught another source's session on the face is discarded, not kept as this broadcast's,
  and a failed or stopped render no longer leaves its frames behind (#35).
- `say` (without `--stream`), `/speak` and the Voices page's Try buffered a TTS reply without limit: they now refuse
  past 10 minutes of speech, saying to send less or stream (#3). The client pacing test has a wider margin (#5).
- `say --voice <name> -p speaker=<n>` against a Wyoming service dropped the voice name: a speaker now goes with the
  voice (Wyoming's `{name, speaker}`), and only a name in the request stands instead of `--voice`.
- **`say` no longer dumps a slow TTS model's speech all at once.** It fetches the whole utterance before claiming the
  face and plays it at the speed of speech (`--stream` keeps the old behaviour for a service faster than real time),
  and a source's pacing clock now starts at its first chunk rather than at connect.

- **Audio no longer runs ahead of the mouth with a slow driver.** Audio2Face answers about 0.3 s behind the audio it
  is given. The server now holds each audio chunk until frames reach its end, and ends a stream whose driver falls
  more than 10 s behind.

[Unreleased]: https://github.com/smk762/argus-mask/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smk762/argus-mask/releases/tag/v0.1.0

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A plate's depth map waits for a busy darkroom behind the GPU broker.** With the argus-iris broker
  (argus-darkroom#53), darkroom answers a full queue with `409 ... already queued or running`, not `already running`.
  `POST /depth/{id}` retried only the second, so a depth map failed the plate instead of waiting. Both are retried now;
  any other 409 (no stored image) is still an answer.

### Added

- **Docs in the hosted demo** (#123). mask.dragonhound.dev/docs/ is the README and `docs/*.md`, rendered at the
  deployed commit with a version header (version, commit, PyPI, GitHub), recorded in `docs/sources.json` with a
  sha256 per file. The build fails closed on an opsec hit, and on clips the engine has changed since. The demo page
  gets a Docs link and a "Made with" line from each clip's record, and unknown paths get a real 404. The demo now
  deploys from `dev`, not `main`.
- **Hosted demo** (#118). `demo/`: a static page that replays recorded clips (a take and its audio) with the viewer's
  own modules, with pause, scrub, captions and what made each clip; `demo/make_clips.py` makes the clips (TTS as
  `say`, Audio2Face as `drive`), `make demo` builds the site, and `pages.yml` deploys it to mask.dragonhound.dev.
- **Gestures** (#62). A waist-up or full-body face talks with its hands: over some phrases one hand or both come up
  in front of the waist, palms open, and strong stressed syllables stroke them down, landing on the syllable. Which
  phrases and which hand follow the face's seed, so a take gestures the same way every time and `record` reproduces
  it. The engine puts each frame's speech cue on the wire for this: `Frame.cue` (a stress, a phrase start or end),
  only on the frames where one began.
- **Full-body stock heads** (#62). `stock-01` revision 6 and `stock-02` revision 3 wear boots (`shoes03`, CC0) and
  come with `full.glb` beside `head.glb` and `waist.glb` (on the `stock-heads` release; `fetch_stock_head.py`
  fetches all three). All three models are rebuilt, so no skin shows through the uniform.
- **Full body** (#62). A preset's `framings` may ask for `full`: the same character head to shoes, nothing cut, with
  twice the triangle budget. It must wear shoes: they are their own slot in the library and the editor (**Shoes**, and
  **Full body too** under Advanced), and a full-body preset without them is refused. Screens show it with
  `?framing=full` (and `record --framing full`): feet together from MakeHuman's A-pose, the soles flat, the same
  breathing and weight shifts as waist-up with the thighs taking back the hips' sway so the feet stay planted, and a
  soft contact shadow under the feet (no shadow map, so on every profile).

### Changed

- **Skin under clothes is hidden by the recipe** (#62). Below the neck, skin a garment covers (cloth within 3 cm out
  along its normal) or pokes through (cloth within 12 mm in) is deleted: under the coverall the shins showed through
  at the knees. Every framing has fewer triangles for it.

- **Accessories** (#61). A preset's `accessories` adds small metallic pieces the recipe makes itself (CC0) and
  places on the dressed body: a `badge` on the chest, collar `pins`, an `earpiece` at the ear, each on the wearer's
  left or right, in a colour, riding on a bone (so the earpiece turns with the head and the badge breathes with the
  chest). The editor has an Accessories panel.
- **More triangles when asked for.** A preset's `max_triangles` (the editor's Triangle budget) now goes up to
  120,000; 45,000 stays the default and the weak-screen budget `inspect` warns about. Waist-up gets 1.5x, within
  the same ceiling. Garments are only thinned to fit whatever budget is set.
- **Garment colourways** (#61). A preset's `garment_colours` recolours a garment at build time: `tint` keeps the
  fabric's shading and detail, `solid` paints it one colour and so removes print the texture carries (logos, emblems).
  The editor's Clothes panel has a Colour row for the outfit worn.
- **Heavier garments fit the budget** (#61). Over the triangle budget, the recipe thins the clothes alone (Blender's
  Decimate, one ratio, never below 40%), carrying a garment's own shapes (a collar's `jawOpen`) over, so the community
  coverall and duty uniform build waist-up within 45,000 triangles.
- **Waist-up stock heads.** `stock-01` and `stock-02` come with a `waist.glb` beside the head (published on the
  `stock-heads` release, same revisions); a lock file lists each extra framing's url and size, and
  `fetch_stock_head.py` downloads and checks every model before writing any.
- **Relaxed arms for waist-up faces.** The arms no longer hang straight with flat hands: the elbows bend a little,
  the forearms turn the palms toward the thighs and the fingers curl. Each arm drifts by about a degree at the
  shoulder and elbow on its own slow period, so the two never move in step.
- **Life below the neck** (#62, step 2b). A waist-up figure breathes (chest and shoulders, a 4.4 s breath), shifts
  its weight slowly (pelvis and spine, two unrelated periods, the head kept level) and lets its chest follow a nod.
  Viewer-side (`rig.js` `BodyLife`), a function of the face's seed and the time: the posed frame's in a recording.
- **Waist-up framing** (#62, step 2a). A preset's `framings: ["waist"]` (the editor's Advanced → **Waist-up too**)
  builds the same face again from the hips up, arms and all, as `waist.glb` beside `head.glb`, with a camera from the
  build's extent; the manifest lists it under `framings` (wire contract 1.5) and the store checks it like the head.
  Screens show it with `?framing=waist` (arms posed down from the A-pose; the head still turns at the neck), and
  `record --framing waist` records it. Saved faces → Review has head / waist up buttons.
- **Rigged stock heads:** `stock-01` revision 5 and `stock-02` revision 2, built with the rigged recipe and
  published on the `stock-heads` release; `make head` and `fetch_stock_head.py` fetch them.
- **Rigged heads: the head turns at the neck** (#62, step 1). The recipe skins every head to MPFB's `game_engine`
  rig (53 bones): skin, body parts and clothes, with the face shapes unchanged. The viewer and the editor's
  previews turn a rigged head through `neck_01` and `head` (40/60, about world axes), so the shoulders stay put
  where the whole bust used to swing. A head without the bones turns whole, as before. About 0.3 MB more per head.
- **Backdrops behind splat faces.** A LAM head now shows `?scene=`, its own default and `record --scene`: the splat
  renderer draws the backdrop behind the head on the same canvas, so recordings stay frame-exact. Over a backdrop
  the camera moves in to a head-and-shoulders shot that ends above LAM's fade at the collarbone, so the head does
  not float. The editor's Saved faces previews it.
- **Plates: backdrops from a still image** (#55). `argus-mask plate generate` asks argus-darkroom for a picture
  under its SFW policy, then for darkroom's depth map of the same record (Depth Anything V2 Small), checks
  both against darkroom's sha256, and writes `background.json` with provenance (`PlateManifest`, wire contract
  1.4). `plate depth` adds the depth map to a plate made without one, and `plate check` checks one. The viewer's
  `scenes/plate.js` animates a plate with built-in effects: drift with parallax over the depth map, sway,
  dappled light, a light pulse, softness and exposure, all functions of the scene's time.
- **`?scene=garden`**: an overgrown glass conservatory, the first plate, from FLUX.1 schnell (Apache-2.0)
  through darkroom, with darkroom's depth map for parallax. `argus-mask record --scene garden` too.
- **Plates of your own, served from the assets** (#55). The server serves each plate in
  `<assets>/backgrounds/<name>/` that passes `plate check` (`GET /scenes`, `GET /scene/{name}/background.json`
  and its images, nothing else), and refuses the rest. `?scene=<name>` and `record --scene <name>` show it.
  `backgrounds` is a reserved face id.
- **A face's own backdrop** (#55). Saved faces → Review → Backdrop previews a scene behind the face and, for an
  exported face, keeps it in `<assets>/scenes.json`; it follows a rename and goes with a delete. Screens show it
  unless they ask for another (`?scene=`, or `?scene=none` for a plain background), and so does `record`
  (`--scene none` to leave it out). `GET /face/{id}/scene` (`FaceScene`).
- **`?scene=lake`**: an alpine lake at dusk, the third scene, a plate made with `argus-mask plate generate`
  through darkroom in one step (FLUX.1 schnell, Apache-2.0, with darkroom's depth map).

### Fixed

- **A recorded frame could be the one before** (#108). `pose()` resolved inside the draw, before the browser had
  presented it, so a screenshot taken at once could show the previous frame (an idle half-blink, the last pose).
  It now resolves two frames later, for meshes and splats. A new test records a real stock head twice (head and
  waist) and allows only the GPU's off-by-one blending of a few pixels; it runs where `assets/stock-01` exists.
- **Two recordings of a real face differed in about 10,000 pixels.** GTAO's denoiser drew its noise texture from
  `Math.random` on each page load, so the ambient occlusion around the eyes, mouth and collar changed between runs.
  It is seeded now; what remains is a few pixels of blended hair, off by one level, from the GPU.
- **The look finds a rigged head's parts.** three.js renames a mesh that shares its name with a bone (`head` →
  `head_1`), which lost the skin treatment; parts are now matched by the name in the file.
- **A splat face from `inspect` kept the wide shot over a backdrop.** `inspect` writes LAM's standard framing into the
  manifest, which counted as the face's own, so the close head-and-shoulders shot never applied. LAM's standard
  framing now counts as not set.
- **A splat face's review on Saved faces timed out** ("did not load in 20 s") when opened while the thumbnails
  were still loading: two splat renderers building at once stall each other, so the page's stages now take turns.
- **Two recordings of one take match at `high`** (#81). Film grain's noise followed the post chain's wall clock, so
  it depended on how long the browser took to draw. In record mode it now follows the posed frame's `t_ms`; live
  screens keep the clock. A test records one take twice in headless Chromium and compares the frames byte for byte.
- **The image reports the release's version** (#88). The 0.2.0 image said `0.2.1.dev0+g8e0618f5c.d20261114` at
  `/health`: `.dockerignore` left the tracked `.github/` out of the build context while keeping `.git`, so hatch-vcs
  saw a dirty tree. `.github/` is back in the context, the release passes the tag to the build as `VERSION`, and the
  release fails if the pushed image's `/health.version` is not the tag.

## [0.2.0] - 2026-11-14

### Added

- **Consented faces and voices** (#26, #83). A real person can be the avatar under a release written for this use:
  `docs/likeness-release.md` (a template for a lawyer to review), `docs/consented-faces.md` (the face capture
  protocol and reference images), `docs/voices.md` § A consenting speaker (the voice session) and `docs/consent.md`.
  - `argus-mask consent new` writes a consent record to `<assets>/.consent/<release_sha256>.json`. It holds the
    hashes of the release, the reference images and the reference recordings, never the documents, media or ID; the
    face, the voice and its deliveries; the uses `avatar`, `video`, `speech` and `training`; dates; and the
    revocation terms. `consent list` shows each record and whether it holds today.
  - `consent revoke` ends a release. Nothing under it is served or spoken from its effective day (a UTC day, as the
    record's dates are), its voices are deleted from the TTS services they were made on and moved to the trash, and
    its faces are moved to the trash. It can be re-run if a service was down.
  - A `consented` face is served only while its record covers it (`avatar`); `record` also needs `video`, which it
    asks of the new `GET /face/{id}/consent`. The Voices page makes a consented voice only with a record that covers
    the voice and its delivery. `say`, `POST /speak`, the Broadcast page and **Try** refuse one that the record no
    longer covers.
  - A face's subject and a voice's speaker must match the record's exactly, spaces at the ends aside.
  - The record and its checks (`ConsentRecord`, `consent_valid`, `consent_matches`) are argus-cortex's
    (`argus_cortex.consent`), shared with curator and darkroom. Mask adds its own rules for the records it writes:
    delivery names as the Voices page uses them (`warm`, not `Warm`), and length limits.
- **The disclosure label.** A consented face's provenance carries `disclosure`, and a stream in a consenting
  person's voice carries it in its header (`SourceHello.disclose`, set by `say` and Broadcast). The viewer shows it on
  every screen, kiosks and `?hud=0` included, and in every recording.
- **Generated faces say how:** `Provenance.generator` (the model, its sha256, the seed and the prompt), for
  `synthetic` and `personal` faces only; `inspect --generator … --generator-sha256 … --generator-seed … --prompt …`.
- **Backdrops** (part of #55). `?scene=ftl` draws stars streaking past at warp behind the face, tuned with
  `scene.speed`, `scene.density` and `scene.tint`, and `argus-mask record --scene ftl` records it. A scene is a
  procedural shader module (`viewer/scenes/`), drawn by `viewer/backdrop.js` into a half-float render target capped
  by the look's profile (480 px at 30 fps on `low`) and used as the scene's background, so depth of field softens
  it; it leans the rim lights toward its own light. In `record` its clock is each frame's `t_ms` (the viewer's
  `pose()` takes it as a fourth argument), so recordings of one take match. Flat tier and mesh faces only: `hld`
  stays black and a splat face refuses a scene. `docs/backgrounds.md` is the recipe for adding one.
- **Hair colour** (part of #59). A preset's `hair_colour` bakes a colour into the hair's texture at build time:
  `eumelanin` (brown to black) and `pheomelanin` (red to ginger), each 0 to 1, as in a physically based hair shader,
  plus an optional `dye` mixed over them by `dye_strength`. The texture keeps its own strand variation and its alpha.
  `parts` can add the brows and lashes. The editor has a Hair colour panel under the hair picker, with a swatch of the
  colour the build will bake. The model is `editor/hair_colour.py`, pure numpy: the recipe loads it by path and the
  tests import it, so both run the same code, and it is part of the build cache's key. Takes and the viewer are
  unchanged. Presets without `hair_colour` build as before.

### Changed

- Wire contract 1.3 (additive): `Provenance.generator` and `disclosure`, `SourceHello.disclose`,
  `StreamHeader.disclosure` (left out of a take when unset, so takes are unchanged), `ConsentVerdict`. Manifests and
  takes from 1.2 read as before.
- Requires argus-cortex `>=0.5.0,<0.6` (for `argus_cortex.consent`).

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

[Unreleased]: https://github.com/smk762/argus-mask/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/smk762/argus-mask/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-mask/releases/tag/v0.1.0

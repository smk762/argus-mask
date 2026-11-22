# CLAUDE.md — argus-mask

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

A talking face for the Argus suite: audio in, facial animation frames out, rendered in a browser on any screen on the LAN

Shared code — the target taxonomy, wire-schema versioning, and the local/remote model contract — lives in **argus-cortex**; import it rather than re-implementing.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Must never happen

Each rule names what checks it. A rule marked **UNENFORCED** has a linked issue for its check; give a new rule a check, not more prose. `test_x::name` is `tests/test_x.py`; `must_never` is `tests/test_must_never.py`.

| Must never happen | Enforced by |
|---|---|
| The committed wire schema drifts from `models.py` | CI `argus-mask schema --check` (run `make schema`, commit the result) |
| The same audio, driver and seed give a different take | `test_engine::test_a_clip_gives_the_same_take_every_time`; `must_never::test_the_engine_reads_no_clock_and_no_unseeded_random` (no clock, `uuid` or unseeded RNG in `ENGINE`'s modules) |
| The viewer reads `Math.random` or the clock in record mode | Partial: `test_record::test_a_real_head_records_the_same_twice`, local only (needs `assets/stock-01` and a browser; skipped in CI). Run it after touching `look.js` or the viewer |
| Mask falls back from `a2f` to `energy` | `test_push::test_a_down_driver_refuses_with_its_reason_and_never_falls_back`, `test_server::test_a2f_down_refuses_the_source_and_sends_no_energy_frames` |
| Sound reaches a viewer before the mouth for it | `test_server::test_audio_is_held_until_the_frames_for_it_exist` |
| A viewer sends more than its status (≤ 1 KB): media, a command | `test_server::test_a_viewer_saying_anything_else_is_ignored` |
| A socket opens for another site, or a write runs for one | `must_never::test_every_socket_refuses_another_site`, `::test_every_write_refuses_another_site` (every route of both apps, so a new one is caught) |
| A page served by a host name is trusted without being listed (DNS rebinding) | `test_server::test_a_screen_served_by_name_needs_the_name_trusted` (cortex 0.6's `cross_site_refuse`; a screen at `http://morpheus:8109` needs `ARGUS_MASK_TRUSTED_HOSTS=morpheus`) |
| A live Audio2Face stream waits on its GPU lease, or takes the local host-wide lock | `test_gpu_tenant::test_a_held_lease_never_delays_the_work_and_is_released_after_it`, `::test_the_live_lease_goes_only_to_the_broker_with_no_fallback` |
| A GPU lease kind is used that `gpu-tenant.yaml` doesn't declare | `test_gpu_tenant::test_the_manifest_validates_and_agrees_with_the_leases_mask_takes` |
| A request's path reaches Blender; an asset no pack licenses is offered | `test_editor::test_an_asset_not_in_the_library_never_reaches_blender`, `::test_clothes_not_in_the_library_never_reach_blender`, `::test_the_library_offers_only_assets_with_a_stated_licence` |
| The editor replaces a face it did not make (no `preset.json`) | `test_editor::test_the_editor_never_replaces_a_face_it_did_not_make`, `::test_export_never_replaces_a_face_the_editor_did_not_make` |
| A guided face's step runs on one that refused, failed, could not check or is stale; anything but an advisory is overruled | `test_guided::test_a_step_cannot_run_on_one_that_did_not_settle`, `::test_nothing_but_an_advisory_can_be_overruled`, `::test_a_record_on_disk_that_overrules_a_refusal_is_refused` (the record's own rules, `editor/guided.py`: a step's route writes through `put` and `overrule`, never to `results`) |
| A head under 18 or undressed is built; a full body without shoes | `test_editor::test_a_preset_under_18_is_refused`, `::test_a_preset_with_no_clothes_is_refused`, `::test_a_full_body_wears_shoes` |
| A consented face or voice is used without a record covering it now, for this use | `test_consent::test_a_consented_face_is_served_only_while_its_record_covers_it`, `::test_speak_refuses_a_revoked_voice_before_any_tts` |
| A consent record's dates are read in local time | `test_consent::test_mask_reads_today_as_cortex_does_in_utc` (mask's today is `utc_today()`) |
| A route serves a consent record (only verdicts) | `must_never::test_no_route_serves_a_consent_record` (every GET route of both apps) |
| The disclosure label is hidden by `?hud` or kiosk | `must_never::test_the_disclosure_label_never_depends_on_the_hud_or_kiosk_switches`; `::test_a_consented_face_is_labelled_with_the_hud_off` in a browser (local: the record extra) |
| A revoked person's voices stay on a TTS service | `test_consent::test_revoking_deletes_the_voices_and_trashes_the_faces` |
| A face without provenance is served; `curated` is accepted; `personal` reads as cleared | `test_faces::test_a_face_without_provenance_is_refused`, `::test_a_curated_face_is_refused`, `::test_a_personal_face_needs_only_its_source_and_is_never_cleared` |
| Uncleared material in the repo, its tests or the demo | `must_never::test_every_face_the_repo_ships_is_cleared`, `test_demo::test_every_clip_in_the_spec_is_a_cleared_stock_head_on_a_built_in_scene`, `test_backgrounds::test_a_shipped_plate_holds_up_and_is_cleared` |
| Text from Broadcast or `/speak` is logged | `test_editor::test_say_speaks_through_the_tts_into_the_face`, `test_push::test_speak_says_text_through_the_configured_tts` |
| An asset pack installs from outside the catalogue, unpinned, or overwrites a file | `test_packs::test_only_a_catalogued_pinned_pack_installs`, `::test_a_download_with_the_wrong_hash_is_refused_and_the_mirror_tried`, `::test_an_installed_file_is_never_replaced` |
| A plate is kept that darkroom's sha256 doesn't match, under an unknown licence, or under a built-in scene's name | `test_backgrounds::test_a_record_that_does_not_hold_up_writes_nothing`, `::test_an_unknown_licence_is_refused_before_darkroom_is_asked`, `::test_a_plate_cannot_take_a_built_in_name` |
| A LAM archive is unpacked to disk, or past its caps | `test_splat::test_an_archive_that_breaks_a_rule_is_refused`, `::test_size_caps_hold_for_each_member_all_of_them_and_the_count` |
| A mesh's bound splats are served that don't match it (files, sha256, parts, SPZ length, triangles) | `test_bound_splats::test_splats_that_break_a_rule_refuse_the_face`, `::test_splats_ride_only_a_mesh_and_only_the_build_they_were_bound_to` |
| Demo clips made by an older engine are deployed | `test_demo::test_the_committed_clips_were_made_by_the_engine_at_head` (re-record with `demo/make_clips.py`) |
| A demo or docs page runs an inline script the CSP doesn't hash | `test_demo::test_the_page_pastes_the_strip_and_allows_its_import_map_by_hash`, `::test_a_docs_page_is_csp_neutral_and_links_stay_on_site` |
| A file Blender loads by path imports the package | `must_never::test_what_blender_loads_by_path_imports_nothing_from_the_package` |
| LAM or VHAP code is copied into the repo | `must_never::test_no_file_carries_a_lam_or_vhap_licence_header`, `::test_no_lam_or_vhap_package_is_in_the_repo_or_imported_outside_the_head_builder` |
| A splat head is made outside `lam/`'s checked pipeline | CI's `lam` job (`test_lam.py`, with CPU torch): the preflight's refusals (`lam/preflight.py`: FLAME 2023 Open by sha256, our embedding), `flame_core` against a synthetic FLAME, `overrides.verify()` against stand-ins; `make_head.py` refuses at run time too |
| A version is hand-edited | Structural: `hatch-vcs` writes `_version.py` (gitignored) from the tag |

## Layout

`src/argus_mask/`: the engine is pydantic, numpy, soxr and cortex; the CLI, server and editor are extras, imported lazily.

- `models.py` is **the API surface**: the wire contract and `render_wire_schema()`. A `gltf` model may carry `splats` (`BoundSplats`, #126 route A): an SPZ and a binding beside it, bound to that build (`mesh_sha256`), with every input's licence; `faces.inspect_bound_splats` checks them.
- `server/app.py`: `perform()` is the one source path (the source socket, `POST /push`, `POST /speak`).
- `performance.py` joins resampler, driver, prosody and idle layer; `demo/docs_page.py`'s `ENGINE` lists the modules a take depends on.
- `drivers/a2f.py`'s docstring is the Audio2Face protocol; `a2f/check.py` checks a running server against it.
- `gpu.py`: the leases mask takes for GPU work it asks of others (#125); `gpu-tenant.yaml` declares them for argus-iris.
- `consent.py`: the record and its checks are cortex's (`argus_cortex.consent`); mask owns what the fields mean, so change them in cortex together with #26.
- `viewer/`: static, no build step. three.js r187, the splat renderer and JSZip are vendored unchanged with their licences. The editor's previews import `look.js`, `motion.js` and `splat.js`.
- `editor/`: a separate optional process. `build_head.py` runs inside Blender, handed over by path, never imported.
- `editor/guided.py`: the build record of a guided face (#163), `<assets>/.guided/<id>/build.json`. `STEPS` is the flow and what each step needs. A step's status is derived (its checks, and the digests of the results it ran on), never stored: add a step's result with `GuidedStore.put`, and it goes stale by itself when an earlier step finds something else.
- `bake/` makes bound splats from a mesh head (#126 route A): Cycles renders, gsplat training on a GPU host, `package.py` into a face; steps and results in `bake/README.md`.
- `a2f/` (the Audio2Face server's image), `lam/` (splat heads from portraits, #126) and `demo/` (the hosted demo, #118) are not part of the package; CI does not lint `a2f/`, `lam/` or `bake/`, and tests only `lam/`'s pure-Python parts (`tests/test_lam.py`, CI's `lam` job). Read `a2f/README.md` before building (about 25 GB of Docker disk) and `lam/README.md` for licences per part.
- `tests/helpers.py` builds every fixture in code: a one-triangle `.glb`, tone WAVs, a fake Audio2Face server, a fake TTS service, `live_server()`.

## Commands

`make install`, `make test`, `make lint` (pinned ruff), `make format`, `make schema`, `make demo`. One test: `uv run --no-sync pytest tests/test_x.py::test_name -q`.

## Conventions & gotchas

- **`src/` layout, `py.typed`, Pydantic v2, `structlog`.** Ruff line-length 120; `make lint` before committing.
- **Any endpoint that does something is a boundary.** There is no auth by default: reaching the port is host access. State-changing routes sit behind `WriteGuard` (origin, read-only); fence any caller-supplied path under a configured root.
- **WebSockets are outside `WriteGuard` and CORS.** Each socket calls `cross_site()` (the same `cross_site_refuse` predicate) by hand.
- **Frame boundaries use integer arithmetic** (`frame * 16000 // fps`): float frame lengths drift at rates that don't divide the sample rate. `drive` and `push` feed the same block size (`performance.BLOCK_MS`), so offline and live agree.
- **Free the face before telling the source it is done** (`stage.free(claim)`, then `done`; only the claim's own token frees it), or a source that reconnects at once is told the face is taken.
- **The viewer handler uses an anyio task group**, not bare asyncio tasks: those made the test client raise `CancelledError` on exit about one run in three.
- **The server's own process uses no GPU**; it leases for the GPU work it asks of others (`gpu.py`, `gpu-tenant.yaml`, #125). `a2f-live` is held in the background for a live a2f stream (`hold`: never waited on, broker only, no fallback: the local lease is a host-wide lock a broadcast would hold for hours); `voice-make` is waited for around a TTS service making a voice. A new lease kind goes in both files (`test_gpu_tenant.py` checks they agree).
- **Two tiers of faces** (README § Personal and cleared): `personal` is served but never cleared; `stock`, `synthetic` and `consented` are cleared. Don't gate personal use on copyright or likeness. Keep `synthetic` honest: a face from a character LoRA is `personal`. The floor is the law: what argus-darkroom refuses is not a face here either. Voices mirror this.
- **The disclosure label** (`Provenance.disclosure`, `StreamHeader.disclosure`) is fixed text from `models.py`, never a field anyone sets. `FaceStore` takes a `clock` and `consent_valid` takes `at`, so tests never depend on today's date.
- **The viewer has little automated test.** After touching `viewer/`, run the server, `argus-mask push` a WAV and watch a browser; `window.__mask` exposes what is drawn. Lip shaping and idle life live in the viewer, never in takes. The look finds parts by name (`head`, `eyes`, `hair`, `wetline`): a recipe change that renames them silently loses skin, catch-lights or soft hair.
- **Splat heads have renderer traps**: read [docs/splat-heads.md](docs/splat-heads.md) before touching `viewer/splat.js`.
- **`lam/`**: `overrides.install()` runs before LAM is imported, and `verify()` must come back empty. The tracker's landmarks and `landmarks/embedding.json` come from the same `ibug68` map: change one, re-measure the other.
- **The demo's `vendor/dh/` is vendored**: don't edit it. `pages.yml` deploys `dev`'s HEAD with full history (the version and the drift check need it).
- Template-managed files (`Makefile`, `Dockerfile`, workflows) have local edits; `copier update` will ask about them.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never open a PR against `main` or push work to it directly: `main` must never be ahead of `dev`. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected and needs a PR, merge the release PR, then immediately merge `main` back into `dev` so the merge commit is not left on `main` alone.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow. Release publishes to PyPI (OIDC trusted publishing) + GHCR on `v*` tags. Run `copier update` to pull template (CI/release/tooling) changes.

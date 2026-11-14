"""The ``argus-mask`` command (the ``[cli]`` extra)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from typer import Argument, Option

from argus_mask.config import DEFAULT_EDITOR_PORT, DEFAULT_FPS, DEFAULT_PORT, ConfigError, Settings

app = typer.Typer(help="A talking face: audio in, animation frames out.", no_args_is_help=True, add_completion=False)

DEFAULT_SCHEMA_PATH = Path("schema/mask-wire.schema.json")
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(1)


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def inspect(
    face: Path = Argument(..., help="A face directory, a .glb file, or a LAM splat head's .zip"),
    write_manifest: bool = Option(
        False, "--write-manifest", help="Write manifest.json beside a .glb or .zip (needs the options below)"
    ),
    face_id: str = Option("", "--id", help="The face's id; default: its directory's name"),
    name: str = Option("", "--name", help="Display name; default: the id"),
    kind: str = Option(
        "personal",
        "--kind",
        help="Provenance kind: personal (your own use, anything; the default), stock, synthetic or consented",
    ),
    source: str = Option("", "--source", help="Provenance: where the model came from"),
    license_: str = Option("", "--license", help="Provenance: the model's licence"),
    notes: str = Option("", "--notes", help="Provenance: anything else worth recording"),
    subject: str = Option("", "--subject", help="Provenance, consented: the person, as the release names them"),
    release: str = Option("", "--release-sha256", help="Provenance, consented: sha256 of the signed release"),
    generator: str = Option("", "--generator", help="Provenance, synthetic: the model that generated the face"),
    generator_sha256: str = Option("", "--generator-sha256", help="Provenance, synthetic: the model's sha256"),
    generator_seed: int | None = Option(None, "--generator-seed", min=0, help="Provenance, synthetic: the seed"),
    prompt: str = Option("", "--prompt", help="Provenance, synthetic: the prompt"),
    distance: float | None = Option(
        None, "--distance", help="Framing: camera distance from the model's origin, in metres (default 0.6; splat 0.7)"
    ),
    height: float | None = Option(
        None, "--height", help="Framing: camera height above the origin, in metres (default 0; splat 1.65)"
    ),
    fov: float | None = Option(
        None, "--fov", help="Framing: vertical field of view, in degrees (default 30; splat 50)"
    ),
    as_json: bool = Option(False, "--json", help="Print the report as JSON"),
) -> None:
    """Check a face against the model contract: which of the 52 ARKit shapes it has, and its size.

    A LAM splat head (.zip) is checked without unpacking it: one folder of known members, sizes in bounds.
    """
    from pydantic import ValidationError

    from argus_mask.faces import (
        MANIFEST_NAME,
        SPLAT_FRAMING,
        FaceError,
        FaceStore,
        SplatReport,
        build_manifest,
        inspect_model,
        model_format,
    )
    from argus_mask.models import Framing, Generator, Provenance

    if not generator and (generator_sha256 or generator_seed is not None or prompt):
        raise _fail("--generator-sha256, --generator-seed and --prompt describe a --generator: name it")
    refusal: str | None = None
    manifest = None
    try:
        if face.is_dir():
            if write_manifest:
                raise _fail("--write-manifest takes the model file (.glb or .zip), not the directory")
            try:
                manifest, model = FaceStore(face.resolve().parent).load(face.resolve().name)
            except FaceError as exc:
                refusal = str(exc)
                models = sorted([*face.glob("*.glb"), *face.glob("*.zip")])
                if len(models) != 1:
                    raise _fail(
                        f"{face}: refused ({refusal}), and it holds {len(models)} .glb or .zip files,"
                        " so none was inspected"
                    ) from exc
                model = models[0]
        else:
            model = face
        report = inspect_model(model)
        if write_manifest:
            default = SPLAT_FRAMING if model_format(model) == "lam-splat" else Framing()
            try:
                manifest = build_manifest(
                    model,
                    face_id=face_id or model.resolve().parent.name,
                    name=name or face_id or model.resolve().parent.name,
                    provenance=Provenance(
                        kind=kind,  # type: ignore[arg-type]
                        source=source,
                        license=license_,
                        notes=notes,
                        subject=subject,
                        release_sha256=release.strip().lower() or None,
                        generator=Generator(
                            model=generator,
                            model_sha256=generator_sha256.strip().lower() or None,
                            seed=generator_seed,
                            prompt=prompt,
                        )
                        if generator
                        else None,
                    ),
                    framing=Framing(
                        distance=default.distance if distance is None else distance,
                        height=default.height if height is None else height,
                        fov=default.fov if fov is None else fov,
                    ),
                )
            except ValidationError as exc:
                raise _fail(
                    "cannot write a manifest: "
                    + "; ".join(
                        e["msg"] + (f" ({'.'.join(map(str, e['loc']))})" if e["loc"] else "") for e in exc.errors()
                    )
                ) from exc
            (model.parent / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (FaceError, OSError) as exc:
        raise _fail(str(exc)) from exc

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "model": str(model),
                    "refused": refusal,
                    "manifest": manifest.model_dump(mode="json") if manifest else None,
                    "present": report.present,
                    "missing": report.missing,
                    "extra": report.extra,
                    "triangles": report.triangles,
                    "textures": report.textures,
                    "meshes": [
                        {"name": m.name, "triangles": m.triangles, "targets": len(m.targets)} for m in report.meshes
                    ],
                    "warnings": report.warnings,
                    **(
                        {
                            "format": "lam-splat",
                            "folder": report.folder,
                            "members": report.members,
                            "splats": report.splats,
                        }
                        if isinstance(report, SplatReport)
                        else {"format": "gltf"}
                    ),
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"model      {model}")
        if manifest:
            p = manifest.provenance
            typer.echo(f"face       {manifest.id} ({manifest.name})")
            typer.echo(
                f"provenance {p.kind}: {p.source}"
                + (f" [{p.license}]" if p.license else "")
                + ("" if p.cleared else " (personal use: not cleared to show or share)")
            )
        typer.echo(
            f"shapes     {len(report.present)} of 52 ARKit shapes"
            + (f", {len(report.extra)} others" if report.extra else "")
        )
        if report.missing:
            typer.echo(f"missing    {', '.join(report.missing)}")
        if isinstance(report, SplatReport):
            typer.echo(
                f"splat head {report.folder}/: {', '.join(report.members)}"
                + (f"; {report.splats} Gaussians" if report.splats is not None else "")
            )
        typer.echo(f"meshes     {len(report.meshes)}, {report.triangles} triangles")
        sizes = ", ".join(f"{s[0]}x{s[1]}" if s else "unknown" for s in report.textures) or "none"
        typer.echo(f"textures   {sizes}")
        for warning in report.warnings:
            typer.echo(f"warning    {warning}")
        if write_manifest:
            typer.echo(f"wrote      {model.parent / MANIFEST_NAME}")
    if refusal:
        raise _fail(f"refused    {refusal}")


@app.command()
def drive(
    audio: Path = Argument(..., help="A WAV file"),
    out: Path = Option(..., "--out", "-o", help="The take file to write (.ndjson)"),
    driver_name: str = Option("", "--driver", help="energy or a2f; default: ARGUS_MASK_DRIVER"),
    fps: int = Option(0, "--fps", help=f"Frames per second; default: ARGUS_MASK_FPS ({DEFAULT_FPS})"),
    seed: int = Option(0, "--seed", help="Seed of the idle layer (blinks, gaze, head sway)"),
    face_id: str = Option("", "--face", help="Recorded in the take's header"),
) -> None:
    """Audio to a take file, offline. The same audio, driver and seed give the same file."""
    from argus_mask.audio import AudioError, read_wav
    from argus_mask.drivers import DriverError, build_driver
    from argus_mask.performance import perform
    from argus_mask.take import render_take

    cfg = _settings()
    try:
        mono, rate = read_wav(audio)
        driver = build_driver(driver_name or cfg.driver, a2f_url=cfg.a2f_url)
    except (AudioError, OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    async def run() -> str:
        if (reason := await driver.check()) is not None:
            raise DriverError(f"the {driver.name} driver is down: {reason}")
        return render_take(*await perform(mono, rate, driver, fps=fps or cfg.fps, seed=seed, face_id=face_id))

    try:
        text = asyncio.run(run())
    except DriverError as exc:
        raise _fail(str(exc)) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(f"{out}: {text.count(chr(10)) - 2} frames from {mono.size / rate:.2f}s of audio ({driver.name})")


def _report(done: dict) -> None:
    typer.echo(f"done: {done.get('frames')} frames, {done.get('duration_ms', 0) / 1000:.2f}s")


@app.command()
def push(
    audio: Path = Argument(..., help="A WAV file"),
    face_id: str = Option(..., "--face", help="The face to animate"),
    url: str = Option(DEFAULT_URL, "--url", help="The argus-mask server"),
    seed: int = Option(0, "--seed", help="Seed of the idle layer"),
) -> None:
    """Play a WAV into a running server, in real time."""
    from argus_mask import client
    from argus_mask.audio import AudioError, read_wav

    try:
        mono, rate = read_wav(audio)
        _report(
            asyncio.run(
                client.push(
                    url, face_id, mono, rate, seed=seed, on_ready=lambda r: typer.echo(f"session {r.get('session_id')}")
                )
            )
        )
    except (AudioError, OSError, client.ClientError) as exc:
        raise _fail(str(exc)) from exc
    except KeyboardInterrupt:
        raise _fail("stopped") from None


@app.command()
def say(
    text: str = Argument(..., help="What to say"),
    face_id: str = Option(..., "--face", help="The face to animate"),
    url: str = Option(DEFAULT_URL, "--url", help="The argus-mask server"),
    tts_url: str = Option("", "--tts-url", help="The TTS stream endpoint; default: ARGUS_MASK_TTS_URL"),
    voice: str = Option(
        "", "--voice", help="The TTS voice: a clone's name or a speaker number; default: ARGUS_MASK_TTS_VOICE"
    ),
    param: list[str] = Option([], "--param", "-p", help="Extra field for the TTS request, key=value (repeatable)"),
    seed: int = Option(0, "--seed", help="Seed of the idle layer"),
    stream: bool = Option(
        False, "--stream", help="Forward audio as the TTS generates it (only for a service faster than real time)"
    ),
    pause: int = Option(
        300,
        "--pause",
        min=0,
        help="Silence between sentences in ms, each sentence its own request; 0 sends the text whole",
    ),
    context: bool = Option(
        False,
        "--context",
        help="Send each sentence with the one before it as context (text and audio; an HTTP service like Miso "
        "takes it). Experimental: with a cloned voice it can make the tone drift rather than hold",
    ),
    as_is: bool = Option(
        False,
        "--as-is",
        help="Send the text as written and play the audio as it came: no numbers as words, no joined short "
        "sentences, no fades or levelling",
    ),
) -> None:
    """Speak text through a TTS service into a running server. The text goes to that service and nowhere else.

    The text goes one sentence at a time with a pause between them (twice as long between paragraphs). The whole
    utterance is fetched before it is played, so a slow TTS model delays the start but never stalls the face.
    Numbers and times are sent as words, a very short sentence goes with the next, and each sentence is faded in
    and out and brought to the same loudness; --as-is turns that off.
    """
    from argus_mask import client

    body: dict[str, object] = {}
    for item in param:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise _fail(f"--param {item!r}: expected key=value")
        try:
            body[key] = json.loads(value)  # numbers and booleans as themselves
        except ValueError:
            body[key] = value
    body["text"] = text
    settings = _settings()
    tts_url = tts_url or settings.tts_url
    from argus_mask.consent import ConsentStore, voice_disclosed, voice_of, voice_refusal
    from argus_mask.faces import local_now

    # A voice made on this machine's Voices page is checked against its consent record; one made elsewhere is not.
    spoken = voice_of(body, voice or settings.tts_voice)
    consents = ConsentStore(settings.assets_dir)
    if why := voice_refusal(settings.voices_dir, consents, spoken, use="avatar", at=local_now()):
        raise _fail(why)
    body = client.voiced(body, voice or settings.tts_voice, tts_url)
    try:
        _report(
            asyncio.run(
                client.say(
                    url,
                    face_id,
                    tts_url,
                    body,
                    seed=seed,
                    disclose=voice_disclosed(settings.voices_dir, spoken),
                    stream=stream,
                    pause_ms=pause,
                    tidy=not as_is,
                    context=context,
                    on_ready=lambda r: typer.echo(f"session {r.get('session_id')}"),
                    on_progress=lambda i, n: typer.echo(f"sentence {i}/{n}", err=True),
                    on_audio=lambda secs: typer.echo(f"speech: {secs:.2f}s, playing"),
                    on_wait=lambda: typer.echo(
                        "waiting for the TTS service (a cold model takes a minute or two)", err=True
                    ),
                )
            )
        )
    except client.ClientError as exc:
        raise _fail(str(exc)) from exc
    except KeyboardInterrupt:
        raise _fail("stopped") from None


@app.command()
def record(
    face_id: str = Option(..., "--face", help="The face to record"),
    out: Path = Option(Path("clip.mp4"), "--out", help="The clip (.mp4, or .webm); its take and WAV go beside it"),
    url: str = Option(DEFAULT_URL, "--url", help="The argus-mask server"),
    width: int = Option(1280, "--width", min=16),
    height: int = Option(720, "--height", min=16),
    display: str = Option("flat", "--display", help="The viewer's display tier: flat or hld"),
    bg: str = Option("", "--bg", help="Background colour, as the viewer takes it (e.g. '#202020')"),
    quality: str = Option("high", "--quality", help="The viewer's look: low, medium or high (lighting, shadows, post)"),
    keep: bool = Option(False, "--keep-frames", help="Keep the rendered PNG frames beside the clip"),
) -> None:
    """Record the next session on a face as a video, offline and frame-exact.

    Waits on the face's viewer socket; run `say` or `push` on that face meanwhile. Then renders every frame of the
    session in a headless browser and muxes them with the audio, so the machine's load never shows in the clip.
    Needs argus-mask[record] and `playwright install chromium-headless-shell`.
    """
    from argus_mask import record as recorder

    def captured(r: recorder.Recording) -> None:
        typer.echo(f"captured {len(r.frames)} frames, {r.duration_ms / 1000:.2f}s; rendering", err=True)

    last = [0]

    def progress(i: int, n: int) -> None:
        if i == n or i * 10 // n > last[0]:
            last[0] = i * 10 // n
            typer.echo(f"  {i}/{n} frames", err=True)

    try:
        r = asyncio.run(
            recorder.record(
                url,
                face_id,
                out,
                width=width,
                height=height,
                display=display,
                bg=bg,
                quality=quality,
                keep=keep,
                on_wait=lambda: typer.echo(f"waiting for a session on {face_id} (run say or push now)", err=True),
                on_captured=captured,
                on_progress=progress,
                on_renderer=lambda r: typer.echo(f"rendering on {r}", err=True),
            )
        )
    except recorder.RecordError as exc:
        raise _fail(str(exc)) from exc
    except KeyboardInterrupt:
        raise _fail("stopped") from None
    typer.echo(f"wrote {out} ({len(r.frames)} frames, {r.duration_ms / 1000:.2f}s)")


@app.command()
def serve(
    host: str = Option("127.0.0.1", "--host", help="Address to bind. Serving the LAN is a deliberate choice."),
    port: int = Option(DEFAULT_PORT, "--port"),
    assets_dir: Path | None = Option(None, "--assets-dir", help="Root of the faces; default: ARGUS_MASK_ASSETS_DIR"),
) -> None:
    """Run the server: the faces, the two sockets and the viewer."""
    from dataclasses import replace

    try:
        import uvicorn

        from argus_mask.server.app import create_app
    except ImportError as exc:
        raise _fail(f"the server needs the [server] extra: {exc}") from exc

    cfg = _settings()
    if assets_dir is not None:
        cfg = replace(cfg, assets_dir=assets_dir)
    uvicorn.run(create_app(cfg), host=host, port=port)


@app.command()
def editor(
    host: str = Option("127.0.0.1", "--host", help="Address to bind. A build starts Blender: keep it local."),
    port: int = Option(DEFAULT_EDITOR_PORT, "--port"),
    assets_dir: Path | None = Option(None, "--assets-dir", help="Root of the faces; default: ARGUS_MASK_ASSETS_DIR"),
    blender: str | None = Option(None, "--blender", help="The Blender binary; default: ARGUS_MASK_BLENDER or blender"),
    blender_user: Path | None = Option(
        None, "--blender-user", help="Blender's user directory with MPFB and its asset packs (BLENDER_USER_RESOURCES)"
    ),
    server_url: str | None = Option(
        None,
        "--server-url",
        help="The argus-mask server, for checks and tests; default: ARGUS_MASK_SERVER_URL or :8109",
    ),
    viewer_url: str | None = Option(
        None, "--viewer-url", help="The origin screens open the viewer at (a TLS proxy on a LAN); default: the server's"
    ),
    recordings_dir: Path | None = Option(
        None,
        "--recordings-dir",
        help="Where the Broadcast page records clips; default: ARGUS_MASK_RECORDINGS_DIR or ./recordings",
    ),
    voices_dir: Path | None = Option(
        None,
        "--voices-dir",
        help="Where the Voices page keeps checked clips and each voice's record; default: ARGUS_MASK_VOICES_DIR "
        "or ./voices",
    ),
) -> None:
    """Run the face editor: build heads from presets with headless Blender and MPFB."""
    from dataclasses import replace

    try:
        import uvicorn

        from argus_mask.config import EditorSettings
        from argus_mask.editor.app import create_editor_app
    except ImportError as exc:
        raise _fail(f"the editor needs the [server] extra: {exc}") from exc

    cfg = _settings()
    if assets_dir is not None:
        cfg = replace(cfg, assets_dir=assets_dir)
    try:
        ed = EditorSettings.from_env()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    if blender is not None:
        ed = replace(ed, blender=blender)
    if blender_user is not None:
        ed = replace(ed, blender_user=blender_user)
    try:
        if server_url is not None:
            ed = replace(ed, server_url=server_url.rstrip("/"))
        if viewer_url is not None:
            ed = replace(ed, viewer_url=viewer_url.rstrip("/"))
        if recordings_dir is not None:
            ed = replace(ed, recordings_dir=recordings_dir)
        if voices_dir is not None:
            ed = replace(ed, voices_dir=voices_dir)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    uvicorn.run(create_editor_app(cfg, ed), host=host, port=port)


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    from argus_mask.models import render_wire_schema

    rendered = render_wire_schema()
    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            typer.echo(f"{output} is stale: run `argus-mask schema` and commit the result.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{output} is up to date.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"wrote {output}")


# ── consent ───────────────────────────────────────────────────────────────────

consent_app = typer.Typer(
    help="Consent records for a real person's face or voice (docs/consent.md). A record holds hashes, never the "
    "release, the reference media or the ID.",
    no_args_is_help=True,
)
app.add_typer(consent_app, name="consent")


def _day(value: str, flag: str):  # noqa: ANN202 - a date
    from datetime import date

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(f"{flag} {value!r}: give a date as YYYY-MM-DD") from exc


def _hash_files(paths: list[Path], flag: str) -> list[str]:
    from argus_mask.faces import sha256_file

    out = []
    for path in paths:
        if not path.is_file():
            raise _fail(f"{flag} {path}: no such file")
        out.append(sha256_file(path))
    return out


@consent_app.command("new")
def consent_new(
    release: Path = Option(..., "--release", help="The signed release (hashed here; never copied or kept)"),
    subject: str = Option(..., "--subject", help="The person, as the release names them"),
    collected_by: str = Option(..., "--collected-by", help="Who ran the signing session"),
    signed_on: str = Option(..., "--signed-on", help="When the release was signed, YYYY-MM-DD"),
    age_checked_by: str = Option(..., "--age-checked-by", help="Who checked the person's age against ID"),
    use: list[str] = Option(..., "--use", help="A use the release covers: avatar, video, speech, training (repeat)"),
    revocable: bool = Option(..., "--revocable/--not-revocable", help="Whether the release can be revoked"),
    on_revoke: str = Option(
        ..., "--on-revoke", help="Recordings already made, on revocation: delete, or keep-labelled"
    ),
    face: bool = Option(False, "--face", help="The release covers the face"),
    voice: bool = Option(False, "--voice", help="The release covers the voice"),
    delivery: list[str] = Option([], "--delivery", help="A voice delivery the release covers (repeat)"),
    reference_image: list[Path] = Option(
        [], "--reference-image", help="A frontal reference image from the signing session (hashed; repeat)"
    ),
    reference_recording: list[Path] = Option(
        [], "--reference-recording", help="The reference clip read at the signing session (hashed; repeat)"
    ),
    age_checked_on: str = Option("", "--age-checked-on", help="When the age was checked; default: the signing day"),
    expires_on: str = Option("", "--expires-on", help="The first day the release no longer holds; default: never"),
    notice_days: int = Option(0, "--notice-days", min=0, help="The notice period for revocation, in days"),
    exclusion: list[str] = Option([], "--exclusion", help="An exclusion beyond the template's list (repeat)"),
    media: list[str] = Option([], "--media", help="A medium the release names (repeat)"),
    territory: list[str] = Option([], "--territory", help="A territory the release names (repeat)"),
    notes: str = Option("", "--notes"),
    assets: Path | None = Option(None, "--assets", help="The assets root; default: ARGUS_MASK_ASSETS_DIR"),
) -> None:
    """Write the consent record for a signed release: what the person agreed to, bound by hashes."""
    from pydantic import ValidationError

    from argus_mask.consent import AgeCheck, ConsentError, ConsentRecord, ConsentStore

    if not release.is_file():
        raise _fail(f"--release {release}: no such file")
    signed = _day(signed_on, "--signed-on")
    try:
        record = ConsentRecord(
            release_sha256=_hash_files([release], "--release")[0],
            subject=subject,
            collected_by=collected_by,
            signed_on=signed,
            expires_on=_day(expires_on, "--expires-on") if expires_on else None,
            age_check=AgeCheck(
                checked_by=age_checked_by,
                checked_on=_day(age_checked_on, "--age-checked-on") if age_checked_on else signed,
            ),
            face=face,
            voice=voice,
            deliveries=delivery,
            uses=use,  # type: ignore[arg-type]
            reference_images=_hash_files(reference_image, "--reference-image"),
            reference_recordings=_hash_files(reference_recording, "--reference-recording"),
            exclusions=exclusion,
            media=media,
            territories=territory,
            revocable=revocable,
            notice_days=notice_days,
            on_revoke=on_revoke,  # type: ignore[arg-type]
            notes=notes,
        )
    except ValidationError as exc:
        raise _fail(
            "cannot write the record: " + "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        ) from exc
    store = ConsentStore(assets or _settings().assets_dir)
    try:
        store.create(record)
    except (ConsentError, OSError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"wrote {store.path(record.release_sha256)}")
    typer.echo(
        f"release {record.release_sha256}: give this hash as the face's --release-sha256, or the voice's release"
    )


@consent_app.command("list")
def consent_list(
    assets: Path | None = Option(None, "--assets", help="The assets root; default: ARGUS_MASK_ASSETS_DIR"),
) -> None:
    """Every consent record, and whether it holds today."""
    from argus_mask.consent import ConsentStore, consent_holds, uses_of
    from argus_mask.faces import local_now

    records, broken = ConsentStore(assets or _settings().assets_dir).all()
    now = local_now()
    for r in records:
        likeness = "face" if r.face else "voice"
        verdict = consent_holds(r, likeness=likeness, at=now, delivery=r.deliveries[0] if r.voice else None)
        state = "holds" if verdict.ok else verdict.reason
        covers = " + ".join(k for k in ("face", "voice") if getattr(r, k))
        extra = f" ({', '.join(r.deliveries)})" if r.voice else ""
        typer.echo(
            f"{r.release_sha256[:16]}  {r.subject}  {covers}{extra}  uses: {', '.join(uses_of(r))}  "
            f"signed {r.signed_on}  {state}"
        )
    for name, why in broken.items():
        typer.echo(f"{name}: unreadable: {why}", err=True)
    if not records and not broken:
        typer.echo("no consent records")


@consent_app.command("revoke")
def consent_revoke(
    release: str = Argument(..., help="The release's sha256, or its first 8+ characters"),
    by: str = Option(..., "--by", help="Who records the revocation"),
    requested_on: str = Option("", "--requested-on", help="When it was asked for; default: today (UTC)"),
    effective_on: str = Option("", "--effective-on", help="When it takes effect; default: the day it was asked for"),
    assets: Path | None = Option(None, "--assets", help="The assets root; default: ARGUS_MASK_ASSETS_DIR"),
    voices: Path | None = Option(None, "--voices", help="The voices directory; default: ARGUS_MASK_VOICES_DIR"),
) -> None:
    """Revoke a release: from its effective day nothing under it is served or spoken; its voices are deleted from
    the TTS services they were made on, and its voices and faces moved to the trash. Safe to run again."""
    import time

    from pydantic import ValidationError

    from argus_mask import client
    from argus_mask.consent import ConsentError, ConsentStore, Revocation, utc_today, withdraw
    from argus_mask.faces import local_now

    settings = _settings()
    assets_dir = assets or settings.assets_dir
    # The UTC day, as cortex reads a record's dates: a local "today" east of UTC would leave the release in force
    # for the rest of the UTC day.
    today = utc_today(local_now())
    requested = _day(requested_on, "--requested-on") if requested_on else today
    try:
        record = ConsentStore(assets_dir).find(release)
        revocation = Revocation(
            requested_on=requested,
            effective_on=_day(effective_on, "--effective-on") if effective_on else requested,
            by=by,
        )
        done = withdraw(
            assets_dir,
            voices or settings.voices_dir,
            record.release_sha256,
            revocation,
            today=today,
            stamp=time.strftime("%Y%m%d-%H%M%S"),
            delete_clone=client.delete_voice,
        )
    except ValidationError as exc:
        raise _fail("; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())) from exc
    except (ConsentError, OSError) as exc:
        raise _fail(str(exc)) from exc
    held = done.record.revocation
    assert held is not None
    if not done.record.revocable:
        typer.echo("note: the release says it cannot be revoked; the revocation is recorded and enforced anyway")
    typer.echo(f"{done.record.subject}: revoked from {held.effective_on} (recorded by {held.by})")
    if done.pending:
        typer.echo(f"nothing removed yet: run this again on or after {held.effective_on} to remove what was made")
        return
    for voice_id in done.voices_removed:
        typer.echo(f"voice {voice_id}: deleted from its TTS service, record and clips to the trash")
    for face_id in done.faces_trashed:
        typer.echo(f"face {face_id}: to the trash")
    typer.echo(
        f"recordings already made: {done.record.on_revoke} (as the release agreed); recordings are not tracked here"
    )
    if done.voices_failed:
        for voice_id, why in done.voices_failed.items():
            typer.echo(f"voice {voice_id}: {why}", err=True)
        raise _fail("some voices are still on their TTS service (refused here, being revoked): run this again")


if __name__ == "__main__":
    app()

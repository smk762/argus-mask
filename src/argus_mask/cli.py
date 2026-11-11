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
    face: Path = Argument(..., help="A face directory, or a .glb file"),
    write_manifest: bool = Option(
        False, "--write-manifest", help="Write manifest.json beside a .glb (needs the options below)"
    ),
    face_id: str = Option("", "--id", help="The face's id; default: its directory's name"),
    name: str = Option("", "--name", help="Display name; default: the id"),
    kind: str = Option("", "--kind", help="Provenance kind: stock or synthetic"),
    source: str = Option("", "--source", help="Provenance: where the model came from"),
    license_: str = Option("", "--license", help="Provenance: the model's licence"),
    notes: str = Option("", "--notes", help="Provenance: anything else worth recording"),
    distance: float = Option(0.6, "--distance", help="Framing: camera distance from the model's origin, in metres"),
    height: float = Option(0.0, "--height", help="Framing: camera height above the origin, in metres"),
    fov: float = Option(30.0, "--fov", help="Framing: vertical field of view, in degrees"),
    as_json: bool = Option(False, "--json", help="Print the report as JSON"),
) -> None:
    """Check a face against the model contract: which of the 52 ARKit shapes it has, and its size."""
    from pydantic import ValidationError

    from argus_mask.faces import MANIFEST_NAME, FaceError, FaceStore, build_manifest, inspect_glb
    from argus_mask.models import Framing, Provenance

    refusal: str | None = None
    manifest = None
    try:
        if face.is_dir():
            if write_manifest:
                raise _fail("--write-manifest takes the .glb file, not the directory")
            try:
                manifest, model = FaceStore(face.resolve().parent).load(face.resolve().name)
            except FaceError as exc:
                refusal = str(exc)
                models = sorted(face.glob("*.glb"))
                if len(models) != 1:
                    raise _fail(
                        f"{face}: refused ({refusal}), and it holds {len(models)} .glb files, so none was inspected"
                    ) from exc
                model = models[0]
        else:
            model = face
        report = inspect_glb(model)
        if write_manifest:
            try:
                manifest = build_manifest(
                    model,
                    face_id=face_id or model.resolve().parent.name,
                    name=name or face_id or model.resolve().parent.name,
                    provenance=Provenance(kind=kind, source=source, license=license_, notes=notes),  # type: ignore[arg-type]
                    framing=Framing(distance=distance, height=height, fov=fov),
                )
            except ValidationError as exc:
                raise _fail(
                    f"cannot write a manifest: {'; '.join(e['msg'] + ' (' + '.'.join(map(str, e['loc'])) + ')' for e in exc.errors())}"
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
                },
                indent=2,
            )
        )
    else:
        typer.echo(f"model      {model}")
        if manifest:
            p = manifest.provenance
            typer.echo(f"face       {manifest.id} ({manifest.name})")
            typer.echo(f"provenance {p.kind}: {p.source} [{p.license}]")
        typer.echo(
            f"shapes     {len(report.present)} of 52 ARKit shapes"
            + (f", {len(report.extra)} others" if report.extra else "")
        )
        if report.missing:
            typer.echo(f"missing    {', '.join(report.missing)}")
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
                    stream=stream,
                    pause_ms=pause,
                    tidy=not as_is,
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


if __name__ == "__main__":
    app()

"""Style templates and darkroom record checks for a guided source (#167)."""

from __future__ import annotations

import hashlib

import pytest

from argus_mask.portraits import (
    CHECKPOINT,
    TEMPLATES,
    PortraitError,
    candidate_for_seed,
    check_source_record,
    generate_body,
    prompt_for,
    refusal,
    seeds,
    subject_line,
)


def test_seeds_parse_ranges_and_lists() -> None:
    assert seeds("501-503") == [501, 502, 503]
    assert seeds("7,9") == [7, 9]
    with pytest.raises(PortraitError, match="at most 64"):
        seeds("1-100")
    with pytest.raises(PortraitError, match="once"):
        seeds("1,1")


def test_subject_refuses_under_eighteen() -> None:
    with pytest.raises(PortraitError, match="18 or over"):
        subject_line(age=17)
    assert "18" in subject_line(age=18, hair="short dark", clothes="a crew-neck top")


def test_prompt_uses_each_style() -> None:
    for style in TEMPLATES:
        text = prompt_for("an adult aged 30, short hair", style)
        assert "short hair" in text
        assert "{" not in text
    with pytest.raises(PortraitError, match="not a portrait style"):
        prompt_for("an adult", "oil")


def test_check_source_record_kind_and_sha(png_bytes: bytes) -> None:
    sha = hashlib.sha256(png_bytes).hexdigest()
    record = {
        "id": "rec1",
        "outcome": "passed",
        "guard": {"enforcing_policy": "default"},
        "image": {"sha256": sha},
        "request": {"loras": [], "final_prompt": "hi", "seed": 1, "checkpoint": {"name": CHECKPOINT}},
    }
    assert check_source_record(record, png_bytes) == "synthetic"
    record["request"]["loras"] = [{"name": "character.safetensors"}]
    assert check_source_record(record, png_bytes) == "personal"
    assert refusal(record) is not None
    with pytest.raises(PortraitError, match="sha256"):
        check_source_record(record, png_bytes + b"x")


def test_generate_body_and_candidate_ids() -> None:
    body = generate_body("a prompt", 501)
    assert body["seed"] == 501
    assert body["checkpoint"] == CHECKPOINT
    assert body["loras"] == []
    assert candidate_for_seed(501) == "s501"


@pytest.fixture
def png_bytes() -> bytes:
    from tests.test_backgrounds import png

    return png(32, 40)

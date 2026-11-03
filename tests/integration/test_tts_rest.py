"""Integration tests against a running nava-tts service (auto-skipped if down)."""
from __future__ import annotations

import httpx
import pytest

from conftest import TTS_URL

pytestmark = pytest.mark.tts_service

LONG_TEXT = ("این یک جملهٔ آزمایشی نسبتاً بلند است که برای بررسی قطعه‌بندی و خط لولهٔ "
             "تولید صدا نوشته شده است. " * 4)


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=TTS_URL, timeout=600) as client:
        yield client


@pytest.fixture(scope="module")
def engines(http):
    return http.get("/api/v1/health").json()["engines"]


def test_health_lists_engines(http):
    body = http.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert set(body["engines"]) >= {"pocket", "mana", "piper"}


@pytest.mark.parametrize("engine", ["mana", "piper"])
def test_speech_streams_valid_pcm(http, engine):
    response = http.post("/api/v1/speech", json={"text": "سلام دنیا", "engine": engine})
    assert response.status_code == 200
    assert response.headers["X-Nava-Engine"] == engine
    rate = int(response.headers["X-Nava-Sample-Rate"])
    assert rate > 8000
    assert len(response.content) > 0 and len(response.content) % 2 == 0  # s16 frames
    assert response.headers["content-type"].startswith("audio/L16")


def test_speech_pocket_default_voice(http, engines):
    response = http.post("/api/v1/speech", json={"text": "سلام", "engine": "pocket"})
    assert response.status_code == 200
    assert len(response.content) > 0


def test_422_when_text_has_no_speakable_content(http):
    # regression: used to return HTTP 200 with an empty stream
    response = http.post("/api/v1/speech", json={"text": "!!!", "engine": "piper"})
    assert response.status_code == 422


def test_422_for_unknown_voice_prompt(http):
    response = http.post("/api/v1/speech", json={"text": "سلام", "engine": "pocket", "voice_prompt": "no_such_voice"})
    assert response.status_code == 422


def test_422_for_pocket_with_speed(http):
    # regression: speed used to be silently ignored for pocket
    response = http.post("/api/v1/speech", json={"text": "سلام", "engine": "pocket", "speed": 2.0})
    assert response.status_code == 422


def test_speed_applies_to_piper(http):
    normal = http.post("/api/v1/speech", json={"text": "سلام دنیا", "engine": "piper", "speed": 1.0})
    fast = http.post("/api/v1/speech", json={"text": "سلام دنیا", "engine": "piper", "speed": 2.0})
    assert normal.status_code == fast.status_code == 200
    assert len(fast.content) < len(normal.content)


def test_pocket_long_text_completes(http):
    # exercises the G2P->synthesis pipeline across many chunks
    response = http.post("/api/v1/speech", json={"text": LONG_TEXT, "engine": "pocket"})
    assert response.status_code == 200
    assert len(response.content) > 16000  # more than a blip

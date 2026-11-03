"""Integration tests against a running nava-stt service (auto-skipped if down)."""
from __future__ import annotations

import httpx
import pytest

from conftest import STT_URL, make_pcm

pytestmark = pytest.mark.stt_service


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=STT_URL, timeout=300) as client:
        yield client


def test_health_endpoints(http):
    for path in ("/healthz", "/api/v1/health"):
        response = http.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_transcribe_returns_result_shape(http):
    response = http.post("/api/v1/transcriptions", content=make_pcm(1.0),
                         headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text", "segments", "language_prob"}
    assert isinstance(body["text"], str)


def test_transcribe_with_query_params(http):
    response = http.post("/api/v1/transcriptions", content=make_pcm(1.0),
                         params={"hotwords": "نوا"}, headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200


def test_422_for_audio_shorter_than_100ms(http):
    response = http.post("/api/v1/transcriptions", content=b"\x00" * 100,
                         headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422


def test_422_for_odd_length_audio(http):
    response = http.post("/api/v1/transcriptions", content=b"\x00" * 3201,
                         headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422


def test_413_rejected_via_content_length_without_buffering(http):
    # 30 MB declared up-front is above the 15-minute default limit (28.8 MB)
    oversized = b"\x00" * 30_000_000
    response = http.post("/api/v1/transcriptions", content=oversized,
                         headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 413


def test_413_aborts_during_chunked_streaming(http):
    def chunks():
        block = b"\x00" * 1_000_000
        for _ in range(31):
            yield block

    response = http.post("/api/v1/transcriptions", content=chunks(),
                         headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 413

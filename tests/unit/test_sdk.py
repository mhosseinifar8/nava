"""SDK unit tests — no network needed (httpx MockTransport + pure helpers)."""
from __future__ import annotations

import json

import httpx
import pytest

from nava import (AsyncNavaClient, NavaClient, NavaServiceError, SpeechResult,
                  TranscriptionResult, client as nava_client)


class TestWsUrl:
    @pytest.mark.parametrize("base,path,expected", [
        ("http://127.0.0.1:8001", "/ws/stt", "ws://127.0.0.1:8001/ws/stt"),
        ("https://nava.example.com", "/ws/tts", "wss://nava.example.com/ws/tts"),
        ("ws://h:1", "/ws/stt", "ws://h:1/ws/stt"),
        ("wss://h", "/ws/tts", "wss://h/ws/tts"),
    ])
    def test_schemes(self, base, path, expected):
        assert nava_client.ws_url(base, path) == expected


class TestCheck:
    def test_success_is_noop(self):
        response = httpx.Response(200, json={"ok": True})
        nava_client.check(response)

    def test_error_raises_with_json_detail(self):
        response = httpx.Response(422, json={"detail": "bad audio"})
        with pytest.raises(NavaServiceError, match="422.*bad audio"):
            nava_client.check(response)

    def test_error_raises_with_text_detail(self):
        response = httpx.Response(500, text="boom")
        with pytest.raises(NavaServiceError, match="500.*boom"):
            nava_client.check(response)


def _mock(transport):
    client = NavaClient()
    client.http = httpx.Client(transport=transport)
    return client


def test_sync_transcribe_parses_result():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/transcriptions"
        assert request.headers["content-type"] == "application/octet-stream"
        assert request.content == b"\x00\x01"
        assert request.url.params["hotwords"] == "نوا"
        return httpx.Response(200, json={"text": "سلام", "segments": [{"start": 0.0, "end": 1.0, "text": "سلام"}], "language_prob": 0.99})

    with _mock(httpx.MockTransport(handler)) as nava:
        result = nava.transcribe(b"\x00\x01", hotwords="نوا")
    assert isinstance(result, TranscriptionResult)
    assert result.text == "سلام" and result.language_prob == 0.99


def test_sync_transcribe_raises_on_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, json={"detail": "too big"})

    with _mock(httpx.MockTransport(handler)) as nava:
        with pytest.raises(NavaServiceError, match="413"):
            nava.transcribe(b"\x00\x01")


def test_sync_synthesize_parses_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"text": "سلام", "engine": "piper", "voice_prompt": "default", "speed": 1.5}
        return httpx.Response(200, content=b"pcm-bytes",
                              headers={"X-Nava-Sample-Rate": "22050", "X-Nava-Engine": "piper"})

    with _mock(httpx.MockTransport(handler)) as nava:
        speech = nava.synthesize("سلام", engine="piper", speed=1.5)
    assert isinstance(speech, SpeechResult)
    assert (speech.pcm, speech.sample_rate, speech.engine) == (b"pcm-bytes", 22050, "piper")


def test_client_reuses_one_http_pool_and_closes():
    client = NavaClient()
    pool = client.http
    assert isinstance(pool, httpx.Client)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"text": "", "segments": [], "language_prob": 1.0})

    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        client.transcribe(b"\x00\x01")
        client.transcribe(b"\x00\x01")
        assert calls == ["/api/v1/transcriptions"] * 2
        assert client.http is not None  # same client object across calls
    finally:
        client.close()
    assert client.http.is_closed


async def test_async_transcribe_and_synthesize():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/transcriptions":
            return httpx.Response(200, json={"text": "t", "segments": [], "language_prob": 0.5})
        return httpx.Response(200, content=b"p", headers={"X-Nava-Sample-Rate": "24000", "X-Nava-Engine": "mana"})

    async with AsyncNavaClient() as nava:
        nava.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await nava.transcribe(b"\x00\x01")
        speech = await nava.synthesize("سلام", engine="mana")
    assert result.text == "t"
    assert (speech.sample_rate, speech.engine) == (24000, "mana")


async def test_chunked_ws_send_splits_large_payloads():
    sent: list[bytes] = []

    class FakeSocket:
        async def send(self, data):
            sent.append(data)

    payload = b"\x00" * (2 * nava_client.WS_FRAME_CHUNK + 5)
    await nava_client._send_chunked(FakeSocket(), payload)
    assert [len(f) for f in sent] == [nava_client.WS_FRAME_CHUNK, nava_client.WS_FRAME_CHUNK, 5]
    assert b"".join(sent) == payload

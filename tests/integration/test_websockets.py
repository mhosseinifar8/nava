"""WebSocket protocol integration tests (auto-skipped when services are down)."""
from __future__ import annotations

import json

import pytest
from websockets.asyncio.client import connect

from conftest import STT_URL, TTS_URL, make_pcm
from nava import AsyncNavaClient, NavaServiceError, client as nava_client


def stt_ws(): return connect(nava_client.ws_url(STT_URL, "/ws/stt"), max_size=None)
def tts_ws(): return connect(nava_client.ws_url(TTS_URL, "/ws/tts"), max_size=None)


@pytest.mark.stt_service
async def test_stt_websocket_full_protocol():
    async with stt_ws() as socket:
        await socket.send(json.dumps({"hotwords": "نوا"}, ensure_ascii=False))
        await socket.send(make_pcm(1.0))
        await socket.send(json.dumps({"action": "eos"}))
        body = json.loads(await socket.recv())
        assert set(body) >= {"text", "segments", "language_prob"}

        # connection is reusable for a second utterance
        await socket.send(make_pcm(1.0))
        await socket.send(json.dumps({"action": "eos"}))
        body2 = json.loads(await socket.recv())
        assert "text" in body2


@pytest.mark.stt_service
async def test_stt_websocket_error_on_empty_utterance():
    async with stt_ws() as socket:
        await socket.send(json.dumps({"action": "eos"}))
        body = json.loads(await socket.recv())
        assert body["status"] == "error"


@pytest.mark.stt_service
async def test_stt_websocket_rejects_non_json_control():
    async with stt_ws() as socket:
        await socket.send("not json")
        body = json.loads(await socket.recv())
        assert body["status"] == "error"


@pytest.mark.stt_service
async def test_stt_websocket_oversized_cumulative_audio_closes_1009():
    # 29 MB > 28.8 MB default limit: server must send an error and close with 1009
    async with stt_ws() as socket:
        block = b"\x00" * 1_000_000
        for _ in range(29):
            await socket.send(block)
        body = json.loads(await socket.recv())
        assert body["status"] == "error"
        with pytest.raises(Exception):
            await socket.recv()


@pytest.mark.stt_service
async def test_sdk_transcribe_ws_chunked_frames():
    # >16 MB in one call: the SDK splits it into 1 MB frames so the server
    # never sees an over-limit single frame
    async with AsyncNavaClient() as nava:
        result = await nava.transcribe_ws(b"\x00" * 17_000_000)
        assert hasattr(result, "text")


@pytest.mark.tts_service
async def test_tts_websocket_streams_then_completes():
    async with tts_ws() as socket:
        await socket.send(json.dumps({"text": "سلام دنیا", "engine": "piper"}, ensure_ascii=False))
        meta = json.loads(await socket.recv())
        assert meta["status"] == "streaming" and meta["sample_rate"] > 8000
        total = 0
        while True:
            frame = await socket.recv()
            if isinstance(frame, bytes):
                assert len(frame) % 2 == 0
                total += len(frame)
                continue
            body = json.loads(frame)
            if body.get("status") == "completed":
                break
            assert body.get("status") != "error"
        assert total > 0


@pytest.mark.tts_service
async def test_tts_websocket_reports_validation_errors():
    async with tts_ws() as socket:
        await socket.send(json.dumps({"text": "سلام", "engine": "pocket", "speed": 2.0}))
        body = json.loads(await socket.recv())
        assert body["status"] == "error"
        assert "speed" in body["error"]


@pytest.mark.tts_service
async def test_tts_websocket_invalid_json_reports_error():
    async with tts_ws() as socket:
        await socket.send("{not valid json")
        body = json.loads(await socket.recv())
        assert body["status"] == "error"


@pytest.mark.tts_service
async def test_sdk_stream_speech_ws_yields_audio():
    async with AsyncNavaClient() as nava:
        chunks = [chunk async for chunk in nava.stream_speech_ws("سلام دنیا", engine="mana")]
    assert chunks and all(c.sample_rate > 8000 for c in chunks)
    assert b"".join(c.pcm for c in chunks)

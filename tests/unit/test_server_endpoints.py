"""Unit tests for FastAPI REST and WebSocket Server Endpoints."""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from nava.server.app import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(duration_s * sample_rate))
    return buf.getvalue()


class TestServerHealth:
    def test_health_routes(self, client: TestClient):
        for path in ("/healthz", "/api/v1/health", "/v1/health"):
            r = client.get(path)
            assert r.status_code == 200
            assert r.json()["status"] == "ok"


class TestModelsEndpoints:
    def test_get_models(self, client: TestClient):
        r = client.get("/v1/models")
        assert r.status_code == 200
        models = r.json()
        assert len(models) >= 21
        model_ids = {m["id"] for m in models}
        assert "shenava-koochik-ctc" in model_ids
        assert "piper-mana-lca" in model_ids

    def test_download_endpoint_validation(self, client: TestClient):
        # Empty body should raise 400
        r = client.post("/v1/models/download", json={})
        assert r.status_code == 400


class TestVoiceCloningEndpoint:
    def test_clone_voice_upload(self, client: TestClient):
        wav = make_wav_bytes(duration_s=2.0)
        files = {"file": ("ref.wav", wav, "audio/wav")}
        data = {"name": "test_speaker"}
        r = client.post("/v1/voices/clone", files=files, data=data)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert "voice_test_speaker" in body["voice_id"]
        assert body["duration_seconds"] >= 1.9


class TestSTTEndpoints:
    def test_transcribe_octet_stream(self, client: TestClient):
        raw_pcm = b"\x00\x00" * 8000
        r = client.post(
            "/v1/audio/transcriptions?model=shenava-koochik-ctc&enable_itn=true",
            content=raw_pcm,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert "itn_text" in data
        assert "segments" in data

    def test_transcribe_legacy_route(self, client: TestClient):
        raw_pcm = b"\x00\x00" * 8000
        r = client.post(
            "/api/v1/transcriptions",
            content=raw_pcm,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200
        assert "text" in r.json()

    def test_transcribe_multipart_form(self, client: TestClient):
        wav = make_wav_bytes(0.5)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {
            "model": "shenava-koochik-ctc",
            "enable_itn": "true",
            "custom_dictionary": json.dumps({"اسنپفود": "اسنپ‌فود"}),
        }
        r = client.post("/v1/audio/transcriptions", files=files, data=data)
        assert r.status_code == 200
        assert "text" in r.json()


class TestTTSEndpoints:
    def test_speech_modern_payload(self, client: TestClient):
        payload = {
            "input": "سلام به پروژه نوا",
            "model": "piper-mana-lca",
            "voice": "mana",
            "speed": 1.0,
            "response_format": "pcm",
        }
        r = client.post("/v1/audio/speech", json=payload)
        assert r.status_code == 200
        assert r.headers["X-Nava-Sample-Rate"] == "22050"
        assert len(r.content) > 0

    def test_speech_wav_format(self, client: TestClient):
        payload = {
            "input": "سلام",
            "model": "ava-82m",
            "response_format": "wav",
        }
        r = client.post("/v1/audio/speech", json=payload)
        assert r.status_code == 200
        assert r.content.startswith(b"RIFF")

    def test_speech_legacy_payload(self, client: TestClient):
        payload = {
            "text": "سلام",
            "engine": "piper",
            "voice_prompt": "default",
            "speed": 1.2,
        }
        r = client.post("/api/v1/speech", json=payload)
        assert r.status_code == 200
        assert r.headers["X-Nava-Engine"] == "piper"
        assert len(r.content) > 0


class TestWebSockets:
    def test_websocket_stt_flow(self, client: TestClient):
        with client.websocket_connect("/v1/ws/stt") as ws:
            # 1. Send handshake
            ws.send_json({
                "model": "shenava-rizeh-streaming-32m",
                "sample_rate": 16000,
                "latency_ms": 80,
                "enable_itn": True,
            })
            resp = ws.receive_json()
            assert resp.get("status") == "configured"

            # 2. Stream audio chunk
            ws.send_bytes(b"\x00\x00" * 4000)

            # 3. Send EOS
            ws.send_json({"action": "eos"})
            final_res = None
            for _ in range(5):
                msg = ws.receive_json()
                if msg.get("type") == "final":
                    final_res = msg
                    break
            assert final_res is not None
            assert final_res.get("type") == "final"
            assert "text" in final_res

    def test_websocket_tts_flow(self, client: TestClient):
        with client.websocket_connect("/v1/ws/tts") as ws:
            # Send text request
            ws.send_json({
                "type": "text",
                "text": "سلام به همه کاربران",
                "model": "piper-mana-lca",
                "voice": "mana",
            })

            # Expect metadata
            meta = ws.receive_json()
            assert meta.get("status") == "streaming"
            assert meta.get("sample_rate") == 22050

            # Expect at least one audio chunk
            chunk = ws.receive_bytes()
            assert len(chunk) > 0

            # Expect completion
            done = ws.receive_json()
            assert done.get("status") == "completed" or done.get("type") == "done"

    def test_websocket_ping_pong(self, client: TestClient):
        with client.websocket_connect("/v1/ws/tts") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp.get("type") == "pong"

        with client.websocket_connect("/v1/ws/stt") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp.get("type") == "pong"


class TestNLPAndOCREndpoints:
    def test_sentiment_endpoint(self, client: TestClient):
        r = client.post("/v1/nlp/sentiment", json={"text": "کیفیت غذا واقعا عالی و تحویل سریع بود"})
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "score" in data
        assert "latency_ms" in data

    def test_sentiment_validation(self, client: TestClient):
        r = client.post("/v1/nlp/sentiment", json={"text": ""})
        assert r.status_code == 422

    def test_ocr_base64_endpoint(self, client: TestClient):
        import base64
        # Minimal 1x1 GIF / PNG bytes
        tiny_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        b64 = base64.b64encode(tiny_png).decode("ascii")
        r = client.post("/v1/ocr/farsi", json={"image_base64": b64})
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert "model" in data


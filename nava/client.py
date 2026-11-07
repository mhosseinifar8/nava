"""Nava Python SDK Client.

Provides synchronous (NavaClient) and asynchronous (AsyncNavaClient) interfaces
for Persian STT, TTS, G2P, and Voice Cloning over REST and WebSockets.
"""
from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

WS_FRAME_CHUNK = 1_000_000  # stay well under server per-frame limits


class NavaServiceError(RuntimeError):
    """Raised when the Nava server returns an error."""
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language_prob: float = 1.0
    itn_text: str | None = None
    rtf: float | None = None
    model: str | None = None


@dataclass(frozen=True)
class SpeechResult:
    pcm: bytes
    sample_rate: int
    engine: str
    voice: str = "default"
    format: str = "pcm"


def check(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise NavaServiceError(f"Nava returned HTTP {response.status_code}: {detail}")


def ws_url(url: str, path: str) -> str:
    base = (
        "wss://" + url[8:]
        if url.startswith("https://")
        else "ws://" + url[7:]
        if url.startswith("http://")
        else url
    )
    return base.rstrip("/") + path


async def _recv_or_raise(socket: Any) -> str | bytes:
    try:
        return await socket.recv()
    except ConnectionClosed as exc:
        raise NavaServiceError(f"connection closed before result arrived: {exc}") from exc


async def _send_chunked(socket: Any, payload: bytes) -> None:
    for i in range(0, len(payload), WS_FRAME_CHUNK):
        await socket.send(payload[i : i + WS_FRAME_CHUNK])


class NavaClient:
    """Synchronous client for Nava Persian Speech Services."""

    def __init__(
        self,
        stt_url: str = "http://127.0.0.1:8001",
        tts_url: str = "http://127.0.0.1:8002",
        *,
        timeout: float = 120.0,
    ):
        self.stt_url = stt_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def __enter__(self) -> NavaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    def transcribe(
        self,
        pcm: bytes,
        *,
        model: str = "shenava-koochik-ctc",
        enable_itn: bool = True,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_dictionary: dict[str, str] | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe speech audio PCM/WAV to Persian text."""
        # Check if server endpoint is /api/v1/transcriptions (legacy or unified)
        url = f"{self.stt_url}/api/v1/transcriptions"

        hw_param = json.dumps(hotwords, ensure_ascii=False) if isinstance(hotwords, (dict, list)) else hotwords
        params: dict[str, Any] = {
            "model": model,
            "enable_itn": enable_itn,
        }
        if hw_param:
            params["hotwords"] = hw_param
        if initial_prompt:
            params["initial_prompt"] = initial_prompt

        response = self.http.post(
            url,
            content=pcm,
            headers={"Content-Type": "application/octet-stream"},
            params=params,
        )
        check(response)
        data = response.json()
        return TranscriptionResult(
            text=data.get("text", ""),
            segments=data.get("segments", []),
            language_prob=float(data.get("language_prob", 1.0)),
            itn_text=data.get("itn_text"),
            rtf=data.get("rtf"),
            model=data.get("model", model),
        )

    def synthesize(
        self,
        text: str,
        *,
        model: str = "piper-mana-lca",
        engine: str | None = None,
        voice: str = "default",
        voice_prompt: str = "default",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
        response_format: str = "pcm",
    ) -> SpeechResult:
        """Synthesize Persian text to speech."""
        url = f"{self.tts_url}/api/v1/speech"
        target_model = engine or model
        target_voice = voice if voice != "default" else voice_prompt

        payload: dict[str, Any] = {
            "text": text,
            "engine": target_model,
            "voice_prompt": target_voice,
            "speed": speed,
        }
        if engine is None and model != "piper-mana-lca":
            payload["model"] = model
        if voice != "default" and voice_prompt == "default":
            payload["voice"] = voice
        if g2p_mode != "lca_albert":
            payload["g2p_mode"] = g2p_mode
        if response_format != "pcm":
            payload["response_format"] = response_format
        if custom_dictionary:
            payload["custom_dictionary"] = custom_dictionary
        if reference_audio:
            payload["reference_audio_base64"] = base64.b64encode(reference_audio).decode("ascii")

        response = self.http.post(url, json=payload)
        check(response)

        rate = int(response.headers.get("X-Nava-Sample-Rate", 22050))
        eng = response.headers.get("X-Nava-Engine", target_model)
        v = response.headers.get("X-Nava-Voice", target_voice)

        return SpeechResult(
            pcm=response.content,
            sample_rate=rate,
            engine=eng,
            voice=v,
            format=response_format,
        )

    def clone_voice(self, audio_bytes: bytes, name: str | None = None) -> str:
        """Upload reference audio (5–30s) and receive cached voice_id."""
        url = f"{self.tts_url}/v1/voices/clone"
        files = {"file": ("reference.wav", audio_bytes, "audio/wav")}
        data = {"name": name} if name else {}
        response = self.http.post(url, files=files, data=data)
        check(response)
        return str(response.json()["voice_id"])

    def list_models(self) -> list[dict[str, Any]]:
        """List all models registered in the service."""
        url = f"{self.stt_url}/v1/models"
        response = self.http.get(url)
        check(response)
        return response.json()

    def download_model(self, model_id: str | None = None, profile: str | None = None) -> dict[str, Any]:
        """Trigger model download."""
        url = f"{self.stt_url}/v1/models/download"
        response = self.http.post(url, json={"model": model_id, "profile": profile})
        check(response)
        return response.json()


class AsyncNavaClient:
    """Asynchronous client for Nava Persian Speech Services."""

    def __init__(
        self,
        stt_url: str = "http://127.0.0.1:8001",
        tts_url: str = "http://127.0.0.1:8002",
        *,
        timeout: float = 120.0,
    ):
        self.stt_url = stt_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> AsyncNavaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def transcribe(
        self,
        pcm: bytes,
        *,
        model: str = "shenava-koochik-ctc",
        enable_itn: bool = True,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_dictionary: dict[str, str] | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        url = f"{self.stt_url}/api/v1/transcriptions"
        hw_param = json.dumps(hotwords, ensure_ascii=False) if isinstance(hotwords, (dict, list)) else hotwords
        params: dict[str, Any] = {
            "model": model,
            "enable_itn": enable_itn,
        }
        if hw_param:
            params["hotwords"] = hw_param
        if initial_prompt:
            params["initial_prompt"] = initial_prompt

        response = await self.http.post(
            url,
            content=pcm,
            headers={"Content-Type": "application/octet-stream"},
            params=params,
        )
        check(response)
        data = response.json()
        return TranscriptionResult(
            text=data.get("text", ""),
            segments=data.get("segments", []),
            language_prob=float(data.get("language_prob", 1.0)),
            itn_text=data.get("itn_text"),
            rtf=data.get("rtf"),
            model=data.get("model", model),
        )

    async def synthesize(
        self,
        text: str,
        *,
        model: str = "piper-mana-lca",
        engine: str | None = None,
        voice: str = "default",
        voice_prompt: str = "default",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
        response_format: str = "pcm",
    ) -> SpeechResult:
        url = f"{self.tts_url}/api/v1/speech"
        target_model = engine or model
        target_voice = voice if voice != "default" else voice_prompt

        payload: dict[str, Any] = {
            "text": text,
            "engine": target_model,
            "voice_prompt": target_voice,
            "speed": speed,
        }
        if engine is None and model != "piper-mana-lca":
            payload["model"] = model
        if voice != "default" and voice_prompt == "default":
            payload["voice"] = voice
        if g2p_mode != "lca_albert":
            payload["g2p_mode"] = g2p_mode
        if response_format != "pcm":
            payload["response_format"] = response_format
        if custom_dictionary:
            payload["custom_dictionary"] = custom_dictionary
        if reference_audio:
            payload["reference_audio_base64"] = base64.b64encode(reference_audio).decode("ascii")

        response = await self.http.post(url, json=payload)
        check(response)

        rate = int(response.headers.get("X-Nava-Sample-Rate", 22050))
        eng = response.headers.get("X-Nava-Engine", target_model)
        v = response.headers.get("X-Nava-Voice", target_voice)

        return SpeechResult(
            pcm=response.content,
            sample_rate=rate,
            engine=eng,
            voice=v,
            format=response_format,
        )

    async def clone_voice(self, audio_bytes: bytes, name: str | None = None) -> str:
        url = f"{self.tts_url}/v1/voices/clone"
        files = {"file": ("reference.wav", audio_bytes, "audio/wav")}
        data = {"name": name} if name else {}
        response = await self.http.post(url, files=files, data=data)
        check(response)
        return str(response.json()["voice_id"])

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self.stt_url}/v1/models"
        response = await self.http.get(url)
        check(response)
        return response.json()

    async def transcribe_ws(
        self,
        pcm: bytes,
        *,
        model: str = "shenava-rizeh-streaming-32m",
        enable_itn: bool = True,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_dictionary: dict[str, str] | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Stream PCM audio to WebSocket and receive final transcription."""
        async with connect(ws_url(self.stt_url, "/ws/stt"), max_size=None) as socket:
            config: dict[str, Any] = {
                "model": model,
                "enable_itn": enable_itn,
            }
            if hotwords:
                config["hotwords"] = hotwords
            if custom_dictionary:
                config["custom_dictionary"] = custom_dictionary
            if initial_prompt:
                config["initial_prompt"] = initial_prompt

            await socket.send(json.dumps(config, ensure_ascii=False))
            await _send_chunked(socket, pcm)
            await socket.send('{"action":"eos"}')

            while True:
                frame = await _recv_or_raise(socket)
                if isinstance(frame, bytes):
                    continue
                body = json.loads(frame)
                if body.get("status") == "error" or body.get("type") == "error":
                    raise NavaServiceError(body.get("error", "WebSocket error"))
                if "text" in body and body.get("type") in ("final", None):
                    return TranscriptionResult(
                        text=body["text"],
                        segments=body.get("segments", []),
                        language_prob=float(body.get("language_prob", 1.0)),
                        itn_text=body.get("itn_text"),
                        rtf=body.get("rtf"),
                        model=body.get("model", model),
                    )

    async def stream_speech_ws(
        self,
        text: str,
        *,
        model: str = "piper-mana-lca",
        engine: str | None = None,
        voice: str = "default",
        voice_prompt: str = "default",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
    ) -> AsyncIterator[SpeechResult]:
        """Low-latency sentence-by-sentence audio streaming over WebSocket."""
        async with connect(ws_url(self.tts_url, "/ws/tts"), max_size=None) as socket:
            target_model = engine or model
            target_voice = voice if voice != "default" else voice_prompt

            config: dict[str, Any] = {
                "type": "text",
                "text": text,
                "model": target_model,
                "engine": target_model,
                "voice": target_voice,
                "voice_prompt": target_voice,
                "speed": speed,
                "g2p_mode": g2p_mode,
            }
            if custom_dictionary:
                config["custom_dictionary"] = custom_dictionary
            if reference_audio:
                config["reference_audio_base64"] = base64.b64encode(reference_audio).decode("ascii")

            await socket.send(json.dumps(config, ensure_ascii=False))
            rate = None
            engine_name = target_model

            while True:
                frame = await _recv_or_raise(socket)
                if isinstance(frame, bytes):
                    if rate is None:
                        raise NavaServiceError("TTS sent audio before metadata")
                    yield SpeechResult(pcm=frame, sample_rate=rate, engine=engine_name, voice=target_voice)
                    continue

                body = json.loads(frame)
                if body.get("status") == "error":
                    raise NavaServiceError(body.get("error", "WebSocket error"))
                if body.get("status") == "streaming":
                    rate = int(body["sample_rate"])
                    engine_name = body.get("engine", target_model)
                if body.get("status") == "completed" or body.get("type") == "done":
                    return

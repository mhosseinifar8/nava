"""REST endpoints for Text-to-Speech Synthesis."""
from __future__ import annotations

import asyncio
import base64
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ..engines.factory import get_tts_engine

router = APIRouter(tags=["TTS"])


class SpeechRequest(BaseModel):
    input: str | None = Field(None, description="Persian text to synthesize")
    text: str | None = Field(None, description="Alias for input (backward compatibility)")
    model: str = Field("piper-mana-lca", description="TTS model ID or engine name")
    engine: str | None = Field(None, description="Alias for model (backward compatibility)")
    voice: str = Field("default", description="Speaker name, character, or voice_id")
    voice_prompt: str | None = Field(None, description="Alias for voice (backward compatibility)")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    g2p_mode: Literal["lca_albert", "homo_ge2pe", "fast_espeak"] = Field(
        "lca_albert", description="G2P pre-processor pipeline mode"
    )
    custom_dictionary: dict[str, str] | None = Field(
        None, description="Custom word/pronunciation overrides"
    )
    reference_audio_base64: str | None = Field(
        None, description="Optional 5-30s reference audio for zero-shot voice cloning"
    )
    response_format: Literal["pcm", "wav", "mp3"] = Field("pcm", description="Output audio format")


@router.post("/v1/audio/speech")
@router.post("/api/v1/speech")
@router.post("/v1/v1/audio/speech")
@router.post("/v1/api/v1/speech")
@router.post("/speech")
@router.post("/audio/speech")
async def generate_speech(payload: SpeechRequest) -> Response:
    """Synthesize Persian text to speech with voice cloning and multi-speaker support."""
    text_content = payload.input or payload.text
    if not text_content or not text_content.strip():
        raise HTTPException(422, "Text content ('input' or 'text') is required and cannot be empty")

    model_name = payload.engine or payload.model
    voice_name = payload.voice_prompt or payload.voice

    ref_audio_bytes: bytes | None = None
    if payload.reference_audio_base64:
        try:
            ref_audio_bytes = base64.b64decode(payload.reference_audio_base64)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 in reference_audio_base64: {e}")

    try:
        engine = get_tts_engine(model_name)
    except KeyError as exc:
        raise HTTPException(400, f"Unsupported TTS model/engine: {model_name}") from exc

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: engine.synthesize(
                text_content,
                voice=voice_name,
                speed=payload.speed,
                g2p_mode=payload.g2p_mode,
                custom_dictionary=payload.custom_dictionary,
                reference_audio=ref_audio_bytes,
            ),
        )
    except Exception as exc:
        raise HTTPException(500, f"Speech synthesis failed: {exc}") from exc

    headers = {
        "X-Nava-Sample-Rate": str(result.sample_rate),
        "X-Nava-Engine": result.engine,
        "X-Nava-Voice": result.voice,
    }

    if payload.response_format == "wav":
        wav_bytes = result.to_wav()
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    else:
        # Default PCM L16
        media_type = f"audio/L16;rate={result.sample_rate};channels=1"
        return Response(content=result.audio, media_type=media_type, headers=headers)


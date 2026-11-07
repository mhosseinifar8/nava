"""REST endpoints for Voice Cloning and Reference Audio Management."""
from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..voices.voice_manager import get_voice_manager

router = APIRouter(tags=["Voice Cloning"])


class CloneRequestJSON(BaseModel):
    reference_audio_base64: str
    name: str | None = None


@router.post("/v1/voices/clone")
async def clone_voice(
    file: UploadFile | None = File(None),
    name: str | None = Form(None),
    body: CloneRequestJSON | None = None,
) -> dict[str, Any]:
    """Upload 5–30s reference audio, normalize to 24kHz mono, and return voice_id."""
    voice_mgr = get_voice_manager()
    audio_bytes: bytes | None = None
    target_name = name

    if file:
        audio_bytes = await file.read()
    elif body:
        try:
            audio_bytes = base64.b64decode(body.reference_audio_base64)
            target_name = body.name or name
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 reference audio: {e}")

    if not audio_bytes:
        raise HTTPException(400, "Reference audio must be provided via file upload or base64 JSON")

    try:
        voice_id, duration_s = voice_mgr.create_voice_profile(audio_bytes, name=target_name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Voice cloning processing error: {exc}") from exc

    return {
        "voice_id": voice_id,
        "status": "ready",
        "duration_seconds": round(duration_s, 2),
        "sample_rate": 24000,
        "message": f"Voice profile registered. Pass voice='{voice_id}' to TTS requests.",
    }


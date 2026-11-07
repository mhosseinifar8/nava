"""REST endpoints for Speech-to-Text Transcriptions."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from ..engines.factory import get_stt_engine

router = APIRouter(tags=["STT"])


def _parse_dict_or_json(val: Any) -> Any:
    if val is None or isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


@router.post("/v1/audio/transcriptions")
@router.post("/api/v1/transcriptions")
async def transcribe_audio(
    request: Request,
    file: UploadFile | None = File(None),
    model: str = Form("shenava-koochik-ctc"),
    enable_itn: bool = Form(True),
    hotwords: str | None = Form(None),
    custom_dictionary: str | None = Form(None),
    initial_prompt: str | None = Form(None),
    # Support query params for octet-stream backward compatibility
    q_model: str | None = Query(None, alias="model"),
    q_enable_itn: bool = Query(True, alias="enable_itn"),
    q_hotwords: str | None = Query(None, alias="hotwords"),
    q_prompt: str | None = Query(None, alias="initial_prompt"),
    content_type: str | None = Header(None),
) -> dict[str, Any]:
    """Transcribe Persian audio with optional ITN, hotwords injection, and custom lexicon."""
    audio_bytes: bytes | None = None

    # Handle application/octet-stream raw binary body
    if content_type and "application/octet-stream" in content_type:
        audio_bytes = await request.body()
        target_model = q_model or model or "shenava-koochik-ctc"
        target_itn = q_enable_itn
        target_hotwords = q_hotwords or hotwords
        target_prompt = q_prompt or initial_prompt
        target_dict = None
    elif file is not None:
        audio_bytes = await file.read()
        target_model = model or q_model or "shenava-koochik-ctc"
        target_itn = enable_itn
        target_hotwords = hotwords or q_hotwords
        target_prompt = initial_prompt or q_prompt
        target_dict = custom_dictionary
    else:
        # Check raw body
        body = await request.body()
        if body:
            audio_bytes = body
            target_model = q_model or model or "shenava-koochik-ctc"
            target_itn = q_enable_itn
            target_hotwords = q_hotwords or hotwords
            target_prompt = q_prompt or initial_prompt
            target_dict = None
        else:
            raise HTTPException(400, "Audio data missing from request")

    if not audio_bytes:
        raise HTTPException(422, "Audio buffer is empty")

    parsed_hotwords = _parse_dict_or_json(target_hotwords)
    parsed_dict = _parse_dict_or_json(target_dict)

    try:
        engine = get_stt_engine(target_model)
    except KeyError as exc:
        raise HTTPException(400, f"Unsupported STT model: {target_model}") from exc

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: engine.transcribe(
                audio_bytes,
                hotwords=parsed_hotwords,
                custom_replacements=parsed_dict if isinstance(parsed_dict, dict) else None,
                enable_itn=target_itn,
                initial_prompt=target_prompt,
            ),
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    return result.to_dict()


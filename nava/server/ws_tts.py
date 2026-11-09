"""Low-Latency Sentence-Streaming WebSocket Text-to-Speech Endpoint."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engines.factory import get_tts_engine

router = APIRouter(tags=["TTS Streaming"])


@router.websocket("/v1/ws/tts")
@router.websocket("/ws/tts")
@router.websocket("/v1/v1/ws/tts")
@router.websocket("/v1/api/v1/ws/tts")
@router.websocket("/api/v1/ws/tts")
@router.websocket("/v1/ws")
@router.websocket("/ws")
async def websocket_tts_endpoint(ws: WebSocket) -> None:
    """Low-latency pipelined sentence-streaming TTS WebSocket handler."""
    await ws.accept()

    config: dict[str, Any] = {
        "model": "piper-mana-lca",
        "voice": "default",
        "speed": 1.0,
        "g2p_mode": "lca_albert",
        "custom_dictionary": None,
        "reference_audio_bytes": None,
    }

    engine = None
    cancel_event = asyncio.Event()

    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                return

            raw_text = frame.get("text") or ""
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_json({"status": "error", "error": "Message must be valid JSON"})
                continue

            msg_type = msg.get("type")

            # Check for cancellation
            if msg_type == "cancel" or msg.get("action") == "cancel":
                cancel_event.set()
                await ws.send_json({"type": "cancelled", "status": "cancelled"})
                continue

            # Update configuration if provided
            for k in ("model", "engine", "voice", "voice_prompt", "speed", "g2p_mode", "custom_dictionary"):
                if k in msg and msg[k] is not None:
                    target_key = "model" if k == "engine" else "voice" if k == "voice_prompt" else k
                    config[target_key] = msg[k]

            if "reference_audio_base64" in msg and msg["reference_audio_base64"]:
                try:
                    config["reference_audio_bytes"] = base64.b64decode(msg["reference_audio_base64"])
                except Exception:
                    pass

            # Check if this frame contains text to synthesize
            text_to_speak = msg.get("text")
            if msg_type == "text" or (text_to_speak and msg_type != "config"):
                cancel_event.clear()
                try:
                    engine = get_tts_engine(config["model"])
                except Exception as e:
                    await ws.send_json({"status": "error", "error": str(e)})
                    continue

                sample_rate = getattr(engine, "sample_rate", 22050)

                # Send streaming start metadata (OpenAI + Nava format)
                await ws.send_json({
                    "status": "streaming",
                    "type": "streaming",
                    "engine": config["model"],
                    "model": config["model"],
                    "sample_rate": sample_rate,
                })

                try:
                    # Stream chunks sentence-by-sentence
                    async for chunk in engine.synthesize_stream(
                        text_to_speak,
                        voice=config["voice"],
                        speed=config["speed"],
                        g2p_mode=config["g2p_mode"],
                        custom_dictionary=config["custom_dictionary"],
                        reference_audio=config["reference_audio_bytes"],
                    ):
                        if cancel_event.is_set():
                            break
                        await ws.send_bytes(chunk)

                    if not cancel_event.is_set():
                        await ws.send_json({"status": "completed", "type": "done"})
                except Exception as exc:
                    await ws.send_json({"status": "error", "error": str(exc)})

    except WebSocketDisconnect:
        return


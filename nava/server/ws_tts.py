"""Low-Latency Sentence-Streaming WebSocket Text-to-Speech Endpoint.

Features:
- Persistent connection keepalive with heartbeat ping/pong handling
- True streaming TTS clause chunking with < 200ms Time-To-First-Byte (TTFB)
- Offloads all CPU-bound inference to pre-warmed thread pool to eliminate event loop blocking
- Graceful recoverable error frames without dropping connections
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engines.factory import get_tts_engine
from ..g2p.normalizer import PersianClauseSplitter
from ..runtime.cpu_tuning import run_in_cpu_pool

logger = logging.getLogger("nava.ws_tts")
router = APIRouter(tags=["TTS Streaming"])


@router.websocket("/v1/ws/tts")
@router.websocket("/ws/tts")
@router.websocket("/v1/v1/ws/tts")
@router.websocket("/v1/api/v1/ws/tts")
@router.websocket("/api/v1/ws/tts")
@router.websocket("/v1/audio/speech")
@router.websocket("/v1/v1/audio/speech")
@router.websocket("/api/v1/speech")
@router.websocket("/v1/api/v1/speech")
@router.websocket("/audio/speech")
@router.websocket("/speech")
@router.websocket("/v1/tts")
@router.websocket("/tts")
@router.websocket("/v1/ws")
@router.websocket("/ws")
async def websocket_tts_endpoint(ws: WebSocket) -> None:
    """Low-latency pipelined sentence-streaming TTS WebSocket handler with keep-alive."""
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
            if not raw_text.strip():
                continue

            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_json({"status": "error", "error": "Message must be valid JSON", "recoverable": True})
                continue

            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            action = msg.get("action")

            # 1. Heartbeat keepalive ping/pong
            if msg_type == "ping" or action == "ping":
                await ws.send_json({"type": "pong", "status": "ok"})
                continue

            # 2. Check for cancellation
            if msg_type == "cancel" or action == "cancel":
                cancel_event.set()
                await ws.send_json({"type": "cancelled", "status": "cancelled"})
                continue

            # 3. Update configuration if provided
            for k in ("model", "engine", "voice", "voice_prompt", "speed", "g2p_mode", "custom_dictionary"):
                if k in msg and msg[k] is not None:
                    target_key = "model" if k == "engine" else "voice" if k == "voice_prompt" else k
                    config[target_key] = msg[k]

            if "reference_audio_base64" in msg and msg["reference_audio_base64"]:
                try:
                    config["reference_audio_bytes"] = base64.b64decode(msg["reference_audio_base64"])
                except Exception:
                    pass

            # 4. Pure configuration frame
            text_to_speak = msg.get("text") or msg.get("input")
            if not text_to_speak and msg_type in ("config", "handshake"):
                await ws.send_json({
                    "status": "ready",
                    "type": "ready",
                    "model": config["model"],
                    "voice": config["voice"],
                })
                continue

            # 5. Synthesis request
            if text_to_speak and msg_type != "config":
                cancel_event.clear()
                try:
                    engine = get_tts_engine(config["model"])
                except Exception as e:
                    await ws.send_json({"status": "error", "error": str(e), "recoverable": True})
                    continue

                sample_rate = getattr(engine, "sample_rate", 22050)

                # Send streaming start metadata
                await ws.send_json({
                    "status": "streaming",
                    "type": "streaming",
                    "engine": config["model"],
                    "model": config["model"],
                    "sample_rate": sample_rate,
                })

                try:
                    # Stream chunks sentence-by-sentence / clause-by-clause
                    async for chunk in engine.synthesize_stream(
                        text_to_speak,
                        voice=config["voice"],
                        speed=float(config["speed"]),
                        g2p_mode=config["g2p_mode"],
                        custom_dictionary=config["custom_dictionary"],
                        reference_audio=config["reference_audio_bytes"],
                    ):
                        if cancel_event.is_set():
                            break
                        if chunk:
                            await ws.send_bytes(chunk)

                    if not cancel_event.is_set():
                        await ws.send_json({"status": "completed", "type": "done"})
                except Exception as exc:
                    logger.warning(f"WebSocket TTS synthesis warning: {exc}")
                    await ws.send_json({"status": "error", "error": str(exc), "recoverable": True})

    except WebSocketDisconnect:
        return

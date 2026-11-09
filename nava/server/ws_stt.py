"""Real-Time Streaming WebSocket Speech-to-Text Endpoint."""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..engines.factory import get_stt_engine

router = APIRouter(tags=["STT Streaming"])


@router.websocket("/v1/ws/stt")
@router.websocket("/ws/stt")
@router.websocket("/v1/v1/ws/stt")
@router.websocket("/v1/api/v1/ws/stt")
@router.websocket("/api/v1/ws/stt")
@router.websocket("/v1/audio/transcriptions")
@router.websocket("/v1/v1/audio/transcriptions")
@router.websocket("/api/v1/transcriptions")
@router.websocket("/v1/api/v1/transcriptions")
@router.websocket("/audio/transcriptions")
@router.websocket("/transcriptions")
@router.websocket("/v1/stt")
@router.websocket("/stt")
@router.websocket("/v1/ws")
@router.websocket("/ws")
async def websocket_stt_endpoint(ws: WebSocket) -> None:
    """Real-Time Streaming Speech-to-Text WebSocket handler."""
    await ws.accept()

    config: dict[str, Any] = {
        "model": "shenava-rizeh-streaming-32m",
        "sample_rate": 16000,
        "latency_ms": 80,
        "enable_itn": True,
        "hotwords": None,
        "custom_dictionary": None,
    }

    engine = None
    stream_obj = None
    accumulated_pcm = bytearray()
    t_start = time.perf_counter()

    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                return

            # Binary audio chunk received
            if frame.get("bytes") is not None:
                chunk = frame["bytes"]
                accumulated_pcm.extend(chunk)

                if engine is None:
                    engine = get_stt_engine(config["model"])

                if hasattr(engine, "feed_audio"):
                    if stream_obj is None:
                        stream_obj = engine.create_stream(
                            hotwords=config["hotwords"],
                            latency_ms=config["latency_ms"],
                        )
                    engine.feed_audio(stream_obj, chunk)
                    partial_text = engine.get_partial(stream_obj)
                    if partial_text:
                        await ws.send_json({"type": "partial", "text": partial_text})
                continue

            # Text / Control message received
            raw_text = frame.get("text") or ""
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_json({"status": "error", "error": "Control frame must be valid JSON"})
                continue

            action = msg.get("action")

            # Check if this is an EOF / EOS event
            if action in ("eos", "eof"):
                if engine is None:
                    engine = get_stt_engine(config["model"])

                if hasattr(engine, "get_final") and stream_obj is not None:
                    raw, itn = engine.get_final(stream_obj, enable_itn=config["enable_itn"])
                    duration_s = max(len(accumulated_pcm) / 32000.0, 0.01)
                    elapsed = time.perf_counter() - t_start
                    rtf = round(elapsed / duration_s, 3)

                    # Send unified modern final event
                    await ws.send_json({
                        "type": "final",
                        "status": "success",
                        "text": raw,
                        "itn_text": itn,
                        "rtf": rtf,
                        "segments": [{"start": 0.0, "end": round(duration_s, 3), "text": itn}],
                        "language_prob": 0.99,
                    })
                else:
                    # Non-streaming engine fallback
                    res = engine.transcribe(
                        bytes(accumulated_pcm),
                        hotwords=config["hotwords"],
                        custom_replacements=config["custom_dictionary"],
                        enable_itn=config["enable_itn"],
                        initial_prompt=config.get("initial_prompt"),
                    )
                    await ws.send_json(res.to_dict())

                accumulated_pcm.clear()
                stream_obj = None
                t_start = time.perf_counter()
                continue

            # Otherwise, update configuration handshake
            for k in ("model", "sample_rate", "latency_ms", "enable_itn", "hotwords", "custom_dictionary", "initial_prompt"):
                if k in msg and msg[k] is not None:
                    config[k] = msg[k]

            # Re-initialize engine if model changed
            try:
                engine = get_stt_engine(config["model"])
                stream_obj = None
                await ws.send_json({
                    "type": "ready",
                    "status": "configured",
                    "model": config["model"],
                    "latency_ms": config["latency_ms"],
                })
            except Exception as e:
                await ws.send_json({"type": "error", "status": "error", "error": str(e)})

    except WebSocketDisconnect:
        return


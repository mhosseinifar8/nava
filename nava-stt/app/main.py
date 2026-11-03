from __future__ import annotations

import asyncio, json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

from .bodylimit import TooLargeError, read_limited
from .settings import Settings

PROMPT = "متن گفتاری فارسی را با املای معیار، نیم‌فاصله و نشانه‌گذاری درست بنویسید."

class Result(BaseModel):
    text: str
    segments: list[dict]
    language_prob: float

class Engine:
    def __init__(self, settings: Settings) -> None:
        if not settings.model_dir.is_dir(): raise RuntimeError(f"Whisper model missing: {settings.model_dir}")
        self.settings = settings
        self.model = WhisperModel(
            str(settings.model_dir), device="cpu", compute_type=settings.compute_type,
            cpu_threads=settings.omp_threads, num_workers=settings.workers,
        )
    def run(self, pcm: bytes, initial_prompt: str | None, hotwords: str | None) -> Result:
        if not pcm or len(pcm) % 2: raise ValueError("audio must be non-empty signed-16 little-endian PCM")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < 1600: raise ValueError("audio is shorter than 100 ms at 16 kHz")
        prompt = initial_prompt or PROMPT
        if hotwords: prompt += f"\nواژه‌های مهم: {hotwords}"
        stream, info = self.model.transcribe(audio, language="fa", beam_size=5, condition_on_previous_text=False, initial_prompt=prompt, hotwords=hotwords, vad_filter=True, vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200})
        segments = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in stream if s.text.strip()]
        return Result(text=" ".join(s["text"] for s in segments), segments=segments, language_prob=round(float(info.language_probability), 5))

settings = Settings.load()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pool size matches WhisperModel num_workers, so parallel requests actually
    # execute in parallel inside CTranslate2 instead of queuing on one model worker.
    app.state.settings = settings
    app.state.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="nava-stt")
    app.state.engine = Engine(settings)
    try: yield
    finally: app.state.pool.shutdown(wait=True, cancel_futures=True)

app = FastAPI(title="Nava STT", version="1.1.0", lifespan=lifespan)

async def infer(pcm: bytes, initial_prompt: str | None = None, hotwords: str | None = None) -> Result:
    if len(pcm) > settings.max_audio_bytes: raise ValueError("audio exceeds NAVA_MAX_STT_AUDIO_BYTES")
    return await asyncio.get_running_loop().run_in_executor(app.state.pool, app.state.engine.run, pcm, initial_prompt, hotwords)

@app.get("/healthz")
@app.get("/api/v1/health")
async def health() -> dict[str, str]: return {"status": "ok", "service": "nava-stt"}

@app.post("/api/v1/transcriptions", response_model=Result, openapi_extra={
    "requestBody": {"content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}, "required": True},
})
async def transcribe(request: Request, initial_prompt: str | None = Query(None), hotwords: str | None = Query(None)) -> Result:
    try:
        pcm = await read_limited(request, settings.max_audio_bytes)
    except TooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    try:
        return await infer(pcm, initial_prompt, hotwords)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

@app.websocket("/ws/stt")
async def websocket_stt(ws: WebSocket) -> None:
    await ws.accept(); config: dict[str, str] = {}; pcm = bytearray()
    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect": return
            if frame.get("bytes") is not None:
                pcm.extend(frame["bytes"])
                if len(pcm) > settings.max_audio_bytes:
                    await ws.send_json({"status":"error","error":"audio exceeds NAVA_MAX_STT_AUDIO_BYTES"}); await ws.close(1009); return
                continue
            try: control = json.loads(frame.get("text") or "")
            except json.JSONDecodeError: await ws.send_json({"status":"error","error":"control frame must be JSON"}); continue
            if control.get("action") != "eos":
                if pcm: await ws.send_json({"status":"error","error":"configuration must precede binary audio"})
                else: config.update({k:str(v) for k,v in control.items() if k in {"initial_prompt","hotwords"} and v is not None})
                continue
            try: await ws.send_json((await infer(bytes(pcm), config.get("initial_prompt"), config.get("hotwords"))).model_dump())
            except ValueError as exc: await ws.send_json({"status":"error","error":str(exc)})
            config.clear(); pcm.clear()
    except WebSocketDisconnect: return

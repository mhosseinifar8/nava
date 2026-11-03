from __future__ import annotations

import asyncio, json, os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

MAX_AUDIO = int(os.getenv("NAVA_MAX_STT_AUDIO_BYTES", str(16_000 * 2 * 60 * 15)))
PROMPT = "متن گفتاری فارسی را با املای معیار، نیم‌فاصله و نشانه‌گذاری درست بنویسید."

class Result(BaseModel):
    text: str
    segments: list[dict]
    language_prob: float

class Engine:
    def __init__(self) -> None:
        model = Path(os.getenv("NAVA_WHISPER_MODEL_DIR", "/models/whisper-large-v3"))
        if not model.is_dir(): raise RuntimeError(f"Whisper model missing: {model}")
        self.model = WhisperModel(str(model), device="cpu", compute_type=os.getenv("NAVA_WHISPER_COMPUTE_TYPE", "float32"), cpu_threads=int(os.getenv("OMP_NUM_THREADS", "4")), num_workers=1)
    def run(self, pcm: bytes, initial_prompt: str | None, hotwords: str | None) -> Result:
        if not pcm or len(pcm) % 2: raise ValueError("audio must be non-empty signed-16 little-endian PCM")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < 1600: raise ValueError("audio is shorter than 100 ms at 16 kHz")
        prompt = initial_prompt or PROMPT
        if hotwords: prompt += f"\nواژه‌های مهم: {hotwords}"
        stream, info = self.model.transcribe(audio, language="fa", beam_size=5, condition_on_previous_text=False, initial_prompt=prompt, hotwords=hotwords, vad_filter=True, vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200})
        segments = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in stream if s.text.strip()]
        return Result(text=" ".join(s["text"] for s in segments), segments=segments, language_prob=round(float(info.language_probability), 5))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ThreadPoolExecutor(max_workers=int(os.getenv("NAVA_INFERENCE_WORKERS", "2")), thread_name_prefix="nava-stt")
    app.state.engine = Engine()
    try: yield
    finally: app.state.pool.shutdown(wait=True, cancel_futures=True)

app = FastAPI(title="Nava STT", version="1.0.0", lifespan=lifespan)

async def infer(pcm: bytes, initial_prompt: str | None = None, hotwords: str | None = None) -> Result:
    if len(pcm) > MAX_AUDIO: raise ValueError("audio exceeds NAVA_MAX_STT_AUDIO_BYTES")
    return await asyncio.get_running_loop().run_in_executor(app.state.pool, app.state.engine.run, pcm, initial_prompt, hotwords)

@app.get("/healthz")
@app.get("/api/v1/health")
async def health() -> dict[str, str]: return {"status": "ok", "service": "nava-stt"}

@app.post("/api/v1/transcriptions", response_model=Result)
async def transcribe(pcm: bytes = Body(..., media_type="application/octet-stream"), initial_prompt: str | None = Query(None), hotwords: str | None = Query(None)) -> Result:
    try: return await infer(pcm, initial_prompt, hotwords)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc

@app.websocket("/ws/stt")
async def websocket_stt(ws: WebSocket) -> None:
    await ws.accept(); config: dict[str, str] = {}; pcm = bytearray()
    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect": return
            if frame.get("bytes") is not None:
                pcm.extend(frame["bytes"])
                if len(pcm) > MAX_AUDIO:
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

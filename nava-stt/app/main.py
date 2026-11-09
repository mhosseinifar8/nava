from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
        self.settings = settings
        self.model = None
        self._find_and_init_model()

    def _find_and_init_model(self) -> None:
        candidates = [
            self.settings.model_dir,
            Path("/models/whisper-large-v3"),
            Path("/models/whisper-turbo-persian-givia"),
            Path("/models/whisper-turbo-persian-gholizadeh"),
            Path("./models/whisper-large-v3"),
        ]
        target_dir = None
        for cand in candidates:
            if cand.is_dir() and ((cand / "model.bin").is_file() or (cand / "model.safetensors").is_file()):
                target_dir = cand
                break

        if target_dir is not None:
            try:
                from faster_whisper import WhisperModel

                self.model = WhisperModel(
                    str(target_dir),
                    device="cpu",
                    compute_type=self.settings.compute_type,
                    cpu_threads=self.settings.omp_threads,
                    num_workers=self.settings.workers,
                )
            except Exception as e:
                print(f"Warning: Faster-Whisper init failed: {e}")

    def run(self, pcm: bytes, initial_prompt: str | None, hotwords: str | None) -> Result:
        if self.model is None:
            self._find_and_init_model()
        if self.model is None:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                str(self.settings.model_dir),
                device="cpu",
                compute_type=self.settings.compute_type,
                cpu_threads=self.settings.omp_threads,
                num_workers=self.settings.workers,
            )
        if not pcm or len(pcm) % 2:
            raise ValueError("audio must be non-empty signed-16 little-endian PCM")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < 1600:
            raise ValueError("audio is shorter than 100 ms at 16 kHz")
        prompt = initial_prompt or PROMPT
        if hotwords:
            prompt += f"\nواژه‌های مهم: {hotwords}"
        stream, info = self.model.transcribe(
            audio,
            language="fa",
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            hotwords=hotwords,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
        )
        segments = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()} for s in stream if s.text.strip()]
        return Result(
            text=" ".join(s["text"] for s in segments),
            segments=segments,
            language_prob=round(float(info.language_probability), 5),
        )


settings = Settings.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="nava-stt")
    try:
        app.state.engine = Engine(settings)
    except Exception as e:
        print(f"Warning: Engine startup deferred: {e}")
        app.state.engine = None
    try:
        yield
    finally:
        app.state.pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Nava STT", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def infer(
    pcm: bytes,
    model_name: str | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    enable_itn: bool = True,
) -> Result:
    if len(pcm) > settings.max_audio_bytes:
        raise ValueError("audio exceeds NAVA_MAX_STT_AUDIO_BYTES")

    target_model = model_name or "shenava-koochik-ctc"

    pool = getattr(app.state, "pool", None)
    local_engine = getattr(app.state, "engine", None)

    # 1. If Faster-Whisper is specifically requested and initialized locally, use it directly
    if target_model in ("faster-whisper", "whisper-large-v3") and local_engine is not None and getattr(local_engine, "model", None) is not None:
        return await asyncio.get_running_loop().run_in_executor(
            pool, local_engine.run, pcm, initial_prompt, hotwords
        )

    # 2. Try unified engine (shenava, nemotron, whisper variants)
    try:
        from nava.engines.factory import get_stt_engine
        stt_eng = get_stt_engine(target_model)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            pool,
            lambda: stt_eng.transcribe(
                pcm,
                hotwords=hotwords,
                enable_itn=enable_itn,
                initial_prompt=initial_prompt,
            ),
        )
        return Result(
            text=res.itn_text if enable_itn else res.text,
            segments=res.segments,
            language_prob=res.language_prob,
        )
    except Exception as exc:
        # 3. Fall back to local Faster-Whisper if available
        if local_engine is not None and getattr(local_engine, "model", None) is not None:
            return await asyncio.get_running_loop().run_in_executor(
                pool, local_engine.run, pcm, initial_prompt, hotwords
            )
        # If neither succeeded, raise informative error
        raise RuntimeError(f"STT inference error for model '{target_model}': {exc}") from exc


@app.get("/healthz")
@app.get("/api/v1/health")
@app.get("/v1/health")
async def health() -> dict[str, Any]:
    all_models = ["faster-whisper"]
    try:
        from nava.models.registry import MODELS
        all_models = sorted(list(set(all_models + [m for m, info in MODELS.items() if info.type == "stt"])))
    except Exception:
        pass
    return {"status": "ok", "service": "nava-stt", "models": all_models}


@app.post("/api/v1/transcriptions")
@app.post("/v1/audio/transcriptions")
@app.post("/v1/v1/audio/transcriptions")
@app.post("/v1/api/v1/transcriptions")
@app.post("/transcriptions")
@app.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile | None = File(None),
    model: str | None = Query(None),
    form_model: str | None = Form(None, alias="model"),
    enable_itn: bool = Query(True),
    form_enable_itn: bool = Form(True, alias="enable_itn"),
    hotwords: str | None = Query(None),
    form_hotwords: str | None = Form(None, alias="hotwords"),
    initial_prompt: str | None = Query(None),
    form_initial_prompt: str | None = Form(None, alias="initial_prompt"),
    content_type: str | None = Header(None),
) -> Result:
    target_model = model or form_model or "shenava-koochik-ctc"
    target_itn = enable_itn and form_enable_itn
    target_hotwords = hotwords or form_hotwords
    target_prompt = initial_prompt or form_initial_prompt

    pcm: bytes | None = None
    if file is not None:
        try:
            pcm = await file.read()
        except Exception as exc:
            raise HTTPException(400, f"Failed to read uploaded file: {exc}") from exc
    else:
        try:
            pcm = await read_limited(request, settings.max_audio_bytes)
        except TooLargeError as exc:
            raise HTTPException(413, str(exc)) from exc

    if not pcm:
        raise HTTPException(422, "Audio content is required")

    try:
        return await infer(
            pcm,
            model_name=target_model,
            initial_prompt=target_prompt,
            hotwords=target_hotwords,
            enable_itn=target_itn,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.websocket("/ws/stt")
@app.websocket("/v1/ws/stt")
@app.websocket("/v1/v1/ws/stt")
@app.websocket("/api/v1/ws/stt")
@app.websocket("/v1/audio/transcriptions")
@app.websocket("/v1/v1/audio/transcriptions")
@app.websocket("/api/v1/transcriptions")
@app.websocket("/v1/api/v1/transcriptions")
@app.websocket("/audio/transcriptions")
@app.websocket("/transcriptions")
@app.websocket("/v1/stt")
@app.websocket("/stt")
@app.websocket("/v1/ws")
@app.websocket("/ws")
async def websocket_stt(ws: WebSocket) -> None:
    await ws.accept()
    config: dict[str, Any] = {
        "model": "shenava-koochik-ctc",
        "initial_prompt": None,
        "hotwords": None,
        "enable_itn": True,
    }
    # Initialize from query parameters if present
    for qk in ("model", "engine", "initial_prompt", "hotwords", "enable_itn"):
        if qk in ws.query_params:
            target_k = "model" if qk == "engine" else qk
            val = ws.query_params[qk]
            if qk == "enable_itn":
                val = val.lower() not in ("false", "0", "no")
            config[target_k] = val

    pcm = bytearray()
    try:
        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                return
            if frame.get("bytes") is not None:
                chunk = frame["bytes"]
                if len(chunk) == 0:
                    # Empty binary frame treated as EOS
                    pass
                else:
                    pcm.extend(chunk)
                    if len(pcm) > settings.max_audio_bytes:
                        await ws.send_json({"status": "error", "error": "audio exceeds NAVA_MAX_STT_AUDIO_BYTES"})
                        await ws.close(1009)
                        return
                    continue

            is_eos = False
            if frame.get("bytes") is not None and len(frame["bytes"]) == 0:
                is_eos = True
            elif frame.get("text"):
                try:
                    control = json.loads(frame.get("text") or "")
                except json.JSONDecodeError:
                    await ws.send_json({"status": "error", "error": "control frame must be JSON"})
                    continue

                if isinstance(control, dict):
                    # Update configuration
                    for k in ("model", "engine", "initial_prompt", "hotwords", "enable_itn"):
                        if k in control and control[k] is not None:
                            target_k = "model" if k == "engine" else k
                            config[target_k] = control[k]

                    if control.get("action") in ("eos", "close") or control.get("type") in ("eos", "done", "close") or control.get("event") == "eos" or control.get("eof"):
                        is_eos = True
                    else:
                        # Acknowledge configuration frame
                        await ws.send_json({"status": "ok", "action": "configured", "model": config["model"]})
                        continue

            if not is_eos:
                continue

            # Process accumulated audio on EOS
            try:
                res = await infer(
                    bytes(pcm),
                    model_name=config.get("model"),
                    initial_prompt=config.get("initial_prompt"),
                    hotwords=config.get("hotwords"),
                    enable_itn=bool(config.get("enable_itn", True)),
                )
                await ws.send_json(res.model_dump())
            except ValueError as exc:
                await ws.send_json({"status": "error", "error": str(exc)})
            except Exception as exc:
                await ws.send_json({"status": "error", "error": str(exc)})

            config.clear()
            config["model"] = "shenava-koochik-ctc"
            config["enable_itn"] = True
            pcm.clear()
    except WebSocketDisconnect:
        return

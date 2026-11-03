from __future__ import annotations

import asyncio, io, threading, wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from piper import PiperVoice, SynthesisConfig
from pocket_tts import TTSModel
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .settings import Settings
from .textpipe import (PHONEME_ALIASES, apply_phoneme_overrides, apply_surface_lexicon,
                       chunk_pocket, fit_token_budget, load_lexicon, normalize, split_sentences)

POCKET_MAX_TOKENS = 16

settings = Settings.load()

class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_text_chars)
    engine: Literal["pocket","mana","piper"] = "pocket"
    voice_prompt: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    speed: float = Field(1.0, ge=0.5, le=2.0)

    @model_validator(mode="after")
    def _pocket_has_no_speed(self) -> "Speech":
        if self.engine == "pocket" and self.speed != 1.0:
            raise ValueError("speed is not supported by the pocket engine; use 1.0")
        return self

class G2P:
    def __init__(self):
        folder = settings.models_dir / "g2p"
        if not folder.is_dir(): raise RuntimeError(f"G2P model missing: {folder}")
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval(); self.lock = threading.Lock()
    @lru_cache(maxsize=4096)
    def run(self, text: str) -> str:
        with self.lock, torch.inference_mode():
            ids = self.tokenizer([text.replace("؟", "").replace("?", "")], add_special_tokens=False, return_tensors="pt")
            out = self.model.generate(**ids, num_beams=5, max_length=512, early_stopping=True)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip().translate(PHONEME_ALIASES)

class Piper:
    def __init__(self, name: str, path):
        if not path.is_file() or not Path(f"{path}.json").is_file(): raise RuntimeError(f"{name} model missing: {path}")
        self.name, self.voice = name, PiperVoice.load(str(path)); self.rate = int(self.voice.config.sample_rate)
    def run(self, text: str, speed: float) -> bytes:
        blob = io.BytesIO()
        with wave.open(blob, "wb") as f: self.voice.synthesize_wav(text, f, syn_config=SynthesisConfig(length_scale=1/speed))
        with wave.open(io.BytesIO(blob.getvalue()), "rb") as f: return f.readframes(f.getnframes())

class Pocket:
    name = "pocket"
    def __init__(self):
        config = settings.models_dir / "pocket/model.yaml"
        if not config.is_file(): raise RuntimeError(f"Pocket model missing: {config}")
        self.model = TTSModel.load_model(config=str(config), temp=0.3, eos_threshold=-2.0); self.rate = int(self.model.sample_rate)
        if not callable(getattr(self.model, "_generate_audio_stream_short_text", None)): raise RuntimeError("Pocket Farsi fork is required")
        self.states: dict[str, object] = {}; self._state("default")
    def _state(self, name: str):
        if name not in self.states:
            path = (settings.models_dir / f"voices/{name}.wav").resolve()
            if (settings.models_dir / "voices").resolve() not in path.parents or not path.is_file(): raise ValueError(f"voice prompt '{name}' is not available")
            from pocket_tts.data.audio import audio_read
            from pocket_tts.data.audio_utils import convert_audio
            audio, rate = audio_read(str(path)); self.states[name] = self.model.get_state_for_audio_prompt(convert_audio(audio[..., :int(rate*5)], rate, self.rate, 1))
        return self.states[name]
    def count_tokens(self, text: str) -> int:
        return len(self.model.flow_lm.conditioner.tokenizer.sp.encode(text))
    def run(self, phonemes: str, voice: str) -> bytes:
        text = phonemes.replace("1", "").strip(); tokens = self.count_tokens(text)
        if not text: raise ValueError("G2P returned no phonemes")
        if tokens > POCKET_MAX_TOKENS: raise ValueError(f"Pocket input exceeds {POCKET_MAX_TOKENS} tokens")
        cap = tokens / 3 + 2
        for _ in range(2):
            parts = list(self.model._generate_audio_stream_short_text(model_state=self._state(voice), text_to_generate=text, frames_after_eos=0, copy_state=True))
            if parts:
                audio = torch.cat(parts).detach().cpu().numpy().reshape(-1)
                if 0.08 < len(audio)/self.rate < cap and np.isfinite(audio).all(): return (np.clip(audio,-1,1)*32767).astype("<i2").tobytes()
        raise RuntimeError("Pocket generated silent or runaway audio twice")

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(settings.omp_threads)
    app.state.settings = settings
    app.state.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="nava-tts")
    app.state.g2p = G2P()
    app.state.engines = {
        "pocket": Pocket(),
        "mana": Piper("mana", settings.models_dir / "mana/fa_IR-mana-medium.onnx"),
        "piper": Piper("piper", settings.models_dir / "piper/fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx"),
    }
    app.state.locks = {name: asyncio.Lock() for name in app.state.engines}
    app.state.lexicon = load_lexicon(settings.lexicon_path)
    try: yield
    finally: app.state.pool.shutdown(wait=True, cancel_futures=True)

app = FastAPI(title="Nava TTS", version="1.1.0", lifespan=lifespan)
async def cpu(fn,*args): return await asyncio.get_running_loop().run_in_executor(app.state.pool,fn,*args)
def gap_bytes(rate: int) -> bytes: return np.zeros(round(rate * 0.150), dtype="<i2").tobytes()

async def produce(payload: Speech) -> AsyncIterator[bytes]:
    engine = app.state.engines[payload.engine]
    text = apply_surface_lexicon(normalize(payload.text), app.state.lexicon)
    if not text: raise ValueError("text has no speakable content")
    gap = gap_bytes(engine.rate); first = True
    if payload.engine == "pocket":
        # Producer-consumer: chunk N+1 phonemisation overlaps chunk N synthesis,
        # so after the first chunk the engine never waits on G2P.
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        async def phonemize() -> None:
            try:
                for piece in chunk_pocket(text):
                    phonemes = await cpu(app.state.g2p.run, piece)
                    phonemes = apply_phoneme_overrides(phonemes, app.state.lexicon)
                    for part in fit_token_budget(phonemes, POCKET_MAX_TOKENS, engine.count_tokens):
                        await queue.put(part)
                await queue.put(None)
            except BaseException as exc:
                await queue.put(exc)
        pump = asyncio.create_task(phonemize())
        try:
            while True:
                item = await queue.get()
                if item is None: break
                if isinstance(item, BaseException): raise item
                async with app.state.locks["pocket"]:
                    audio = await cpu(engine.run, item, payload.voice_prompt)
                if not first: yield gap
                yield audio; first = False
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
    else:
        # mana/piper phonemise internally — no neural G2P call at all.
        for piece in split_sentences(text):
            async with app.state.locks[payload.engine]:
                audio = await cpu(engine.run, piece, payload.speed)
            if not first: yield gap
            yield audio; first = False

@app.get("/healthz")
@app.get("/api/v1/health")
async def health(): return {"status":"ok","service":"nava-tts","engines":list(app.state.engines)}
@app.post("/api/v1/speech")
async def speech(payload: Speech):
    engine = app.state.engines[payload.engine]
    # Synthesise the first chunk before the response starts, so clients get a
    # real HTTP error status instead of 200 with an empty/truncated stream.
    stream = produce(payload)
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        raise HTTPException(422, "text has no speakable content") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    async def body() -> AsyncIterator[bytes]:
        yield first
        async for chunk in stream: yield chunk
    return StreamingResponse(body(),media_type=f"audio/L16;rate={engine.rate};channels=1",headers={"X-Nava-Sample-Rate":str(engine.rate),"X-Nava-Engine":payload.engine})
@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame=await ws.receive()
            if frame["type"]=="websocket.disconnect": return
            try:
                payload=Speech.model_validate_json(frame.get("text") or ""); engine=app.state.engines[payload.engine]
                await ws.send_json({"status":"streaming","engine":payload.engine,"sample_rate":engine.rate})
                async for audio in produce(payload): await ws.send_bytes(audio)
                await ws.send_json({"status":"completed"})
            except Exception as exc: await ws.send_json({"status":"error","error":str(exc)})
    except WebSocketDisconnect: return

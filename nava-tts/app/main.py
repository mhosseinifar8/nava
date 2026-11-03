from __future__ import annotations

import asyncio, io, json, os, re, threading, wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from piper import PiperVoice, SynthesisConfig
from pocket_tts import TTSModel
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = lambda: Path(os.getenv("NAVA_MODELS_DIR", "/models"))
MAX_TEXT = int(os.getenv("NAVA_MAX_TTS_TEXT_CHARS", "20000"))
CHARS = str.maketrans({"ي":"ی", "ى":"ی", "ك":"ک", "ؤ":"و", "إ":"ا", "أ":"ا", "ة":"ه", "ۀ":"هٔ"})
PHONEMES = str.maketrans({"/":"a", "a":"A", "@":"?", "$":"S", "c":"C"})

class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    engine: Literal["pocket","mana","piper"] = "pocket"
    voice_prompt: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    speed: float = Field(1.0, ge=0.5, le=2.0)

class G2P:
    def __init__(self):
        folder = ROOT() / "g2p"
        if not folder.is_dir(): raise RuntimeError(f"G2P model missing: {folder}")
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval(); self.lock = threading.Lock()
    @lru_cache(maxsize=4096)
    def run(self, text: str) -> str:
        with self.lock, torch.inference_mode():
            ids = self.tokenizer([text.replace("؟", "").replace("?", "")], add_special_tokens=False, return_tensors="pt")
            out = self.model.generate(**ids, num_beams=5, max_length=512, early_stopping=True)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip().translate(PHONEMES)

class Piper:
    def __init__(self, name: str, path: Path):
        if not path.is_file() or not Path(f"{path}.json").is_file(): raise RuntimeError(f"{name} model missing: {path}")
        self.name, self.voice = name, PiperVoice.load(str(path)); self.rate = int(self.voice.config.sample_rate)
    def run(self, text: str, speed: float) -> bytes:
        blob = io.BytesIO()
        with wave.open(blob, "wb") as f: self.voice.synthesize_wav(text, f, syn_config=SynthesisConfig(length_scale=1/speed))
        with wave.open(io.BytesIO(blob.getvalue()), "rb") as f: return f.readframes(f.getnframes())

class Pocket:
    name = "pocket"
    def __init__(self):
        config = ROOT() / "pocket/model.yaml"
        if not config.is_file(): raise RuntimeError(f"Pocket model missing: {config}")
        self.model = TTSModel.load_model(config=str(config), temp=0.3, eos_threshold=-2.0); self.rate = int(self.model.sample_rate)
        if not callable(getattr(self.model, "_generate_audio_stream_short_text", None)): raise RuntimeError("Pocket Farsi fork is required")
        self.states: dict[str, object] = {}; self._state("default")
    def _state(self, name: str):
        if name not in self.states:
            path = (ROOT() / f"voices/{name}.wav").resolve()
            if (ROOT() / "voices").resolve() not in path.parents or not path.is_file(): raise ValueError(f"voice prompt '{name}' is not available")
            from pocket_tts.data.audio import audio_read
            from pocket_tts.data.audio_utils import convert_audio
            audio, rate = audio_read(str(path)); self.states[name] = self.model.get_state_for_audio_prompt(convert_audio(audio[..., :int(rate*5)], rate, self.rate, 1))
        return self.states[name]
    def run(self, phonemes: str, voice: str) -> bytes:
        text = phonemes.replace("1", "").strip(); tokens = len(self.model.flow_lm.conditioner.tokenizer.sp.encode(text))
        if not text: raise ValueError("G2P returned no phonemes")
        if tokens > 16: raise ValueError("Pocket input exceeds 16 tokens")
        cap = tokens / 3 + 2
        for _ in range(2):
            parts = list(self.model._generate_audio_stream_short_text(model_state=self._state(voice), text_to_generate=text, frames_after_eos=0, copy_state=True))
            if parts:
                audio = torch.cat(parts).detach().cpu().numpy().reshape(-1)
                if 0.08 < len(audio)/self.rate < cap and np.isfinite(audio).all(): return (np.clip(audio,-1,1)*32767).astype("<i2").tobytes()
        raise RuntimeError("Pocket generated silent or runaway audio twice")

def clean(text: str) -> str: return re.sub(r"\s+", " ", re.sub(r"[^\w\s،؛.!؟,:()\-]", " ", text.translate(CHARS).replace("\u200c"," "))).strip()
def split(text: str, pocket: bool) -> list[str]:
    sentences = [x.strip() for x in re.split(r"(?<=[.،؛!؟\n])\s*", text) if x.strip()]
    return [" ".join(s.split()[i:i+12]) for s in sentences for i in range(0, len(s.split()), 12)] if pocket else sentences

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ThreadPoolExecutor(max_workers=int(os.getenv("NAVA_INFERENCE_WORKERS","2")), thread_name_prefix="nava-tts")
    app.state.g2p = G2P(); app.state.engines = {"pocket":Pocket(), "mana":Piper("mana",ROOT()/"mana/fa_IR-mana-medium.onnx"), "piper":Piper("piper",ROOT()/"piper/fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx")}; app.state.locks = {x:asyncio.Lock() for x in app.state.engines}
    try: yield
    finally: app.state.pool.shutdown(wait=True, cancel_futures=True)

app = FastAPI(title="Nava TTS", version="1.0.0", lifespan=lifespan)
async def cpu(fn,*args): return await asyncio.get_running_loop().run_in_executor(app.state.pool,fn,*args)
async def produce(payload: Speech) -> AsyncIterator[bytes]:
    text=clean(payload.text)
    if not text: raise ValueError("text has no speakable content")
    engine=app.state.engines[payload.engine]; first=True
    for piece in split(text,payload.engine=="pocket"):
        phonemes=await cpu(app.state.g2p.run,piece)
        async with app.state.locks[payload.engine]: audio=await cpu(engine.run,phonemes,payload.voice_prompt) if payload.engine=="pocket" else await cpu(engine.run,piece,payload.speed)
        if not first: yield np.zeros(round(engine.rate*.150),dtype="<i2").tobytes()
        yield audio; first=False

@app.get("/healthz")
@app.get("/api/v1/health")
async def health(): return {"status":"ok","service":"nava-tts","engines":list(app.state.engines)}
@app.post("/api/v1/speech")
async def speech(payload: Speech):
    engine=app.state.engines[payload.engine]
    return StreamingResponse(produce(payload),media_type=f"audio/L16;rate={engine.rate};channels=1",headers={"X-Nava-Sample-Rate":str(engine.rate),"X-Nava-Engine":payload.engine})
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

from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import os

try:
    import onnxruntime as ort
    _orig_SessionOptions = ort.SessionOptions

    class _PatchedSessionOptions(_orig_SessionOptions):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            num_threads = int(os.getenv("OMP_NUM_THREADS", os.getenv("TTS_OMP_THREADS", "2")))
            self.intra_op_num_threads = max(1, num_threads)
            self.inter_op_num_threads = 1

    ort.SessionOptions = _PatchedSessionOptions
except Exception:
    pass

try:
    import torch
except ImportError:
    torch = None
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .pocket_config import localize_pocket_config
from .settings import Settings
from .textpipe import (
    PHONEME_ALIASES,
    apply_phoneme_overrides,
    apply_surface_lexicon,
    chunk_pocket,
    fit_token_budget,
    load_lexicon,
    normalize,
    split_sentences,
)

POCKET_MAX_TOKENS = 16

settings = Settings.load()


class Speech(BaseModel):
    input: str | None = Field(None, description="Persian text to synthesize")
    text: str | None = Field(None, description="Alias for input")
    model: str | None = Field(None, description="TTS model ID or engine name")
    engine: str = Field("pocket", description="TTS model ID or engine name")
    voice: str | None = Field(None, description="Speaker name, character, or voice_id")
    voice_prompt: str = Field("default", description="Alias for voice")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    g2p_mode: str = Field("lca_albert", description="G2P pre-processor pipeline mode")
    custom_dictionary: dict[str, str] | None = None
    reference_audio_base64: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "model" in data and "engine" not in data:
                data["engine"] = data["model"]
            elif "engine" in data and "model" not in data:
                data["model"] = data["engine"]
            if "input" in data and "text" not in data:
                data["text"] = data["input"]
            elif "text" in data and "input" not in data:
                data["input"] = data["text"]
            if "voice" in data and "voice_prompt" not in data:
                data["voice_prompt"] = data["voice"]
            elif "voice_prompt" in data and "voice" not in data:
                data["voice"] = data["voice_prompt"]
        return data

    @model_validator(mode="after")
    def _pocket_speed_check(self) -> "Speech":
        if self.engine in ("pocket", "pocket-tts-farsi") and self.speed != 1.0:
            self.speed = 1.0
        return self


class G2P:
    def __init__(self):
        folder = settings.models_dir / "g2p"
        if not folder.is_dir():
            folder = settings.models_dir / "persian-ezafe-albert"
        if not folder.is_dir():
            self.tokenizer = None
            self.model = None
            return
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, extra_special_tokens={})
            self.model = T5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval()
            self.lock = threading.Lock()
        except Exception:
            self.tokenizer = None
            self.model = None

    @lru_cache(maxsize=4096)
    def run(self, text: str) -> str:
        if self.model is None or self.tokenizer is None:
            return text
        with self.lock, torch.inference_mode():
            ids = self.tokenizer([text.replace("؟", "").replace("?", "")], add_special_tokens=False, return_tensors="pt")
            out = self.model.generate(**ids, num_beams=5, max_length=512, early_stopping=True)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip().translate(PHONEME_ALIASES)


class Piper:
    def __init__(self, name: str, path: Path):
        json_p = Path(f"{path}.json")
        if not path.is_file() or not json_p.is_file():
            raise RuntimeError(f"{name} model missing: {path}")
        from piper import PiperVoice
        self.name, self.voice = name, PiperVoice.load(str(path))
        self.rate = int(self.voice.config.sample_rate)

    def run(self, text: str, speed: float) -> bytes:
        from piper import SynthesisConfig
        blob = io.BytesIO()
        with wave.open(blob, "wb") as f:
            self.voice.synthesize_wav(text, f, syn_config=SynthesisConfig(length_scale=1 / speed))
        with wave.open(io.BytesIO(blob.getvalue()), "rb") as f:
            return f.readframes(f.getnframes())


class Pocket:
    name = "pocket"

    def __init__(self):
        config = settings.models_dir / "pocket/model.yaml"
        pocket_dir = (settings.models_dir / "pocket").resolve()
        if not config.is_file():
            config = settings.models_dir / "pocket-tts-farsi/model.yaml"
            pocket_dir = (settings.models_dir / "pocket-tts-farsi").resolve()
        if not config.is_file():
            raise RuntimeError(f"Pocket model missing: {config}")
        config = localize_pocket_config(config, pocket_dir)
        from pocket_tts import TTSModel
        self.model = TTSModel.load_model(config=str(config), temp=0.3, eos_threshold=-2.0)
        self.rate = int(self.model.sample_rate)
        self.states: dict[str, object] = {}
        try:
            self._state("default")
        except Exception:
            pass

    def _state(self, name: str):
        if name not in self.states:
            path = (settings.models_dir / f"voices/{name}.wav").resolve()
            if not path.is_file():
                # Check default voice
                path = (settings.models_dir / "voices/default.wav").resolve()
            if not path.is_file():
                return None
            from pocket_tts.data.audio import audio_read
            from pocket_tts.data.audio_utils import convert_audio
            audio, rate = audio_read(str(path))
            self.states[name] = self.model.get_state_for_audio_prompt(convert_audio(audio[..., : int(rate * 5)], rate, self.rate, 1))
        return self.states[name]

    def count_tokens(self, text: str) -> int:
        return len(self.model.flow_lm.conditioner.tokenizer.sp.encode(text))

    def run(self, phonemes: str, voice: str) -> bytes:
        text = phonemes.replace("1", "").strip()
        tokens = self.count_tokens(text)
        if not text:
            raise ValueError("G2P returned no phonemes")
        if tokens > POCKET_MAX_TOKENS:
            raise ValueError(f"Pocket input exceeds {POCKET_MAX_TOKENS} tokens")
        cap = tokens / 3 + 2
        st = self._state(voice)
        for _ in range(2):
            parts = list(self.model._generate_audio_stream_short_text(model_state=st, text_to_generate=text, frames_after_eos=0, copy_state=True))
            if parts:
                audio = torch.cat(parts).detach().cpu().numpy().reshape(-1)
                if 0.08 < len(audio) / self.rate < cap and np.isfinite(audio).all():
                    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        raise RuntimeError("Pocket generated silent or runaway audio twice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch is not None:
        torch.set_num_threads(settings.omp_threads)
    app.state.settings = settings
    app.state.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="nava-tts")
    try:
        app.state.g2p = G2P()
    except Exception:
        app.state.g2p = None

    engines = {}
    # 1. Pocket
    for candidate in [settings.models_dir / "pocket/model.yaml", settings.models_dir / "pocket-tts-farsi/model.yaml"]:
        if candidate.is_file():
            try:
                engines["pocket"] = Pocket()
                break
            except Exception as e:
                print(f"Warning: Pocket init failed: {e}")

    # 2. Mana
    for candidate in [settings.models_dir / "mana/fa_IR-mana-medium.onnx", settings.models_dir / "piper-mana-lca/fa_IR-mana-medium.onnx"]:
        if candidate.is_file():
            try:
                engines["mana"] = Piper("mana", candidate)
                break
            except Exception as e:
                print(f"Warning: Mana init failed: {e}")

    # 3. Piper
    for candidate in [
        settings.models_dir / "piper/fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx",
        settings.models_dir / "piper-voices-fa/ganji/fa_IR-ganji-medium.onnx",
    ]:
        if candidate.is_file():
            try:
                engines["piper"] = Piper("piper", candidate)
                break
            except Exception as e:
                print(f"Warning: Piper init failed: {e}")

    app.state.engines = engines
    app.state.locks = {name: asyncio.Lock() for name in ("pocket", "mana", "piper")}
    try:
        app.state.lexicon = load_lexicon(settings.lexicon_path)
    except Exception:
        app.state.lexicon = {}

    try:
        yield
    finally:
        app.state.pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Nava TTS", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(app.state.pool, fn, *args)


def gap_bytes(rate: int) -> bytes:
    return np.zeros(round(rate * 0.150), dtype="<i2").tobytes()


async def produce(payload: Speech) -> AsyncIterator[bytes]:
    effective_engine = payload.model or payload.engine or "pocket"
    effective_voice = payload.voice or payload.voice_prompt or "default"
    text_content = payload.input or payload.text
    if not text_content or not text_content.strip():
        raise ValueError("text has no speakable content")

    # Try unified Nava Engine Manager first (supports matcha, ava, pocket, piper, etc.)
    try:
        from nava.engines.factory import get_tts_engine
        engine = get_tts_engine(effective_engine)
        ref_bytes = None
        if payload.reference_audio_base64:
            try:
                import base64
                ref_bytes = base64.b64decode(payload.reference_audio_base64)
            except Exception:
                pass
        async for chunk in engine.synthesize_stream(
            text_content,
            voice=effective_voice,
            speed=payload.speed,
            g2p_mode=payload.g2p_mode,
            custom_dictionary=payload.custom_dictionary,
            reference_audio=ref_bytes,
        ):
            yield chunk
        return
    except Exception as exc:
        # Check if local fallback engine is available
        alias_map = {"piper-mana-lca": "mana", "piper-voices-fa": "piper", "pocket-tts-farsi": "pocket"}
        local_key = alias_map.get(effective_engine, effective_engine)
        if local_key not in app.state.engines:
            # If unified engine failed due to missing model, re-raise with clear guidance
            raise RuntimeError(f"Engine '{effective_engine}' error: {exc}") from exc
        effective_engine = local_key

    # Local fallback synthesis for pocket, mana, piper
    legacy_engine = app.state.engines[effective_engine]
    text = apply_surface_lexicon(normalize(text_content), getattr(app.state, "lexicon", {}))
    if not text:
        raise ValueError("text has no speakable content")
    gap = gap_bytes(legacy_engine.rate)
    first = True
    if effective_engine == "pocket":
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def phonemize() -> None:
            try:
                for piece in chunk_pocket(text):
                    phonemes = piece
                    if getattr(app.state, "g2p", None) is not None:
                        phonemes = await cpu(app.state.g2p.run, piece)
                        phonemes = apply_phoneme_overrides(phonemes, app.state.lexicon)
                    for part in fit_token_budget(phonemes, POCKET_MAX_TOKENS, legacy_engine.count_tokens):
                        await queue.put(part)
                await queue.put(None)
            except BaseException as exc:
                await queue.put(exc)

        pump = asyncio.create_task(phonemize())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                lock = app.state.locks.get("pocket")
                if lock:
                    async with lock:
                        audio = await cpu(legacy_engine.run, item, effective_voice)
                else:
                    audio = await cpu(legacy_engine.run, item, effective_voice)
                if not first:
                    yield gap
                yield audio
                first = False
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
    else:
        for piece in split_sentences(text):
            lock = app.state.locks.get(effective_engine)
            if lock:
                async with lock:
                    audio = await cpu(legacy_engine.run, piece, payload.speed)
            else:
                audio = await cpu(legacy_engine.run, piece, payload.speed)
            if not first:
                yield gap
            yield audio
            first = False


@app.get("/healthz")
@app.get("/api/v1/health")
@app.get("/v1/health")
async def health():
    all_engines = list(app.state.engines.keys())
    try:
        from nava.models.registry import MODELS
        all_engines = sorted(list(set(all_engines + [m for m, info in MODELS.items() if info.type == "tts"])))
    except Exception:
        pass
    return {"status": "ok", "service": "nava-tts", "engines": all_engines}


@app.post("/api/v1/speech")
@app.post("/v1/audio/speech")
@app.post("/v1/v1/audio/speech")
@app.post("/v1/api/v1/speech")
@app.post("/speech")
@app.post("/audio/speech")
async def speech(payload: Speech):
    text_content = payload.input or payload.text
    if not text_content or not text_content.strip():
        raise HTTPException(422, "Text content ('input' or 'text') is required and cannot be empty")

    engine_name = payload.model or payload.engine or "pocket"
    rate = 22050
    try:
        from nava.engines.factory import get_tts_engine
        eng = get_tts_engine(engine_name)
        rate = getattr(eng, "sample_rate", 22050)
    except Exception:
        if engine_name in app.state.engines:
            rate = app.state.engines[engine_name].rate

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
        async for chunk in stream:
            yield chunk

    return StreamingResponse(
        body(),
        media_type=f"audio/L16;rate={rate};channels=1",
        headers={"X-Nava-Sample-Rate": str(rate), "X-Nava-Engine": engine_name},
    )


@app.websocket("/ws/tts")
@app.websocket("/v1/ws/tts")
@app.websocket("/v1/v1/ws/tts")
@app.websocket("/api/v1/ws/tts")
@app.websocket("/v1/audio/speech")
@app.websocket("/v1/v1/audio/speech")
@app.websocket("/api/v1/speech")
@app.websocket("/v1/api/v1/speech")
@app.websocket("/audio/speech")
@app.websocket("/speech")
@app.websocket("/v1/tts")
@app.websocket("/tts")
@app.websocket("/v1/ws")
@app.websocket("/ws")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    current_config: dict[str, Any] = {
        "engine": "pocket",
        "voice": "default",
        "speed": 1.0,
        "g2p_mode": "lca_albert",
        "custom_dictionary": None,
    }
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
                await ws.send_json({"status": "error", "error": "control frame must be JSON"})
                continue

            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            action = msg.get("action")
            if msg_type == "ping" or action == "ping":
                await ws.send_json({"type": "pong", "status": "ok"})
                continue

            # Update configuration if fields are provided
            for k in ("model", "engine", "voice", "voice_prompt", "speed", "g2p_mode", "custom_dictionary"):
                if k in msg and msg[k] is not None:
                    target_k = "engine" if k == "model" else "voice" if k == "voice_prompt" else k
                    current_config[target_k] = msg[k]

            text_to_speak = msg.get("text") or msg.get("input")

            # If this is purely a configuration frame (no text to synthesize), acknowledge and wait for next frame
            if not text_to_speak and msg.get("action") != "synthesize":
                await ws.send_json({"status": "ok", "action": "configured", "engine": current_config["engine"]})
                continue

            if not text_to_speak:
                await ws.send_json({"status": "error", "error": "text is required for synthesis"})
                continue

            try:
                engine_name = current_config.get("engine", "pocket")
                voice_name = current_config.get("voice", "default")
                speed = float(current_config.get("speed", 1.0))

                rate = 22050
                try:
                    from nava.engines.factory import get_tts_engine
                    eng = get_tts_engine(engine_name)
                    rate = getattr(eng, "sample_rate", 22050)
                except Exception:
                    if engine_name in app.state.engines:
                        rate = app.state.engines[engine_name].rate

                await ws.send_json({"status": "streaming", "engine": engine_name, "sample_rate": rate})

                speech_obj = Speech(
                    text=text_to_speak,
                    engine=engine_name,
                    voice_prompt=voice_name,
                    speed=speed,
                    g2p_mode=current_config.get("g2p_mode", "lca_albert"),
                    custom_dictionary=current_config.get("custom_dictionary"),
                )
                async for audio in produce(speech_obj):
                    await ws.send_bytes(audio)
                await ws.send_json({"status": "completed"})
            except Exception as exc:
                await ws.send_json({"status": "error", "error": str(exc)})
    except WebSocketDisconnect:
        return

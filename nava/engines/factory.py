"""Nava Unified Engine Registry and Factory.

Provides thread-safe lazy-loading and singleton management for STT and TTS engines.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from ..models.registry import get_model
from .base import BaseSTTEngine, BaseStreamingSTTEngine, BaseTTSEngine
from .stt_nemotron import NemotronStreamingEngine
from .stt_shenava import ShenavaCTCEngine, ShenavaRNNTEngine, ShenavaStreamingEngine
from .stt_whisper import FasterWhisperEngine, WhisperSherpaEngine
from .tts_ava import AvaEngine
from .tts_matcha import MatchaEngine
from .tts_piper import PiperEngine
from .tts_pocket import PocketEngine


class EngineManager:
    """Thread-safe manager for lazy loading of STT and TTS engines."""

    def __init__(self, models_dir: Path | str | None = None):
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            env_dir = os.getenv("NAVA_MODELS_DIR")
            self.models_dir = (Path(env_dir) if env_dir else Path("./models")).resolve()
        self._stt_engines: dict[str, BaseSTTEngine] = {}
        self._tts_engines: dict[str, BaseTTSEngine] = {}
        self._lock = threading.Lock()

    def get_stt_engine(self, model_id: str = "shenava-koochik-ctc") -> BaseSTTEngine:
        """Get or lazily instantiate an STT engine."""
        model_info = get_model(model_id)
        canon_id = model_info.id

        with self._lock:
            if canon_id in self._stt_engines:
                return self._stt_engines[canon_id]

            model_path = model_info.get_local_path(self.models_dir)

            if canon_id in ("shenava-koochik-ctc", "shenava-115m-flagship"):
                engine = ShenavaCTCEngine(canon_id, model_path)
            elif canon_id == "shenava-koochik-v1.5-rnnt":
                engine = ShenavaRNNTEngine(canon_id, model_path)
            elif canon_id in ("shenava-rizeh-streaming-32m", "shenava-rizeh-pizeh-6.9m"):
                engine = ShenavaStreamingEngine(canon_id, model_path)
            elif canon_id == "nemotron-streaming-farsi":
                engine = NemotronStreamingEngine(canon_id, model_path)
            elif canon_id == "whisper-turbo-persian-sherpa":
                engine = WhisperSherpaEngine(canon_id, model_path)
            elif canon_id in (
                "whisper-turbo-persian-givia",
                "whisper-turbo-persian-gholizadeh",
                "whisper-large-v3",
            ):
                engine = FasterWhisperEngine(canon_id, model_path)
            else:
                engine = ShenavaCTCEngine(canon_id, model_path)

            self._stt_engines[canon_id] = engine
            return engine

    def get_tts_engine(self, model_id: str = "piper-mana-lca") -> BaseTTSEngine:
        """Get or lazily instantiate a TTS engine."""
        model_info = get_model(model_id)
        canon_id = model_info.id

        with self._lock:
            if canon_id in self._tts_engines:
                return self._tts_engines[canon_id]

            model_path = model_info.get_local_path(self.models_dir)

            if canon_id in ("piper-mana-lca", "piper-voices-fa"):
                engine = PiperEngine(canon_id, model_path)
            elif canon_id == "ava-82m":
                engine = AvaEngine.get_singleton(canon_id, model_path)
            elif canon_id in ("matcha-khadijah-fa-en", "matcha-musa-fa-en"):
                engine = MatchaEngine(canon_id, model_path)
            elif canon_id in ("pocket-tts-farsi", "nava-fa-clone"):
                engine = PocketEngine(canon_id, model_path)
            else:
                engine = PiperEngine(canon_id, model_path)

            self._tts_engines[canon_id] = engine
            return engine


_GLOBAL_ENGINE_MGR = EngineManager()


def get_stt_engine(model_id: str = "shenava-koochik-ctc") -> BaseSTTEngine:
    return _GLOBAL_ENGINE_MGR.get_stt_engine(model_id)


def get_tts_engine(model_id: str = "piper-mana-lca") -> BaseTTSEngine:
    return _GLOBAL_ENGINE_MGR.get_tts_engine(model_id)


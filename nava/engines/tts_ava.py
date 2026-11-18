"""Ava-82M 24kHz Kokoro Persian TTS Engine Adapter.

Implements StyleTTS2/iSTFTNet 24kHz architecture (xmanii/Ava-82M) via `ava_tts`
with warm singleton retention in memory, pronunciations.json integration,
and custom dictionary merge.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.pipeline import preprocess_tts_text
from ..runtime.cpu_tuning import run_in_cpu_pool
from .base import (
    BaseTTSEngine,
    ModelNotInstalledError,
    TTSResult,
    allow_dummy_audio_fallback,
    float32_to_pcm16,
    make_silence_pcm,
    wav_bytes_to_pcm16,
)
from .tts_piper import _split_into_sentences

logger = logging.getLogger("nava.tts")


class AvaEngine(BaseTTSEngine):
    """Ava-82M Studio Quality Persian TTS (24kHz)."""

    _instance: AvaEngine | None = None  # Warm singleton

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self._ava_model: Any = None
        self.pronunciations: dict[str, str] = {}
        self._load_pronunciations()

    @classmethod
    def get_singleton(cls, model_id: str, model_dir: Path) -> AvaEngine:
        """Get or initialize warm singleton in memory."""
        if cls._instance is None or cls._instance.model_dir != Path(model_dir):
            cls._instance = cls(model_id, model_dir)
        return cls._instance

    def _load_pronunciations(self) -> None:
        p_file = self.model_dir / "pronunciations.json"
        if p_file.is_file():
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.pronunciations = {str(k): str(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load Ava pronunciations.json: {e}")

    def _get_ava_model(self) -> Any:
        if self._ava_model is not None:
            return self._ava_model

        try:
            from ava_tts import Ava
        except ImportError as exc:
            raise ModelNotInstalledError(
                self.model_id,
                f"ava_tts package not installed — install via: "
                f"pip install 'https://huggingface.co/xmanii/Ava-82M/resolve/main/ava_tts-0.2.0-py3-none-any.whl' ({exc})",
            ) from exc

        try:
            # Prefer loading from local model_dir when weights exist on disk
            has_local = (self.model_dir / "model.pth").is_file() or (self.model_dir / "config.json").is_file()
            if has_local and hasattr(Ava, "from_pretrained"):
                try:
                    self._ava_model = Ava.from_pretrained(str(self.model_dir))
                except Exception:
                    self._ava_model = None

            if self._ava_model is None and hasattr(Ava, "from_pretrained"):
                try:
                    self._ava_model = Ava.from_pretrained("xmanii/Ava-82M")
                except Exception:
                    self._ava_model = None

            if self._ava_model is None:
                self._ava_model = Ava()

            if hasattr(self._ava_model, "sample_rate") and self._ava_model.sample_rate:
                self.sample_rate = int(self._ava_model.sample_rate)
            return self._ava_model
        except Exception as exc:
            logger.error(f"Failed to initialize Ava-82M model: {exc}")
            raise RuntimeError(f"Ava-82M initialization failed: {exc}") from exc

    def warmup(self) -> None:
        """Pre-warm Ava-82M model into memory if ava_tts is installed."""
        try:
            self._get_ava_model()
        except Exception as e:
            logger.warning(f"Ava warmup skipped: {e}")

    def _synthesize_via_cli(self, sentence: str, speed: float) -> bytes | None:
        ava_bin = shutil.which("ava")
        if not ava_bin:
            return None

        tmp_fd, tmp_wav_str = tempfile.mkstemp(prefix="nava-ava-", suffix=".wav")
        os.close(tmp_fd)
        tmp_wav = Path(tmp_wav_str)
        try:
            cmd = [ava_bin, sentence, "-o", str(tmp_wav)]
            if abs(speed - 1.0) > 1e-3:
                cmd.extend(["--speed", f"{speed:.2f}"])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            if proc.returncode != 0 or not tmp_wav.is_file() or tmp_wav.stat().st_size <= 44:
                raise RuntimeError(f"ava CLI failed (exit {proc.returncode}): {(proc.stderr or '')[-300:]}")
            pcm, sr = wav_bytes_to_pcm16(tmp_wav.read_bytes())
            self.sample_rate = sr
            return pcm
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except OSError:
                pass

    def _synthesize_sentence(
        self,
        sentence: str,
        voice: str,
        speed: float,
        merged_dict: dict[str, str],
    ) -> bytes:
        if not sentence or not sentence.strip():
            return b""

        try:
            ava = self._get_ava_model()

            # 1. Direct in-memory synthesis methods if provided by ava_tts
            for method_name in ("synthesize", "generate", "__call__"):
                fn = getattr(ava, method_name, None)
                if not callable(fn):
                    continue
                if method_name == "__call__" and not hasattr(type(ava), "__call__"):
                    continue
                try:
                    sig = inspect.signature(fn)
                    kwargs: dict[str, Any] = {}
                    if "speed" in sig.parameters:
                        kwargs["speed"] = speed
                    out = fn(sentence, **kwargs)
                    if isinstance(out, tuple) and len(out) == 2:
                        wav_arr, sr = out
                        if isinstance(sr, (int, float)) and sr > 0:
                            self.sample_rate = int(sr)
                        pcm = float32_to_pcm16(wav_arr)
                        if pcm:
                            return pcm
                    elif out is not None and not isinstance(out, (str, Path)):
                        pcm = float32_to_pcm16(out)
                        if pcm:
                            return pcm
                except Exception:
                    continue

            # 2. Official ava.save(text, path) API
            if hasattr(ava, "save") and callable(ava.save):
                tmp_fd, tmp_wav_str = tempfile.mkstemp(prefix="nava-ava-", suffix=".wav")
                os.close(tmp_fd)
                tmp_wav = Path(tmp_wav_str)
                try:
                    sig = inspect.signature(ava.save)
                    kwargs = {}
                    if "speed" in sig.parameters:
                        kwargs["speed"] = speed
                    ava.save(sentence, str(tmp_wav), **kwargs)
                    if not tmp_wav.is_file() or tmp_wav.stat().st_size <= 44:
                        raise RuntimeError(f"Ava.save produced empty WAV for '{sentence[:60]}'")
                    pcm, sr = wav_bytes_to_pcm16(tmp_wav.read_bytes())
                    self.sample_rate = sr
                    return pcm
                finally:
                    try:
                        tmp_wav.unlink(missing_ok=True)
                    except OSError:
                        pass

            raise RuntimeError("ava_tts.Ava object has no supported synthesize/generate/save method")
        except (ModelNotInstalledError, RuntimeError) as exc:
            # Try ava CLI binary as secondary path if package import failed
            if isinstance(exc, ModelNotInstalledError):
                cli_pcm = self._synthesize_via_cli(sentence, speed)
                if cli_pcm:
                    return cli_pcm
            if allow_dummy_audio_fallback():
                duration = max(len(sentence) * 0.08 / max(speed, 0.1), 0.2)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Ava-82M '{self.model_id}' synthesis failed: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Ava-82M '{self.model_id}' unexpected error: {exc}")
            raise

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "ava",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
    ) -> TTSResult:
        merged_dict = dict(self.pronunciations)
        if custom_dictionary:
            merged_dict.update(custom_dictionary)

        processed = preprocess_tts_text(
            text,
            custom_dictionary=merged_dict,
            g2p_mode=g2p_mode,
            enable_ezafe=False,
        )

        sentences = _split_into_sentences(processed)
        chunks: list[bytes] = []
        silence_gap: bytes | None = None

        for s in sentences:
            chunk = self._synthesize_sentence(s, voice, speed, merged_dict)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if chunk:
                if chunks:
                    chunks.append(silence_gap)
                chunks.append(chunk)

        full_audio = b"".join(chunks) if chunks else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="ava",
            voice=voice,
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str = "ava",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
    ) -> AsyncIterator[bytes]:
        merged_dict = dict(self.pronunciations)
        if custom_dictionary:
            merged_dict.update(custom_dictionary)

        processed = preprocess_tts_text(
            text,
            custom_dictionary=merged_dict,
            g2p_mode=g2p_mode,
            enable_ezafe=False,
        )

        sentences = _split_into_sentences(processed)
        silence_gap: bytes | None = None
        emitted = False

        for s in sentences:
            chunk = await run_in_cpu_pool(self._synthesize_sentence, s, voice, speed, merged_dict)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True

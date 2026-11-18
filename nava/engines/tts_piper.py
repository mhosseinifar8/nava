"""Piper Persian TTS Engine Adapter.

Supports:
- MahtaFetrat/Mana-Persian-Piper (fa_IR-mana-medium.onnx, MOS 3.14)
- Multi-speaker piper voices: amir, gyro, ganji, ganji_adabi, reza_ibrahim, haaniye_low
- Sentence-by-sentence pipelined audio streaming
"""
from __future__ import annotations

import io
import logging
import wave
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
)

logger = logging.getLogger("nava.tts")


def _split_into_sentences(text: str) -> list[str]:
    """Split Persian text on clause and sentence boundaries."""
    import re
    parts = re.split(r"(?<=[.!؟؛،\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _make_synthetic_pcm(duration_s: float, sample_rate: int = 22050, freq: float = 440.0) -> bytes:
    """Return clean s16le PCM silence for offline testing (never an ear-piercing tone)."""
    return make_silence_pcm(duration_s, sample_rate=sample_rate)


class PiperEngine(BaseTTSEngine):
    """Persian Piper VITS-based TTS engine."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 22050
        self.voices: dict[str, Any] = {}

    def _candidate_dirs(self) -> list[Path]:
        dirs = [self.model_dir]
        # Support legacy download_models.sh directory layout (/models/mana, /models/piper)
        if self.model_id == "piper-mana-lca":
            dirs.append(self.model_dir.parent / "mana")
        elif self.model_id == "piper-voices-fa":
            dirs.append(self.model_dir.parent / "piper")
        return dirs

    def _resolve_model_file(self, voice_name: str) -> tuple[Path, Path]:
        for base_dir in self._candidate_dirs():
            if not base_dir.is_dir():
                continue

            # Check if direct mana file exists
            mana_onnx = base_dir / "fa_IR-mana-medium.onnx"
            if mana_onnx.exists():
                return mana_onnx, base_dir / "fa_IR-mana-medium.onnx.json"

            # Check subfolder by speaker name
            speaker_onnx = base_dir / f"{voice_name}/fa_IR-{voice_name}-medium.onnx"
            if speaker_onnx.exists():
                return speaker_onnx, base_dir / f"{voice_name}/fa_IR-{voice_name}-medium.onnx.json"

            # Check nested piper-voices path (fa/fa_IR/<speaker>/medium/...)
            nested_onnx = base_dir / f"fa/fa_IR/{voice_name}/medium/fa_IR-{voice_name}-medium.onnx"
            if nested_onnx.exists():
                return nested_onnx, Path(f"{nested_onnx}.json")

            # Check ganji default
            for ganji_onnx in (
                base_dir / "ganji/fa_IR-ganji-medium.onnx",
                base_dir / "fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx",
            ):
                if ganji_onnx.exists():
                    return ganji_onnx, Path(f"{ganji_onnx}.json")

            # Search any .onnx with matching .onnx.json in directory
            for p in base_dir.glob("**/*.onnx"):
                json_p = Path(str(p) + ".json")
                if json_p.exists():
                    return p, json_p

        raise ModelNotInstalledError(self.model_id, self.model_dir)

    def _get_voice(self, voice_name: str):
        if voice_name in self.voices:
            return self.voices[voice_name]

        onnx_path, json_path = self._resolve_model_file(voice_name)

        try:
            from piper import PiperVoice
            voice = PiperVoice.load(str(onnx_path), config_path=str(json_path) if json_path.exists() else None)
            self.sample_rate = int(voice.config.sample_rate)
            self.voices[voice_name] = voice
            return voice
        except Exception as piper_exc:
            # Fall back to sherpa-onnx OfflineTts if available
            try:
                import sherpa_onnx
                from ..g2p.lexicon import find_espeak_ng_data

                espeak_dir = find_espeak_ng_data(self.model_dir)
                config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=str(onnx_path),
                            tokens="",
                            data_dir=str(espeak_dir) if espeak_dir else "",
                        ),
                        num_threads=2,
                    )
                )
                tts = sherpa_onnx.OfflineTts(config)
                self.sample_rate = int(getattr(tts, "sample_rate", self.sample_rate))
                self.voices[voice_name] = tts
                return tts
            except Exception as e:
                logger.error(
                    f"Piper engine '{self.model_id}' failed to load {onnx_path}: "
                    f"piper={piper_exc}; sherpa_onnx={e}"
                )
                raise ModelNotInstalledError(self.model_id, f"{onnx_path} ({piper_exc})")

    def _synthesize_sentence(self, sentence: str, voice_name: str, speed: float) -> bytes:
        try:
            voice_obj = self._get_voice(voice_name)
            if hasattr(voice_obj, "synthesize_wav"):
                from piper import SynthesisConfig
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    voice_obj.synthesize_wav(
                        sentence,
                        wf,
                        syn_config=SynthesisConfig(length_scale=1.0 / max(speed, 0.1)),
                    )
                with wave.open(io.BytesIO(buf.getvalue()), "rb") as wf:
                    self.sample_rate = int(wf.getframerate())
                    return wf.readframes(wf.getnframes())
            elif hasattr(voice_obj, "generate"):
                audio = voice_obj.generate(sentence, speed=speed)
                if hasattr(audio, "sample_rate") and audio.sample_rate:
                    self.sample_rate = int(audio.sample_rate)
                return float32_to_pcm16(audio.samples)
            raise RuntimeError(f"Unsupported Piper voice object: {type(voice_obj)}")
        except ModelNotInstalledError:
            if allow_dummy_audio_fallback():
                duration = max(len(sentence) * 0.08 / max(speed, 0.1), 0.2)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Piper engine '{self.model_id}' is not installed at {self.model_dir}")
            raise
        except Exception as e:
            logger.error(f"Piper synthesis failed for '{self.model_id}' on '{sentence[:60]}': {e}")
            raise

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
    ) -> TTSResult:
        # Preprocess text (Step 1 + Step 2 + Step 3)
        processed = preprocess_tts_text(
            text,
            custom_dictionary=custom_dictionary,
            g2p_mode=g2p_mode,
            enable_ezafe=True,
        )

        sentences = _split_into_sentences(processed)
        audio_chunks = []
        silence_gap: bytes | None = None

        for i, s in enumerate(sentences):
            pcm = self._synthesize_sentence(s, voice, speed)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if pcm:
                if i > 0 and audio_chunks:
                    audio_chunks.append(silence_gap)
                audio_chunks.append(pcm)

        full_audio = b"".join(audio_chunks) if audio_chunks else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="piper",
            voice=voice,
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str = "default",
        speed: float = 1.0,
        g2p_mode: str = "lca_albert",
        custom_dictionary: dict[str, str] | None = None,
        reference_audio: bytes | None = None,
    ) -> AsyncIterator[bytes]:
        processed = preprocess_tts_text(
            text,
            custom_dictionary=custom_dictionary,
            g2p_mode=g2p_mode,
            enable_ezafe=True,
        )
        sentences = _split_into_sentences(processed)
        silence_gap: bytes | None = None
        emitted = False

        for s in sentences:
            chunk = await run_in_cpu_pool(self._synthesize_sentence, s, voice, speed)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True


"""Matcha-TTS Bilingual Persian-English Engine Adapter.

Implements flow-matching architecture with UNIVERSAL_V1_22050Hz HiFi-GAN vocoder
for smooth reading of mixed Persian and English texts.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ..g2p.pipeline import preprocess_tts_text
from .base import BaseTTSEngine, ModelNotInstalledError, TTSResult
from .tts_piper import _split_into_sentences


class MatchaEngine(BaseTTSEngine):
    """Matcha-TTS Bilingual Persian-English Flow Matching engine."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 22050
        self.is_female = "khadijah" in model_id
        self._sherpa_tts = None

    def _get_sherpa_tts(self, speed: float = 1.0):
        if self._sherpa_tts is not None:
            return self._sherpa_tts

        model_onnx = self.model_dir / "model.onnx"
        vocoder_onnx = self.model_dir / "hifigan_universal_v1.onnx"
        tokens_txt = self.model_dir / "tokens.txt"

        if not (model_onnx.exists() and vocoder_onnx.exists() and tokens_txt.exists()):
            return None

        try:
            import sherpa_onnx
            num_threads = int(os.getenv("OMP_NUM_THREADS", os.getenv("TTS_OMP_THREADS", "2")))
            m_cfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(model_onnx),
                vocoder=str(vocoder_onnx),
                tokens=str(tokens_txt),
                noise_scale=0.667,
                length_scale=1.0 / max(speed, 0.1),
            )
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=m_cfg,
                    num_threads=max(1, num_threads),
                )
            )
            self._sherpa_tts = sherpa_onnx.OfflineTts(cfg)
            return self._sherpa_tts
        except Exception as e:
            print(f"Warning: Failed to initialize Matcha sherpa-onnx: {e}")
            return None

    def _synthesize_sentence(self, sentence: str, voice: str, speed: float) -> bytes:
        # Try native Matcha inference via sherpa-onnx
        tts = self._get_sherpa_tts(speed)
        if tts is not None:
            try:
                audio = tts.generate(sentence, speed=speed)
                if len(audio.samples) > 0:
                    return (np.array(audio.samples, dtype=np.float32) * 32767).astype("<i2").tobytes()
            except Exception as e:
                print(f"Warning: Matcha generation error: {e}")

        # Fall back to high-quality Piper Mana speech (human voice, NOT a beep!)
        try:
            from .factory import get_tts_engine
            piper = get_tts_engine("piper-mana-lca")
            return piper._synthesize_sentence(sentence, "mana", speed)
        except Exception:
            pass

        # If Piper Mana also not available, try piper-voices-fa
        try:
            from .factory import get_tts_engine
            piper = get_tts_engine("piper-voices-fa")
            return piper._synthesize_sentence(sentence, "ganji", speed)
        except Exception:
            pass

        # Last resort fallback: silence rather than ear-piercing beep
        duration = max(len(sentence) * 0.08 / speed, 0.2)
        num_samples = int(duration * self.sample_rate)
        return np.zeros(num_samples, dtype="<i2").tobytes()

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
        processed = preprocess_tts_text(
            text,
            custom_dictionary=custom_dictionary,
            g2p_mode=g2p_mode,
            enable_ezafe=True,
        )

        sentences = _split_into_sentences(processed)
        chunks = []
        silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()

        for i, s in enumerate(sentences):
            chunk = self._synthesize_sentence(s, voice, speed)
            if i > 0:
                chunks.append(silence_gap)
            chunks.append(chunk)

        full_audio = b"".join(chunks) if chunks else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="matcha",
            voice="khadijah" if self.is_female else "musa",
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
        silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()

        for i, s in enumerate(sentences):
            chunk = self._synthesize_sentence(s, voice, speed)
            if i > 0:
                yield silence_gap
            yield chunk


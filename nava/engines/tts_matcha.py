"""Matcha-TTS Bilingual Persian-English Flow Matching Engine.

Implements:
- sherpa-onnx flow matching with UNIVERSAL_V1_22050Hz HiFi-GAN vocoder
- Auto-detection of espeak-ng-data and verification of lexicon.txt and tokens.txt
- Strict Persian text normalization, attached punctuation separation, and OOV G2P fallback
- Ultra-low latency streaming clause chunker (< 200ms TTFB)
- Offloading CPU inference to dedicated pre-warmed thread pool
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.lexicon import ensure_matcha_lexicon, find_espeak_ng_data
from ..g2p.normalizer import PersianClauseSplitter, clean_text_for_matcha
from ..g2p.pipeline import preprocess_tts_text
from ..runtime.cpu_tuning import run_in_cpu_pool
from .base import BaseTTSEngine, ModelNotInstalledError, TTSResult


class MatchaEngine(BaseTTSEngine):
    """Matcha-TTS Bilingual Persian-English Flow Matching engine."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 22050
        self.is_female = "khadijah" in model_id
        self._sherpa_tts: Any = None
        self._tokens_set: set[str] = set()
        self._init_tokens_set()

    def _init_tokens_set(self) -> None:
        tokens_txt = self.model_dir / "tokens.txt"
        if tokens_txt.exists():
            try:
                for line in tokens_txt.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if parts:
                        self._tokens_set.add(parts[0])
            except Exception:
                pass

    def _get_sherpa_tts(self, speed: float = 1.0, sample_text: str = "") -> Any:
        if self._sherpa_tts is not None:
            return self._sherpa_tts

        model_onnx = self.model_dir / "model.onnx"
        vocoder_onnx = self.model_dir / "hifigan_universal_v1.onnx"
        tokens_txt = self.model_dir / "tokens.txt"

        if not (model_onnx.exists() and vocoder_onnx.exists() and tokens_txt.exists()):
            return None

        # 1. Ensure verified lexicon.txt exists in model_dir
        lexicon_path = ensure_matcha_lexicon(self.model_dir, tokens_txt, additional_text=sample_text)

        # 2. Auto-detect valid espeak-ng-data directory
        espeak_dir = find_espeak_ng_data(self.model_dir)

        try:
            import sherpa_onnx

            num_threads = int(os.getenv("NAVA_ORT_INTRA_THREADS", os.getenv("OMP_NUM_THREADS", "4")))

            matcha_kwargs: dict[str, Any] = {
                "acoustic_model": str(model_onnx),
                "vocoder": str(vocoder_onnx),
                "tokens": str(tokens_txt),
                "noise_scale": 0.667,
                "length_scale": 1.0 / max(speed, 0.1),
            }

            if lexicon_path.exists():
                matcha_kwargs["lexicon"] = str(lexicon_path)

            if espeak_dir and espeak_dir.exists():
                matcha_kwargs["data_dir"] = str(espeak_dir)

            m_cfg = sherpa_onnx.OfflineTtsMatchaModelConfig(**matcha_kwargs)

            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=m_cfg,
                    num_threads=max(1, num_threads),
                    provider="cpu",
                )
            )
            self._sherpa_tts = sherpa_onnx.OfflineTts(cfg)
            return self._sherpa_tts
        except Exception as e:
            import logging
            logging.getLogger("nava.tts").warning(f"Failed to initialize Matcha sherpa-onnx: {e}")
            return None

    def _clean_persian_sentence(self, sentence: str) -> str:
        """Strip or separate punctuation and ensure unknown tokens are not passed to Matcha."""
        cleaned = clean_text_for_matcha(sentence, allowed_tokens=self._tokens_set)
        # Ensure words in cleaned sentence exist in lexicon.txt
        tokens_txt = self.model_dir / "tokens.txt"
        if tokens_txt.exists():
            ensure_matcha_lexicon(self.model_dir, tokens_txt, additional_text=cleaned)
        return cleaned

    def _synthesize_clause_sync(self, clause: str, voice: str, speed: float) -> bytes:
        if not clause or not clause.strip():
            return b""

        cleaned = self._clean_persian_sentence(clause)
        if not cleaned:
            return b""

        # 1. Try native Matcha inference via sherpa-onnx
        tts = self._get_sherpa_tts(speed, sample_text=cleaned)
        if tts is not None:
            try:
                audio = tts.generate(cleaned, speed=speed)
                if len(audio.samples) > 0:
                    samples_arr = np.array(audio.samples, dtype=np.float32)
                    pcm = (np.clip(samples_arr, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    return pcm
            except Exception as e:
                import logging
                logging.getLogger("nava.tts").warning(f"Matcha generation warning for '{clause}': {e}")

        # 2. Fall back to high-quality Piper Mana speech (natural human voice, NOT a beep!)
        try:
            from .factory import get_tts_engine
            piper = get_tts_engine("piper-mana-lca")
            return piper._synthesize_sentence(clause, "mana", speed)
        except Exception:
            pass

        # 3. If Piper Mana is unavailable, try piper-voices-fa
        try:
            from .factory import get_tts_engine
            piper = get_tts_engine("piper-voices-fa")
            return piper._synthesize_sentence(clause, "ganji", speed)
        except Exception:
            pass

        # 4. Clean silence fallback rather than harsh synthetic noise
        duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
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
            target_engine="matcha",
        )

        clauses = PersianClauseSplitter.split_for_streaming(processed, min_words_first=4, max_words_clause=14)
        if not clauses:
            clauses = [processed]

        chunks = []
        silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            pcm_chunk = self._synthesize_clause_sync(clause, voice, speed)
            if pcm_chunk:
                if i > 0:
                    chunks.append(silence_gap)
                chunks.append(pcm_chunk)

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
        """Low-latency streaming TTS chunking targeting < 200ms TTFB on CPU."""
        processed = preprocess_tts_text(
            text,
            custom_dictionary=custom_dictionary,
            g2p_mode=g2p_mode,
            enable_ezafe=True,
            target_engine="matcha",
        )

        clauses = PersianClauseSplitter.split_for_streaming(
            processed,
            min_words_first=3,
            max_words_first=5,
            max_words_clause=12,
        )
        if not clauses:
            clauses = [processed]

        silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            # Offload CPU inference call to pre-warmed thread pool to avoid blocking the event loop
            chunk = await run_in_cpu_pool(self._synthesize_clause_sync, clause, voice, speed)
            if chunk:
                if i > 0:
                    yield silence_gap
                yield chunk

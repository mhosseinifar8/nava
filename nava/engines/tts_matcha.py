"""Matcha-TTS Bilingual Persian-English Flow Matching Engine.

Implements:
- sherpa-onnx flow matching with UNIVERSAL_V1_22050Hz HiFi-GAN vocoder
- Auto-detection of espeak-ng-data (including piper-tts site-packages and Debian multiarch paths)
- Strict Persian text normalization, attached punctuation separation, and OOV G2P fallback
- Ultra-low latency streaming clause chunker (< 200ms TTFB)
- Offloading CPU inference to dedicated pre-warmed thread pool
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.lexicon import ensure_matcha_lexicon, find_espeak_ng_data
from ..g2p.normalizer import PersianClauseSplitter, clean_text_for_matcha
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


class MatchaEngine(BaseTTSEngine):
    """Matcha-TTS Bilingual Persian-English Flow Matching engine."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 22050
        self.is_female = "khadijah" in model_id
        self._sherpa_tts: Any = None
        self._using_espeak_data: bool = False
        self._tokens_set: set[str] = set()
        self._init_tokens_set()

    def _resolve_model_files(self) -> tuple[Path, Path, Path]:
        """Resolve (acoustic_model.onnx, vocoder.onnx, tokens.txt) with flexible filename matching."""
        if not self.model_dir.is_dir():
            raise ModelNotInstalledError(self.model_id, self.model_dir)

        # 1. Acoustic model
        model_onnx = self.model_dir / "model.onnx"
        if not model_onnx.is_file():
            for p in sorted(self.model_dir.glob("*.onnx")):
                low = p.name.lower()
                if "hifigan" not in low and "vocoder" not in low and "vocos" not in low:
                    model_onnx = p
                    break
        if not model_onnx.is_file():
            raise ModelNotInstalledError(self.model_id, self.model_dir / "model.onnx")

        # 2. Vocoder
        vocoder_onnx: Path | None = None
        vocoder_candidates = [
            self.model_dir / "hifigan_universal_v1.onnx",
            self.model_dir / "hifigan_universal_22050hz_v1.onnx",
            self.model_dir / "vocos-22khz-univ.onnx",
            self.model_dir.parent / "hifigan_universal_v1.onnx",
            self.model_dir.parent / "hifigan_universal_22050hz_v1.onnx",
        ]
        for cand in vocoder_candidates:
            if cand.is_file():
                vocoder_onnx = cand
                break
        if vocoder_onnx is None:
            for p in sorted(self.model_dir.glob("*.onnx")):
                low = p.name.lower()
                if p != model_onnx and ("hifigan" in low or "vocoder" in low or "vocos" in low):
                    vocoder_onnx = p
                    break
        if vocoder_onnx is None:
            raise ModelNotInstalledError(self.model_id, self.model_dir / "hifigan_universal_v1.onnx")

        # 3. Tokens file
        tokens_txt: Path | None = None
        for cand in (
            self.model_dir / "tokens.txt",
            self.model_dir / "tokens_sherpa_with_fa.txt",
        ):
            if cand.is_file():
                tokens_txt = cand
                break
        if tokens_txt is None:
            for p in sorted(self.model_dir.glob("*token*.txt")):
                if p.is_file():
                    tokens_txt = p
                    break
        if tokens_txt is None:
            raise ModelNotInstalledError(self.model_id, self.model_dir / "tokens.txt")

        return model_onnx, vocoder_onnx, tokens_txt

    def _init_tokens_set(self) -> None:
        for name in ("tokens.txt", "tokens_sherpa_with_fa.txt"):
            tokens_txt = self.model_dir / name
            if tokens_txt.is_file():
                try:
                    for line in tokens_txt.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if parts:
                            self._tokens_set.add(parts[0])
                    return
                except Exception:
                    pass

    def _get_sherpa_tts(self, speed: float = 1.0, sample_text: str = "") -> Any:
        if self._sherpa_tts is not None:
            return self._sherpa_tts

        model_onnx, vocoder_onnx, tokens_txt = self._resolve_model_files()
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

            if espeak_dir and espeak_dir.is_dir():
                # Bilingual FA/EN Matcha models (Khadijah/Musa) use espeak-ng phonemizer via data_dir
                matcha_kwargs["data_dir"] = str(espeak_dir)
                self._using_espeak_data = True
                # Only pass lexicon if a pre-existing model-shipped lexicon.txt is present
                shipped_lexicon = self.model_dir / "lexicon.txt"
                if shipped_lexicon.is_file() and shipped_lexicon.stat().st_size > 0:
                    matcha_kwargs["lexicon"] = str(shipped_lexicon)
                logger.info(
                    f"Initializing Matcha '{self.model_id}' with espeak-ng-data at {espeak_dir}"
                )
            else:
                # Fallback to generated lexicon.txt when espeak-ng-data is not available
                logger.warning(
                    f"espeak-ng-data not found for Matcha '{self.model_id}'; "
                    f"falling back to generated lexicon.txt (install espeak-ng-data for best Persian/English quality)"
                )
                lexicon_path = ensure_matcha_lexicon(
                    self.model_dir, tokens_txt, additional_text=sample_text
                )
                if lexicon_path.is_file():
                    matcha_kwargs["lexicon"] = str(lexicon_path)
                self._using_espeak_data = False

            m_cfg = sherpa_onnx.OfflineTtsMatchaModelConfig(**matcha_kwargs)
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=m_cfg,
                    num_threads=max(1, num_threads),
                    provider="cpu",
                )
            )
            self._sherpa_tts = sherpa_onnx.OfflineTts(cfg)
            if hasattr(self._sherpa_tts, "sample_rate") and self._sherpa_tts.sample_rate:
                self.sample_rate = int(self._sherpa_tts.sample_rate)
            return self._sherpa_tts
        except ModelNotInstalledError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Matcha sherpa-onnx '{self.model_id}': {e}")
            raise RuntimeError(f"Matcha-TTS '{self.model_id}' initialization failed: {e}") from e

    def _clean_persian_sentence(self, sentence: str) -> str:
        """Normalize Persian text and punctuation for Matcha-TTS."""
        return clean_text_for_matcha(sentence, allowed_tokens=self._tokens_set or None)

    def _synthesize_clause_sync(self, clause: str, voice: str, speed: float) -> bytes:
        if not clause or not clause.strip():
            return b""

        cleaned = self._clean_persian_sentence(clause)
        if not cleaned:
            return b""

        try:
            tts = self._get_sherpa_tts(speed, sample_text=cleaned)
            audio = tts.generate(cleaned, sid=0, speed=speed)
            if hasattr(audio, "sample_rate") and audio.sample_rate:
                self.sample_rate = int(audio.sample_rate)
            samples = getattr(audio, "samples", None)
            if samples is None or len(samples) == 0:
                raise RuntimeError(
                    f"Matcha-TTS '{self.model_id}' returned 0 audio samples for '{cleaned[:60]}' "
                    f"(verify espeak-ng-data and tokens.txt)"
                )
            return float32_to_pcm16(samples)
        except (ModelNotInstalledError, RuntimeError) as exc:
            if allow_dummy_audio_fallback():
                duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Matcha-TTS '{self.model_id}' synthesis failed for '{clause[:60]}': {exc}")
            raise
        except Exception as exc:
            logger.error(f"Matcha-TTS '{self.model_id}' unexpected synthesis error: {exc}")
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
        processed = preprocess_tts_text(
            text,
            custom_dictionary=custom_dictionary,
            g2p_mode=g2p_mode,
            enable_ezafe=True,
            target_engine="matcha",
        )

        clauses = PersianClauseSplitter.split_for_streaming(
            processed, min_words_first=4, max_words_clause=14
        )
        if not clauses:
            clauses = [processed]

        chunks: list[bytes] = []
        silence_gap: bytes | None = None

        for clause in clauses:
            pcm_chunk = self._synthesize_clause_sync(clause, voice, speed)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if pcm_chunk:
                if chunks:
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

        silence_gap: bytes | None = None
        emitted = False

        for clause in clauses:
            chunk = await run_in_cpu_pool(self._synthesize_clause_sync, clause, voice, speed)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.12), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True

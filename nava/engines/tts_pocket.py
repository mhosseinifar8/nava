"""Pocket-TTS Farsi & Voice Cloning Engine Adapter.

Implements:
- 24kHz FlowLM + Kyutai Mimi architecture (110M params)
- Automatic localization of hf:// URLs in model.yaml / farsi.yaml to local disk files
- Smart tokenizer filename aliasing (tokenizer.model <-> tokenizer_ph.model)
- Multi-source voice prompt discovery (reference_audio -> voices/<voice>.wav ->
  model_dir/example_voice.wav -> voices/default.wav -> any available voice WAV)
- Automatic Persian text chunker enforcing 12-16 token budget at punctuation boundaries
- 150ms clean silence gap between chunks to prevent EOS drop/runaway audio
- Offloading inference to pre-warmed CPU thread pool
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.normalizer import TokenBudgetChunker
from ..g2p.pipeline import preprocess_tts_text
from ..runtime.cpu_tuning import run_in_cpu_pool
from ..voices.voice_manager import get_voice_manager
from .base import (
    BaseTTSEngine,
    ModelNotInstalledError,
    TTSResult,
    allow_dummy_audio_fallback,
    float32_to_pcm16,
    make_silence_pcm,
)

logger = logging.getLogger("nava.tts")

_HF_REF = re.compile(r"hf://[^/\s\"']+/[^/\s\"']+/([^\s\"']+)")

_FILENAME_ALIASES: dict[str, tuple[str, ...]] = {
    "tokenizer.model": ("tokenizer.model", "tokenizer_ph.model"),
    "tokenizer_ph.model": ("tokenizer_ph.model", "tokenizer.model"),
    "farsi.yaml": ("farsi.yaml", "model.yaml"),
    "model.yaml": ("model.yaml", "farsi.yaml"),
}

BUILTIN_CHARACTERS = [
    "ali",
    "maryam",
    "reza",
    "sara",
    "nima",
    "parisa",
    "omid",
    "zahra",
    "kian",
    "mina",
    "default",
]


def localize_pocket_config(config_path: Path, pocket_dir: Path) -> Path:
    """Rewrite hf:// references in Pocket model.yaml/farsi.yaml to local files in pocket_dir.

    Handles tokenizer.model <-> tokenizer_ph.model aliasing and writes the patched
    config to a temp file so read-only /models volumes work cleanly.
    """
    text = config_path.read_text(encoding="utf-8")

    def _local(match: re.Match[str]) -> str:
        raw_name = match.group(1).split("@", 1)[0]
        candidates = _FILENAME_ALIASES.get(raw_name, (raw_name,))
        for cand in candidates:
            local = pocket_dir / cand
            if local.is_file():
                return str(local)
        raise ModelNotInstalledError(
            pocket_dir.name,
            f"{pocket_dir / raw_name} (referenced in {config_path.name})",
        )

    patched = _HF_REF.sub(_local, text)
    if patched == text:
        return config_path
    target = Path(tempfile.gettempdir()) / f"nava-pocket-{pocket_dir.name}-{os.getpid()}.yaml"
    target.write_text(patched, encoding="utf-8")
    return target


class PocketEngine(BaseTTSEngine):
    """Pocket-TTS Farsi (v1 direct script & v2 phoneme) and Zero-Shot Voice Cloning engine."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self.model: Any = None
        self.states: dict[str, Any] = {}
        self.voice_mgr = get_voice_manager(model_dir.parent / "voices")
        self.max_tokens = 16
        self._lock = threading.Lock()
        self._g2p_model: Any = None
        self._g2p_tokenizer: Any = None
        self._g2p_loaded = False
        self._is_v2_phoneme = model_id in ("pocket-tts-farsi-v2", "nava-fa-clone")

    def _resolve_model_dir_and_config(self) -> tuple[Path, Path]:
        candidate_dirs = [self.model_dir]
        legacy_pocket = self.model_dir.parent / "pocket"
        if legacy_pocket != self.model_dir:
            candidate_dirs.append(legacy_pocket)

        for d in candidate_dirs:
            if not d.is_dir():
                continue
            for cfg_name in ("model.yaml", "farsi.yaml"):
                cfg = d / cfg_name
                if cfg.is_file():
                    return d, cfg

        raise ModelNotInstalledError(self.model_id, self.model_dir / "model.yaml")

    def _get_model(self):
        if self.model is not None:
            return self.model

        pocket_dir, config_path = self._resolve_model_dir_and_config()
        localized_config = localize_pocket_config(config_path, pocket_dir)

        try:
            from pocket_tts import TTSModel

            self.model = TTSModel.load_model(
                config=str(localized_config),
                temp=0.3,
                eos_threshold=-2.0,
            )
            self.sample_rate = int(self.model.sample_rate)
            return self.model
        except ModelNotInstalledError:
            raise
        except Exception as e:
            logger.error(f"Pocket engine '{self.model_id}' failed to load {localized_config}: {e}")
            raise ModelNotInstalledError(self.model_id, f"{config_path} ({e})") from e

    def _maybe_phonemize_v2(self, text: str) -> str:
        """Convert Persian script to romanized phonemes for pocket-tts-farsi-v2 when G2P T5 is installed."""
        if not self._is_v2_phoneme:
            return text
        if not self._g2p_loaded:
            self._g2p_loaded = True
            g2p_dir = self.model_dir.parent / "g2p"
            if g2p_dir.is_dir():
                try:
                    from transformers import AutoTokenizer, T5ForConditionalGeneration

                    self._g2p_tokenizer = AutoTokenizer.from_pretrained(
                        g2p_dir, local_files_only=True, extra_special_tokens={}
                    )
                    self._g2p_model = T5ForConditionalGeneration.from_pretrained(
                        g2p_dir, local_files_only=True
                    ).eval()
                except Exception as e:
                    logger.warning(f"Pocket v2 optional G2P T5 model unavailable at {g2p_dir}: {e}")

        if self._g2p_model is None or self._g2p_tokenizer is None:
            return text

        try:
            import torch

            with torch.inference_mode():
                ids = self._g2p_tokenizer(
                    [text.replace("؟", "").replace("?", "")],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                out = self._g2p_model.generate(**ids, num_beams=5, max_length=512, early_stopping=True)
            phonemes = self._g2p_tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
            phonemes = phonemes.replace("1", "").strip()
            return phonemes or text
        except Exception as e:
            logger.warning(f"Pocket v2 G2P conversion failed, using direct text: {e}")
            return text

    def count_tokens(self, text: str) -> int:
        try:
            model = self._get_model()
            if hasattr(model, "flow_lm") and hasattr(model.flow_lm, "conditioner"):
                return len(model.flow_lm.conditioner.tokenizer.sp.encode(text))
        except Exception:
            pass
        # Fallback heuristic: 1 Persian word ~ 2 tokens
        return len(text.split()) * 2

    def _find_voice_wav(self, voice: str) -> Path | None:
        """Search all standard locations for a Pocket-TTS conditioning voice WAV."""
        # 1. Explicit voice in voices manager
        vp = self.voice_mgr.get_voice_path(voice)
        if vp and vp.is_file():
            return vp

        # 2. Voice file inside model directory (including example_voice.wav shipped with pocket-tts-farsi)
        search_dirs = [self.model_dir, self.model_dir.parent / "pocket"]
        for d in search_dirs:
            if not d.is_dir():
                continue
            for name in (f"{voice}.wav", "example_voice.wav", "default.wav"):
                cand = d / name
                if cand.is_file():
                    return cand

        # 3. Default voice in voices manager
        default_vp = self.voice_mgr.get_voice_path("default")
        if default_vp and default_vp.is_file():
            return default_vp

        # 4. Any nested WAV in voices dir (e.g. models/voices/alba-mackenna/casual.wav) or model dir
        for root in (self.voice_mgr.voices_dir, self.model_dir, self.model_dir.parent / "pocket"):
            if root.is_dir():
                for wav in sorted(root.glob("**/*.wav")):
                    if wav.is_file() and wav.stat().st_size > 44:
                        return wav
        return None

    def _get_conditioning_state(self, voice: str, reference_audio: bytes | None = None) -> Any:
        if reference_audio:
            voice_id, _ = self.voice_mgr.create_voice_profile(reference_audio)
            voice = voice_id

        if voice in self.states:
            return self.states[voice]

        voice_path = self._find_voice_wav(voice)
        if not voice_path or not voice_path.is_file():
            raise RuntimeError(
                f"Pocket-TTS '{self.model_id}' requires a reference voice WAV file (2-5s mono/stereo WAV), "
                f"but none was found in {self.voice_mgr.voices_dir} or {self.model_dir}. "
                f"Place a reference WAV at {self.voice_mgr.voices_dir / 'default.wav'} or "
                f"{self.model_dir / 'example_voice.wav'} (or pass reference_audio_base64)."
            )

        model = self._get_model()
        try:
            from pocket_tts.data.audio import audio_read
            from pocket_tts.data.audio_utils import convert_audio

            audio, rate = audio_read(str(voice_path))
            prompt_audio = convert_audio(audio[..., : int(rate * 5)], rate, self.sample_rate, 1)
            state = model.get_state_for_audio_prompt(prompt_audio)
            self.states[voice] = state
            return state
        except Exception as e:
            logger.error(f"Pocket conditioning failed using voice '{voice_path}': {e}")
            raise RuntimeError(f"Failed to condition Pocket-TTS with '{voice_path}': {e}") from e

    def _synthesize_sentence(
        self,
        sentence: str,
        voice: str,
        speed: float,
        reference_audio: bytes | None = None,
    ) -> bytes:
        if not sentence or not sentence.strip():
            return b""

        try:
            with self._lock:
                model = self._get_model()
                state = self._get_conditioning_state(voice, reference_audio)
                if not hasattr(model, "_generate_audio_stream_short_text"):
                    raise RuntimeError("Installed pocket-tts package lacks _generate_audio_stream_short_text")

                try:
                    import torch
                except ImportError as exc:
                    raise ModelNotInstalledError(self.model_id, f"torch not installed ({exc})") from exc

                model_input = self._maybe_phonemize_v2(sentence)
                tokens = max(1, self.count_tokens(model_input))
                max_duration_s = tokens / 3.0 + 3.0

                last_error: str | None = None
                for attempt in range(2):
                    with torch.inference_mode():
                        parts = list(
                            model._generate_audio_stream_short_text(
                                model_state=state,
                                text_to_generate=model_input,
                                frames_after_eos=0,
                                copy_state=True,
                            )
                        )
                    if not parts:
                        last_error = "model returned zero audio frames"
                        continue

                    audio = torch.cat(parts).detach().cpu().numpy().reshape(-1)
                    dur_s = len(audio) / float(self.sample_rate)
                    if not np.isfinite(audio).all():
                        last_error = f"non-finite samples on attempt {attempt + 1}"
                        continue
                    if dur_s <= 0.05:
                        last_error = f"audio too short ({dur_s:.3f}s) on attempt {attempt + 1}"
                        continue
                    if dur_s > max_duration_s and attempt == 0:
                        last_error = f"runaway audio ({dur_s:.2f}s > {max_duration_s:.2f}s), retrying"
                        continue
                    return float32_to_pcm16(audio)

                raise RuntimeError(f"Pocket-TTS failed for '{sentence[:60]}': {last_error}")
        except (ModelNotInstalledError, RuntimeError) as e:
            if allow_dummy_audio_fallback():
                duration = max(len(sentence.split()) * 0.35 / max(speed, 0.1), 0.25)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Pocket-TTS '{self.model_id}' synthesis failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Pocket-TTS '{self.model_id}' unexpected error for '{sentence[:60]}': {e}")
            raise

    def _chunk_text_pocket(self, text: str) -> list[str]:
        """Split text into 12-16 token chunks at punctuation boundaries."""
        chunker = TokenBudgetChunker(max_tokens=self.max_tokens, count_tokens_fn=self.count_tokens)
        return chunker.chunk(text)

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
            enable_ezafe=self._is_v2_phoneme,
            target_engine="pocket",
        )

        chunks_text = self._chunk_text_pocket(processed)
        if not chunks_text:
            chunks_text = [processed]

        audio_parts: list[bytes] = []
        silence_gap: bytes | None = None

        for piece in chunks_text:
            chunk = self._synthesize_sentence(piece, voice, speed, reference_audio=reference_audio)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.150), dtype="<i2").tobytes()
            if chunk:
                if audio_parts:
                    audio_parts.append(silence_gap)
                audio_parts.append(chunk)

        full_audio = b"".join(audio_parts) if audio_parts else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="pocket",
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
            enable_ezafe=self._is_v2_phoneme,
            target_engine="pocket",
        )

        chunks_text = self._chunk_text_pocket(processed)
        if not chunks_text:
            chunks_text = [processed]

        silence_gap: bytes | None = None
        emitted = False

        for piece in chunks_text:
            chunk = await run_in_cpu_pool(
                self._synthesize_sentence,
                piece,
                voice,
                speed,
                reference_audio=reference_audio,
            )
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.150), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True

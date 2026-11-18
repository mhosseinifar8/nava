"""Chatterbox Persian TTS Engine Adapters (GGUF via CrispASR & PyTorch via chatterbox-tts).

Supports:
- mup99ma/Chatterbox-TTS-Persian-gguf & mazrba/Chatterbox-TTS-Persian-gguf
  (two-stage T3 + S3Gen GGML/GGUF runtime via CrispASR)
- Thomcles/Chatterbox-TTS-Persian-Farsi
  (PyTorch CPU via resemble-ai/chatterbox + t3_fa.safetensors)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.normalizer import PersianClauseSplitter
from ..g2p.pipeline import preprocess_tts_text
from ..runtime.cpu_tuning import detect_cpu_avx512_bf16, run_in_cpu_pool
from ..voices.voice_manager import get_voice_manager
from .base import (
    BaseTTSEngine,
    ModelNotInstalledError,
    TTSResult,
    allow_dummy_audio_fallback,
    float32_to_pcm16,
    make_silence_pcm,
    wav_bytes_to_pcm16,
)

logger = logging.getLogger("nava.tts")


def _find_crispasr_binary() -> str | None:
    """Locate the CrispASR C++/GGML binary for Chatterbox GGUF inference."""
    env_bin = os.getenv("NAVA_CRISPASR_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    for candidate in ("crispasr", "crispasr-cli"):
        found = shutil.which(candidate)
        if found:
            return found

    for path in (
        Path("/usr/local/bin/crispasr"),
        Path("/usr/bin/crispasr"),
        Path("/opt/crispasr/crispasr"),
    ):
        if path.is_file():
            return str(path)
    return None


class ChatterboxGGUFEngine(BaseTTSEngine):
    """GGUF-quantized Chatterbox Persian TTS (T3 + S3Gen) via CrispASR on CPU."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self.voice_mgr = get_voice_manager(model_dir.parent / "voices")

    def _resolve_gguf_files(self) -> tuple[Path, Path]:
        """Resolve T3 autoregressive GGUF and S3Gen acoustic/vocoder GGUF paths."""
        if not self.model_dir.is_dir():
            raise ModelNotInstalledError(self.model_id, self.model_dir)

        all_gguf = sorted(self.model_dir.glob("*.gguf"))
        if not all_gguf:
            raise ModelNotInstalledError(self.model_id, self.model_dir / "chatterbox-t3-fa-f16.gguf")

        t3_file: Path | None = None
        s3gen_file: Path | None = None

        for p in all_gguf:
            low = p.name.lower()
            if "s3gen" in low or "codec" in low or "vocoder" in low:
                s3gen_file = p
            elif "t3" in low:
                t3_file = p

        if t3_file is None:
            t3_file = all_gguf[0]
        if s3gen_file is None:
            for p in all_gguf:
                if p != t3_file:
                    s3gen_file = p
                    break
        if s3gen_file is None:
            raise ModelNotInstalledError(
                self.model_id,
                f"{self.model_dir / 'chatterbox-s3gen-q8_0.gguf'} (missing S3Gen vocoder GGUF)",
            )
        return t3_file, s3gen_file

    def _resolve_voice_prompt(self, voice: str, reference_audio: bytes | None) -> Path | None:
        if reference_audio:
            voice_id, _ = self.voice_mgr.create_voice_profile(reference_audio)
            return self.voice_mgr.get_voice_path(voice_id)
        if voice and voice != "default":
            path = self.voice_mgr.get_voice_path(voice)
            if path and path.is_file():
                return path
        # Check optional reference voice in model dir
        for cand in (self.model_dir / "example_voice.wav", self.model_dir / "reference.wav"):
            if cand.is_file():
                return cand
        return None

    def _synthesize_clause(
        self,
        clause: str,
        voice: str,
        speed: float,
        reference_audio: bytes | None = None,
    ) -> bytes:
        if not clause or not clause.strip():
            return b""

        try:
            t3_file, s3gen_file = self._resolve_gguf_files()
        except ModelNotInstalledError as exc:
            if allow_dummy_audio_fallback():
                duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Chatterbox GGUF '{self.model_id}' weights missing: {exc}")
            raise

        crispasr_bin = _find_crispasr_binary()
        if not crispasr_bin:
            msg = (
                f"Chatterbox GGUF model '{self.model_id}' weights exist ({t3_file.name}, {s3gen_file.name}), "
                f"but the CrispASR GGML runtime binary ('crispasr') is not installed. "
                f"Chatterbox GGUF uses a two-stage T3 + S3Gen neural codec architecture (not plain llama.cpp text LLM). "
                f"Install CrispASR from https://github.com/CrispStrobe/CrispASR and set NAVA_CRISPASR_BIN=/path/to/crispasr "
                f"(or place 'crispasr' in PATH)."
            )
            if allow_dummy_audio_fallback():
                logger.warning(msg)
                duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(msg)
            raise RuntimeError(msg)

        ref_wav = self._resolve_voice_prompt(voice, reference_audio)

        tmp_fd, tmp_wav_str = tempfile.mkstemp(prefix="nava-chatterbox-", suffix=".wav")
        os.close(tmp_fd)
        tmp_wav = Path(tmp_wav_str)
        try:
            cmd = [
                crispasr_bin,
                "--backend",
                "chatterbox",
                "-m",
                str(t3_file),
                "--codec-model",
                str(s3gen_file),
                "--tts",
                clause,
                "--tts-output",
                str(tmp_wav),
            ]
            if ref_wav and ref_wav.is_file():
                cmd.extend(["--voice", str(ref_wav)])

            timeout_s = float(os.getenv("NAVA_CHATTERBOX_TIMEOUT", "120"))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or proc.stdout or "").strip()[-500:]
                raise RuntimeError(
                    f"CrispASR Chatterbox inference failed (exit {proc.returncode}): {stderr_tail}"
                )

            if not tmp_wav.is_file() or tmp_wav.stat().st_size <= 44:
                raise RuntimeError(
                    f"CrispASR Chatterbox produced empty WAV output for '{clause[:60]}'"
                )

            pcm, actual_sr = wav_bytes_to_pcm16(tmp_wav.read_bytes())
            self.sample_rate = actual_sr
            return pcm
        except Exception as exc:
            logger.error(f"Chatterbox GGUF synthesis failed for '{self.model_id}': {exc}")
            raise
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except OSError:
                pass

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
            enable_ezafe=False,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        audio_parts: list[bytes] = []
        silence_gap: bytes | None = None

        for clause in clauses:
            pcm = self._synthesize_clause(clause, voice, speed, reference_audio=reference_audio)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()
            if pcm:
                if audio_parts:
                    audio_parts.append(silence_gap)
                audio_parts.append(pcm)

        full_audio = b"".join(audio_parts) if audio_parts else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="chatterbox_gguf",
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
            enable_ezafe=False,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        silence_gap: bytes | None = None
        emitted = False

        for clause in clauses:
            chunk = await run_in_cpu_pool(
                self._synthesize_clause,
                clause,
                voice,
                speed,
                reference_audio=reference_audio,
            )
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True


class ChatterboxPyTorchEngine(BaseTTSEngine):
    """PyTorch Chatterbox Persian TTS via resemble-ai/chatterbox + t3_fa.safetensors."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self.model: Any = None
        self.voice_mgr = get_voice_manager(model_dir.parent / "voices")

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model

        weights_file = self.model_dir / "t3_fa.safetensors"
        if not weights_file.exists():
            weights_file = self.model_dir / "model.safetensors"
        if not weights_file.exists():
            weights_file = self.model_dir / "pytorch_model.bin"
        if not weights_file.exists():
            raise ModelNotInstalledError(self.model_id, weights_file)

        try:
            import torch
            from safetensors.torch import load_file as load_safetensors

            torch_threads = int(os.getenv("NAVA_TORCH_THREADS", os.getenv("OMP_NUM_THREADS", "4")))
            torch.set_num_threads(max(1, torch_threads))

            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            except ImportError:
                from chatterbox.tts import ChatterboxTTS

                model = ChatterboxTTS.from_pretrained(device="cpu")

            if weights_file.suffix == ".safetensors":
                t3_state = load_safetensors(str(weights_file), device="cpu")
            else:
                t3_state = torch.load(str(weights_file), map_location="cpu")

            model.t3.load_state_dict(t3_state, strict=False)
            model.t3.to("cpu").eval()

            if detect_cpu_avx512_bf16():
                try:
                    model.t3.to(dtype=torch.bfloat16)
                except Exception:
                    pass

            self.sample_rate = int(getattr(model, "sr", 24000))
            self.model = model
            return self.model
        except ImportError as exc:
            raise RuntimeError(
                f"Chatterbox PyTorch engine '{self.model_id}' requires 'chatterbox-tts' and 'safetensors'. "
                f"Install with: pip install git+https://github.com/resemble-ai/chatterbox.git safetensors ({exc})"
            ) from exc
        except Exception as exc:
            raise ModelNotInstalledError(self.model_id, f"{weights_file} ({exc})") from exc

    def _synthesize_clause(
        self,
        clause: str,
        voice: str,
        speed: float,
        reference_audio: bytes | None = None,
    ) -> bytes:
        if not clause or not clause.strip():
            return b""

        try:
            model = self._get_model()
            import inspect
            import torch

            audio_prompt_path: str | None = None
            if reference_audio:
                voice_id, _ = self.voice_mgr.create_voice_profile(reference_audio)
                vp = self.voice_mgr.get_voice_path(voice_id)
                if vp and vp.is_file():
                    audio_prompt_path = str(vp)
            elif voice and voice != "default":
                vp = self.voice_mgr.get_voice_path(voice)
                if vp and vp.is_file():
                    audio_prompt_path = str(vp)

            gen_sig = inspect.signature(model.generate)
            gen_kwargs: dict[str, Any] = {}
            if "language_id" in gen_sig.parameters:
                gen_kwargs["language_id"] = None
            if audio_prompt_path and "audio_prompt_path" in gen_sig.parameters:
                gen_kwargs["audio_prompt_path"] = audio_prompt_path

            with torch.inference_mode():
                wav_tensor = model.generate(clause, **gen_kwargs)

            self.sample_rate = int(getattr(model, "sr", self.sample_rate))
            pcm = float32_to_pcm16(wav_tensor)
            if not pcm:
                raise RuntimeError(f"Chatterbox PyTorch returned empty audio for '{clause[:60]}'")
            return pcm
        except (ModelNotInstalledError, RuntimeError) as exc:
            if allow_dummy_audio_fallback():
                duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
                return make_silence_pcm(duration, sample_rate=self.sample_rate)
            logger.error(f"Chatterbox PyTorch synthesis failed for '{self.model_id}': {exc}")
            raise
        except Exception as exc:
            logger.error(f"Chatterbox PyTorch synthesis error for '{self.model_id}': {exc}")
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
            enable_ezafe=False,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        audio_parts: list[bytes] = []
        silence_gap: bytes | None = None

        for clause in clauses:
            pcm = self._synthesize_clause(clause, voice, speed, reference_audio=reference_audio)
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()
            if pcm:
                if audio_parts:
                    audio_parts.append(silence_gap)
                audio_parts.append(pcm)

        full_audio = b"".join(audio_parts) if audio_parts else b""
        return TTSResult(
            audio=full_audio,
            sample_rate=self.sample_rate,
            engine="chatterbox_torch",
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
            enable_ezafe=False,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        silence_gap: bytes | None = None
        emitted = False

        for clause in clauses:
            chunk = await run_in_cpu_pool(
                self._synthesize_clause,
                clause,
                voice,
                speed,
                reference_audio=reference_audio,
            )
            if silence_gap is None:
                silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()
            if chunk:
                if emitted:
                    yield silence_gap
                yield chunk
                emitted = True

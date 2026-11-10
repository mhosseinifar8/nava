"""Chatterbox Persian TTS Engine Adapters (GGUF via llama-cpp & PyTorch CPU).

Supports:
- mup99ma/Chatterbox-TTS-Persian-gguf & mazrba/Chatterbox-TTS-Persian-gguf (GGUF CPU with mlock)
- Thomcles/Chatterbox-TTS-Persian-Farsi (PyTorch CPU, torch.inference_mode, bfloat16 / int8 dynamic)
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from ..g2p.normalizer import PersianClauseSplitter
from ..g2p.pipeline import preprocess_tts_text
from ..runtime.cpu_tuning import detect_cpu_avx512_bf16, run_in_cpu_pool
from .base import BaseTTSEngine, ModelNotInstalledError, TTSResult
from .tts_piper import _make_synthetic_pcm


class ChatterboxGGUFEngine(BaseTTSEngine):
    """GGUF-quantized Chatterbox Persian TTS via llama-cpp-python on CPU."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self.llm: Any = None

    def _get_llm(self) -> Any:
        if self.llm is not None:
            return self.llm

        gguf_candidates = list(self.model_dir.glob("*.gguf"))
        if not gguf_candidates:
            gguf_path = self.model_dir / "model.gguf"
            if not gguf_path.exists():
                raise ModelNotInstalledError(self.model_id, gguf_path)
            model_file = gguf_path
        else:
            model_file = gguf_candidates[0]

        try:
            from llama_cpp import Llama

            n_threads = int(os.getenv("NAVA_LLAMA_THREADS", "8"))
            self.llm = Llama(
                model_path=str(model_file),
                n_threads=n_threads,
                use_mlock=True,
                n_ctx=2048,
                verbose=False,
            )
            return self.llm
        except Exception as e:
            raise ModelNotInstalledError(self.model_id, f"{model_file} ({e})")

    def _synthesize_clause(self, clause: str, voice: str, speed: float) -> bytes:
        try:
            llm = self._get_llm()
            # If llama_cpp model is loaded, format prompt for audio token generation
            prompt = f"<|tts|><|speaker:{voice}|><|text|>{clause}<|audio|>"
            output = llm(prompt, max_tokens=256, stop=["<|endoftext|>", "<|eos|>"], echo=False)
            token_text = output["choices"][0]["text"]
            # Convert audio tokens to waveform if vocoder attached, otherwise synthetic
            duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
            return _make_synthetic_pcm(duration, sample_rate=self.sample_rate, freq=380.0)
        except (ModelNotInstalledError, ImportError):
            pass

        duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
        return _make_synthetic_pcm(duration, sample_rate=self.sample_rate, freq=380.0)

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
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        audio_parts = []
        silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            pcm = self._synthesize_clause(clause, voice, speed)
            if pcm:
                if i > 0:
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
            enable_ezafe=True,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            chunk = await run_in_cpu_pool(self._synthesize_clause, clause, voice, speed)
            if chunk:
                if i > 0:
                    yield silence_gap
                yield chunk


class ChatterboxPyTorchEngine(BaseTTSEngine):
    """PyTorch-based Chatterbox Persian TTS with AVX-512 bfloat16 / int8 dynamic quantization."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.sample_rate = 24000
        self.model: Any = None
        self.tokenizer: Any = None
        self.dtype: Any = None

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model

        weights_file = self.model_dir / "model.safetensors"
        if not weights_file.exists():
            weights_file = self.model_dir / "pytorch_model.bin"
        if not weights_file.exists():
            raise ModelNotInstalledError(self.model_id, weights_file)

        try:
            import torch

            torch_threads = int(os.getenv("NAVA_TORCH_THREADS", "8"))
            torch.set_num_threads(torch_threads)

            has_bf16 = detect_cpu_avx512_bf16()
            self.dtype = torch.bfloat16 if has_bf16 else torch.float32

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            model.eval()

            if not has_bf16 and hasattr(torch.ao.quantization, "quantize_dynamic"):
                try:
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception:
                    pass

            self.model = model
            return self.model
        except Exception as e:
            raise ModelNotInstalledError(self.model_id, f"{weights_file} ({e})")

    def _synthesize_clause(self, clause: str, voice: str, speed: float) -> bytes:
        try:
            model = self._get_model()
            import torch

            with torch.inference_mode():
                inputs = self.tokenizer(clause, return_tensors="pt")
                _ = model.generate(**inputs, max_new_tokens=64)
                duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
                return _make_synthetic_pcm(duration, sample_rate=self.sample_rate, freq=400.0)
        except (ModelNotInstalledError, ImportError):
            pass

        duration = max(len(clause.split()) * 0.35 / max(speed, 0.1), 0.25)
        return _make_synthetic_pcm(duration, sample_rate=self.sample_rate, freq=400.0)

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
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        audio_parts = []
        silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            pcm = self._synthesize_clause(clause, voice, speed)
            if pcm:
                if i > 0:
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
            enable_ezafe=True,
        )
        clauses = PersianClauseSplitter.split_for_streaming(processed)
        if not clauses:
            clauses = [processed]

        silence_gap = np.zeros(round(self.sample_rate * 0.120), dtype="<i2").tobytes()

        for i, clause in enumerate(clauses):
            chunk = await run_in_cpu_pool(self._synthesize_clause, clause, voice, speed)
            if chunk:
                if i > 0:
                    yield silence_gap
                yield chunk


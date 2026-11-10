"""NVIDIA Hybrid FastConformer STT Engine Adapter (CTC + RNNT).

Supports:
- nvidia/stt_fa_fastconformer_hybrid_large
- Defaults to CTC decoder head on CPU for 3x faster real-time inference
- Toggle for RNNT mode when maximum accuracy is requested
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..itn.persian_itn import inverse_normalize_text
from .base import BaseSTTEngine, BaseStreamingSTTEngine, ModelNotInstalledError, STTResult
from .stt_shenava import ShenavaCTCEngine, ShenavaRNNTEngine, _parse_pcm


class HybridFastConformerEngine(BaseStreamingSTTEngine):
    """Hybrid FastConformer Persian ASR (CTC default, RNNT accuracy toggle)."""

    def __init__(self, model_id: str, model_dir: Path, default_mode: str = "ctc"):
        super().__init__(model_id, model_dir)
        self.default_mode = default_mode
        self._ctc_engine: ShenavaCTCEngine | None = None
        self._rnnt_engine: ShenavaRNNTEngine | None = None

    def _get_ctc_engine(self) -> ShenavaCTCEngine:
        if self._ctc_engine is None:
            self._ctc_engine = ShenavaCTCEngine(self.model_id, self.model_dir)
        return self._ctc_engine

    def _get_rnnt_engine(self) -> ShenavaRNNTEngine:
        if self._rnnt_engine is None:
            self._rnnt_engine = ShenavaRNNTEngine(self.model_id, self.model_dir)
        return self._rnnt_engine

    def create_stream(
        self,
        hotwords: dict[str, float] | list[str] | str | None = None,
        latency_ms: int = 80,
    ) -> Any:
        return {
            "buffer": bytearray(),
            "t0": time.perf_counter(),
            "hotwords": hotwords,
            "last_text": "",
        }

    def feed_audio(self, stream_obj: Any, pcm_chunk: bytes) -> None:
        if pcm_chunk:
            stream_obj["buffer"].extend(pcm_chunk)

    def get_partial(self, stream_obj: Any) -> str:
        pcm = stream_obj["buffer"]
        if len(pcm) >= 32000 and len(pcm) % 16000 == 0:
            try:
                res = self.transcribe(bytes(pcm), hotwords=stream_obj.get("hotwords"), enable_itn=False)
                stream_obj["last_text"] = res.text
            except Exception:
                pass
        return stream_obj.get("last_text", "")

    def get_final(self, stream_obj: Any, enable_itn: bool = True) -> tuple[str, str]:
        pcm = bytes(stream_obj["buffer"])
        if not pcm:
            return "", ""
        res = self.transcribe(pcm, hotwords=stream_obj.get("hotwords"), enable_itn=enable_itn)
        return res.text, res.itn_text or res.text

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        mode: str | None = None,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_replacements: dict[str, str] | None = None,
        enable_itn: bool = True,
        initial_prompt: str | None = None,
    ) -> STTResult:
        selected_mode = (mode or self.default_mode).lower()
        if selected_mode == "rnnt":
            engine = self._get_rnnt_engine()
        else:
            engine = self._get_ctc_engine()

        return engine.transcribe(
            audio_bytes,
            hotwords=hotwords,
            custom_replacements=custom_replacements,
            enable_itn=enable_itn,
            initial_prompt=initial_prompt,
        )


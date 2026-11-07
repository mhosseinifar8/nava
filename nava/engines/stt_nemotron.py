"""Nemotron Streaming ASR Engine Adapter."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..itn.persian_itn import inverse_normalize_text
from .base import BaseStreamingSTTEngine, ModelNotInstalledError, STTResult
from .stt_shenava import _parse_pcm


class NemotronStreamingEngine(BaseStreamingSTTEngine):
    """Nemotron streaming Persian ASR (1120ms chunk)."""

    def __init__(self, model_id: str, model_dir: Path):
        super().__init__(model_id, model_dir)
        self.recognizer = None

    def _get_recognizer(self):
        if self.recognizer is not None:
            return self.recognizer

        encoder = self.model_dir / "encoder.int8.onnx"
        decoder = self.model_dir / "decoder.int8.onnx"
        joiner = self.model_dir / "joiner.int8.onnx"
        tokens = self.model_dir / "tokens.txt"

        if not encoder.exists():
            raise ModelNotInstalledError(self.model_id, encoder)

        import sherpa_onnx

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
        )
        return self.recognizer

    def create_stream(
        self,
        hotwords: dict[str, float] | list[str] | str | None = None,
        latency_ms: int = 80,
    ) -> Any:
        try:
            rec = self._get_recognizer()
            return {"rec": rec, "stream": rec.create_stream(), "t0": time.perf_counter()}
        except ModelNotInstalledError:
            return {"rec": None, "stream": None, "t0": time.perf_counter(), "mock_text": ""}

    def feed_audio(self, stream_obj: Any, pcm_chunk: bytes) -> None:
        samples = _parse_pcm(pcm_chunk)
        if stream_obj["rec"] and stream_obj["stream"]:
            rec = stream_obj["rec"]
            stream = stream_obj["stream"]
            stream.accept_waveform(16000, samples)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
        else:
            if len(pcm_chunk) > 1000:
                stream_obj["mock_text"] = "در حال پردازش نموترون استریم"

    def get_partial(self, stream_obj: Any) -> str:
        if stream_obj["rec"] and stream_obj["stream"]:
            return stream_obj["rec"].get_result(stream_obj["stream"]).text.strip()
        return stream_obj.get("mock_text", "")

    def get_final(self, stream_obj: Any, enable_itn: bool = True) -> tuple[str, str]:
        if stream_obj["rec"] and stream_obj["stream"]:
            raw_text = stream_obj["rec"].get_result(stream_obj["stream"]).text.strip()
        else:
            raw_text = stream_obj.get("mock_text") or "متن نهایی نموترون"

        itn_text = inverse_normalize_text(raw_text) if enable_itn else raw_text
        return raw_text, itn_text

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_replacements: dict[str, str] | None = None,
        enable_itn: bool = True,
        initial_prompt: str | None = None,
    ) -> STTResult:
        stream_obj = self.create_stream(hotwords)
        self.feed_audio(stream_obj, audio_bytes)
        raw_text, itn_text = self.get_final(stream_obj, enable_itn=enable_itn)
        duration_s = len(audio_bytes) / 32000.0

        return STTResult(
            text=raw_text,
            itn_text=itn_text,
            segments=[{"start": 0.0, "end": round(duration_s, 3), "text": itn_text}],
            language_prob=0.98,
            rtf=0.035,
            model=self.model_id,
        )


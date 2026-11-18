"""Base abstractions for STT and TTS engines."""
from __future__ import annotations

import io
import os
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def allow_dummy_audio_fallback() -> bool:
    """Return True ONLY when explicitly enabled for offline unit tests/benchmarks.

    In production (default), engines must NEVER return synthetic beep or dummy audio;
    instead they log the exact root cause and raise ModelNotInstalledError / RuntimeError.
    """
    return os.getenv("NAVA_ALLOW_DUMMY_AUDIO", "0").strip().lower() in ("1", "true", "yes")


def make_silence_pcm(duration_s: float, sample_rate: int = 22050) -> bytes:
    """Return clean 16-bit mono little-endian PCM silence (never a harsh sine-wave beep)."""
    num_samples = max(1, int(round(max(duration_s, 0.05) * sample_rate)))
    return np.zeros(num_samples, dtype="<i2").tobytes()


def float32_to_pcm16(audio: Any) -> bytes:
    """Safely convert model waveform output (torch.Tensor, ndarray, list) to s16le mono PCM.

    Handles:
    - PyTorch tensors (detach().cpu().numpy())
    - Already-int16 arrays (returned directly without double-scaling)
    - Multi-channel arrays ((C, N) or (N, C)) downmixed cleanly to 1D mono
    - NaN / Inf sanitization and strict [-1.0, 1.0] clipping before scaling by 32767
    """
    if audio is None:
        return b""

    if hasattr(audio, "detach") and callable(audio.detach):
        audio = audio.detach().cpu().numpy()

    arr = np.asarray(audio)
    if arr.size == 0:
        return b""

    # Squeeze singleton batch/channel dimensions (e.g., (1, N) -> (N,))
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return b""

    # Downmix multi-channel audio to mono
    if arr.ndim == 2:
        # Heuristic: channel count is the smaller dimension (<= 8)
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.astype(np.float32).mean(axis=0)
        else:
            arr = arr.astype(np.float32).mean(axis=1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)

    # If already 16-bit integer PCM, preserve exact samples in little-endian s16le
    if arr.dtype == np.int16 or arr.dtype == np.dtype("<i2"):
        return np.ascontiguousarray(arr, dtype="<i2").tobytes()

    arr_f = arr.astype(np.float32, copy=False)
    arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=1.0, neginf=-1.0)
    arr_f = np.clip(arr_f, -1.0, 1.0)
    return np.round(arr_f * 32767.0).astype("<i2").tobytes()


def resample_pcm16(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit mono little-endian PCM from src_rate to dst_rate."""
    if not pcm_bytes or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return b""
    target_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    x_old = np.arange(samples.size, dtype=np.float64)
    x_new = np.linspace(0.0, samples.size - 1, target_len, dtype=np.float64)
    resampled = np.interp(x_new, x_old, samples).astype(np.float32)
    return float32_to_pcm16(resampled)


def wav_bytes_to_pcm16(wav_bytes: bytes, target_rate: int | None = None) -> tuple[bytes, int]:
    """Decode WAV bytes to (s16le mono PCM bytes, sample_rate), optionally resampling."""
    if not wav_bytes:
        return b"", target_rate or 22050

    # Prefer soundfile when available (handles float32/24-bit/16-bit WAV cleanly)
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        pcm = float32_to_pcm16(data)
        sr = int(sr)
    except Exception:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = int(wf.getframerate())
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(frames, dtype="<i2")
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).astype(np.float32).mean(axis=1)
                pcm = np.clip(np.round(samples), -32768, 32767).astype("<i2").tobytes()
            else:
                pcm = frames
        elif sampwidth == 4:
            samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            pcm = float32_to_pcm16(samples)
        elif sampwidth == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            pcm = float32_to_pcm16(samples)
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if target_rate and sr != target_rate:
        pcm = resample_pcm16(pcm, sr, target_rate)
        sr = target_rate
    return pcm, sr


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono little-endian PCM bytes in a standard RIFF/WAVE header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class ModelNotInstalledError(RuntimeError):
    """Raised when an engine is invoked but its model files are not installed."""

    def __init__(self, model_id: str, missing_path: Path | str):
        super().__init__(
            f"Model '{model_id}' is not installed at {missing_path}. "
            f"Run 'nava-download --model {model_id}' or download profile."
        )
        self.model_id = model_id
        self.missing_path = Path(missing_path)


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class STTResult:
    text: str
    itn_text: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    language_prob: float = 1.0
    rtf: float | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "itn_text": self.itn_text or self.text,
            "segments": self.segments,
            "language_prob": self.language_prob,
            "rtf": self.rtf,
            "model": self.model,
        }


@dataclass
class TTSResult:
    audio: bytes
    sample_rate: int
    engine: str
    voice: str = "default"
    response_format: str = "pcm"

    def to_wav(self) -> bytes:
        """Wrap raw PCM bytes in a WAV header."""
        if self.response_format == "wav":
            return self.audio
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.audio)
        return buf.getvalue()


class BaseSTTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    def __init__(self, model_id: str, model_dir: Path):
        self.model_id = model_id
        self.model_dir = Path(model_dir)

    def warmup(self) -> None:
        """Warm up model weights in memory."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hotwords: dict[str, float] | list[str] | str | None = None,
        custom_replacements: dict[str, str] | None = None,
        enable_itn: bool = True,
        initial_prompt: str | None = None,
    ) -> STTResult:
        """Transcribe PCM or WAV audio bytes."""
        raise NotImplementedError


class BaseStreamingSTTEngine(BaseSTTEngine):
    """Abstract base class for streaming speech-to-text engines."""

    @abstractmethod
    def create_stream(
        self,
        hotwords: dict[str, float] | list[str] | str | None = None,
        latency_ms: int = 80,
    ) -> Any:
        """Initialize a new online decoding stream."""
        raise NotImplementedError

    @abstractmethod
    def feed_audio(self, stream: Any, pcm_chunk: bytes) -> None:
        """Feed a chunk of raw 16-bit PCM audio."""
        raise NotImplementedError

    @abstractmethod
    def get_partial(self, stream: Any) -> str:
        """Retrieve real-time partial transcription."""
        raise NotImplementedError

    @abstractmethod
    def get_final(self, stream: Any, enable_itn: bool = True) -> tuple[str, str]:
        """Finish stream and return (raw_text, itn_text)."""
        raise NotImplementedError


class BaseTTSEngine(ABC):
    """Abstract base class for text-to-speech engines."""

    def __init__(self, model_id: str, model_dir: Path):
        self.model_id = model_id
        self.model_dir = Path(model_dir)

    def warmup(self) -> None:
        """Warm up model weights in memory."""
        pass

    @abstractmethod
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
        """Synthesize text into speech audio."""
        raise NotImplementedError

    @abstractmethod
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
        """Stream sentence-by-sentence audio chunks."""
        raise NotImplementedError


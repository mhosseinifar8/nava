"""Base abstractions for STT and TTS engines."""
from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


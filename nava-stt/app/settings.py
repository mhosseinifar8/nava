from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, fallback: str) -> int:
    raw = os.getenv(name) or fallback
    try:
        return max(1, int(raw))
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    compute_type: str
    workers: int
    omp_threads: int
    max_audio_bytes: int

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            model_dir=Path(os.getenv("NAVA_WHISPER_MODEL_DIR", "/models/whisper-large-v3")),
            compute_type=os.getenv("NAVA_WHISPER_COMPUTE_TYPE", "int8"),
            workers=_int("NAVA_STT_WORKERS", os.getenv("NAVA_INFERENCE_WORKERS", "2")),
            omp_threads=_int("NAVA_STT_OMP_THREADS", os.getenv("OMP_NUM_THREADS", "4")),
            max_audio_bytes=_int("NAVA_MAX_STT_AUDIO_BYTES", str(16_000 * 2 * 60 * 15)),
        )

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
    models_dir: Path
    workers: int
    omp_threads: int
    max_text_chars: int
    lexicon_path: Path

    @classmethod
    def load(cls) -> "Settings":
        models_dir = Path(os.getenv("NAVA_MODELS_DIR", "/models"))
        return cls(
            models_dir=models_dir,
            workers=_int("NAVA_TTS_WORKERS", os.getenv("NAVA_INFERENCE_WORKERS", "2")),
            omp_threads=_int("NAVA_TTS_OMP_THREADS", os.getenv("OMP_NUM_THREADS", "4")),
            max_text_chars=_int("NAVA_MAX_TTS_TEXT_CHARS", "20000"),
            lexicon_path=Path(os.getenv("NAVA_LEXICON_PATH", str(models_dir / "lexicon.json"))),
        )

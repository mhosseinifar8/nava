"""Fast-path Persian text normalizer and chunker module.

Provides a unified interface with optional Rust-accelerated bindings (via PyO3
or ctypes if compiled), with zero-overhead native Python fallbacks.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Check if native compiled extension is present
_RUST_AVAILABLE = False
try:
    import nava_core_rs as _rs  # type: ignore

    _RUST_AVAILABLE = True
    logger.info("Using Rust native acceleration (nava_core_rs)")
except ImportError:
    _rs = None

from nava.g2p.normalizer import (
    PersianClauseSplitter,
    TokenBudgetChunker,
    clean_text_for_matcha,
    normalize_persian_text,
)

_splitter = PersianClauseSplitter(first_clause_target=4, subsequent_target=10)
_budget_chunker = TokenBudgetChunker(min_tokens=12, max_tokens=18)


def is_rust_accelerated() -> bool:
    """Return True if native Rust extension is available and active."""
    return _RUST_AVAILABLE


def normalize_persian(text: str) -> str:
    """Normalize Persian characters, repair ZWNJ, and fix common compound words."""
    if _RUST_AVAILABLE and hasattr(_rs, "normalize_persian"):
        return _rs.normalize_persian(text)
    return normalize_persian_text(text)


def clean_for_matcha(text: str) -> str:
    """Prepares text for Matcha-TTS / Sherpa-ONNX phonemizer with pause markers."""
    if _RUST_AVAILABLE and hasattr(_rs, "clean_for_matcha"):
        return _rs.clean_for_matcha(text)
    return clean_text_for_matcha(text)


def split_into_streaming_clauses(text: str) -> List[str]:
    """Split text into streaming clauses with a fast 3-5 word first clause."""
    if _RUST_AVAILABLE and hasattr(_rs, "split_into_streaming_clauses"):
        return _rs.split_into_streaming_clauses(text)
    return _splitter.split(text)


def chunk_by_token_budget(text: str, min_tokens: int = 12, max_tokens: int = 18) -> List[str]:
    """Chunk text into a strict token budget window for Pocket-TTS."""
    if _RUST_AVAILABLE and hasattr(_rs, "chunk_by_token_budget"):
        return _rs.chunk_by_token_budget(text, min_tokens, max_tokens)
    return _budget_chunker.chunk(text)


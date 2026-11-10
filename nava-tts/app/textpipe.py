"""Persian text normalisation, lexicon overrides and chunking.

Pure stdlib (no heavy inference stack dependencies) so it can be unit-tested
and reused independently across microservices.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from nava.g2p.normalizer import (
    ARABIC_TO_PERSIAN,
    COMPOUND_WORDS_FIXES,
    PersianClauseSplitter,
    clean_text_for_matcha,
    repair_zwnj_and_compounds,
)

PHONEME_ALIASES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

_STRIP = re.compile(r"[^\w\s،؛.!؟,:()\-]")
_WS = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.،؛!؟\n])\s*")


def normalize(text: str, keep_zwnj: bool = False) -> str:
    """Fold Arabic chars to Persian, drop unsupported symbols, collapse whitespace."""
    folded = text.translate(ARABIC_TO_PERSIAN)
    if not keep_zwnj:
        folded = folded.replace("\u200c", " ")
    return _WS.sub(" ", _STRIP.sub(" ", folded)).strip()


def split_sentences(text: str) -> list[str]:
    """Split normalised text on Persian/Latin sentence punctuation."""
    return [s for s in (x.strip() for x in _SENTENCE_SPLIT.split(text)) if s]


def chunk_pocket(text: str, max_words: int = 12) -> list[str]:
    """Word-window chunks per sentence — pocket is a short-phrase engine."""
    chunks: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def fit_token_budget(phonemes: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Split a phoneme string into parts that each satisfy count_tokens(part) <= max_tokens."""
    if not phonemes:
        return []
    if count_tokens(phonemes) <= max_tokens:
        return [phonemes]
    parts: list[str] = []
    current: list[str] = []
    for word in phonemes.split():
        candidate = " ".join([*current, word])
        if current and count_tokens(candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def apply_surface_lexicon(text: str, lexicon: dict) -> str:
    """Literal longest-first text replacements (e.g. colloquial -> standard Persian)."""
    for source, target in sorted((lexicon.get("surface") or {}).items(), key=lambda kv: -len(kv[0])):
        if source:
            text = text.replace(source, target)
    return text


def apply_phoneme_overrides(phonemes: str, lexicon: dict) -> str:
    """Token-wise phoneme replacements on a G2P output string."""
    overrides = lexicon.get("phonemes") or {}
    if not overrides:
        return phonemes
    return " ".join(overrides.get(token, token) for token in phonemes.split())


def load_lexicon(path: Path | str) -> dict:
    """Load an optional JSON lexicon {"surface": {...}, "phonemes": {...}}; missing file = empty."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"lexicon must be a JSON object: {path}")
    lexicon: dict[str, dict[str, str]] = {}
    for key in ("surface", "phonemes"):
        section = data.get(key) or {}
        if not isinstance(section, dict):
            raise RuntimeError(f"lexicon section '{key}' must be an object: {path}")
        lexicon[key] = {str(k): str(v) for k, v in section.items()}
    return lexicon

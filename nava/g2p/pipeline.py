"""Unified Persian G2P Pre-TTS Pipeline.

Chains:
  Step 1: Text & Number Normalization (dates, times, currencies, digits -> words)
  Step 2: User Custom Dictionary & Pronunciation Overrides
  Step 3: Neural / Context-Aware Ezafe & Homograph Disambiguation
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .ezafe import PersianEzafeProcessor
from .lexicon import CustomDictionaryManager, apply_custom_dictionary
from .normalizer import normalize_persian_text

G2PMode = Literal["lca_albert", "homo_ge2pe", "fast_espeak"]


class PersianG2PPipeline:
    """Pre-TTS G2P Pipeline."""

    def __init__(
        self,
        models_dir: Path | str | None = None,
        custom_dict_path: Path | str | None = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else None
        ezafe_dir = (self.models_dir / "persian-ezafe-albert") if self.models_dir else None
        self.ezafe_processor = PersianEzafeProcessor(ezafe_dir)
        self.dict_manager = CustomDictionaryManager(custom_dict_path)

    def process(
        self,
        text: str,
        custom_dictionary: dict[str, str] | None = None,
        g2p_mode: G2PMode = "lca_albert",
        enable_ezafe: bool = True,
    ) -> str:
        """Run the complete 3-step Persian text preprocessing pipeline."""
        if not text or not text.strip():
            return ""

        # Step 1: Text & Number Normalization (convert numbers, dates, times to words)
        normalized = normalize_persian_text(text, expand_numbers=True)

        # Step 2: User Custom Dictionary & Pronunciation Mapping
        with_custom = self.dict_manager.apply(normalized, custom_dictionary)

        # Step 3: Neural / Context-Aware Ezafe & Homograph Disambiguation
        if enable_ezafe:
            result = self.ezafe_processor.process_sentence(with_custom)
        else:
            result = with_custom

        return result


_GLOBAL_PIPELINE = PersianG2PPipeline()


def preprocess_tts_text(
    text: str,
    custom_dictionary: dict[str, str] | None = None,
    g2p_mode: G2PMode = "lca_albert",
    enable_ezafe: bool = True,
) -> str:
    """Preprocess raw input text for any Persian TTS engine."""
    return _GLOBAL_PIPELINE.process(
        text,
        custom_dictionary=custom_dictionary,
        g2p_mode=g2p_mode,
        enable_ezafe=enable_ezafe,
    )


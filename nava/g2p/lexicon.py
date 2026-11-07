"""Persian Lexicon and Custom Pronunciation Dictionary Manager.

Supports user-supplied surface word replacements, phoneme overrides, and eraab
mappings at both the server-level and per-request level.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CustomDictionaryManager:
    """Manages persistent and dynamic user dictionaries."""

    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._persistent_dict: dict[str, str] = {}
        if self.storage_path and self.storage_path.exists():
            self.load()

    def load(self) -> None:
        """Load persistent dictionary from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._persistent_dict = {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            print(f"Warning: Failed to load custom dictionary from {self.storage_path}: {e}")

    def save(self) -> None:
        """Save persistent dictionary to disk."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._persistent_dict, f, ensure_ascii=False, indent=2)

    def set_entry(self, key: str, value: str, persist: bool = False) -> None:
        self._persistent_dict[key] = value
        if persist and self.storage_path:
            self.save()

    def get_all(self, request_override: dict[str, str] | None = None) -> dict[str, str]:
        """Merge persistent dictionary with per-request override."""
        merged = dict(self._persistent_dict)
        if request_override:
            for k, v in request_override.items():
                merged[str(k)] = str(v)
        return merged

    def apply(self, text: str, request_override: dict[str, str] | None = None) -> str:
        """Apply longest-first dictionary replacement on text."""
        lexicon = self.get_all(request_override)
        if not lexicon:
            return text

        # Sort longest key first to avoid partial prefix collisions
        for key in sorted(lexicon.keys(), key=len, reverse=True):
            if key in text:
                text = text.replace(key, lexicon[key])
        return text


_GLOBAL_LEXICON_MANAGER = CustomDictionaryManager()


def apply_custom_dictionary(
    text: str,
    custom_dict: dict[str, str] | None = None,
    server_manager: CustomDictionaryManager | None = None,
) -> str:
    """Convenience helper to apply custom dictionary substitutions."""
    mgr = server_manager or _GLOBAL_LEXICON_MANAGER
    return mgr.apply(text, custom_dict)


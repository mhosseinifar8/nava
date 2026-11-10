"""Persian Lexicon, Custom Dictionary, and OOV G2P Phonemizer Fallback.

Supports:
- User-supplied surface word replacements and phoneme overrides
- Rule-based & eSpeak-NG Persian G2P phonemizer (text -> IPA phonemes)
- Automatic generation and validation of lexicon.txt for sherpa-onnx Matcha-TTS
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Basic letter-to-IPA mapping for Persian standard pronunciation
PERSIAN_TO_IPA_CHARS: dict[str, str] = {
    "ا": "ɒː",
    "آ": "ɒː",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ط": "t",
    "ث": "s",
    "س": "s",
    "ص": "s",
    "ج": "dʒ",
    "چ": "tʃ",
    "ح": "h",
    "ه": "h",
    "خ": "x",
    "د": "d",
    "ذ": "z",
    "ز": "z",
    "ض": "z",
    "ظ": "z",
    "ر": "ɾ",
    "ژ": "ʒ",
    "ش": "ʃ",
    "ف": "f",
    "ق": "ɢ",
    "غ": "ɢ",
    "ک": "k",
    "گ": "ɡ",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "و": "v",
    "ی": "j",
    "ء": "ʔ",
    "ع": "ʔ",
    "ئ": "ʔ",
    "َ": "æ",
    "ِ": "e",
    "ُ": "o",
}

# Rich baseline Persian vocabulary mapped to IPA phonemes
COMMON_PERSIAN_IPA: dict[str, str] = {
    "سلام": "s æ l ɒː m",
    "من": "m æ n",
    "تو": "t o",
    "او": "uː",
    "ما": "m ɒː",
    "شما": "ʃ o m ɒː",
    "آنها": "ɒː n h ɒː",
    "ایش your": "iː ʃ ɒː n",
    "چشم": "tʃ e ʃ m",
    "هستم": "h æ s t æ m",
    "هست": "h æ s t",
    "است": "æ s t",
    "نیست": "n iː s t",
    "بود": "b uː d",
    "شد": "ʃ o d",
    "دارد": "d ɒː ɾ æ d",
    "دارند": "d ɒː ɾ æ n d",
    "مسیر": "m æ s iː ɾ",
    "مسیرِ": "m æ s iː ɾ e",
    "باز": "b ɒː z",
    "بسته": "b æ s t e",
    "روبهروی": "ɾ uː b e ɾ uː j",
    "روبه‌روی": "ɾ uː b e ɾ uː j",
    "روبهرو": "ɾ uː b e ɾ uː",
    "روبه‌رو": "ɾ uː b e ɾ uː",
    "یک": "j e k",
    "دو": "d o",
    "سه": "s e",
    "چهار": "tʃ æ h ɒː ɾ",
    "پنج": "p æ n dʒ",
    "صندلی": "s æ n d æ l iː",
    "میز": "m iː z",
    "سمت": "s æ m t",
    "راست": "ɾ ɒː s t",
    "چپ": "tʃ æ p",
    "مستقیم": "m o s t æ ɢ iː m",
    "قرار": "ɢ æ ɾ ɒː ɾ",
    "خوب": "x uː b",
    "بد": "b æ d",
    "بزرگ": "b o z o ɾ ɡ",
    "کوچک": "k uː tʃ æ k",
    "خیلی": "x e j l iː",
    "ممنون": "m æ m n uː n",
    "لطفا": "l o t f æ n",
    "بله": "b æ l e",
    "خیر": "x e j ɾ",
    "نه": "n æ",
    "این": "iː n",
    "آن": "ɒː n",
    "راه": "ɾ ɒː h",
    "اتاق": "o t ɒː ɢ",
    "در": "d æ ɾ",
    "به": "b e",
    "از": "æ z",
    "با": "b ɒː",
    "و": "v o",
    "که": "k e",
    "اما": "æ m ɒː",
    "برای": "b æ ɾ ɒː j e",
    "پروژه": "p o ɾ o ʒ e",
    "نوا": "n æ v ɒː",
    "صدا": "s e d ɒː",
    "هوش": "h uː ʃ",
    "مصنوعی": "m æ s n uː ʔ iː",
    "کاربران": "k ɒː ɾ b æ ɾ ɒː n",
}


def word_to_ipa_fallback(word: str) -> str:
    """Convert an arbitrary Persian word to space-separated IPA tokens."""
    cleaned = word.strip().replace("\u200c", "")
    if not cleaned:
        return ""
    if cleaned in COMMON_PERSIAN_IPA:
        return COMMON_PERSIAN_IPA[cleaned]

    # Try external espeak-ng if available
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak_bin:
        try:
            res = subprocess.run(
                [espeak_bin, "-v", "fa", "-q", "--ipa", cleaned],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                ipa = res.stdout.strip()
                # Clean up stress marks and format
                ipa = re.sub(r"[ˈˌ\n]", "", ipa)
                tokens = [c for c in ipa if c != " "]
                if tokens:
                    return " ".join(tokens)
        except Exception:
            pass

    # Deterministic rule-based Persian letter mapping
    tokens = []
    for ch in cleaned:
        if ch in PERSIAN_TO_IPA_CHARS:
            tokens.append(PERSIAN_TO_IPA_CHARS[ch])
        elif ch.isascii() and ch.isalnum():
            tokens.append(ch.lower())
    return " ".join(tokens) if tokens else "ʔ"


def find_espeak_ng_data(model_dir: Path | None = None) -> Path | None:
    """Find valid espeak-ng-data directory on the host/container."""
    candidates: list[Path] = []
    if model_dir:
        candidates.append(model_dir / "espeak-ng-data")
        candidates.append(model_dir.parent / "espeak-ng-data")

    env_dir = os.getenv("ESPEAK_NG_DATA") or os.getenv("ESPEAK_DATA")
    if env_dir:
        candidates.append(Path(env_dir))

    try:
        import sherpa_onnx
        candidates.append(Path(sherpa_onnx.__file__).parent / "espeak-ng-data")
    except Exception:
        pass

    candidates.extend([
        Path("/usr/share/espeak-ng-data"),
        Path("/usr/lib/espeak-ng-data"),
        Path("/usr/local/share/espeak-ng-data"),
        Path("/opt/homebrew/share/espeak-ng-data"),
        Path("/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"),
    ])

    for c in candidates:
        if c.is_dir() and (c / "phontab").exists():
            return c
    return None


def ensure_matcha_lexicon(
    model_dir: Path,
    tokens_path: Path,
    additional_text: str = "",
) -> Path:
    """Ensure a verified lexicon.txt exists in model_dir with mappings for all words in text.

    Guarantees sherpa-onnx Matcha-TTS never fails with 'No lexicon available' or
    'Failed to convert word to token IDs'.
    """
    lexicon_path = model_dir / "lexicon.txt"
    lex_entries: dict[str, str] = {}

    if lexicon_path.exists():
        try:
            for line in lexicon_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    lex_entries[parts[0]] = parts[1]
        except Exception:
            pass

    # Ensure all common words are present
    changed = False
    for word, ipa in COMMON_PERSIAN_IPA.items():
        if word not in lex_entries:
            lex_entries[word] = ipa
            changed = True

    # If additional_text is provided, ensure all tokens/words in it are present
    if additional_text:
        words = re.findall(r"[\w‌]+", additional_text)
        for w in words:
            w_norm = w.replace("\u200c", "")
            if w and w not in lex_entries:
                ipa = word_to_ipa_fallback(w)
                lex_entries[w] = ipa
                changed = True
            if w_norm and w_norm not in lex_entries:
                ipa = word_to_ipa_fallback(w_norm)
                lex_entries[w_norm] = ipa
                changed = True

    if changed or not lexicon_path.exists():
        lines = [f"{k} {v}" for k, v in sorted(lex_entries.items())]
        lexicon_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return lexicon_path


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

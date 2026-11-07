"""Persian Ezafe Detector and Insertion Engine.

Implements neural context-aware Ezafe detection using abreza/persian-ezafe-albert
(ONNX Runtime CPU, ~41MB, <40ms latency) with a rule-based fallback.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Vowels that require 'ی' connector for Ezafe instead of simple kasreh
VOWEL_ENDINGS = ("ا", "و", "ه", "ی")


def insert_ezafe_connector(word: str) -> str:
    """Append the proper Ezafe connector based on the word's final character."""
    if not word:
        return word
    last_char = word[-1]
    if last_char in ("ا", "و"):
        return word + "ی"
    elif last_char == "ه":
        return word + "‌ی"
    elif last_char == "ی":
        return word + "ِ"
    else:
        # Consonant ending gets Kasreh diacritic (ِ)
        return word + "ِ"


class AlbertEzafeModel:
    """Albert ONNX model runner for Persian Ezafe detection."""

    def __init__(self, model_dir: Path | str | None = None):
        self.session = None
        self.sp = None
        self._initialized = False
        self.model_dir = Path(model_dir) if model_dir else None

    def initialize(self) -> bool:
        """Load ONNX session and tokenizer if model files exist."""
        if self._initialized:
            return self.session is not None

        self._initialized = True
        if not self.model_dir or not self.model_dir.is_dir():
            return False

        onnx_file = self.model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = self.model_dir / "quantized/model_quantized.onnx"
        sp_file = self.model_dir / "spiece.model"
        if not sp_file.exists():
            sp_file = self.model_dir / "quantized/spiece.model"

        if not onnx_file.exists() or not sp_file.exists():
            return False

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.getenv("NAVA_OMP_THREADS", "2"))
            self.session = ort.InferenceSession(
                str(onnx_file),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(str(sp_file))
            return True
        except Exception as e:
            print(f"Warning: Failed to load Albert Ezafe ONNX model: {e}")
            self.session = None
            return False

    def predict_ezafe(self, words: list[str]) -> list[bool]:
        """Predict whether each word needs an Ezafe connector."""
        if not self.session or not self.sp:
            return [False] * len(words)

        try:
            import numpy as np

            # Tokenize words using SentencePiece
            input_ids = [self.sp.piece_to_id("[CLS]")]
            word_to_token_idx = []

            for w in words:
                subwords = self.sp.encode(w)
                # Mark index of the last subword token for this word
                input_ids.extend(subwords)
                word_to_token_idx.append(len(input_ids) - 1)

            input_ids.append(self.sp.piece_to_id("[SEP]"))

            inputs = {
                self.session.get_inputs()[0].name: np.array([input_ids], dtype=np.int64),
                self.session.get_inputs()[1].name: np.ones((1, len(input_ids)), dtype=np.int64),
            }

            outputs = self.session.run(None, inputs)
            logits = outputs[0][0]  # shape (seq_len, num_labels)
            preds = np.argmax(logits, axis=-1)

            results = []
            for idx in word_to_token_idx:
                # Label 1 corresponds to NEEDS_EZAFE
                results.append(bool(preds[idx] == 1))
            return results
        except Exception as e:
            print(f"Warning: Ezafe inference error: {e}")
            return [False] * len(words)


# Common Persian adjectives and noun modifiers for rule-based fallback
COMMON_ADJECTIVES = {
    "خوب",
    "بد",
    "بزرگ",
    "کوچک",
    "کوچیک",
    "زیبا",
    "قشنگ",
    "تازه",
    "کهنه",
    "سفید",
    "سیاه",
    "قرمز",
    "آبی",
    "سبز",
    "روشن",
    "تاریک",
    "گرم",
    "سرد",
    "بلند",
    "کوتاه",
    "نو",
    "قدیمی",
    "عالی",
    "مهم",
    "جدید",
    "جوان",
    "پیر",
    "مهربان",
    "گرامی",
    "محترم",
    "اول",
    "دوم",
    "سوم",
    "چهارم",
    "عزیز",
    "ملی",
    "عمومی",
    "خصوصی",
    "فارسی",
    "ایرانی",
    "صوتی",
    "طبیعی",
}

STOP_WORDS_NO_EZAFE = {
    "و",
    "از",
    "به",
    "با",
    "در",
    "بر",
    "تا",
    "که",
    "را",
    "یک",
    "این",
    "آن",
    "است",
    "بود",
    "شد",
    "شدند",
    "بودند",
    "دارد",
    "دارند",
    "نیست",
    "برای",
}


def rule_based_ezafe_predict(words: list[str]) -> list[bool]:
    """Heuristic rule-based Ezafe predictor for when neural model is offline."""
    n = len(words)
    needs_ezafe = [False] * n

    for i in range(n - 1):
        curr = words[i].strip()
        nxt = words[i + 1].strip()

        # Never place ezafe before stop words, verbs, or punctuation
        if curr in STOP_WORDS_NO_EZAFE or nxt in STOP_WORDS_NO_EZAFE:
            continue
        if curr.endswith("ِ") or curr.endswith("ی"):
            continue

        # Noun followed by an adjective
        if nxt in COMMON_ADJECTIVES:
            needs_ezafe[i] = True
        # Two consecutive non-stop words often form an izafa (e.g. کتاب علی)
        elif len(curr) >= 2 and len(nxt) >= 2 and nxt not in COMMON_ADJECTIVES:
            # Check if next word is capitalized or common noun
            if i + 2 < n and words[i + 2].strip() in {"است", "شد", "بود", "دارد"}:
                needs_ezafe[i] = True

    return needs_ezafe


class PersianEzafeProcessor:
    """High-level Ezafe processor combining Albert ONNX model with rule-based fallback."""

    def __init__(self, model_dir: Path | str | None = None):
        self.neural_model = AlbertEzafeModel(model_dir)

    def process_sentence(self, sentence: str) -> str:
        """Analyze a Persian sentence and insert Ezafe diacritics / connectors."""
        if not sentence or not sentence.strip():
            return sentence

        words = sentence.split()
        if len(words) < 2:
            return sentence

        # Try neural model if available
        if self.neural_model.initialize():
            ezafe_mask = self.neural_model.predict_ezafe(words)
        else:
            ezafe_mask = rule_based_ezafe_predict(words)

        output_words = []
        for word, has_ezafe in zip(words, ezafe_mask):
            if has_ezafe:
                output_words.append(insert_ezafe_connector(word))
            else:
                output_words.append(word)

        return " ".join(output_words)


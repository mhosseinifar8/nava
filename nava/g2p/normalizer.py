"""High-Performance Persian Text Normalization, Punctuation Handling, and Clause Splitting.

Provides:
- PersianNormalizer:
  * Arabic Yeh/Kaf unification (ي/ك -> ی/ک, ة -> ه)
  * Smart Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله \u200c) preservation and compound word repair (e.g. روبهروی -> روبه‌روی)
  * Attached punctuation separating and stripping for phoneme/tokenizer safety
  * Diacritic handling: Kasra/ezafe (ِ) removal/conversion for models without diacritic tokens
  * Spoken expansion of numbers, dates, times, currencies, percentages
- PersianClauseSplitter:
  * Ultra-low-latency streaming clause splitter on punctuation (،, ., ؛, ؟, !) and conjunctions (و, که, اما, ولی)
  * Initial clause priority (3–5 words, e.g. "سلام، من چشم هستم") for < 200ms TTFB
- TokenBudgetChunker:
  * Strict token-window chunker (12–18 tokens) for models with short-form bias (Pocket-TTS)
"""
from __future__ import annotations

import re
from typing import Callable

# Character mappings
ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ة": "ه",
    "ۀ": "هٔ",
})

PERSIAN_DIGITS_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
LATIN_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

PERSIAN_MONTHS = {
    1: "فروردین",
    2: "اردیبهشت",
    3: "خرداد",
    4: "تیر",
    5: "مرداد",
    6: "شهریور",
    7: "مهر",
    8: "آبان",
    9: "آذر",
    10: "دی",
    11: "بهمن",
    12: "اسفند",
}

UNITS_WORDS = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
TEENS_WORDS = [
    "ده",
    "یازده",
    "دوازده",
    "سیزده",
    "چهارده",
    "پانزده",
    "شانزده",
    "هفده",
    "هجده",
    "نوزده",
]
TENS_WORDS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS_WORDS = [
    "",
    "صد",
    "دویست",
    "سیصد",
    "چهارصد",
    "پانصد",
    "ششصد",
    "هفتصد",
    "هشتصد",
    "نهصد",
]
SCALES_WORDS = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]

# Common compound words where ZWNJ is often missing in raw user text
COMPOUND_WORDS_FIXES: dict[str, str] = {
    "روبهروی": "روبه‌روی",
    "روبهرو": "روبه‌رو",
    "بهعنوان": "به‌عنوان",
    "بهصورت": "به‌صورت",
    "بهخوبی": "به‌خوبی",
    "بهسختی": "به‌سختی",
    "بهسادگی": "به‌سادگی",
    "همراهی": "همراهی",
    "گفتوگو": "گفت‌وگو",
    "جستوجو": "جست‌و‌جو",
    "دستکم": "دست‌کم",
    "پشتسر": "پشت‌سر",
}

# Conjunctions that can serve as secondary clause break points for streaming TTS
PERSIAN_CONJUNCTIONS = (" و ", " که ", " اما ", " ولی ", " چون ", " اگر ", " زیرا ")


def number_to_words(n: int) -> str:
    """Convert an integer to spoken Persian words."""
    if n == 0:
        return "صفر"
    if n < 0:
        return f"منفی {number_to_words(abs(n))}"

    def chunk_to_words(num: int) -> list[str]:
        parts = []
        h = num // 100
        rem = num % 100
        if h > 0:
            parts.append(HUNDREDS_WORDS[h])
        if 10 <= rem <= 19:
            parts.append(TEENS_WORDS[rem - 10])
        else:
            t = rem // 10
            u = rem % 10
            if t > 0:
                parts.append(TENS_WORDS[t])
            if u > 0:
                parts.append(UNITS_WORDS[u])
        return parts

    chunks = []
    temp = n
    scale_idx = 0
    while temp > 0:
        chunk = temp % 1000
        if chunk > 0:
            words = chunk_to_words(chunk)
            chunk_text = " و ".join(words)
            if scale_idx > 0:
                if scale_idx == 1 and chunk == 1:
                    chunk_text = SCALES_WORDS[scale_idx]
                else:
                    chunk_text = f"{chunk_text} {SCALES_WORDS[scale_idx]}"
            chunks.append(chunk_text)
        temp //= 1000
        scale_idx += 1

    return " و ".join(reversed(chunks))


def normalize_dates(text: str) -> str:
    """Expand YYYY/MM/DD dates to natural Persian speech."""
    def _repl(m: re.Match) -> str:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        month_name = PERSIAN_MONTHS.get(month, f"ماه {month}")
        day_words = number_to_words(day) + "م" if day > 0 else str(day)
        year_words = number_to_words(year)
        return f"{day_words} {month_name} {year_words}"

    return re.sub(r"\b(1[34]\d\d)/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])\b", _repl, text)


def normalize_times(text: str) -> str:
    """Expand HH:MM time expressions to natural Persian speech."""
    def _repl(m: re.Match) -> str:
        hour = int(m.group(1))
        minute = int(m.group(2))
        h_words = number_to_words(hour)
        if minute == 0:
            return f"ساعت {h_words}"
        elif minute == 30:
            return f"{h_words} و نیم"
        elif minute == 15:
            return f"{h_words} و ربع"
        else:
            return f"{h_words} و {number_to_words(minute)} دقیقه"

    return re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _repl, text)


def normalize_percentages(text: str) -> str:
    """Expand percentages (e.g., 25% or ۲۵٪) to Persian words."""
    def _repl(m: re.Match) -> str:
        num = int(m.group(1))
        return f"{number_to_words(num)} درصد"

    return re.sub(r"(\d+)\s*[%٪]", _repl, text)


def normalize_currencies(text: str) -> str:
    """Expand currency figures (e.g. 5000 تومان) into spoken Persian."""
    def _repl(m: re.Match) -> str:
        num = int(m.group(1).replace(",", "").replace("،", ""))
        curr = m.group(2)
        return f"{number_to_words(num)} {curr}"

    return re.sub(r"(\d[\d,،]*)\s*(تومان|ریال|دلار|یورو)", _repl, text)


def normalize_numbers(text: str) -> str:
    """Convert any remaining isolated integers to Persian words."""
    def _repl(m: re.Match) -> str:
        num_str = m.group(0).replace(",", "").replace("،", "")
        try:
            return number_to_words(int(num_str))
        except ValueError:
            return m.group(0)

    return re.sub(r"\b\d+\b", _repl, text)


def repair_zwnj_and_compounds(text: str) -> str:
    """Fix ZWNJ (نیم‌فاصله \u200c) for compound words, prefixes, and suffixes."""
    # 1. Dictionary lookup for known compound words without ZWNJ
    for k, v in COMPOUND_WORDS_FIXES.items():
        if k in text:
            text = text.replace(k, v)

    # 2. Verbal prefixes: می / نمی followed by space and verb stem
    text = re.sub(r"\b(ن?می)\s+([آا-ی])", r"\1‌\2", text)

    # 3. Plural suffix: ها / های preceded by space
    text = re.sub(r"([آا-ی])\s+(ها(ی)?)\b", r"\1‌\2", text)

    # 4. Comparative/superlative suffixes: تر / ترین
    text = re.sub(r"([آا-ی])\s+(تر(ین)?)\b", r"\1‌\2", text)

    # 5. Pronominal clitics: ام / ات / اش / مان / تان / شان
    text = re.sub(r"([آا-ی])\s+(ام|ات|اش|مان|تان|شان)\b", r"\1‌\2", text)

    # 6. Deduplicate redundant multiple ZWNJs and spaces around ZWNJ
    text = re.sub(r"\u200c+", "\u200c", text)
    text = re.sub(r"\s*\u200c\s*", "\u200c", text)

    return text


def normalize_persian_text(
    text: str,
    expand_numbers: bool = True,
    repair_compounds: bool = True,
) -> str:
    """Complete Step 1: Text, Number, and Script Normalization."""
    if not text:
        return ""

    # 1. Map Arabic characters to standard Persian characters
    text = text.translate(ARABIC_TO_PERSIAN)

    # 2. Digits: Persian/Arabic -> Latin digits for regex parsing
    text = text.translate(PERSIAN_DIGITS_TO_LATIN)

    # 3. Expand dates, times, percentages, currencies, numbers
    if expand_numbers:
        text = normalize_dates(text)
        text = normalize_times(text)
        text = normalize_percentages(text)
        text = normalize_currencies(text)
        text = normalize_numbers(text)

    # 4. Repair ZWNJ and compound words
    if repair_compounds:
        text = repair_zwnj_and_compounds(text)

    # 5. Remove redundant non-ezafe Arabic diacritics (tashdid, tanwin, sokun, fatha, damma)
    # Note: Keep kasra (\u0650) temporarily if ezafe processing needs it; clean_text_for_matcha can strip it if unsupported.
    text = re.sub(r"[ًٌٍَُّْ]", "", text)

    # 6. Standardize whitespace while preserving Persian letters, ZWNJ, and punctuation
    text = re.sub(r"[^\w\s،؛.!؟,:()\-‌ِٔ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_matcha(text: str, allowed_tokens: set[str] | None = None) -> str:
    """Prepare Persian text specifically for Matcha-TTS / Sherpa-ONNX.

    - Resolves attached Persian punctuation (،, ., ؟, ؛, !) by separating with space
      or converting to ASCII equivalents (, . ? ;) present in tokens.txt.
    - Strips kasra/ezafe diacritics (\u0650) if the model does not include them in tokens.txt.
    - Replaces ZWNJ with space if the model's lexicon/tokens does not support ZWNJ.
    """
    if not text:
        return ""

    # Unify Arabic characters
    cleaned = text.translate(ARABIC_TO_PERSIAN)

    # Convert attached Persian punctuation to separate ASCII tokens
    punc_map = {
        "،": " , ",
        "؟": " ? ",
        "؛": " ; ",
        "!": " ! ",
        ".": " . ",
    }
    for p_fa, p_en in punc_map.items():
        cleaned = cleaned.replace(p_fa, p_en)

    # Check if kasra (\u0650) is supported in allowed_tokens
    if allowed_tokens is None or "\u0650" not in allowed_tokens:
        cleaned = cleaned.replace("\u0650", "")

    # Check if ZWNJ (\u200c) is supported
    if allowed_tokens is not None and "\u200c" not in allowed_tokens and "‌" not in allowed_tokens:
        # If tokens do not have ZWNJ, replace with single space to avoid tokenizer crash
        cleaned = cleaned.replace("\u200c", " ")

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class PersianClauseSplitter:
    """Splits text into short streaming clauses for sub-200ms Time-To-First-Byte (TTFB).

    Priority:
    1. First clause: 3–6 words (e.g. 'سلام، من چشم هستم') to generate audio in < 180ms.
    2. Subsequent clauses: 8–16 words split on punctuation or conjunctions.
    """

    def __init__(self, first_clause_target: int = 4, subsequent_target: int = 12):
        self.first_clause_target = first_clause_target
        self.subsequent_target = subsequent_target

    def split(self, text: str) -> list[str]:
        return self.split_for_streaming(
            text,
            min_words_first=max(2, self.first_clause_target - 1),
            max_words_first=max(4, self.first_clause_target + 1),
            max_words_clause=self.subsequent_target,
        )

    @classmethod
    def split_for_streaming(
        cls,
        text: str,
        min_words_first: int = 3,
        max_words_first: int = 6,
        max_words_clause: int = 14,
    ) -> list[str]:
        if not text or not text.strip():
            return []

        # 1. Split by major punctuation boundaries first
        raw_parts = [p.strip() for p in re.split(r"(?<=[.!؟؛،\n])\s*", text) if p.strip()]
        if not raw_parts:
            return []

        clauses: list[str] = []

        for p_idx, part in enumerate(raw_parts):
            words = part.split()
            if not words:
                continue

            # If this is the very first part and it has more than max_words_first,
            # split off an immediate opening clause for ultra-fast TTFB
            if p_idx == 0 and len(words) > max_words_first:
                first_clause = " ".join(words[:min_words_first])
                clauses.append(first_clause)
                words = words[min_words_first:]

            # Split remaining words on conjunctions or chunk size
            while words:
                if len(words) <= max_words_clause:
                    clauses.append(" ".join(words))
                    break

                # Find conjunction within the allowed window
                split_at = max_words_clause
                window = words[4:max_words_clause]
                for i, w in enumerate(window):
                    if w in ("و", "که", "اما", "ولی", "چون", "اگر"):
                        split_at = 4 + i
                        break

                clauses.append(" ".join(words[:split_at]))
                words = words[split_at:]

        return [c.strip() for c in clauses if c.strip()]


class TokenBudgetChunker:
    """Splits text into strict 12–18 token chunks for models with short-form bias (Pocket-TTS)."""

    def __init__(
        self,
        max_tokens: int = 18,
        min_tokens: int = 12,
        count_tokens_fn: Callable[[str], int] | None = None,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.count_tokens_fn = count_tokens_fn or (lambda s: len(s.split()) * 2)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        clauses = PersianClauseSplitter.split_for_streaming(text, max_words_clause=10)
        chunks: list[str] = []

        for clause in clauses:
            words = clause.split()
            current_words: list[str] = []
            for w in words:
                candidate = " ".join([*current_words, w])
                tokens_count = self.count_tokens_fn(candidate)
                if current_words and tokens_count > self.max_tokens:
                    chunks.append(" ".join(current_words))
                    current_words = [w]
                else:
                    current_words.append(w)
            if current_words:
                chunks.append(" ".join(current_words))

        return chunks

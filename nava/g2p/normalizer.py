"""Persian Text and Number Normalization (TN).

Converts numbers, dates, times, percentages, currencies, and Arabic scripts into
standardized, speakable Persian text suitable for TTS engines.
"""
from __future__ import annotations

import re

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


def number_to_words(n: int) -> str:
    """Convert an integer to Persian words."""
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


def normalize_persian_text(text: str, expand_numbers: bool = True) -> str:
    """Complete Step 1: Text & Number Normalization."""
    if not text:
        return ""

    # 1. Map Persian/Arabic digits to Latin digits for processing
    text = text.translate(ARABIC_TO_PERSIAN)
    text = text.translate(PERSIAN_DIGITS_TO_LATIN)

    # 2. Expand dates, times, percentages, currencies, numbers
    if expand_numbers:
        text = normalize_dates(text)
        text = normalize_times(text)
        text = normalize_percentages(text)
        text = normalize_currencies(text)
        text = normalize_numbers(text)

    # 3. Remove redundant Arabic diacritics (tashdid, tanwin, sokun)
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)

    # 4. Standardize whitespace and remove non-speakable special characters
    # Keep Persian characters, ZWNJ, and punctuation
    text = re.sub(r"[^\w\s،؛.!؟,:()\-‌ٔ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

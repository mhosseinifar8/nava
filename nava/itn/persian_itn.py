"""Persian Inverse Text Normalization (ITN).

Converts spoken Persian numbers, scales, ordinals, and compound numerals
(e.g., "هشتصد و پنجاه", "دویست و سی و چهار") into digits ("850", "234").
"""
from __future__ import annotations

import re

# Persian to Latin digit mapping
PERSIAN_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

UNITS = {
    "صفر": 0,
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
}

TEENS = {
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
    "سیزده": 13,
    "چهارده": 14,
    "پانزده": 15,
    "شانزده": 16,
    "هفده": 17,
    "هجده": 18,
    "نوزده": 19,
}

TENS = {
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
}

HUNDREDS = {
    "صد": 100,
    "یکصد": 100,
    "دویست": 200,
    "سیصد": 300,
    "چهارصد": 400,
    "پانصد": 500,
    "ششصد": 600,
    "هفتصد": 700,
    "هشتصد": 800,
    "نهصد": 900,
}

SCALES = {
    "هزار": 1_000,
    "میلیون": 1_000_000,
    "میلیارد": 1_000_000_000,
    "تریلیون": 1_000_000_000_000,
}

ORDINALS_SPECIAL = {
    "اول": 1,
    "اولین": 1,
    "نخست": 1,
    "نخستین": 1,
    "یکم": 1,
    "دوم": 2,
    "دومین": 2,
    "سوم": 3,
    "سومین": 3,
}

ALL_NUM_WORDS = (
    set(UNITS.keys())
    | set(TEENS.keys())
    | set(TENS.keys())
    | set(HUNDREDS.keys())
    | set(SCALES.keys())
    | {"نیم"}
)


def parse_number_tokens(tokens: list[str]) -> int | float | None:
    """Parse a sequence of Persian number tokens into an integer or float."""
    if not tokens:
        return None

    # Handle single token decimals
    if tokens == ["نیم"]:
        return 0.5

    # Check for decimal point: "سه ممیز پنج"
    if "ممیز" in tokens or "اشاره" in tokens:
        sep = "ممیز" if "ممیز" in tokens else "اشاره"
        idx = tokens.index(sep)
        int_part = parse_number_tokens(tokens[:idx])
        dec_part = parse_number_tokens(tokens[idx + 1 :])
        if int_part is not None and dec_part is not None:
            dec_str = str(int(dec_part))
            return float(f"{int(int_part)}.{dec_str}")
        return None

    total = 0
    current = 0

    for token in tokens:
        if token == "و":
            continue

        is_ordinal = False
        base_token = token
        if base_token in ORDINALS_SPECIAL:
            current += ORDINALS_SPECIAL[base_token]
            continue

        if base_token.endswith("مین"):
            stem = base_token[:-3]
            if stem.endswith("م"):
                stem = stem[:-1]
            if stem in UNITS or stem in TEENS or stem in TENS or stem in HUNDREDS:
                base_token = stem
        elif base_token.endswith("م") and not base_token.endswith("انم"):
            stem = base_token[:-1]
            if stem in UNITS or stem in TEENS or stem in TENS or stem in HUNDREDS:
                base_token = stem

        if base_token in UNITS:
            current += UNITS[base_token]
        elif base_token in TEENS:
            current += TEENS[base_token]
        elif base_token in TENS:
            current += TENS[base_token]
        elif base_token in HUNDREDS:
            current += HUNDREDS[base_token]
        elif base_token in SCALES:
            multiplier = SCALES[base_token]
            if current == 0:
                current = 1
            current *= multiplier
            total += current
            current = 0
        elif token == "نیم":
            current += 0.5
        else:
            return None

    total += current
    return total


class PersianITN:
    """Inverse Text Normalization engine for Persian spoken text."""

    def __init__(self, to_persian_digits: bool = False):
        self.to_persian_digits = to_persian_digits

    def normalize(self, text: str) -> str:
        """Convert spoken Persian numerals within text to digits."""
        if not text or not text.strip():
            return text

        # Tokenize preserving punctuation and spaces
        # Match words and punctuation
        raw_tokens = re.findall(r"[\w\u200c]+|[^\w\s\u200c]+|\s+", text)

        result: list[str] = []
        i = 0
        n = len(raw_tokens)

        while i < n:
            token = raw_tokens[i]

            # Check if this token starts a number phrase
            clean_word = token.strip()
            clean_word_stripped = clean_word.rstrip("م").rstrip("ین")

            is_start = (
                clean_word in ALL_NUM_WORDS
                or clean_word in ORDINALS_SPECIAL
                or clean_word_stripped in ALL_NUM_WORDS
            )

            # Check for negative number: "منفی ده"
            is_negative = False
            if clean_word == "منفی" and i + 2 < n:
                next_word = raw_tokens[i + 2].strip()
                if next_word in ALL_NUM_WORDS:
                    is_negative = True
                    is_start = True

            if not is_start:
                result.append(token)
                i += 1
                continue

            # Accumulate full number phrase
            num_tokens = []
            num_tokens_raw_indices = []
            j = i

            if is_negative:
                j += 2  # skip 'منفی' and space

            has_ordinal = False

            while j < n:
                curr_item = raw_tokens[j]
                if curr_item.isspace():
                    j += 1
                    continue

                curr_clean = curr_item.strip()

                # Check ordinal suffix
                if curr_clean in ORDINALS_SPECIAL or (
                    (curr_clean.endswith("م") or curr_clean.endswith("مین"))
                    and curr_clean.rstrip("م").rstrip("ین") in ALL_NUM_WORDS
                ):
                    has_ordinal = True

                if (
                    curr_clean in ALL_NUM_WORDS
                    or curr_clean == "و"
                    or curr_clean in {"ممیز", "اشاره"}
                    or curr_clean in ORDINALS_SPECIAL
                    or curr_clean.rstrip("م").rstrip("ین") in ALL_NUM_WORDS
                ):
                    num_tokens.append(curr_clean)
                    num_tokens_raw_indices.append(j)
                    j += 1
                else:
                    break

            # If trailing token is 'و', rewind it
            while num_tokens and num_tokens[-1] == "و":
                num_tokens.pop()
                num_tokens_raw_indices.pop()

            parsed = parse_number_tokens(num_tokens) if num_tokens else None

            if parsed is not None:
                # Format number
                if isinstance(parsed, float) and parsed.is_integer():
                    parsed = int(parsed)

                if is_negative:
                    parsed = -parsed

                num_str = str(parsed)
                if has_ordinal:
                    num_str += "م"

                if self.to_persian_digits:
                    num_str = num_str.translate(LATIN_TO_PERSIAN)

                result.append(num_str)
                # advance i past the matched span
                i = (num_tokens_raw_indices[-1] + 1) if num_tokens_raw_indices else j
            else:
                result.append(token)
                i += 1

        return "".join(result)


_GLOBAL_ITN = PersianITN(to_persian_digits=False)


def inverse_normalize_text(text: str, to_persian_digits: bool = False) -> str:
    """Convenience helper for Persian ITN."""
    if to_persian_digits:
        return PersianITN(to_persian_digits=True).normalize(text)
    return _GLOBAL_ITN.normalize(text)

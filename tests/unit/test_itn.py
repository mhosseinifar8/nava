"""Unit tests for Persian Inverse Text Normalization (ITN)."""
from __future__ import annotations

import pytest

from nava.itn import PersianITN, inverse_normalize_text


class TestPersianITN:
    def test_units_and_tens(self):
        assert inverse_normalize_text("صفر") == "0"
        assert inverse_normalize_text("یک") == "1"
        assert inverse_normalize_text("نه") == "9"
        assert inverse_normalize_text("ده") == "10"
        assert inverse_normalize_text("یازده") == "11"
        assert inverse_normalize_text("بیست") == "20"
        assert inverse_normalize_text("سی") == "30"

    def test_compound_numbers(self):
        assert inverse_normalize_text("هشتصد و پنجاه") == "850"
        assert inverse_normalize_text("دویست و سی و چهار") == "234"
        assert inverse_normalize_text("هزار و نهصد و نود و نه") == "1999"
        assert inverse_normalize_text("دو میلیون و پانصد هزار") == "2500000"

    def test_ordinals(self):
        assert inverse_normalize_text("اول") == "1م"
        assert inverse_normalize_text("دوم") == "2م"
        assert inverse_normalize_text("سوم") == "3م"
        assert inverse_normalize_text("چهارم") == "4م"
        assert inverse_normalize_text("بیستم") == "20م"

    def test_negative_and_decimals(self):
        assert inverse_normalize_text("منفی ده") == "-10"
        assert inverse_normalize_text("منفی صد و پنجاه") == "-150"
        assert inverse_normalize_text("سه ممیز پنج") == "3.5"
        assert inverse_normalize_text("نیم") == "0.5"

    def test_in_context_sentences(self):
        text = "قیمت این محصول هشتصد و پنجاه تومان است."
        expected = "قیمت این محصول 850 تومان است."
        assert inverse_normalize_text(text) == expected

        text2 = "من در طبقه سوم یا چهارم زندگی می‌کنم."
        expected2 = "من در طبقه 3م یا 4م زندگی می‌کنم."
        assert inverse_normalize_text(text2) == expected2

    def test_persian_digits_option(self):
        assert inverse_normalize_text("هشتصد و پنجاه", to_persian_digits=True) == "۸۵۰"
        assert inverse_normalize_text("بیست و پنج", to_persian_digits=True) == "۲۵"


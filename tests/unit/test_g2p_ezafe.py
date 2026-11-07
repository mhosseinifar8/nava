"""Unit tests for Persian G2P, Text Normalization, Custom Dictionary, and Ezafe."""
from __future__ import annotations

import pytest

from nava.g2p import (CustomDictionaryManager, PersianEzafeProcessor,
                      apply_custom_dictionary, insert_ezafe_connector,
                      normalize_dates, normalize_numbers,
                      normalize_percentages, normalize_persian_text,
                      normalize_times, number_to_words, preprocess_tts_text,
                      rule_based_ezafe_predict)


class TestTextNormalization:
    def test_number_to_words(self):
        assert number_to_words(0) == "صفر"
        assert number_to_words(850) == "هشتصد و پنجاه"
        assert number_to_words(1403) == "هزار و چهارصد و سه"
        assert number_to_words(2500000) == "دو میلیون و پانصد هزار"

    def test_dates_and_times(self):
        text = "در تاریخ 1403/08/18 جلسه برگزار شد."
        norm = normalize_dates(text)
        assert "هجدهم آبان هزار و چهارصد و سه" in norm

        time_text = "ساعت 14:30 است."
        norm_time = normalize_times(time_text)
        assert "چهارده و نیم" in norm_time

    def test_percentages_and_currencies(self):
        text = "رشد ۲۵٪ حاصل شد."
        norm = normalize_percentages(text)
        assert "بیست و پنج درصد" in norm

        curr = "مبلغ ۵۰۰۰ تومان پرداخت شد."
        norm_curr = normalize_persian_text(curr)
        assert "پنج هزار تومان" in norm_curr

    def test_arabic_folding(self):
        raw = "يك كتاب زيبا در مكّه"
        norm = normalize_persian_text(raw, expand_numbers=False)
        assert "یک" in norm
        assert "کتاب" in norm
        assert "مکه" in norm


class TestCustomDictionary:
    def test_lexicon_replacement(self):
        mgr = CustomDictionaryManager()
        text = "من از اسنپفود سفارش دادم."
        replaced = mgr.apply(text, {"اسنپفود": "اسنپ‌فود"})
        assert replaced == "من از اسنپ‌فود سفارش دادم."

    def test_phoneme_or_eraab_override(self):
        text = "او یک مرد شجاع است."
        replaced = apply_custom_dictionary(text, {"مرد": "مَرد"})
        assert "مَرد" in replaced


class TestEzafeInsertion:
    def test_connector_types(self):
        # Consonant ending receives Kasreh (ِ)
        assert insert_ezafe_connector("کتاب") == "کتابِ"
        # Vowel ending 'ه' receives "ه‌ی"
        assert insert_ezafe_connector("خانه") == "خانه‌ی"
        # Vowel ending 'ا' receives "ی"
        assert insert_ezafe_connector("آشنا") == "آشنای"

    def test_rule_based_prediction(self):
        words = ["کتاب", "خوب"]
        preds = rule_based_ezafe_predict(words)
        assert preds == [True, False]

    def test_pipeline_integration(self):
        text = "کتاب خوب و خانه بزرگ"
        result = preprocess_tts_text(text, enable_ezafe=True)
        assert "کتابِ خوب" in result
        assert "خانه‌ی بزرگ" in result


"""Persian text pipeline: normalisation, chunking, token budgets, lexicon."""
from __future__ import annotations

import pytest


class TestNormalize:
    def test_folds_arabic_chars_to_persian(self, textpipe):
        assert textpipe.normalize("كتابي") == "کتابی"

    def test_zwnj_becomes_space(self, textpipe):
        assert textpipe.normalize("می\u200cروم") == "می روم"

    def test_strips_unsupported_symbols_keeps_persian_punctuation(self, textpipe):
        assert textpipe.normalize("«سلام» دنیا؛ خوبی؟") == "سلام دنیا؛ خوبی؟"

    def test_collapses_whitespace(self, textpipe):
        assert textpipe.normalize("سلام   \n\t دنیا") == "سلام دنیا"

    def test_keeps_word_chars_and_ascii(self, textpipe):
        assert textpipe.normalize("Nava version 2 (beta)") == "Nava version 2 (beta)"


class TestSplitSentences:
    def test_splits_on_persian_punctuation(self, textpipe):
        assert textpipe.split_sentences("سلام. دنیا؟ بریم، هست") == ["سلام.", "دنیا؟", "بریم،", "هست"]

    def test_no_punctuation_is_one_sentence(self, textpipe):
        assert textpipe.split_sentences("یک جمله پیوسته") == ["یک جمله پیوسته"]

    def test_empty(self, textpipe):
        assert textpipe.split_sentences("") == []


class TestChunkPocket:
    def test_chunks_are_at_most_max_words(self, textpipe):
        text = " ".join(f"w{i}" for i in range(30))
        chunks = textpipe.chunk_pocket(text, max_words=12)
        assert [len(c.split()) for c in chunks] == [12, 12, 6]

    def test_chunking_is_per_sentence(self, textpipe):
        chunks = textpipe.chunk_pocket("یک دو سه. چهار پنج شش.", max_words=2)
        assert chunks == ["یک دو", "سه.", "چهار پنج", "شش."]

    def test_short_text_single_chunk(self, textpipe):
        assert textpipe.chunk_pocket("سلام دنیا") == ["سلام دنیا"]


class TestFitTokenBudget:
    def test_already_within_budget_untouched(self, textpipe):
        counter = lambda s: len(s.split()) * 2  # noqa: E731
        assert textpipe.fit_token_budget("a b c", 6, counter) == ["a b c"]

    def test_splits_into_budget_parts(self, textpipe):
        counter = lambda s: len(s.split()) * 2  # noqa: E731
        parts = textpipe.fit_token_budget("a b c d e", 6, counter)
        assert parts == ["a b c", "d e"]
        assert all(counter(p) <= 6 for p in parts)

    def test_single_overlong_word_passes_through(self, textpipe):
        counter = lambda s: len(s)  # noqa: E731
        assert textpipe.fit_token_budget("abcdefgh", 4, counter) == ["abcdefgh"]

    def test_empty(self, textpipe):
        assert textpipe.fit_token_budget("", 16, lambda s: 0) == []

    def test_pocket_budget_end_to_end(self, textpipe):
        # every produced synthesis piece fits the 16-token pocket budget
        counter = lambda s: len(s.split()) * 3  # noqa: E731
        text = " ".join(f"کلمه{i}" for i in range(40))
        pieces = [p for chunk in textpipe.chunk_pocket(text)
                  for p in textpipe.fit_token_budget(chunk, 16, counter)]
        assert pieces and all(counter(p) <= 16 for p in pieces)
        assert " ".join(pieces).split() == text.split()


class TestLexicon:
    def test_surface_longest_match_first(self, textpipe):
        lexicon = {"surface": {"foo": "X", "foo bar": "Y"}}
        assert textpipe.apply_surface_lexicon("say foo bar now", lexicon) == "say Y now"

    def test_surface_multiple_and_overlapping(self, textpipe):
        lexicon = {"surface": {"می خوام": "می خواهم", "خوام": "خواهم"}}
        assert textpipe.apply_surface_lexicon("من می خوام بروم", lexicon) == "من می خواهم بروم"

    def test_empty_lexicon_is_noop(self, textpipe):
        assert textpipe.apply_surface_lexicon("سلام", {}) == "سلام"

    def test_phoneme_overrides_tokenwise(self, textpipe):
        lexicon = {"phonemes": {"@": "?"}}
        assert textpipe.apply_phoneme_overrides("s @ l A m", lexicon) == "s ? l A m"

    def test_load_lexicon_missing_file_is_empty(self, textpipe, tmp_path):
        assert textpipe.load_lexicon(tmp_path / "nope.json") == {}

    def test_load_lexicon_valid(self, textpipe, tmp_path):
        path = tmp_path / "lex.json"
        path.write_text('{"surface": {"a": "b"}, "phonemes": {"c": "d"}}', encoding="utf-8")
        assert textpipe.load_lexicon(path) == {"surface": {"a": "b"}, "phonemes": {"c": "d"}}

    def test_load_lexicon_rejects_bad_section(self, textpipe, tmp_path):
        path = tmp_path / "lex.json"
        path.write_text('{"surface": [1, 2]}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="surface"):
            textpipe.load_lexicon(path)

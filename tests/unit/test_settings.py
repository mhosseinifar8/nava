"""Service settings: env precedence, fallbacks and validation."""
from __future__ import annotations

import pytest

STT_ENV = ["NAVA_STT_WORKERS", "NAVA_STT_OMP_THREADS", "NAVA_INFERENCE_WORKERS",
           "OMP_NUM_THREADS", "NAVA_WHISPER_COMPUTE_TYPE", "NAVA_WHISPER_MODEL_DIR",
           "NAVA_MAX_STT_AUDIO_BYTES"]
TTS_ENV = ["NAVA_TTS_WORKERS", "NAVA_TTS_OMP_THREADS", "NAVA_INFERENCE_WORKERS",
           "OMP_NUM_THREADS", "NAVA_MODELS_DIR", "NAVA_MAX_TTS_TEXT_CHARS", "NAVA_LEXICON_PATH"]


def test_stt_explicit_values(monkeypatch, stt_settings):
    monkeypatch.setenv("NAVA_STT_WORKERS", "3")
    monkeypatch.setenv("NAVA_STT_OMP_THREADS", "5")
    monkeypatch.setenv("NAVA_WHISPER_COMPUTE_TYPE", "int8_float32")
    s = stt_settings.Settings.load()
    assert (s.workers, s.omp_threads, s.compute_type) == (3, 5, "int8_float32")


def test_stt_falls_back_to_infrastructure_vars(monkeypatch, stt_settings):
    for name in STT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NAVA_INFERENCE_WORKERS", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    s = stt_settings.Settings.load()
    assert (s.workers, s.omp_threads) == (7, 3)


def test_stt_defaults(monkeypatch, stt_settings):
    for name in STT_ENV:
        monkeypatch.delenv(name, raising=False)
    s = stt_settings.Settings.load()
    assert s.compute_type == "int8"
    assert s.workers == 2
    assert s.omp_threads == 4
    assert s.max_audio_bytes == 16_000 * 2 * 60 * 15  # 15 minutes of s16le 16 kHz


def test_stt_rejects_non_integer(monkeypatch, stt_settings):
    monkeypatch.setenv("NAVA_STT_WORKERS", "two")
    with pytest.raises(RuntimeError, match="NAVA_STT_WORKERS"):
        stt_settings.Settings.load()


def test_stt_clamps_to_minimum_one(monkeypatch, stt_settings):
    monkeypatch.setenv("NAVA_STT_WORKERS", "0")
    assert stt_settings.Settings.load().workers == 1


def test_tts_explicit_values(monkeypatch, tts_settings):
    monkeypatch.setenv("NAVA_MODELS_DIR", "/tmp/models-x")
    monkeypatch.setenv("NAVA_TTS_WORKERS", "4")
    monkeypatch.setenv("NAVA_TTS_OMP_THREADS", "6")
    monkeypatch.setenv("NAVA_MAX_TTS_TEXT_CHARS", "500")
    s = tts_settings.Settings.load()
    assert (s.workers, s.omp_threads, s.max_text_chars) == (4, 6, 500)
    assert str(s.models_dir) == "/tmp/models-x"
    assert s.lexicon_path == s.models_dir / "lexicon.json"


def test_tts_lexicon_path_override(monkeypatch, tts_settings):
    monkeypatch.setenv("NAVA_LEXICON_PATH", "/tmp/lex/custom.json")
    assert str(tts_settings.Settings.load().lexicon_path) == "/tmp/lex/custom.json"


def test_tts_defaults(monkeypatch, tts_settings):
    for name in TTS_ENV:
        monkeypatch.delenv(name, raising=False)
    s = tts_settings.Settings.load()
    assert (s.workers, s.omp_threads, s.max_text_chars) == (2, 4, 20000)

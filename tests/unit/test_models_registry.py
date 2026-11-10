"""Unit tests for Model Registry and Profiles."""
from __future__ import annotations

import pytest

from nava.models import (MODELS, get_model, get_profile_models, list_models)


class TestModelRegistry:
    def test_all_expected_models_registered(self):
        expected_models = [
            "shenava-koochik-ctc",
            "shenava-koochik-v1.5-rnnt",
            "shenava-115m-flagship",
            "shenava-rizeh-streaming-32m",
            "shenava-rizeh-pizeh-6.9m",
            "nemotron-streaming-farsi",
            "whisper-turbo-persian-sherpa",
            "whisper-turbo-persian-givia",
            "whisper-turbo-persian-gholizadeh",
            "vosk-fa-light",
            "whisper-large-v3",
            "persian-ezafe-albert",
            "homo-ge2pe-persian",
            "homo-t5",
            "piper-mana-lca",
            "piper-voices-fa",
            "ava-82m",
            "matcha-khadijah-fa-en",
            "matcha-musa-fa-en",
            "pocket-tts-farsi",
            "nava-fa-clone",
        ]
        for mid in expected_models:
            assert mid in MODELS, f"Model {mid} missing from registry"

    def test_aliases(self):
        assert get_model("pocket").id == "pocket-tts-farsi"
        assert get_model("pocket-v2").id == "pocket-tts-farsi-v2"
        assert get_model("mana").id == "piper-mana-lca"
        assert get_model("piper").id == "piper-voices-fa"
        assert get_model("shenava").id == "shenava-koochik-ctc"
        assert get_model("whisper").id == "whisper-turbo-persian-sherpa"
        assert get_model("parakeet").id == "parakeet-ctc-109m-farsi"
        assert get_model("fastconformer").id == "fastconformer-hybrid-large"
        assert get_model("chatterbox").id == "chatterbox-tts-persian-gguf"
        assert get_model("sentiment").id == "snappfood-sentiment-analysis"
        assert get_model("ocr").id == "olmocr-farsi-lora"

    def test_suite_models_registered(self):
        suite = [
            "pocket-tts-farsi-v2",
            "pocket-tts-farsi",
            "chatterbox-tts-persian-gguf",
            "chatterbox-tts-persian-mazrba",
            "chatterbox-tts-persian-torch",
            "nemotron-asr-streaming-farsi",
            "nemotron-streaming-podcast",
            "parakeet-ctc-109m-farsi",
            "fastconformer-hybrid-large",
            "whisper-large-v3-farsi-cv17",
            "snappfood-sentiment-analysis",
            "olmocr-farsi-lora",
        ]
        for mid in suite:
            assert mid in MODELS, f"Suite model {mid} missing from registry"
            model = get_model(mid)
            assert model.repo_id != ""

    def test_profiles(self):
        minimal = get_profile_models("minimal")
        assert any(m.id == "shenava-koochik-ctc" for m in minimal)
        assert any(m.id == "piper-mana-lca" for m in minimal)
        assert any(m.id == "persian-ezafe-albert" for m in minimal)

        realtime = get_profile_models("realtime")
        assert any(m.id == "shenava-rizeh-streaming-32m" for m in realtime)
        assert any(m.id == "nemotron-streaming-farsi" for m in realtime)

        high_acc = get_profile_models("high-accuracy")
        assert any(m.id == "shenava-koochik-v1.5-rnnt" for m in high_acc)
        assert any(m.id == "ava-82m" for m in high_acc)

    def test_list_models_returns_dicts(self):
        models = list_models()
        assert len(models) >= 21
        for m in models:
            assert "id" in m
            assert "type" in m
            assert "installed" in m
            assert "features" in m


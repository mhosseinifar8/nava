"""Shared fixtures for Nava tests.

Unit tests run anywhere (no heavy deps, modules loaded by file path).
Integration tests are skipped automatically when the services are not running.
"""
from __future__ import annotations

import importlib.util
import math
import os
import socket
import struct
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
STT_URL = os.getenv("NAVA_TEST_STT_URL", "http://127.0.0.1:8001")
TTS_URL = os.getenv("NAVA_TEST_TTS_URL", "http://127.0.0.1:8002")

# make the nava SDK importable without installing it
sys.path.insert(0, str(ROOT / "nava_sdk"))
sys.path.insert(0, str(ROOT))


def _load_module(name: str, relpath: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def textpipe() -> ModuleType:
    return _load_module("nava_textpipe", "nava-tts/app/textpipe.py")


@pytest.fixture(scope="session")
def bodylimit() -> ModuleType:
    return _load_module("nava_bodylimit", "nava-stt/app/bodylimit.py")


@pytest.fixture(scope="session")
def stt_settings() -> ModuleType:
    return _load_module("nava_stt_settings", "nava-stt/app/settings.py")


@pytest.fixture(scope="session")
def tts_settings() -> ModuleType:
    return _load_module("nava_tts_settings", "nava-tts/app/settings.py")


def make_pcm(seconds: float = 1.0, freq: float = 440.0, rate: int = 16000) -> bytes:
    """s16le mono PCM tone — valid STT input shape (VAD may hear no speech)."""
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))


def _alive(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=0.5):
            return True
    except OSError:
        return False


STT_UP = _alive(STT_URL)
TTS_UP = _alive(TTS_URL)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "stt_service" in item.keywords and not STT_UP:
            item.add_marker(pytest.mark.skip(reason=f"nava-stt not reachable at {STT_URL}"))
        if "tts_service" in item.keywords and not TTS_UP:
            item.add_marker(pytest.mark.skip(reason=f"nava-tts not reachable at {TTS_URL}"))

"""Unit tests for Voice Cloning and Reference Audio Management."""
from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import pytest

from nava.voices import VoiceProfileManager


def make_test_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n = int(duration_s * sample_rate)
        frames = b"".join(
            struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440.0 * i / sample_rate)))
            for i in range(n)
        )
        wf.writeframes(frames)
    return buf.getvalue()


class TestVoiceCloning:
    def test_voice_cloning_profile_creation(self, tmp_path: Path):
        mgr = VoiceProfileManager(voices_dir=tmp_path)
        wav_bytes = make_test_wav(duration_s=2.5)

        voice_id, dur = mgr.create_voice_profile(wav_bytes, name="parsa")
        assert "parsa" in voice_id
        assert 2.4 <= dur <= 2.6

        # Check saved file exists
        saved_file = mgr.get_voice_path(voice_id)
        assert saved_file is not None
        assert saved_file.exists()

        # Check sample rate is normalized to 24000
        with wave.open(str(saved_file), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1

    def test_too_short_audio_rejected(self, tmp_path: Path):
        mgr = VoiceProfileManager(voices_dir=tmp_path)
        wav_bytes = make_test_wav(duration_s=0.2)
        with pytest.raises(ValueError, match="too short"):
            mgr.create_voice_profile(wav_bytes)


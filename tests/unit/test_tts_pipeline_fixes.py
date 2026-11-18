"""Unit tests verifying fixes for non-Piper TTS beep/fallback, PCM/WAV pipeline,
Pocket config localization, Matcha espeak-ng-data discovery, Chatterbox CrispASR
execution, and REST/WebSocket chunk buffering.
"""
from __future__ import annotations

import io
import os
import stat
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from nava.engines.base import (
    ModelNotInstalledError,
    float32_to_pcm16,
    pcm16_to_wav_bytes,
    resample_pcm16,
    wav_bytes_to_pcm16,
)
from nava.engines.tts_ava import AvaEngine
from nava.engines.tts_chatterbox import ChatterboxGGUFEngine, ChatterboxPyTorchEngine
from nava.engines.tts_matcha import MatchaEngine
from nava.engines.tts_piper import PiperEngine
from nava.engines.tts_pocket import PocketEngine, localize_pocket_config
from nava.g2p.lexicon import find_espeak_ng_data
from nava.server.app import app

# Import the audio analyzer utility
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_tts_audio import analyze_tts_audio, load_audio_samples  # noqa: E402


def _make_speech_like_signal(sample_rate: int = 24000, duration_s: float = 1.0) -> np.ndarray:
    """Create a multi-harmonic, amplitude-modulated waveform (mimics speech spectrum & syllables)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    # Syllable-like amplitude envelope (3-4 Hz modulation with pauses)
    envelope = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t) * np.cos(2 * np.pi * 1.1 * t), 0.0, 1.0)
    envelope[int(0.45 * sample_rate) : int(0.55 * sample_rate)] = 0.0  # brief word pause
    # Rich harmonic formants + mild fricative noise
    rng = np.random.default_rng(42)
    signal = (
        0.35 * np.sin(2 * np.pi * 140.0 * t)
        + 0.25 * np.sin(2 * np.pi * 420.0 * t)
        + 0.20 * np.sin(2 * np.pi * 980.0 * t)
        + 0.15 * np.sin(2 * np.pi * 2150.0 * t)
        + 0.05 * rng.standard_normal(t.size).astype(np.float32)
    )
    return (signal * envelope * 0.7).astype(np.float32)


class TestNoSyntheticBeepInProduction:
    """Production default (NAVA_ALLOW_DUMMY_AUDIO=0) must NEVER emit a synthetic beep."""

    @pytest.mark.parametrize(
        "engine_cls,model_id",
        [
            (ChatterboxGGUFEngine, "chatterbox-tts-persian-gguf"),
            (ChatterboxPyTorchEngine, "chatterbox-tts-persian-torch"),
            (PocketEngine, "pocket-tts-farsi"),
            (MatchaEngine, "matcha-khadijah-fa-en"),
            (MatchaEngine, "matcha-musa-fa-en"),
            (AvaEngine, "ava-82m"),
            (PiperEngine, "piper-mana-lca"),
        ],
    )
    def test_missing_model_raises_error_never_beep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_cls, model_id: str
    ):
        monkeypatch.delenv("NAVA_ALLOW_DUMMY_AUDIO", raising=False)
        engine = engine_cls(model_id, tmp_path / model_id)
        with pytest.raises((ModelNotInstalledError, RuntimeError)):
            engine.synthesize("سلام تست صدا")

    def test_rest_endpoint_returns_503_not_beep_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NAVA_ALLOW_DUMMY_AUDIO", raising=False)
        client = TestClient(app)
        for model_id in (
            "chatterbox-tts-persian-gguf",
            "pocket-tts-farsi",
            "matcha-khadijah-fa-en",
            "ava-82m",
        ):
            r = client.post(
                "/v1/audio/speech",
                json={"model": model_id, "input": "سلام تست", "response_format": "wav"},
            )
            assert r.status_code == 503, f"Expected 503 for uninstalled {model_id}, got {r.status_code}"

    def test_websocket_returns_error_frame_not_beep_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NAVA_ALLOW_DUMMY_AUDIO", raising=False)
        client = TestClient(app)
        with client.websocket_connect("/v1/ws/tts") as ws:
            ws.send_json({"model": "chatterbox-tts-persian-gguf", "text": "سلام تست"})
            msg = ws.receive_json()
            assert msg["status"] == "error"
            assert msg.get("error")

    def test_dummy_mode_emits_clean_silence_never_sine_beep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("NAVA_ALLOW_DUMMY_AUDIO", "1")
        for engine in (
            ChatterboxGGUFEngine("chatterbox-tts-persian-gguf", tmp_path / "cb"),
            PocketEngine("pocket-tts-farsi", tmp_path / "pk"),
            MatchaEngine("matcha-khadijah-fa-en", tmp_path / "ma"),
            AvaEngine("ava-82m", tmp_path / "av"),
            PiperEngine("piper-mana-lca", tmp_path / "pi"),
        ):
            res = engine.synthesize("سلام دنیا")
            samples = np.frombuffer(res.audio, dtype="<i2")
            assert len(samples) > 0
            # Must be clean silence (all zeros), NEVER a 380Hz/440Hz beep
            assert int(np.max(np.abs(samples))) == 0


class TestAudioConversionPipeline:
    """Verify float32 -> int16 PCM conversion, clipping, mono downmix, and WAV wrapping."""

    def test_float32_clipping_and_nan_safety(self):
        raw = np.array([-2.5, -1.0, 0.0, 0.5, 1.0, 3.0, np.nan, np.inf, -np.inf], dtype=np.float32)
        pcm = float32_to_pcm16(raw)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert ints.tolist() == [-32767, -32767, 0, 16384, 32767, 32767, 0, 32767, -32767]

    def test_stereo_downmix_to_mono(self):
        # Shape (2, 4) channels-first stereo
        stereo = np.array([[0.5, -0.5, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        pcm = float32_to_pcm16(stereo)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert len(ints) == 4
        assert abs(int(ints[0]) - 16384) <= 1
        assert int(ints[1]) == 0
        assert abs(int(ints[2]) - 16384) <= 1

    def test_int16_passthrough_no_double_scaling(self):
        orig = np.array([-1000, 0, 12345, -20000], dtype="<i2")
        pcm = float32_to_pcm16(orig)
        assert np.frombuffer(pcm, dtype="<i2").tolist() == orig.tolist()

    def test_resample_22050_to_24000(self):
        src_sr, dst_sr = 22050, 24000
        pcm_in = float32_to_pcm16(_make_speech_like_signal(src_sr, 0.5))
        pcm_out = resample_pcm16(pcm_in, src_sr, dst_sr)
        out_samples = len(pcm_out) // 2
        assert abs(out_samples - int(0.5 * dst_sr)) <= 2

    def test_wav_roundtrip_header(self):
        sig = _make_speech_like_signal(24000, 0.25)
        pcm = float32_to_pcm16(sig)
        wav_data = pcm16_to_wav_bytes(pcm, 24000)
        assert wav_data.startswith(b"RIFF")
        pcm2, sr = wav_bytes_to_pcm16(wav_data)
        assert sr == 24000
        assert pcm2 == pcm


class TestPocketFixes:
    """Verify Pocket-TTS hf:// localization, tokenizer aliasing, and example_voice discovery."""

    def test_localize_pocket_config_resolves_tokenizer_alias(self, tmp_path: Path):
        pocket_dir = tmp_path / "pocket-tts-farsi"
        pocket_dir.mkdir()
        (pocket_dir / "model.safetensors").write_bytes(b"weights")
        # Registry saves tokenizer.model as tokenizer_ph.model
        (pocket_dir / "tokenizer_ph.model").write_bytes(b"spm")
        cfg = pocket_dir / "model.yaml"
        cfg.write_text(
            "weights_path: hf://mehdi-hf/pocket-tts-farsi/model.safetensors@abc123\n"
            "flow_lm:\n"
            "  tokenizer_path: hf://mehdi-hf/pocket-tts-farsi/tokenizer.model@abc123\n",
            encoding="utf-8",
        )

        localized = localize_pocket_config(cfg, pocket_dir)
        patched_text = localized.read_text(encoding="utf-8")
        assert "hf://" not in patched_text
        assert str(pocket_dir / "model.safetensors") in patched_text
        assert str(pocket_dir / "tokenizer_ph.model") in patched_text

    def test_pocket_finds_example_voice_in_model_dir(self, tmp_path: Path):
        pocket_dir = tmp_path / "pocket-tts-farsi"
        pocket_dir.mkdir()
        example_wav = pocket_dir / "example_voice.wav"
        example_wav.write_bytes(pcm16_to_wav_bytes(b"\x00\x00" * 24000, 24000))

        engine = PocketEngine("pocket-tts-farsi", pocket_dir)
        found = engine._find_voice_wav("default")
        assert found == example_wav

    def test_nava_fa_clone_reuses_sibling_pocket_example_voice(self, tmp_path: Path):
        # nava-fa-clone has no WAV of its own, but sibling pocket-tts-farsi has example_voice.wav
        sibling = tmp_path / "pocket-tts-farsi"
        sibling.mkdir()
        example_wav = sibling / "example_voice.wav"
        example_wav.write_bytes(pcm16_to_wav_bytes(b"\x00\x00" * 24000, 24000))

        clone_dir = tmp_path / "nava-fa-clone"
        clone_dir.mkdir()
        engine = PocketEngine("nava-fa-clone", clone_dir)
        found = engine._find_voice_wav("default")
        assert found == example_wav

    def test_pocket_auto_bootstraps_default_voice_via_piper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from nava.engines.base import TTSResult

        clone_dir = tmp_path / "nava-fa-clone"
        clone_dir.mkdir()
        piper_dir = tmp_path / "piper-mana-lca"
        piper_dir.mkdir()
        (piper_dir / "fa_IR-mana-medium.onnx").write_bytes(b"onnx")
        (piper_dir / "fa_IR-mana-medium.onnx.json").write_text("{}", encoding="utf-8")

        speech_pcm = float32_to_pcm16(_make_speech_like_signal(22050, 1.5))
        monkeypatch.setattr(
            "nava.engines.tts_piper.PiperEngine.synthesize",
            lambda self, text, **kw: TTSResult(audio=speech_pcm, sample_rate=22050, engine="piper", voice="mana"),
        )

        engine = PocketEngine("nava-fa-clone", clone_dir)
        bootstrapped = engine._bootstrap_default_voice_wav()
        assert bootstrapped is not None and bootstrapped.is_file()
        pcm, sr = wav_bytes_to_pcm16(bootstrapped.read_bytes())
        assert sr == 24000
        assert len(pcm) > 24000 * 2  # > 1 second of 24kHz audio


class TestMatchaFixes:
    """Verify Matcha flexible file resolution and espeak-ng-data discovery."""

    def test_find_espeak_ng_data_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        espeak_dir = tmp_path / "espeak-ng-data"
        espeak_dir.mkdir()
        (espeak_dir / "phontab").write_bytes(b"data")
        monkeypatch.setenv("ESPEAK_NG_DATA", str(espeak_dir))
        assert find_espeak_ng_data() == espeak_dir

    def test_matcha_resolves_original_hf_filenames(self, tmp_path: Path):
        m_dir = tmp_path / "matcha-khadijah-fa-en"
        m_dir.mkdir()
        (m_dir / "matcha-fa_en-khadijah-22050-3.onnx").write_bytes(b"onnx")
        (m_dir / "hifigan_universal_22050hz_v1.onnx").write_bytes(b"voc")
        (m_dir / "tokens_sherpa_with_fa.txt").write_text("a 0\nb 1\n", encoding="utf-8")

        engine = MatchaEngine("matcha-khadijah-fa-en", m_dir)
        acoustic, vocoder, tokens = engine._resolve_model_files()
        assert acoustic.name == "matcha-fa_en-khadijah-22050-3.onnx"
        assert vocoder.name == "hifigan_universal_22050hz_v1.onnx"
        assert tokens.name == "tokens_sherpa_with_fa.txt"


class TestChatterboxCrispASRAndStreamingBuffering:
    """Verify Chatterbox GGUF CrispASR invocation and non-repeating streaming chunks."""

    def test_chatterbox_gguf_missing_crispasr_reports_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("NAVA_ALLOW_DUMMY_AUDIO", raising=False)
        monkeypatch.setattr("nava.engines.tts_chatterbox._find_crispasr_binary", lambda: None)

        cb_dir = tmp_path / "chatterbox-tts-persian-gguf"
        cb_dir.mkdir()
        (cb_dir / "chatterbox-t3-fa-f16.gguf").write_bytes(b"gguf-t3")
        (cb_dir / "chatterbox-s3gen-q8_0.gguf").write_bytes(b"gguf-s3gen")

        engine = ChatterboxGGUFEngine("chatterbox-tts-persian-gguf", cb_dir)
        with pytest.raises(RuntimeError, match="CrispASR"):
            engine.synthesize("سلام تست")

    @pytest.mark.asyncio
    async def test_chatterbox_gguf_crispasr_produces_real_non_tone_speech_and_distinct_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Simulate a real CrispASR binary producing distinct speech WAVs per clause."""
        cb_dir = tmp_path / "chatterbox-tts-persian-gguf"
        cb_dir.mkdir()
        (cb_dir / "chatterbox-t3-fa-f16.gguf").write_bytes(b"gguf-t3")
        (cb_dir / "chatterbox-s3gen-q8_0.gguf").write_bytes(b"gguf-s3gen")

        # Create a mock crispasr executable that writes speech-like audio unique to the --tts text
        fake_bin = tmp_path / "fake_crispasr.py"
        fake_bin.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, sys, wave\n"
            "import numpy as np\n"
            "args = sys.argv[1:]\n"
            "text = args[args.index('--tts') + 1]\n"
            "out = args[args.index('--tts-output') + 1]\n"
            "sr = 24000\n"
            "seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)\n"
            "rng = np.random.default_rng(seed)\n"
            "t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)\n"
            "env = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t), 0.0, 1.0)\n"
            "env[int(0.25 * sr):int(0.32 * sr)] = 0.0\n"
            "f0 = 130.0 + (seed % 50)\n"
            "sig = (0.35*np.sin(2*np.pi*f0*t) + 0.25*np.sin(2*np.pi*3*f0*t) + "
            "0.2*np.sin(2*np.pi*7*f0*t) + 0.08*rng.standard_normal(t.size)) * env * 0.7\n"
            "pcm = np.round(np.clip(sig, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()\n"
            "with wave.open(out, 'wb') as wf:\n"
            "    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm)\n",
            encoding="utf-8",
        )
        fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("NAVA_CRISPASR_BIN", str(fake_bin))

        engine = ChatterboxGGUFEngine("chatterbox-tts-persian-gguf", cb_dir)
        chunks = []
        async for chunk in engine.synthesize_stream("سلام دنیا. حال شما چطور است؟"):
            chunks.append(chunk)

        # Expect: clause1, silence_gap, clause2 -> chunks 0 and 2 must be distinct (no repeated stale chunk!)
        assert len(chunks) >= 3
        assert chunks[0] != chunks[2]

        # Verify combined output passes audio speech analysis (NOT a pure sine tone)
        wav_path = tmp_path / "chatterbox_test.wav"
        wav_path.write_bytes(pcm16_to_wav_bytes(b"".join(chunks), engine.sample_rate))
        samples, sr, ch = load_audio_samples(wav_path)
        assert sr == 24000
        assert ch == 1
        report = analyze_tts_audio(samples, sr)
        assert report["verdict"] == "PASS_SPEECH_SIGNAL"
        assert report["is_real_speech"] is True
        assert report["is_pure_tone"] is False


class TestAudioAnalyzerUtility:
    """Verify verify_tts_audio.py accurately catches continuous sine wave beeps."""

    def test_detects_440hz_beep(self):
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        beep = 0.25 * np.sin(2 * np.pi * 380.0 * t)
        report = analyze_tts_audio(beep, sr)
        assert report["verdict"] == "FAIL_CONTINUOUS_TONE_BEEP"
        assert report["is_pure_tone"] is True
        assert report["is_real_speech"] is False
        assert abs(report["dominant_hz"] - 380.0) < 5.0

    def test_detects_silence(self):
        silence = np.zeros(24000, dtype=np.float32)
        report = analyze_tts_audio(silence, 24000)
        assert report["verdict"] == "FAIL_SILENCE"
        assert report["is_silence"] is True

    def test_passes_speech_like_signal(self):
        speech = _make_speech_like_signal(24000, 1.0)
        report = analyze_tts_audio(speech, 24000)
        assert report["verdict"] == "PASS_SPEECH_SIGNAL"
        assert report["is_real_speech"] is True
        assert report["is_pure_tone"] is False


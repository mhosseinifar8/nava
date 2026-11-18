#!/usr/bin/env python3
"""Verify that synthesized TTS audio (.wav or raw s16le .pcm) contains real speech
rather than silence or a continuous synthetic sine-wave beep.

Usage:
    python3 scripts/verify_tts_audio.py chatterbox_test.wav
    python3 scripts/verify_tts_audio.py chatterbox_test.pcm --sample-rate 24000
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np


def load_audio_samples(path: Path, sample_rate: int = 24000) -> tuple[np.ndarray, int, int]:
    """Load WAV or raw s16le mono PCM as float32 samples in [-1, 1].

    Returns (samples_float32, sample_rate, channels).
    """
    raw = path.read_bytes()
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV (sampwidth=2), got {sampwidth}")
        arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return arr, int(sr), int(channels)

    # Raw s16le mono PCM
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr, int(sample_rate), 1


def analyze_tts_audio(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Analyze waveform statistics to detect silence, clipping, or pure sine-wave beeps."""
    if samples.size == 0:
        return {
            "verdict": "FAIL_EMPTY",
            "is_real_speech": False,
            "is_pure_tone": False,
            "is_silence": True,
            "sample_rate": sample_rate,
            "num_samples": 0,
            "duration_s": 0.0,
            "rms": 0.0,
            "peak": 0.0,
        }

    duration_s = float(samples.size) / float(sample_rate)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(samples)))
    is_silence = rms < 1e-3

    # FFT spectral dominance check (pure sine wave concentrates >80% energy in a narrow bin)
    centered = samples - float(np.mean(samples))
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    total_energy = float(np.sum(spectrum)) + 1e-12
    peak_idx = int(np.argmax(spectrum))
    dominant_hz = float(freqs[peak_idx])
    band_mask = np.abs(freqs - dominant_hz) <= 20.0
    tone_energy_ratio = float(np.sum(spectrum[band_mask]) / total_energy)

    # Temporal amplitude envelope modulation across 50ms windows
    win_len = max(1, int(0.050 * sample_rate))
    n_wins = samples.size // win_len
    if n_wins >= 3:
        frames = samples[: n_wins * win_len].reshape(n_wins, win_len)
        win_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        mean_win_rms = float(np.mean(win_rms)) + 1e-12
        envelope_cv = float(np.std(win_rms) / mean_win_rms)
    else:
        envelope_cv = 0.0

    # Pure sine tone / continuous beep signature:
    # >80% spectral power concentrated within +-20Hz of a single frequency AND flat amplitude envelope
    is_pure_tone = (not is_silence) and (tone_energy_ratio > 0.80 and envelope_cv < 0.15)

    if is_silence:
        verdict = "FAIL_SILENCE"
    elif is_pure_tone:
        verdict = "FAIL_CONTINUOUS_TONE_BEEP"
    else:
        verdict = "PASS_SPEECH_SIGNAL"

    return {
        "verdict": verdict,
        "is_real_speech": (not is_silence and not is_pure_tone),
        "is_pure_tone": is_pure_tone,
        "is_silence": is_silence,
        "sample_rate": sample_rate,
        "num_samples": int(samples.size),
        "duration_s": round(duration_s, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "dominant_hz": round(dominant_hz, 2),
        "tone_energy_ratio": round(tone_energy_ratio, 4),
        "envelope_cv": round(envelope_cv, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TTS audio output is real speech (not beep/silence).")
    parser.add_argument("audio_path", type=Path, help="Path to .wav or raw s16le .pcm file")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate for raw PCM (default: 24000)")
    args = parser.parse_args()

    if not args.audio_path.is_file():
        print(f"ERROR: File not found: {args.audio_path}", file=sys.stderr)
        return 2

    samples, sr, channels = load_audio_samples(args.audio_path, sample_rate=args.sample_rate)
    report = analyze_tts_audio(samples, sr)
    report["channels"] = channels
    report["file"] = str(args.audio_path)
    print(json.dumps(report, indent=2))
    return 0 if report["is_real_speech"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


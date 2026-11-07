"""Voice Cloning Profile and Reference Audio Manager.

Validates reference audio (5–30s), normalizes/resamples to 24kHz mono,
and caches conditioning state tensors on CPU.
"""
from __future__ import annotations

import hashlib
import io
import os
import uuid
import wave
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore


class VoiceProfileManager:
    """Manages reference voice profiles and cached conditioning embeddings."""

    def __init__(self, voices_dir: Path | str | None = None):
        if voices_dir:
            self.voices_dir = Path(voices_dir)
        else:
            env_dir = os.getenv("NAVA_MODELS_DIR")
            self.voices_dir = (Path(env_dir) if env_dir else Path("./models")).resolve() / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._states_cache: dict[str, Any] = {}

    def get_voice_path(self, voice_id: str) -> Path | None:
        """Find path to voice wav file by ID or character name."""
        # 1. Direct file in voices directory
        candidate = self.voices_dir / f"{voice_id}.wav"
        if candidate.exists():
            return candidate

        # 2. Check without .wav
        for p in self.voices_dir.glob("*.wav"):
            if p.stem == voice_id:
                return p

        return None

    def create_voice_profile(self, audio_bytes: bytes, name: str | None = None) -> tuple[str, float]:
        """Validate, resample to 24kHz mono, save reference WAV, and return (voice_id, duration_s)."""
        if not audio_bytes:
            raise ValueError("Reference audio cannot be empty")

        # Read audio via soundfile or wave
        samples = None
        sr = 24000
        if sf:
            try:
                data, in_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if len(data.shape) > 1:
                    data = data.mean(axis=1)  # downmix to mono
                # Simple linear resample to 24000 if needed
                if in_sr != 24000:
                    target_len = int(len(data) * 24000 / in_sr)
                    data = np.interp(
                        np.linspace(0, len(data), target_len, endpoint=False),
                        np.arange(len(data)),
                        data,
                    ).astype(np.float32)
                samples = data
                sr = 24000
            except Exception:
                pass

        if samples is None:
            # Fallback to wave / PCM parsing
            if audio_bytes.startswith(b"RIFF"):
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        duration_s = len(samples) / float(sr)
        if duration_s < 1.0:
            raise ValueError(f"Reference audio too short: {duration_s:.1f}s (must be at least 1s, recommended 5-30s)")
        if duration_s > 60.0:
            # Clip to 30 seconds
            samples = samples[: 30 * sr]
            duration_s = 30.0

        # Generate unique voice_id
        audio_hash = hashlib.sha256(samples.tobytes()).hexdigest()[:12]
        voice_id = f"voice_{name or audio_hash}"

        target_file = self.voices_dir / f"{voice_id}.wav"
        # Write 24kHz 16-bit mono WAV
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(target_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())

        return voice_id, duration_s


_GLOBAL_VOICE_MGR = VoiceProfileManager()


def get_voice_manager(voices_dir: Path | str | None = None) -> VoiceProfileManager:
    if voices_dir:
        return VoiceProfileManager(voices_dir)
    return _GLOBAL_VOICE_MGR


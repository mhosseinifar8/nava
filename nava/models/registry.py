"""Nava Model Registry.

Defines all supported CPU-optimized Persian STT, TTS, G2P, and Voice Cloning models,
their metadata, Hugging Face repositories, file mappings, and profile memberships.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModelType = Literal["stt", "tts", "g2p", "vocoder"]
ProfileName = Literal["minimal", "realtime", "high-accuracy", "all"]


@dataclass(frozen=True)
class ModelFile:
    filename: str
    target_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    subfolder: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    type: ModelType
    repo_id: str
    files: tuple[ModelFile, ...]
    size_mb: float
    ram_mb: float
    rtf_class: float
    streaming: bool = False
    voice_cloning: bool = False
    bilingual_en: bool = False
    hotwords: bool = False
    sample_rate: int = 16000
    profiles: tuple[ProfileName, ...] = ("all",)
    description: str = ""
    engine_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_installed(self, models_dir: Path | str | None = None) -> bool:
        base = _resolve_models_dir(models_dir) / self.id
        if not base.is_dir():
            return False
        for mf in self.files:
            rel = mf.target_path or mf.filename
            # If target_path is specified, check it
            p = base / rel
            if not p.exists():
                return False
        return True

    def get_local_path(self, models_dir: Path | str | None = None) -> Path:
        return _resolve_models_dir(models_dir) / self.id

    def to_dict(self, models_dir: Path | str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "repo_id": self.repo_id,
            "size_mb": self.size_mb,
            "ram_mb": self.ram_mb,
            "rtf_class": self.rtf_class,
            "installed": self.is_installed(models_dir),
            "sample_rate": self.sample_rate,
            "profiles": list(self.profiles),
            "description": self.description,
            "engine": self.engine_name,
            "features": {
                "streaming": self.streaming,
                "voice_cloning": self.voice_cloning,
                "bilingual_en": self.bilingual_en,
                "hotwords": self.hotwords,
            },
        }


def _resolve_models_dir(models_dir: Path | str | None = None) -> Path:
    if models_dir:
        return Path(models_dir)
    env_dir = os.getenv("NAVA_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("./models").resolve()


# ==============================================================================
# Model Definitions
# ==============================================================================

MODELS: dict[str, ModelInfo] = {}


def _register(model: ModelInfo) -> None:
    MODELS[model.id] = model


# ------------------------------------------------------------------------------
# A. Persian STT Models (CPU-Optimized)
# ------------------------------------------------------------------------------

_register(
    ModelInfo(
        id="shenava-koochik-ctc",
        name="Shenava Koochik v1.0 CTC",
        type="stt",
        repo_id="Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx",
        files=(
            ModelFile("model.onnx"),
            ModelFile("tokens.txt"),
            ModelFile("persian_itn.py"),
        ),
        size_mb=480.0,
        ram_mb=600.0,
        rtf_class=0.05,
        streaming=False,
        hotwords=False,
        sample_rate=16000,
        profiles=("minimal", "high-accuracy", "all"),
        description="FastConformer CTC Persian acoustic model with 8.12% WER",
        engine_name="shenava_ctc",
        metadata={"normalize_type": ""},
    )
)

_register(
    ModelInfo(
        id="shenava-koochik-v1.5-rnnt",
        name="Shenava Koochik v1.5 RNNT",
        type="stt",
        repo_id="Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx",
        files=(
            ModelFile("encoder.int8.onnx"),
            ModelFile("decoder.int8.onnx"),
            ModelFile("joiner.int8.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=280.0,
        ram_mb=450.0,
        rtf_class=0.06,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="FastConformer Transducer (RNNT) with NeMo LSTM prediction head",
        engine_name="shenava_rnnt",
        metadata={"model_type": "nemo_transducer"},
    )
)

_register(
    ModelInfo(
        id="shenava-115m-flagship",
        name="Shenava Flagship 115M FastConformer",
        type="stt",
        repo_id="Reza2kn/shenava-fa-fastconformer-115m",
        files=(
            ModelFile("model.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=460.0,
        ram_mb=750.0,
        rtf_class=0.08,
        streaming=False,
        hotwords=False,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="Flagship 115M FastConformer Persian model (7.29% WER)",
        engine_name="shenava_ctc",
    )
)

_register(
    ModelInfo(
        id="shenava-rizeh-streaming-32m",
        name="Shenava Rizeh 32M Streaming",
        type="stt",
        repo_id="Reza2kn/Shenava-Rizeh-v1.0-sherpa-onnx",
        files=(
            ModelFile("encoder.int8.onnx"),
            ModelFile("decoder.int8.onnx"),
            ModelFile("joiner.int8.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=85.0,
        ram_mb=200.0,
        rtf_class=0.027,
        streaming=True,
        hotwords=True,
        sample_rate=16000,
        profiles=("realtime", "minimal", "all"),
        description="32M parameter zero-latency streaming ASR (RTF ~ 0.027)",
        engine_name="shenava_streaming",
        metadata={"att_context_sizes": [[70, 0], [70, 1], [70, 6], [70, 13]]},
    )
)

_register(
    ModelInfo(
        id="shenava-rizeh-pizeh-6.9m",
        name="Shenava Rizeh Pizeh 6.9M Edge",
        type="stt",
        repo_id="Reza2kn/Shenava-Rizeh-Pizeh-v1.0-sherpa-onnx",
        files=(
            ModelFile("encoder.int8.onnx"),
            ModelFile("decoder.int8.onnx"),
            ModelFile("joiner.int8.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=28.0,
        ram_mb=90.0,
        rtf_class=0.018,
        streaming=True,
        hotwords=True,
        sample_rate=16000,
        profiles=("realtime", "all"),
        description="Ultra-lightweight 6.9M edge/wake-word streaming model (RTF ~ 0.018)",
        engine_name="shenava_streaming",
        metadata={"att_context_sizes": [[70, 0], [70, 1], [70, 6], [70, 13]]},
    )
)

_register(
    ModelInfo(
        id="nemotron-streaming-farsi",
        name="Nemotron Streaming Persian 1120ms",
        type="stt",
        repo_id="mentisialog/sherpa-onnx-nemotron-asr-streaming-farsi-1120ms-int8",
        files=(
            ModelFile("encoder.int8.onnx"),
            ModelFile("decoder.int8.onnx"),
            ModelFile("joiner.int8.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=120.0,
        ram_mb=250.0,
        rtf_class=0.035,
        streaming=True,
        hotwords=True,
        sample_rate=16000,
        profiles=("realtime", "all"),
        description="Nemotron 1120ms chunk streaming Persian ASR",
        engine_name="nemotron_streaming",
    )
)

_register(
    ModelInfo(
        id="whisper-turbo-persian-sherpa",
        name="Whisper Large-v3-Turbo Persian (Sherpa Hybrid)",
        type="stt",
        repo_id="getsovona/sherpa-onnx-whisper-turbo-persian",
        files=(
            ModelFile("persian-encoder.int8.onnx"),
            ModelFile("persian-decoder.onnx"),
            ModelFile("persian-tokens.txt"),
        ),
        size_mb=950.0,
        ram_mb=1800.0,
        rtf_class=0.12,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "minimal", "all"),
        description="Whisper Turbo INT8 encoder + FP32 decoder for Persian accuracy",
        engine_name="whisper_sherpa",
        metadata={"task": "transcribe", "language": "fa", "silence_pad_s": 1.0},
    )
)

_register(
    ModelInfo(
        id="whisper-turbo-persian-givia",
        name="Whisper Large-v3-Turbo Persian (Givia Colloquial)",
        type="stt",
        repo_id="givia/whisper-large-v3-turbo-persian",
        files=(
            ModelFile("model.bin"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
        ),
        size_mb=1600.0,
        ram_mb=2200.0,
        rtf_class=0.14,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="Colloquial and punctuated Persian Whisper Large-v3-Turbo",
        engine_name="whisper_faster",
    )
)

_register(
    ModelInfo(
        id="whisper-turbo-persian-gholizadeh",
        name="Whisper Large-v3-Turbo Persian (Gholizadeh Robust)",
        type="stt",
        repo_id="MohammadGholizadeh/whisper-large-v3-turbo-persian",
        files=(
            ModelFile("model.bin"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
        ),
        size_mb=1600.0,
        ram_mb=2200.0,
        rtf_class=0.14,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="Noise-robust Persian Whisper Large-v3-Turbo",
        engine_name="whisper_faster",
    )
)

_register(
    ModelInfo(
        id="vosk-fa-light",
        name="Vosk Persian Light",
        type="stt",
        repo_id="alphacep/vosk-model-fa",
        files=(ModelFile("vosk-model-fa.zip"),),
        size_mb=45.0,
        ram_mb=120.0,
        rtf_class=0.02,
        streaming=True,
        sample_rate=16000,
        profiles=("all",),
        description="Standard lightweight Persian Vosk model",
        engine_name="vosk",
    )
)

_register(
    ModelInfo(
        id="whisper-large-v3",
        name="Faster Whisper Large-v3",
        type="stt",
        repo_id="Systran/faster-whisper-large-v3",
        files=(
            ModelFile("model.bin"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
            ModelFile("vocabulary.json"),
        ),
        size_mb=1500.0,
        ram_mb=2500.0,
        rtf_class=0.18,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("all",),
        description="Standard CTranslate2 Whisper Large-v3",
        engine_name="whisper_faster",
    )
)

# ------------------------------------------------------------------------------
# B. Persian G2P (Ezafe & Homograph) Preprocessor Models
# ------------------------------------------------------------------------------

_register(
    ModelInfo(
        id="persian-ezafe-albert",
        name="Persian Ezafe Albert Quantized ONNX",
        type="g2p",
        repo_id="abreza/persian-ezafe-albert",
        files=(
            ModelFile("model_quantized.onnx", subfolder="quantized"),
            ModelFile("spiece.model", subfolder="quantized"),
            ModelFile("config.json", subfolder="quantized"),
        ),
        size_mb=41.0,
        ram_mb=100.0,
        rtf_class=0.005,
        streaming=False,
        profiles=("minimal", "realtime", "high-accuracy", "all"),
        description="Fast context-aware Ezafe detector (41MB ONNX, 37ms CPU latency)",
        engine_name="ezafe_albert",
    )
)

_register(
    ModelInfo(
        id="homo-ge2pe-persian",
        name="Homo-GE2PE Persian",
        type="g2p",
        repo_id="MahtaFetrat/Homo-GE2PE-Persian",
        files=(
            ModelFile("model-weights/homo-ge2pe.zip"),
            ModelFile("assets/GE2PE.py"),
            ModelFile("assets/Parsivar.zip"),
        ),
        size_mb=180.0,
        ram_mb=350.0,
        rtf_class=0.03,
        streaming=False,
        profiles=("high-accuracy", "all"),
        description="Neural homograph disambiguator and G2P model",
        engine_name="homo_ge2pe",
    )
)

_register(
    ModelInfo(
        id="homo-t5",
        name="Homo-T5 Persian Homograph Disambiguator",
        type="g2p",
        repo_id="MahtaFetrat/Homo-GE2PE-Persian",
        files=(ModelFile("model-weights/homo-t5.zip"),),
        size_mb=240.0,
        ram_mb=450.0,
        rtf_class=0.04,
        streaming=False,
        profiles=("high-accuracy", "all"),
        description="Seq2seq neural homograph disambiguator T5",
        engine_name="homo_t5",
    )
)

# ------------------------------------------------------------------------------
# C. Persian TTS Models (CPU-Optimized)
# ------------------------------------------------------------------------------

_register(
    ModelInfo(
        id="piper-mana-lca",
        name="Mana Persian Piper (LCA)",
        type="tts",
        repo_id="MahtaFetrat/Mana-Persian-Piper",
        files=(
            ModelFile("fa_IR-mana-medium.onnx"),
            ModelFile("fa_IR-mana-medium.onnx.json"),
        ),
        size_mb=65.0,
        ram_mb=150.0,
        rtf_class=0.04,
        streaming=True,
        sample_rate=22050,
        profiles=("minimal", "realtime", "high-accuracy", "all"),
        description="Highest MOS 3.14 studio-quality real-time CPU TTS with Ezafe and HomoFast",
        engine_name="piper",
        metadata={"voice_id": "mana"},
    )
)

_register(
    ModelInfo(
        id="piper-voices-fa",
        name="Persian Piper Multi-Speaker Voices",
        type="tts",
        repo_id="rhasspy/piper-voices",
        files=(
            ModelFile(
                "fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx",
                target_path="ganji/fa_IR-ganji-medium.onnx",
            ),
            ModelFile(
                "fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx.json",
                target_path="ganji/fa_IR-ganji-medium.onnx.json",
            ),
        ),
        size_mb=180.0,
        ram_mb=200.0,
        rtf_class=0.04,
        streaming=True,
        sample_rate=22050,
        profiles=("minimal", "all"),
        description="Multi-speaker Persian Piper voices (amir, gyro, ganji, ganji_adabi, reza_ibrahim, haaniye)",
        engine_name="piper",
        metadata={
            "speakers": [
                "amir",
                "gyro",
                "ganji",
                "ganji_adabi",
                "reza_ibrahim",
                "haaniye_low",
            ]
        },
    )
)

_register(
    ModelInfo(
        id="ava-82m",
        name="Ava-82M Kokoro Persian",
        type="tts",
        repo_id="xmanii/Ava-82M",
        files=(
            ModelFile("model.onnx"),
            ModelFile("config.json"),
            ModelFile("pronunciations.json"),
        ),
        size_mb=320.0,
        ram_mb=600.0,
        rtf_class=0.09,
        streaming=True,
        sample_rate=24000,
        profiles=("high-accuracy", "all"),
        description="24kHz studio-quality Kokoro-82M Persian architecture with warm singleton",
        engine_name="ava",
    )
)

_register(
    ModelInfo(
        id="matcha-khadijah-fa-en",
        name="Matcha Khadijah FA-EN Bilingual (Female)",
        type="tts",
        repo_id="mah92/Khadijah-FA_EN-Matcha-TTS-Model",
        files=(
            ModelFile("model.onnx"),
            ModelFile("hifigan_universal_v1.onnx"),
        ),
        size_mb=180.0,
        ram_mb=400.0,
        rtf_class=0.08,
        streaming=False,
        bilingual_en=True,
        sample_rate=22050,
        profiles=("all",),
        description="Bilingual Persian-English female flow-matching TTS with Universal HiFi-GAN",
        engine_name="matcha",
        metadata={"gender": "female"},
    )
)

_register(
    ModelInfo(
        id="matcha-musa-fa-en",
        name="Matcha Musa FA-EN Bilingual (Male)",
        type="tts",
        repo_id="mah92/Musa-FA_EN-Matcha-TTS-Model",
        files=(
            ModelFile("model.onnx"),
            ModelFile("hifigan_universal_v1.onnx"),
        ),
        size_mb=180.0,
        ram_mb=400.0,
        rtf_class=0.08,
        streaming=False,
        bilingual_en=True,
        sample_rate=22050,
        profiles=("all",),
        description="Bilingual Persian-English male flow-matching TTS with Universal HiFi-GAN",
        engine_name="matcha",
        metadata={"gender": "male"},
    )
)

_register(
    ModelInfo(
        id="pocket-tts-farsi",
        name="Pocket-TTS Farsi v2 (with Voice Cloning)",
        type="tts",
        repo_id="mehdi-hf/pocket-tts-farsi-v2",
        files=(
            ModelFile("model.yaml"),
            ModelFile("weights.pt"),
        ),
        size_mb=450.0,
        ram_mb=850.0,
        rtf_class=0.15,
        streaming=True,
        voice_cloning=True,
        sample_rate=24000,
        profiles=("minimal", "high-accuracy", "all"),
        description="24kHz FlowLM + Kyutai Mimi with zero-shot voice cloning and 10 built-in characters",
        engine_name="pocket",
        metadata={
            "characters": [
                "ali",
                "maryam",
                "reza",
                "sara",
                "nima",
                "parisa",
                "omid",
                "zahra",
                "kian",
                "mina",
                "default",
            ]
        },
    )
)

_register(
    ModelInfo(
        id="nava-fa-clone",
        name="Nava-FA Zero-Shot Voice Clone",
        type="tts",
        repo_id="zorost/nava-fa",
        files=(
            ModelFile("model.yaml"),
            ModelFile("weights.pt"),
        ),
        size_mb=450.0,
        ram_mb=850.0,
        rtf_class=0.15,
        streaming=True,
        voice_cloning=True,
        sample_rate=24000,
        profiles=("all",),
        description="24kHz FlowLM zero-shot voice cloning engine for Persian",
        engine_name="pocket",
    )
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_model(model_id: str) -> ModelInfo:
    """Retrieve model info by ID or raise KeyError."""
    if model_id not in MODELS:
        # Check aliases
        aliases = {
            "pocket": "pocket-tts-farsi",
            "mana": "piper-mana-lca",
            "piper": "piper-voices-fa",
            "shenava": "shenava-koochik-ctc",
            "whisper": "whisper-turbo-persian-sherpa",
            "faster-whisper": "whisper-large-v3",
            "albert": "persian-ezafe-albert",
            "ava": "ava-82m",
        }
        if model_id in aliases and aliases[model_id] in MODELS:
            return MODELS[aliases[model_id]]
        raise KeyError(f"Unknown model ID: {model_id}. Available: {list(MODELS.keys())}")
    return MODELS[model_id]


def list_models(models_dir: Path | str | None = None, model_type: ModelType | None = None) -> list[dict[str, Any]]:
    """List all models with installation status and feature details."""
    result = []
    for model in MODELS.values():
        if model_type and model.type != model_type:
            continue
        result.append(model.to_dict(models_dir))
    return result


def get_profile_models(profile: ProfileName) -> list[ModelInfo]:
    """Return all models belonging to a given profile."""
    if profile == "all":
        return list(MODELS.values())
    return [m for m in MODELS.values() if profile in m.profiles]


def resolve_model_path(model_id: str, models_dir: Path | str | None = None) -> Path:
    """Return local path to a model directory."""
    return get_model(model_id).get_local_path(models_dir)


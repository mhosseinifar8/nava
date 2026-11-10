"""Nava Model Registry.

Defines all supported CPU-optimized Persian STT, TTS, G2P, NLP, and OCR models,
their metadata, Hugging Face repositories, file mappings, and profile memberships.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModelType = Literal["stt", "tts", "g2p", "vocoder", "nlp", "ocr"]
ProfileName = Literal["minimal", "realtime", "high-accuracy", "all"]


@dataclass(frozen=True)
class ModelFile:
    filename: str
    target_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    subfolder: str | None = None
    repo_id: str | None = None


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
        if not self.files:
            return False
        for mf in self.files:
            rel = mf.target_path or mf.filename
            p = base / rel
            if not p.is_file() or p.stat().st_size == 0:
                return False
            if mf.size_bytes and p.stat().st_size != mf.size_bytes:
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
        id="parakeet-ctc-109m-farsi",
        name="Parakeet-CTC 109M Persian (Primary Ultra-Fast)",
        type="stt",
        repo_id="Peacockery/parakeet-ctc-109m-farsi",
        files=(
            ModelFile("model.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=218.0,
        ram_mb=350.0,
        rtf_class=0.015,
        streaming=True,
        hotwords=False,
        sample_rate=16000,
        profiles=("minimal", "realtime", "all"),
        description="Primary Ultra-Fast <150ms latency CTC Persian ASR engine",
        engine_name="parakeet_ctc",
    )
)

_register(
    ModelInfo(
        id="fastconformer-hybrid-large",
        name="NVIDIA FastConformer Hybrid Large (CTC + RNNT)",
        type="stt",
        repo_id="nvidia/stt_fa_fastconformer_hybrid_large",
        files=(
            ModelFile("model.onnx"),
            ModelFile("tokens.txt"),
        ),
        size_mb=460.0,
        ram_mb=900.0,
        rtf_class=0.045,
        streaming=True,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="NVIDIA Hybrid FastConformer with default 3x CTC CPU speedup and RNNT accuracy toggle",
        engine_name="fastconformer_hybrid",
    )
)

_register(
    ModelInfo(
        id="nemotron-streaming-farsi",
        name="Nemotron Streaming Persian ASR",
        type="stt",
        repo_id="mehdi-hf/nemotron-asr-streaming-farsi",
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
        description="Mehdi-HF Nemotron streaming Persian ASR (1120ms chunk)",
        engine_name="nemotron_streaming",
    )
)
# Also register under full name mehdi-hf/nemotron-asr-streaming-farsi
_register(
    ModelInfo(
        id="nemotron-asr-streaming-farsi",
        name="Nemotron Streaming Persian ASR",
        type="stt",
        repo_id="mehdi-hf/nemotron-asr-streaming-farsi",
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
        description="Mehdi-HF Nemotron streaming Persian ASR (1120ms chunk)",
        engine_name="nemotron_streaming",
    )
)

_register(
    ModelInfo(
        id="nemotron-streaming-podcast",
        name="Nemotron Streaming Persian Podcast ASR",
        type="stt",
        repo_id="MohammadGholizadeh/nemotron-asr-streaming-farsi-podcast",
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
        profiles=("realtime", "high-accuracy", "all"),
        description="Gholizadeh Nemotron streaming conversational & podcast ASR",
        engine_name="nemotron_streaming",
    )
)

_register(
    ModelInfo(
        id="whisper-large-v3-farsi-cv17",
        name="Whisper Large-v3 Persian CV17 (CTranslate2 INT8)",
        type="stt",
        repo_id="mohammadjavadnasri/whisper-large-v3-farsi-cv17",
        files=(
            ModelFile("model.bin"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
            ModelFile("vocabulary.json"),
        ),
        size_mb=1550.0,
        ram_mb=2400.0,
        rtf_class=0.11,
        streaming=False,
        hotwords=True,
        sample_rate=16000,
        profiles=("high-accuracy", "all"),
        description="Fine-tuned Whisper Large-v3 on Common Voice 17 with INT8 CPU optimization",
        engine_name="whisper_faster",
    )
)

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
        repo_id="Reza2kn/shenava-fa-fastconformer-115m-onnx-fp16",
        files=(
            ModelFile("fastconformer_v4_ctc_fixed2005_len_fp16_ingraphnorm_embedded.onnx", target_path="model.onnx"),
            ModelFile("115m_tokens.json", target_path="tokens.json"),
        ),
        size_mb=230.0,
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
            ModelFile("model.onnx"),
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
            ModelFile("model.onnx"),
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
        id="whisper-turbo-persian-sherpa",
        name="Whisper Large-v3-Turbo Persian (Sherpa Hybrid)",
        type="stt",
        repo_id="csukuangfj/sherpa-onnx-whisper-turbo",
        files=(
            ModelFile("turbo-encoder.int8.onnx", target_path="persian-encoder.int8.onnx"),
            ModelFile("turbo-decoder.onnx", target_path="persian-decoder.onnx"),
            ModelFile("turbo-tokens.txt", target_path="persian-tokens.txt"),
        ),
        size_mb=1250.0,
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
            ModelFile("model.safetensors"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
        ),
        size_mb=3100.0,
        ram_mb=3500.0,
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
            ModelFile("model.safetensors"),
            ModelFile("tokenizer.json"),
            ModelFile("config.json"),
        ),
        size_mb=3100.0,
        ram_mb=3500.0,
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
        repo_id="rhasspy/vosk-models",
        files=(
            ModelFile("vosk-model-small-fa-0.5.zip", subfolder="fa", target_path="vosk-model-fa.zip"),
        ),
        size_mb=60.0,
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
# B. Persian G2P Preprocessor Models
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
        id="pocket-tts-farsi-v2",
        name="Pocket-TTS Farsi v2 (FlowLM 24kHz)",
        type="tts",
        repo_id="mehdi-hf/pocket-tts-farsi-v2",
        files=(
            ModelFile("model.yaml"),
            ModelFile("model.safetensors"),
            ModelFile("tokenizer_ph.model"),
        ),
        size_mb=440.0,
        ram_mb=850.0,
        rtf_class=0.15,
        streaming=True,
        voice_cloning=True,
        sample_rate=24000,
        profiles=("minimal", "realtime", "high-accuracy", "all"),
        description="24kHz FlowLM + Kyutai Mimi v2 with superior prosody and zero-shot voice cloning",
        engine_name="pocket",
        metadata={
            "characters": ["ali", "maryam", "reza", "sara", "nima", "parisa", "omid", "zahra", "kian", "mina", "default"]
        },
    )
)

_register(
    ModelInfo(
        id="pocket-tts-farsi",
        name="Pocket-TTS Farsi v1 (Direct Script)",
        type="tts",
        repo_id="mehdi-hf/pocket-tts-farsi",
        files=(
            ModelFile("model.yaml"),
            ModelFile("model.safetensors"),
            ModelFile("tokenizer_ph.model"),
        ),
        size_mb=440.0,
        ram_mb=850.0,
        rtf_class=0.15,
        streaming=True,
        voice_cloning=True,
        sample_rate=24000,
        profiles=("minimal", "all"),
        description="24kHz FlowLM Persian direct script TTS with 10 built-in characters",
        engine_name="pocket",
    )
)

_register(
    ModelInfo(
        id="chatterbox-tts-persian-gguf",
        name="Chatterbox TTS Persian GGUF",
        type="tts",
        repo_id="mup99ma/Chatterbox-TTS-Persian-gguf",
        files=(
            ModelFile("model.gguf"),
        ),
        size_mb=320.0,
        ram_mb=650.0,
        rtf_class=0.08,
        streaming=True,
        sample_rate=24000,
        profiles=("realtime", "all"),
        description="GGUF-quantized Chatterbox Persian TTS via llama-cpp-python with mlock on CPU",
        engine_name="chatterbox_gguf",
    )
)

_register(
    ModelInfo(
        id="chatterbox-tts-persian-mazrba",
        name="Chatterbox TTS Persian GGUF (Mazrba)",
        type="tts",
        repo_id="mazrba/Chatterbox-TTS-Persian-gguf",
        files=(
            ModelFile("model.gguf"),
        ),
        size_mb=320.0,
        ram_mb=650.0,
        rtf_class=0.08,
        streaming=True,
        sample_rate=24000,
        profiles=("all",),
        description="Alternative GGUF-quantized Chatterbox Persian model",
        engine_name="chatterbox_gguf",
    )
)

_register(
    ModelInfo(
        id="chatterbox-tts-persian-torch",
        name="Chatterbox TTS Persian PyTorch",
        type="tts",
        repo_id="Thomcles/Chatterbox-TTS-Persian-Farsi",
        files=(
            ModelFile("model.safetensors"),
            ModelFile("config.json"),
            ModelFile("tokenizer.json"),
        ),
        size_mb=680.0,
        ram_mb=1200.0,
        rtf_class=0.12,
        streaming=True,
        sample_rate=24000,
        profiles=("high-accuracy", "all"),
        description="PyTorch Chatterbox Persian TTS with AVX-512 bfloat16 / int8 dynamic quantization",
        engine_name="chatterbox_torch",
    )
)

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
            "speakers": ["amir", "gyro", "ganji", "ganji_adabi", "reza_ibrahim", "haaniye_low"]
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
            ModelFile("model.pth"),
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
            ModelFile("matcha-fa_en-khadijah-22050-3.onnx", target_path="model.onnx"),
            ModelFile("tokens_sherpa_with_fa.txt", target_path="tokens.txt"),
            ModelFile(
                "hifigan_universal_22050hz_v1.onnx",
                repo_id="mah92/hifigan_for_sherpa",
                target_path="hifigan_universal_v1.onnx",
            ),
        ),
        size_mb=130.0,
        ram_mb=400.0,
        rtf_class=0.08,
        streaming=True,
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
            ModelFile("matcha-fa_en-musa-22050-3.onnx", target_path="model.onnx"),
            ModelFile("tokens_sherpa_with_fa.txt", target_path="tokens.txt"),
            ModelFile(
                "hifigan_universal_22050hz_v1.onnx",
                repo_id="mah92/hifigan_for_sherpa",
                target_path="hifigan_universal_v1.onnx",
            ),
        ),
        size_mb=130.0,
        ram_mb=400.0,
        rtf_class=0.08,
        streaming=True,
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
        id="nava-fa-clone",
        name="Nava-FA Zero-Shot Voice Clone",
        type="tts",
        repo_id="mehdi-hf/pocket-tts-farsi-v2",
        files=(
            ModelFile("model.yaml"),
            ModelFile("model.safetensors"),
            ModelFile("tokenizer_ph.model"),
        ),
        size_mb=440.0,
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

# ------------------------------------------------------------------------------
# D. Specialized Persian NLP & Vision/OCR Models
# ------------------------------------------------------------------------------

_register(
    ModelInfo(
        id="snappfood-sentiment-analysis",
        name="Snappfood Persian Sentiment Analysis",
        type="nlp",
        repo_id="ParsiAI/snappfood-sentiment-analysis",
        files=(
            ModelFile("model.onnx"),
            ModelFile("config.json"),
            ModelFile("tokenizer.json"),
        ),
        size_mb=110.0,
        ram_mb=180.0,
        rtf_class=0.005,
        streaming=False,
        profiles=("all",),
        description="Low-latency Persian text sentiment classification (< 15ms ONNX INT8)",
        engine_name="snappfood_sentiment",
    )
)

_register(
    ModelInfo(
        id="olmocr-farsi-lora",
        name="olmOCR Farsi LoRA Document OCR",
        type="ocr",
        repo_id="larshiakzemil/olmocr-farsi-lora",
        files=(
            ModelFile("adapter_model.safetensors"),
            ModelFile("adapter_config.json"),
        ),
        size_mb=120.0,
        ram_mb=12000.0,
        rtf_class=0.50,
        streaming=False,
        profiles=("high-accuracy", "all"),
        description="PEFT LoRA Persian document OCR (Qwen2.5-VL-7B backend, build_no_anchoring_v4_yaml_prompt)",
        engine_name="olmocr_farsi",
    )
)


# ==============================================================================
# Helper Functions & Aliases
# ==============================================================================

ALIASES: dict[str, str] = {
    # TTS aliases
    "pocket": "pocket-tts-farsi",
    "pocket-v2": "pocket-tts-farsi-v2",
    "pocket-v1": "pocket-tts-farsi",
    "chatterbox": "chatterbox-tts-persian-gguf",
    "chatterbox-gguf": "chatterbox-tts-persian-gguf",
    "chatterbox-torch": "chatterbox-tts-persian-torch",
    "mana": "piper-mana-lca",
    "piper": "piper-voices-fa",
    "ava": "ava-82m",
    "matcha": "matcha-khadijah-fa-en",
    "matcha-khadijah": "matcha-khadijah-fa-en",
    "matcha-musa": "matcha-musa-fa-en",
    # STT aliases
    "parakeet": "parakeet-ctc-109m-farsi",
    "parakeet-ctc": "parakeet-ctc-109m-farsi",
    "fastconformer": "fastconformer-hybrid-large",
    "fastconformer-large": "fastconformer-hybrid-large",
    "shenava": "shenava-koochik-ctc",
    "nemotron": "nemotron-asr-streaming-farsi",
    "nemotron-podcast": "nemotron-streaming-podcast",
    "whisper": "whisper-turbo-persian-sherpa",
    "whisper-cv17": "whisper-large-v3-farsi-cv17",
    "faster-whisper": "whisper-large-v3",
    # NLP / OCR
    "sentiment": "snappfood-sentiment-analysis",
    "snappfood": "snappfood-sentiment-analysis",
    "ocr": "olmocr-farsi-lora",
    "olmocr": "olmocr-farsi-lora",
    # Preprocessor
    "albert": "persian-ezafe-albert",
}


def get_model(model_id: str) -> ModelInfo:
    """Retrieve model info by ID or alias, or raise KeyError."""
    if model_id in MODELS:
        return MODELS[model_id]
    if model_id in ALIASES and ALIASES[model_id] in MODELS:
        return MODELS[ALIASES[model_id]]
    # Check lowercase / normalized
    norm = model_id.lower().strip()
    if norm in MODELS:
        return MODELS[norm]
    if norm in ALIASES and ALIASES[norm] in MODELS:
        return MODELS[ALIASES[norm]]
    raise KeyError(f"Unknown model ID: {model_id}. Available: {list(MODELS.keys())}")


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

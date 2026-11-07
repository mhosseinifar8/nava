# Nava (نوا) — Production-Grade Persian Multi-Engine Speech Platform & SDK

**nava-fa-speech** is an offline, CPU-optimized Persian speech platform and Python SDK. It provides unified, production-ready Speech-to-Text (STT), Text-to-Speech (TTS), Inverse Text Normalization (ITN), Context-Aware G2P / Ezafe detection, and Zero-Shot Voice Cloning over REST and low-latency WebSockets.

---

## Key Features

- **Automated Model Registry & Downloader CLI (`nava-download` / `scripts/download_models.py`):**
  - Download by profile (`--profile minimal | realtime | high-accuracy | all`) or by model (`--model <id>`).
  - Checksum verification, resume support, and progress tracking from Hugging Face & Releases.
- **Top Persian STT Engines on CPU:**
  - **Shenava Family (FastConformer):**
    - `shenava-koochik-ctc`: CTC acoustic model with 8.12% WER; per-feature normalization disabled (`normalize_type=""`).
    - `shenava-koochik-v1.5-rnnt`: Transducer model using NeMo LSTM prediction network (`model_type="nemo_transducer"`).
    - `shenava-115m-flagship`: Flagship 115M FastConformer Persian model (7.29% WER).
    - `shenava-rizeh-streaming-32m`: Zero-latency streaming ASR ($\text{RTF} \approx 0.027$) with configurable latency context (`0ms`, `80ms`, `480ms`, `1040ms`).
    - `shenava-rizeh-pizeh-6.9m`: Ultra-light 6.9M edge/wake-word streaming model ($\text{RTF} \approx 0.018$).
  - **Nemotron Streaming:** `nemotron-streaming-farsi` (1120ms chunk streaming).
  - **Whisper Large-v3-Turbo Persian:**
    - `whisper-turbo-persian-sherpa`: Hybrid INT8 encoder + FP32 decoder with automatic 1.0s silence padding and 2s sliding window chunking.
    - `whisper-turbo-persian-givia` & `whisper-turbo-persian-gholizadeh`: Punctuated, colloquial, and noise-robust models.
    - `vosk-fa-light`: Lightweight edge Vosk model.
- **Top Persian TTS Engines & G2P Pipeline:**
  - **Context-Aware G2P & Ezafe Pipeline:**
    - Step 1: Text & Number Normalization (dates, times, percentages, currencies, digits to words).
    - Step 2: Custom Pronunciation Dictionary (server JSON + per-request override).
    - Step 3: Neural Ezafe detection (`abreza/persian-ezafe-albert` ONNX <40ms CPU latency) inserting proper kasreh / `‌ی` connectors.
  - **Engines:**
    - `piper-mana-lca`: Studio-quality MOS 3.14 real-time CPU TTS with Ezafe.
    - `piper-voices-fa`: Multi-speaker voices (`amir`, `gyro`, `ganji`, `ganji_adabi`, `reza_ibrahim`, `haaniye_low`).
    - `ava-82m`: 24kHz Kokoro-82M Persian architecture kept in memory as a warm singleton.
    - `matcha-khadijah-fa-en` & `matcha-musa-fa-en`: Bilingual Persian-English flow-matching with HiFi-GAN 22050Hz vocoder.
    - `pocket-tts-farsi` & `nava-fa-clone`: 24kHz FlowLM with 10 built-in characters and **Zero-Shot Voice Cloning** on CPU from 5–30s reference audio.
- **REST & Real-Time WebSocket Protocols:**
  - OpenAI-compatible endpoints with Persian extensions (`hotwords`, `enable_itn`, `custom_dictionary`, `reference_audio_base64`).
  - Full backward compatibility with Nava v1 endpoints.

---

## Installation

```bash
# Install SDK and CLI from repo root
pip install -e .

# Or install the SDK subpackage directly
pip install -e ./nava_sdk
```

---

## Model Downloader CLI

```bash
# List all 21 models and their installation status
nava-download --list

# Download by pre-configured profile
nava-download --profile minimal         # Fast starter set: Koochik CTC, Albert Ezafe, Mana Piper, Pocket
nava-download --profile realtime        # Low-latency streaming: Rizeh 32M, Nemotron, Albert Ezafe, Mana
nava-download --profile high-accuracy   # Highest precision: Koochik RNNT, Whisper Turbo Sherpa, Ava-82M
nava-download --profile all             # Complete collection

# Download a specific model
nava-download --model shenava-koochik-ctc
nava-download --model piper-mana-lca
nava-download --model ava-82m
```

---

## Quick Start (Python SDK)

```python
from nava import NavaClient

# Connect to Nava Server (default: http://127.0.0.1:8000)
with NavaClient() as nava:
    # 1. Speech-to-Text (STT) with ITN and Hotwords
    with open("audio.wav", "rb") as f:
        pcm = f.read()
    
    result = nava.transcribe(
        pcm,
        model="shenava-koochik-ctc",
        enable_itn=True,                    # Converts "هشتصد و پنجاه" -> "850"
        hotwords={"دیجیکالا": 2.0},
        custom_dictionary={"اسنب فود": "اسنپ‌فود"},
    )
    print("Transcript:", result.text)
    print("With ITN:", result.itn_text)

    # 2. Text-to-Speech (TTS) with Ezafe and Custom Lexicon
    speech = nava.synthesize(
        "کتاب خوب و خانه بزرگ در تاریخ 1403/08/18 افتتاح شد.",
        model="piper-mana-lca",
        speed=1.0,
        g2p_mode="lca_albert",
    )
    with open("output.pcm", "wb") as f:
        f.write(speech.pcm)

    # 3. Zero-Shot Voice Cloning
    with open("reference_voice.wav", "rb") as f:
        ref_bytes = f.read()

    voice_id = nava.clone_voice(ref_bytes, name="my_speaker")
    cloned_speech = nava.synthesize(
        "سلام، این متن با صدای شبیه‌سازی‌شده خوانده می‌شود.",
        model="pocket-tts-farsi",
        voice=voice_id,
    )
```

### Async & Real-Time WebSockets

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient() as nava:
        # Real-time sentence streaming TTS over WebSocket
        async for chunk in nava.stream_speech_ws(
            "سلام به سامانه پیشرفته نوا خوش آمدید. این سامانه به طور زنده صدا تولید می‌کند.",
            model="piper-mana-lca",
        ):
            print(f"Received chunk: {len(chunk.pcm)} bytes at {chunk.sample_rate} Hz")

asyncio.run(main())
```

---

## Server & API Endpoints

Launch the server:
```bash
nava-server --host 0.0.0.0 --port 8000
```

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/models` | List all models, RAM footprint, RTF class, and download status |
| `POST` | `/v1/models/download` | Trigger synchronous or background download (`{"model": "...", "profile": "..."}`) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible + Nava STT (PCM/WAV, `enable_itn`, `hotwords`, `custom_dictionary`) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible + Nava TTS (`input`, `model`, `voice`, `g2p_mode`, `speed`, `response_format`) |
| `POST` | `/v1/voices/clone` | Upload 5–30s reference WAV, normalize to 24kHz mono, return cached `voice_id` |
| `GET` | `/healthz` | Health check |

*Backward-compatible aliases (`/api/v1/transcriptions`, `/api/v1/speech`, `/api/v1/health`) are fully supported.*

### WebSocket API

- **`WS /v1/ws/stt` (Streaming STT):**
  - **Handshake (JSON):** `{"model": "shenava-rizeh-streaming-32m", "sample_rate": 16000, "latency_ms": 80, "enable_itn": true, "hotwords": {"چشم": 2.5}}`
  - **Stream (Binary):** Stream 16-bit PCM chunks.
  - **Events (JSON):** Server emits real-time `{"type": "partial", "text": "..."}` and `{"type": "final", "text": "...", "itn_text": "...", "rtf": 0.027}` on EOF.
- **`WS /v1/ws/tts` (Sentence-Streaming TTS):**
  - **Handshake (JSON):** `{"model": "piper-mana-lca", "voice": "mana", "speed": 1.0, "g2p_mode": "lca_albert"}`
  - **Text Frames (JSON):** `{"type": "text", "text": "..."}` and `{"type": "cancel"}`.
  - **Audio Stream (Binary):** Sentence-by-sentence low-latency PCM audio stream.

---

## Running Tests

```bash
# Run unit tests (83 passing tests in ~0.3s, no model download required)
pytest tests/unit

# Run all tests (including integration tests against live service)
pytest tests
```

---

## License

MIT License. Designed and maintained for Persian speech AI community.

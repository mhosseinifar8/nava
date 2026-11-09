# Nava ( نوا ) — Production-Grade Persian Multi-Engine Speech Platform & SDK

<p align="center">
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/v/nava-fa-speech.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/pyversions/nava-fa-speech.svg" alt="Python versions"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-stt"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--stt-2496ED?logo=docker&logoColor=white" alt="Docker STT"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-tts"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--tts-2496ED?logo=docker&logoColor=white" alt="Docker TTS"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava-2496ED?logo=docker&logoColor=white" alt="Docker Server"></a>
  <a href="https://github.com/mhosseinifar8/nava/actions/workflows/docker-publish.yml"><img src="https://github.com/mhosseinifar8/nava/actions/workflows/docker-publish.yml/badge.svg" alt="Docker Build & Publish"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Inference-CPU%20Optimized-green.svg" alt="CPU Optimized"></a>
</p>

**Nava (نوا)** is an offline, CPU-optimized Persian speech platform and Python SDK (`nava-fa-speech`). It provides unified, production-ready **Speech-to-Text (STT)**, **Text-to-Speech (TTS)**, **Inverse Text Normalization (ITN)**, **Context-Aware G2P / Neural Ezafe detection**, and **Zero-Shot Voice Cloning** over REST and low-latency WebSockets.

Designed to operate seamlessly on standard CPU servers without requiring expensive GPUs, Nava delivers real-time speech processing for high-throughput microservices and edge deployments.

---

## Key Features

- **Automated Model Registry & Downloader CLI (`nava-download`):**
  - Download by profile (`--profile minimal | realtime | high-accuracy | all`) or by individual model id.
  - SHA256 checksum verification, multi-stream download, resume support, and progress bars.
  - Built-in proxy (`--proxy`) and Hugging Face token (`--hf-token`) support for restricted networks.
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
- **Production-Grade Microservices & Unified Deployment:**
  - Independent `nava-stt` (port 8001) and `nava-tts` (port 8002) microservices or all-in-one `nava-server` (port 8000).
  - Flexible route compatibility: standard `/v1/audio/...`, `/api/v1/...`, and double-prefix `/v1/v1/...` to eliminate client configuration mismatches.
  - WebSocket control handshakes (initial configuration without mandatory text frames, live partials, sentence streaming).

---

## Installation

### 1. Official Python SDK from PyPI (`nava-fa-speech`)

For client applications, web backends, and bots:

```bash
pip install nava-fa-speech
```

*Note: Both `import nava` and `import nava_fa_speech` are fully supported.*

### 2. Full Platform from Source

To run the local server or modify engines:

```bash
git clone https://github.com/mhosseinifar8/nava.git
cd nava
pip install -e .
```

---

## Docker Hub Official Images

Prebuilt, production-ready `linux/amd64` Docker images are automatically built and published to Docker Hub on every commit to `main` and release tag:

| Service | Docker Hub Image | Port | Description |
|---|---|---|---|
| **STT Microservice** | [`mhosseinifar8/nava-stt:latest`](https://hub.docker.com/r/mhosseinifar8/nava-stt) | `8001` | FastConformer Shenava, Whisper-Turbo, Nemotron, Vosk |
| **TTS Microservice** | [`mhosseinifar8/nava-tts:latest`](https://hub.docker.com/r/mhosseinifar8/nava-tts) | `8002` | Piper Mana, Matcha-TTS, Ava Kokoro, Pocket-TTS & Voice Cloning |
| **Unified Platform** | [`mhosseinifar8/nava:latest`](https://hub.docker.com/r/mhosseinifar8/nava) | `8000` | Complete platform with all STT, TTS, G2P, and Model Manager |

### Quick Run with Docker Compose

```bash
# 1. Clone repository
git clone https://github.com/mhosseinifar8/nava.git
cd nava

# 2. Download starter models to ./models
pip install nava-fa-speech
nava-download minimal

# 3. Pull and launch STT & TTS microservices (ports 8001 & 8002)
docker compose pull
docker compose up -d

# Or launch the unified single-port platform (port 8000)
docker compose --profile unified up -d
```

### Standalone Docker Run

```bash
# Run STT Service
docker run -d \
  --name nava-stt \
  -p 8001:8001 \
  -v $(pwd)/models:/models \
  --restart unless-stopped \
  mhosseinifar8/nava-stt:latest

# Run TTS Service
docker run -d \
  --name nava-tts \
  -p 8002:8002 \
  -v $(pwd)/models:/models \
  --restart unless-stopped \
  mhosseinifar8/nava-tts:latest

# Run Unified Server
docker run -d \
  --name nava \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  --restart unless-stopped \
  mhosseinifar8/nava:latest
```

### Proxy & Restricted Network Configuration (.env)

When running behind restrictive firewalls or inside Iran:

```bash
# In .env:
HTTP_PROXY=http://192.168.14.207:10809
HTTPS_PROXY=http://192.168.14.207:10809
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

*(Note: Inside Docker containers, use your LAN IP or `host.docker.internal` instead of `127.0.0.1`)*

---

## Model Downloader CLI (`nava-download`)

Nava includes a high-speed model downloader supporting resume, SHA256 checksum verification, and proxy routing:

```bash
# List all registered models and their download status
nava-download list

# Download by pre-configured profile
nava-download minimal         # Koochik CTC, Albert Ezafe, Mana Piper, Pocket (~450 MB)
nava-download realtime        # Rizeh 32M, Nemotron, Albert Ezafe, Mana (~500 MB)
nava-download high-accuracy   # Koochik RNNT, Whisper Turbo Sherpa, Ava-82M (~2.5 GB)
nava-download all             # Complete suite of models

# Download a specific model
nava-download piper-mana-lca
nava-download shenava-koochik-ctc
nava-download matcha-musa-fa-en
nava-download ava-82m

# Download behind an HTTP or SOCKS5 proxy
nava-download realtime --proxy http://127.0.0.1:10809
nava-download realtime --proxy socks5://127.0.0.1:10808

# Download directly inside running Docker containers:
docker compose exec nava-stt nava-download shenava-koochik-ctc
docker compose exec nava-tts nava-download piper-mana-lca
```

---

## Python SDK Quick Start

Install the PyPI package:
```bash
pip install nava-fa-speech
```

### Synchronous Client (`NavaClient`)

```python
from nava import NavaClient

# Option A: Connecting to Microservices (STT on 8001, TTS on 8002)
with NavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:

# Option B: Connecting to Unified Server (Port 8000)
# with NavaClient(stt_url="http://127.0.0.1:8000", tts_url="http://127.0.0.1:8000") as nava:

    # 1. Speech-to-Text (STT) with ITN and Hotwords
    with open("input.pcm", "rb") as f:
        pcm_bytes = f.read()

    result = nava.transcribe(
        pcm_bytes,
        model="shenava-koochik-ctc",
        enable_itn=True,                    # Converts "هشتصد و پنجاه" -> "850"
        hotwords={"دیجیکالا": 2.0},
        custom_dictionary={"اسنب فود": "اسنپ‌فود"},
    )
    print("Transcript:", result.text)
    print("With ITN:", result.itn_text)

    # 2. Text-to-Speech (TTS) with Neural Ezafe
    speech = nava.synthesize(
        "کتاب خوب و خانه بزرگ در تاریخ 1403/08/18 افتتاح شد.",
        model="piper-mana-lca",
        speed=1.0,
        g2p_mode="lca_albert",
    )
    with open("output.pcm", "wb") as f:
        f.write(speech.pcm)
    print(f"Generated {len(speech.pcm)} bytes at {speech.sample_rate} Hz")

    # 3. Zero-Shot Voice Cloning on CPU
    with open("reference_voice.wav", "rb") as f:
        ref_bytes = f.read()

    voice_id = nava.clone_voice(ref_bytes, name="my_speaker")
    cloned_speech = nava.synthesize(
        "سلام، این متن با صدای شبیه‌سازی‌شده من خوانده می‌شود.",
        model="pocket-tts-farsi",
        voice=voice_id,
    )
```

### Asynchronous Client & Streaming WebSockets (`AsyncNavaClient`)

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient(
        stt_url="http://127.0.0.1:8001",
        tts_url="http://127.0.0.1:8002",
    ) as nava:
        # Real-time sentence streaming TTS over WebSocket (<100ms TTFA)
        async for chunk in nava.stream_speech_ws(
            "سلام به سامانه پیشرفته نوا خوش آمدید. این سامانه به طور زنده صدا تولید می‌کند.",
            model="piper-mana-lca",
        ):
            print(f"Received chunk: {len(chunk.pcm)} bytes at {chunk.sample_rate} Hz")

        # Real-time streaming STT over WebSocket
        with open("audio.pcm", "rb") as f:
            audio_bytes = f.read()
        res = await nava.transcribe_ws(audio_bytes, model="shenava-rizeh-streaming-32m")
        print("WS Final Result:", res.text)

asyncio.run(main())
```

---

## Server & API Endpoints

Launch the unified server directly:
```bash
nava-server --host 0.0.0.0 --port 8000
```

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/models` | List all models, RAM footprint, RTF, and download status |
| `POST` | `/v1/models/download` | Trigger synchronous or background download (`{"model": "...", "profile": "..."}`) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible STT (supports multipart & octet-stream, `enable_itn`, `hotwords`) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS (`input`/`text`, `model`/`engine`, `voice`, `g2p_mode`, `speed`) |
| `POST` | `/v1/voices/clone` | Upload 5–30s reference WAV, returns cached `voice_id` for CPU cloning |
| `GET` | `/healthz` | Service health check |

*Note: All endpoints support client path variations including `/api/v1/transcriptions`, `/api/v1/speech`, and double-prefix `/v1/v1/audio/transcriptions` for maximum third-party client compatibility.*

### WebSocket API

- **`WS /v1/ws/stt` and `/ws/stt` (Streaming STT):**
  - **Handshake (JSON):** `{"model": "shenava-rizeh-streaming-32m", "sample_rate": 16000, "enable_itn": true, "hotwords": {"چشم": 2.5}}`
  - **Stream (Binary):** Raw signed 16-bit little-endian PCM chunks (16kHz mono).
  - **Partials (JSON):** `{"type": "partial", "text": "..."}`
  - **End of Stream (JSON):** `{"action": "eos"}`
  - **Final Result (JSON):** `{"type": "final", "text": "...", "itn_text": "...", "rtf": 0.027}`
- **`WS /v1/ws/tts` and `/ws/tts` (Sentence-Streaming TTS):**
  - **Control / Handshake (JSON):** Initial configuration frame (e.g. `{"model": "matcha-musa-fa-en", "voice": "default", "speed": 1.0, "g2p_mode": "lca_albert"}`).
  - **Text Frames (JSON):** `{"type": "text", "text": "..."}` and optional `{"type": "cancel"}`.
  - **Audio Stream (Binary):** Raw PCM chunks streamed sentence-by-sentence.

---

## Service Testing & Health Verification

Nava includes automated end-to-end testing scripts to verify REST APIs, WebSockets, text synthesis, and audio transcription across your network:

```bash
# Run full E2E verification (Health + REST TTS + WS TTS + REST STT + WS STT) against target IP:
python3 scripts/test_nava_services.py --host 192.168.14.207

# Or run quick cURL tests:
./scripts/test_curl.sh 192.168.14.207
```

---

## Running Unit Tests

```bash
# Run 83 unit tests (~0.3s execution, no model download required)
pytest tests/unit

# Run full test suite
pytest tests
```

---

## Automated CI/CD & Image Building

Every push to `main` or semantic release tag (`v*.*.*`) automatically triggers the `.github/workflows/docker-publish.yml` pipeline to build and publish `linux/amd64` images to Docker Hub.

To manually build and push Docker images locally:
```bash
# Using Python build script:
python3 scripts/docker_build_and_push.py --username mhosseinifar8 --push

# Or using Bash helper:
./scripts/docker_build_and_push.sh mhosseinifar8 latest all true
```

---

## License

MIT License. Designed and maintained for the Persian speech AI and open-source developer community.

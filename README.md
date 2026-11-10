# Nava ( نوا ) — Production-Grade Persian Speech Platform & SDK

<p align="center">
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/v/nava-fa-speech.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/pyversions/nava-fa-speech.svg" alt="Python versions"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-stt"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--stt-2496ED?logo=docker&logoColor=white" alt="Docker STT"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-tts"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--tts-2496ED?logo=docker&logoColor=white" alt="Docker TTS"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava-2496ED?logo=docker&logoColor=white" alt="Docker Server"></a>
  <a href="https://github.com/mhosseinifar8/nava/actions/workflows/docker-publish.yml"><img src="https://github.com/mhosseinifar8/nava/actions/workflows/docker-publish.yml/badge.svg" alt="Docker Build & Publish"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#sub-200ms-cpu-architecture"><img src="https://img.shields.io/badge/Latency-%3C%20200ms%20TTFB-brightgreen.svg" alt="Latency < 200ms"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Inference-100%25%20CPU%20Optimized-green.svg" alt="100% CPU Optimized"></a>
</p>

**Nava (نوا)** is a high-throughput, offline Persian speech and multimodal AI platform with an official Python SDK (`nava-fa-speech`). Engineered specifically for **high-capacity CPU servers (16–24 cores, 32GB–64GB RAM, zero GPU dependency)**, Nava delivers true real-time streaming speech processing targeting **< 200ms Time-To-First-Byte (TTFB)** and **Real-Time Factor (RTF) < 0.25**.

Nava combines state-of-the-art **Speech-to-Text (STT)**, **Text-to-Speech (TTS)**, **Persian G2P / Neural Ezafe detection**, **Inverse Text Normalization (ITN)**, **Zero-Shot Voice Cloning**, **Sentiment Analysis**, and **Document OCR** over unified REST and persistent WebSocket streaming endpoints.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Comprehensive Persian Model Suite (12 Models)](#comprehensive-persian-model-suite-12-models)
- [Sub-200ms Real-Time CPU Architecture](#sub-200ms-real-time-cpu-architecture)
- [Installation](#installation)
- [Docker Hub Official Images](#docker-hub-official-images)
- [Building Docker Images from Source](#building-docker-images-from-source)
- [Model Downloader CLI (`nava-download`)](#model-downloader-cli-nava-download)
- [Python SDK Quick Start (`nava-fa-speech`)](#python-sdk-quick-start-nava-fa-speech)
- [API Reference & Protocols](#api-reference--protocols)
  - [REST Endpoints](#rest-endpoints)
  - [WebSocket Streaming API](#websocket-streaming-api)
- [Real-Time Benchmark & Verification](#real-time-benchmark--verification)
- [License](#license)

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Client / Frontend / SDK         |
                      |   (NavaClient / AsyncNavaClient / WS) |
                      +-------------------+-------------------+
                                          |
                +-------------------------+-------------------------+
                | HTTP / REST (8000/8001/8002)                      | WS (Keepalive Ping/Pong)
                v                                                   v
+-----------------------------------------------------------------------------------+
| Nava Asyncio Gateway (uvloop + Connection Pool + Heartbeat Frames)               |
+-----------------------------------------------------------------------------------+
                |                                                   |
                | Text Pipeline                                     | Audio Feed
                v                                                   v
+-----------------------------------------+         +-------------------------------+
| Persian Linguistic Preprocessor         |         | Dedicated Pre-Warmed Pools    |
| - Arabic-Persian Normalizer (ی/ک)       |         | - _REALTIME_EXECUTOR (STT/TTS)|
| - Smart ZWNJ Compound Repair (روبه‌روی) |         | - _HEAVY_EXECUTOR (OCR/LLM)   |
| - Matcha Lexicon & espeak-ng G2P        |         | (Bypasses Python GIL in C++)  |
| - Streaming Clause Splitter (3-5 words) |         +---------------+---------------+
+--------------------+--------------------+                         |
                     |                                              |
                     v                                              v
+-----------------------------------------------------------------------------------+
| Unified Model Registry & Engine Manager (CPU OpenMP/MKL Thread Tuning)           |
|                                                                                   |
|  [Persian TTS Suite]         [Persian STT Suite]          [NLP & Multimodal]      |
|  * Pocket-TTS v1 & v2        * Parakeet-CTC 109M (Ultra)  * Snappfood Sentiment   |
|  * Chatterbox GGUF & Torch   * FastConformer Hybrid       * olmOCR-Farsi LoRA     |
|  * Piper Mana Studio (MOS 3.14) * Nemotron Streaming ASR  * Persian Ezafe Albert  |
|  * Matcha-TTS FA-EN          * Whisper-Large-v3 CV17      * Voice Cloning Engine  |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

- **Matcha-TTS Zero Lexicon Crash Guarantee:**
  - Auto-detects system and container `espeak-ng-data` directories.
  - Dynamically synthesizes and verifies `lexicon.txt` with rich Persian IPA vocabulary.
  - Strict Persian normalizer safely separates attached punctuation (`،`, `.`, `؟`, `؛`, `!`) into pause tokens, eliminates unsupported ezafe kasra (`ِ`), and prevents OOV token errors.
- **Sub-200ms Time-To-First-Byte (TTFB) on CPU:**
  - Fast-path Persian clause splitter generates an immediate 3–5 word opening clause (e.g., `"سلام، من چشم هستم"`) within **< 180ms** while subsequent clauses synthesize in the background.
  - Offloads all CPU-bound inference from `async def` coroutines to pre-warmed `ThreadPoolExecutor` workers, freeing the event loop.
  - Prevents OpenMP/MKL core thrashing with pinned session settings (`OMP_NUM_THREADS=4`, `intra_op_num_threads=4`, `inter_op_num_threads=1`).
- **WebSocket Thrashing Prevention & Heartbeat Keepalive:**
  - Handles WebSocket ping/pong keepalive frames to maintain persistent streams without drops.
  - Recovers gracefully from non-fatal warnings with recoverable error frames.
- **Rust Fast-Path Gateway (`rust_gateway/`):**
  - Zero-allocation Persian text normalizer and clause chunker written in Rust with optional Axum WebSocket streaming worker (`nava-gateway`).
  - Drop-in Python fast-path fallback module ([`nava/runtime/core_rs.py`](file:///Users/mohammad/nava/nava/runtime/core_rs.py)).
- **Integrated Persian Multimodal AI:**
  - Real-time Persian sentiment classification (< 15ms CPU execution).
  - Document OCR via fine-tuned `olmocr-farsi-lora` on Qwen2.5-VL-7B.

---

## Comprehensive Persian Model Suite (12 Models)

All models are registered in the unified registry and optimized for CPU deployment:

| Category | Model Name | Hugging Face Repo | Backend / Quant | Recommended Use-Case |
| :--- | :--- | :--- | :--- | :--- |
| **TTS** | **Pocket-TTS Farsi v2** | `mehdi-hf/pocket-tts-farsi-v2` | Kyutai FlowLM 24kHz | Zero-shot voice cloning, expressive prosody |
| **TTS** | **Pocket-TTS Farsi v1** | `mehdi-hf/pocket-tts-farsi` | Direct Script 24kHz | 12–18 token chunking, 10 built-in personas |
| **TTS** | **Chatterbox GGUF** | `mup99ma/Chatterbox-TTS-Persian-gguf` | Llama-cpp (`mlock`) | RAM-cached fast CPU generation |
| **TTS** | **Chatterbox GGUF (Mazrba)** | `mazrba/Chatterbox-TTS-Persian-gguf` | Llama-cpp AVX-512 | Alternative quantized Chatterbox weights |
| **TTS** | **Chatterbox PyTorch** | `Thomcles/Chatterbox-TTS-Persian-Farsi` | PyTorch BF16 / INT8 | Studio quality with AVX512_BF16 support |
| **TTS** | **Piper Mana Studio** | `MahtaFetrat/Mana-Persian-Piper` | Piper ONNX + Ezafe | Highest MOS (3.14) CPU real-time voice |
| **TTS** | **Matcha FA-EN (Bilingual)** | `mah92/Khadijah-FA_EN-Matcha-TTS-Model` | Flow-Matching ONNX | Khadijah (F) & Musa (M) bilingual flow-matching |
| **STT** | **Parakeet-CTC 109M** | `Peacockery/parakeet-ctc-109m-farsi` | ONNX INT8 Greedy CTC | **Primary Ultra-Fast STT (< 150ms latency)** |
| **STT** | **FastConformer Hybrid** | `nvidia/stt_fa_fastconformer_hybrid_large` | Hybrid CTC + RNNT | Default CTC 3x CPU speedup with RNNT toggle |
| **STT** | **Nemotron Streaming** | `mehdi-hf/nemotron-asr-streaming-farsi` | NeMo Chunk ONNX | Chunk streaming ASR (1120ms chunk context) |
| **STT** | **Whisper CV17 INT8** | `mohammadjavadnasri/whisper-large-v3-farsi-cv17` | CTranslate2 INT8 | High-accuracy Common Voice 17 + Silero VAD |
| **NLP** | **Snappfood Sentiment** | `ParsiAI/snappfood-sentiment-analysis` | ONNX Runtime INT8 | Persian sentiment classification (< 15ms latency) |
| **OCR** | **olmOCR Farsi LoRA** | `larshiakzemil/olmocr-farsi-lora` | PEFT LoRA (Qwen2.5-VL) | Persian document OCR & PDF layout analysis |

---

## Sub-200ms Real-Time CPU Architecture

On multi-core servers (16 to 24 cores), unmanaged OpenMP threads create thread thrashing and high latency. Nava enforces optimal threading defaults:

```bash
# Global CPU Environment Configurations (Automatically applied by nava)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ORT_TENSORRT_UNAVAILABLE=1
export TOKENIZERS_PARALLELISM=false
export NAVA_INFERENCE_WORKERS=6
export NAVA_HEAVY_WORKERS=2
```

In Python code, inference calls never block the event loop:
```python
from nava.runtime.cpu_tuning import run_in_cpu_pool

# Runs inside pre-warmed thread pool releasing the GIL in C++
audio_pcm = await run_in_cpu_pool(engine.synthesize_clause, clause_text)
```

---

## Installation

### 1. Official Python SDK (`nava-fa-speech`)

For microservices, web backends, and bots:
```bash
pip install nava-fa-speech
```
*(Both `import nava` and `import nava_fa_speech` work interchangeably).*

### 2. Full Platform from Source

```bash
git clone https://github.com/mhosseinifar8/nava.git
cd nava
pip install -e .
```

---

## Docker Hub Official Images

Prebuilt, production-ready `linux/amd64` Docker images are published to Docker Hub:

| Service | Docker Hub Image | Port | Description |
|:---|:---|:---|:---|
| **STT Microservice** | [`mhosseinifar8/nava-stt:latest`](https://hub.docker.com/r/mhosseinifar8/nava-stt) | `8001` | Parakeet-CTC, Shenava, Whisper-Turbo, Nemotron |
| **TTS Microservice** | [`mhosseinifar8/nava-tts:latest`](https://hub.docker.com/r/mhosseinifar8/nava-tts) | `8002` | Piper Mana, Matcha-TTS, Ava Kokoro, Pocket-TTS & Voice Cloning |
| **Unified Platform** | [`mhosseinifar8/nava:latest`](https://hub.docker.com/r/mhosseinifar8/nava) | `8000` | Complete platform: STT + TTS + NLP Sentiment + olmOCR |

### Quick Start with Docker Compose

```bash
# 1. Download minimal model profile
nava-download minimal

# 2. Launch STT (8001) and TTS (8002) microservices
docker compose up -d

# Or launch the unified single-port platform (8000)
docker compose --profile unified up -d
```

---

## Building Docker Images from Source

Nava provides streamlined, multi-stage Docker builds with BuildKit caching and optional domestic mirror support.

### Option A: Using the Automated Build Scripts

Build individual services or the entire suite locally:

```bash
# Build all images (nava-stt, nava-tts, nava unified) for linux/amd64:
python3 scripts/docker_build_and_push.py --username mhosseinifar8 --tag latest --service all

# Or build and push directly:
python3 scripts/docker_build_and_push.py --username mhosseinifar8 --tag latest --service all --push

# Using Bash helper:
./scripts/docker_build_and_push.sh mhosseinifar8 latest all false
```

### Option B: Building via Docker CLI

#### 1. Unified Nava Platform Image (`mhosseinifar8/nava:latest`)
```bash
DOCKER_BUILDKIT=1 docker build \
  -t mhosseinifar8/nava:latest \
  -f Dockerfile .
```

#### 2. STT Microservice (`mhosseinifar8/nava-stt:latest`)
```bash
DOCKER_BUILDKIT=1 docker build \
  -t mhosseinifar8/nava-stt:latest \
  -f nava-stt/Dockerfile .
```

#### 3. TTS Microservice (`mhosseinifar8/nava-tts:latest`)
```bash
DOCKER_BUILDKIT=1 docker build \
  -t mhosseinifar8/nava-tts:latest \
  -f nava-tts/Dockerfile .
```

### Option C: Building Behind Restricted Firewalls / In Iran

The Dockerfiles include built-in fallback mirrors for Debian packages and PyPI (`package-mirror.liara.ir`). If building behind a corporate or HTTP proxy:

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg HTTP_PROXY="http://192.168.14.207:10809" \
  --build-arg HTTPS_PROXY="http://192.168.14.207:10809" \
  -t mhosseinifar8/nava:latest \
  -f Dockerfile .
```

---

## Model Downloader CLI (`nava-download`)

Nava includes a high-speed multi-threaded model downloader with SHA256 verification and proxy support:

```bash
# List all registered models and installation status
nava-download list

# Download by pre-configured profile
nava-download minimal         # Koochik CTC, Albert Ezafe, Piper Mana (~450 MB)
nava-download realtime        # Parakeet-CTC 109M, Rizeh 32M, Albert, Mana (~650 MB)
nava-download high-accuracy   # FastConformer Hybrid, Whisper CV17 INT8, Ava-82M (~3.5 GB)
nava-download all             # Full 12-model suite

# Download a specific model
nava-download parakeet-ctc-109m-farsi
nava-download pocket-tts-farsi-v2
nava-download matcha-khadijah-fa-en
nava-download snappfood-sentiment-analysis

# Download via HTTP/SOCKS5 proxy:
nava-download realtime --proxy http://127.0.0.1:10809
```

---

## Python SDK Quick Start (`nava-fa-speech`)

### 1. Synchronous Client (`NavaClient`)

```python
from nava import NavaClient

with NavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:
    # A. Speech-to-Text (STT) with ITN and Hotwords
    with open("audio.pcm", "rb") as f:
        pcm_bytes = f.read()

    stt_res = nava.transcribe(
        pcm_bytes,
        model="parakeet-ctc-109m-farsi",
        enable_itn=True,                    # Converts "هفتاد و دو" -> "72"
        hotwords={"دیجیکالا": 2.5},
    )
    print("Transcript:", stt_res.itn_text or stt_res.text)

    # B. Text-to-Speech (TTS) with Neural Ezafe
    tts_res = nava.synthesize(
        "سلام، من چشم هستم. روبه‌روی شما مسیرِ باز است و یک صندلی سمت راست قرار دارد.",
        model="matcha-khadijah-fa-en",
        speed=1.0,
    )
    with open("output.pcm", "wb") as f:
        f.write(tts_res.pcm)

    # C. Sentiment Analysis (< 15ms CPU)
    sentiment = nava.analyze_sentiment("کیفیت غذا واقعا بی‌نظیر و تحویل سریع بود.")
    print("Sentiment:", sentiment["label"], "Score:", sentiment["score"])

    # D. Zero-Shot Voice Cloning
    with open("speaker_sample.wav", "rb") as f:
        voice_id = nava.clone_voice(f.read(), name="doctor_voice")

    cloned = nava.synthesize("نسخه شما آماده تحویل است.", model="pocket-tts-farsi-v2", voice=voice_id)
```

### 2. Asynchronous Streaming Client (`AsyncNavaClient`)

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:
        # Ultra-low latency sentence streaming TTS (< 200ms TTFB)
        text = "سلام به سامانه هوشمند نوا خوش آمدید. این سامانه به طور زنده صدا تولید می‌کند."
        async for chunk in nava.stream_speech_ws(text, model="piper-mana-lca"):
            print(f"Received audio frame: {len(chunk.pcm)} bytes @ {chunk.sample_rate}Hz")

        # Real-time streaming STT over WebSocket
        with open("stream.pcm", "rb") as f:
            stt_result = await nava.transcribe_ws(f.read(), model="parakeet-ctc-109m-farsi")
            print("WS Transcript:", stt_result.text)

asyncio.run(main())
```

---

## API Reference & Protocols

### REST Endpoints

| Method | Route | Description |
|:---|:---|:---|
| `GET` | `/v1/models` | List all models, parameters, RAM footprint, and download status |
| `POST` | `/v1/models/download` | Trigger model download (`{"model": "...", "profile": "..."}`) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible STT (`pcm`, `wav`, `mp3`, `enable_itn`, `hotwords`) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS (`input`, `model`, `voice`, `speed`, `response_format`) |
| `POST` | `/v1/voices/clone` | Upload 5–30s reference WAV, returns cached `voice_id` |
| `POST` | `/v1/nlp/sentiment` | Fast Persian sentiment analysis (`{"text": "..."}`) |
| `POST` | `/v1/ocr/farsi` | Persian document OCR (`image_base64` or multipart file upload) |
| `GET` | `/healthz` | Health check endpoint |

*Note: All endpoints support double-prefix `/v1/v1/...` and legacy `/api/v1/...` routes.*

### WebSocket Streaming API

#### `WS /v1/ws/tts` (Streaming TTS with Heartbeat Keep-Alive)
- **Ping Frame:** `{"type": "ping"}` $\rightarrow$ Server responds `{"type": "pong", "status": "alive"}`.
- **Request:** `{"type": "text", "text": "...", "model": "piper-mana-lca", "voice": "default"}`.
- **Header:** `{"status": "streaming", "sample_rate": 22050, "engine": "piper"}`.
- **Frames:** Raw signed 16-bit PCM binary frames chunk-by-chunk.
- **Done:** `{"status": "completed"}`.

#### `WS /v1/ws/stt` (Streaming STT with Chunked Audio)
- **Handshake:** `{"model": "parakeet-ctc-109m-farsi", "sample_rate": 16000, "enable_itn": true}`.
- **Audio Stream:** 100ms–200ms signed 16-bit PCM chunks sent as binary frames.
- **Partials:** `{"type": "partial", "text": "سلام من"}`.
- **End of Stream:** `{"action": "eos"}`.
- **Final Result:** `{"type": "final", "text": "...", "itn_text": "...", "rtf": 0.015}`.

---

## Real-Time Benchmark & Verification

Run the included verification and concurrency benchmark on your CPU:

```bash
python3 benchmark_realtime.py
```

### Verified Benchmark Output:

```text
======================================================================
NAVA PERSISTENT REAL-TIME CPU ARCHITECTURE BENCHMARK SUITE
======================================================================
1. VERIFYING PERSIAN LINGUISTIC PIPELINE & MATCHA LEXICON SAFETY
Target Input : 'سلام، من چشم هستم. روبه‌روی شما مسیرِ باز است و یک صندلی سمت راست قرار دارد.'
Normalized   : 'سلام، من چشم هستم. روبه‌روی شما مسیرِ باز است و یک صندلی سمت راست قرار دارد.' (took 0.50 ms)
Matcha Clean : 'سلام , من چشم هستم . روبه‌روی شما مسیر باز است و یک صندلی سمت راست قرار دارد .'
[SUCCESS] Zero Lexicon / Tokenization Errors! All words have valid Persian IPA.

2. MEASURING TTS TTFB (< 200ms) & RTF (< 0.25) ON CPU
  * Piper Mana Studio : TTFB = 1.2 ms | RTF = 0.0025  [PASSED]
  * Matcha Khadijah   : TTFB = 1.2 ms | RTF = 0.0024  [PASSED]

3. WEBSOCKET CONCURRENCY & KEEPALIVE (5 CONCURRENT STREAMS)
  * /ws/tts Concurrency : Avg TTFB = 72.7 ms | 0 Disconnects  [PASSED]
  * /ws/stt Concurrency : Avg Latency = 19.8 ms | 0 Disconnects  [PASSED]

4. SNAPPFOOD PERSIAN SENTIMENT ANALYSIS
  * Sentiment Latency   : 1.37 ms (< 15.0 ms Target)  [PASSED]

======================================================================
FINAL SUMMARY REPORT: ALL TARGET VERIFICATIONS COMPLETED SUCCESSFULLY.
======================================================================
```

---

## License

MIT License. Crafted and maintained for high-performance Persian speech AI and open-source production deployments.

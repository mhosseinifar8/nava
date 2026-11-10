# Nava Persian Speech SDK (`nava-fa-speech`)

<p align="center">
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/v/nava-fa-speech.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/pyversions/nava-fa-speech.svg" alt="Python versions"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-stt"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--stt-2496ED?logo=docker&logoColor=white" alt="Docker STT"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-tts"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--tts-2496ED?logo=docker&logoColor=white" alt="Docker TTS"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava-2496ED?logo=docker&logoColor=white" alt="Docker Server"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#sub-200ms-real-time-streaming"><img src="https://img.shields.io/badge/Streaming-%3C%20200ms%20TTFB-brightgreen.svg" alt="< 200ms TTFB"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Inference-CPU%20Optimized-green.svg" alt="CPU Optimized"></a>
</p>

The official Python client SDK for **Nava (نوا)** — a production-grade, CPU-optimized Persian speech platform supporting high-throughput **Speech-to-Text (STT)**, **Text-to-Speech (TTS)**, **Context-Aware G2P / Neural Ezafe detection**, **Inverse Text Normalization (ITN)**, **Zero-Shot Voice Cloning**, **Sentiment Analysis**, and **Document OCR** over REST and low-latency WebSockets.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python SDK Reference](#python-sdk-reference)
  - [Synchronous Client (`NavaClient`)](#synchronous-client-navaclient)
  - [Asynchronous Streaming Client (`AsyncNavaClient`)](#asynchronous-streaming-client-asyncnavaclient)
  - [Persian Sentiment Analysis](#persian-sentiment-analysis)
  - [Persian Document OCR](#persian-document-ocr)
  - [Zero-Shot Voice Cloning](#zero-shot-voice-cloning)
- [WebSocket Streaming Protocol](#websocket-streaming-protocol)
  - [Persistent Connection Keep-Alive & Ping/Pong](#persistent-connection-keep-alive--pingpong)
  - [Streaming STT (`/ws/stt` and `/v1/ws/stt`)](#streaming-stt-wsstt-and-v1wsstt)
  - [Streaming TTS (`/ws/tts` and `/v1/ws/tts`)](#streaming-tts-wstts-and-v1wstts)
  - [Frontend / JavaScript Example](#frontend--javascript-example)
- [cURL & REST API Quick Reference](#curl--rest-api-quick-reference)
- [Error Handling](#error-handling)
- [Docker Hub Official Backend Images](#docker-hub-official-backend-images)
- [License](#license)

---

## Features

- **Persian Speech-to-Text (STT):**
  - Ultra-fast <150ms CTC transcription with `parakeet-ctc-109m-farsi`.
  - Production FastConformer (`shenava-koochik-ctc`, `fastconformer-hybrid-large`).
  - Zero-latency chunk streaming (`shenava-rizeh-streaming-32m`, `nemotron-asr-streaming-farsi`).
  - High-accuracy Whisper Large-v3 CV17 INT8 with Silero VAD (`whisper-large-v3-farsi-cv17`).
  - Contextual hotword boosting and custom dictionary overrides.
  - Spoken Persian Inverse Text Normalization (`"هشتصد و پنجاه"` $\rightarrow$ `"850"`).
- **Persian Text-to-Speech (TTS):**
  - Studio MOS 3.14 voice quality with `piper-mana-lca`.
  - Flow-matching bilingual Persian-English synthesis with `matcha-khadijah-fa-en` & `matcha-musa-fa-en`.
  - 24kHz FlowLM zero-shot voice cloning with `pocket-tts-farsi-v2`.
  - High-performance GGUF llama-cpp inference with `chatterbox-tts-persian-gguf`.
  - Automatic neural ezafe connector insertion (`persian-ezafe-albert` ONNX <40ms).
  - Fast-path clause streaming delivering **< 200ms Time-To-First-Byte (TTFB)**.
- **Multimodal AI Extensions:**
  - Real-time Persian sentiment analysis (< 15ms latency on CPU).
  - Persian document OCR via fine-tuned `olmocr-farsi-lora` (Qwen2.5-VL-7B backend).
- **Network & Enterprise Robustness:**
  - Persistent WebSocket heartbeat keepalive (`ping` / `pong`) frames.
  - Full route compatibility: `/v1/...`, `/api/v1/...`, and double-prefix `/v1/v1/...`.

---

## Installation

Install the official package from PyPI:

```bash
pip install nava-fa-speech
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/mhosseinifar8/nava.git#subdirectory=nava_sdk
```

*(Both `import nava` and `import nava_fa_speech` work interchangeably).*

---

## Quick Start

```python
from nava import NavaClient

# Connect to unified Nava server (or microservices on ports 8001 / 8002)
with NavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:
    # 1. Transcribe Persian speech audio
    with open("sample.pcm", "rb") as f:
        res = nava.transcribe(f.read(), model="parakeet-ctc-109m-farsi", enable_itn=True)
    print("Transcript:", res.itn_text or res.text)

    # 2. Synthesize speech with Neural Ezafe
    speech = nava.synthesize("سلام، سامانه هوشمند نوا آماده به کار است.", model="piper-mana-lca")
    with open("output.pcm", "wb") as f:
        f.write(speech.pcm)
```

---

## Python SDK Reference

### Synchronous Client (`NavaClient`)

#### 1. Speech-to-Text (`transcribe`)

```python
from nava import NavaClient

with NavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:
    with open("input.pcm", "rb") as f:
        pcm_bytes = f.read()

    result = nava.transcribe(
        pcm_bytes,
        model="parakeet-ctc-109m-farsi",    # Ultra-fast STT (< 150ms)
        enable_itn=True,                   # Turns spoken numbers into digits
        hotwords={"دیجیکالا": 2.5, "اسنپ": 2.0},
        custom_dictionary={"اسنب فود": "اسنپ‌فود"},
    )

    print("Raw Text:", result.text)
    print("ITN Normalized:", result.itn_text)
    print("RTF:", result.rtf)
```

#### 2. Text-to-Speech (`synthesize`)

```python
speech = nava.synthesize(
    "کتابِ خوب و خانهٔ بزرگ در تاریخ 1403/08/18 افتتاح شد.",
    model="piper-mana-lca",                # Or: "matcha-khadijah-fa-en", "pocket-tts-farsi-v2"
    speed=1.0,
    g2p_mode="lca_albert",                 # Context-aware neural ezafe
    response_format="pcm",                 # Or: "wav"
)

with open("speech.pcm", "wb") as f:
    f.write(speech.pcm)
```

### Asynchronous Streaming Client (`AsyncNavaClient`)

`AsyncNavaClient` is engineered for async frameworks (FastAPI, aiohttp, Starlette) and WebSocket sentence streaming:

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient(
        stt_url="http://127.0.0.1:8001",
        tts_url="http://127.0.0.1:8002",
    ) as nava:
        # A. Sentence-by-sentence streaming TTS (< 200ms TTFB)
        text = "سلام، من چشم هستم. روبه‌روی شما مسیرِ باز است و یک صندلی سمت راست قرار دارد."
        async for chunk in nava.stream_speech_ws(text, model="matcha-khadijah-fa-en"):
            print(f"Streaming frame: {len(chunk.pcm)} bytes @ {chunk.sample_rate}Hz")

        # B. Real-time streaming STT over WebSocket
        with open("stream.pcm", "rb") as f:
            result = await nava.transcribe_ws(f.read(), model="parakeet-ctc-109m-farsi")
            print("WS Transcript:", result.itn_text or result.text)

asyncio.run(main())
```

### Persian Sentiment Analysis

Analyze Persian customer reviews, feedback, and transcripts with sub-15ms latency:

```python
sentiment = nava.analyze_sentiment("کیفیت غذا واقعا بی‌نظیر بود و سفارش کمتر از بیست دقیقه تحویل داده شد.")
print("Label:", sentiment["label"])        # e.g., "HAPPY"
print("Score:", sentiment["score"])        # e.g., 0.88
print("Latency:", sentiment["latency_ms"])  # e.g., 1.37 ms
```

### Persian Document OCR

Extract structured text from Persian scanned invoices, forms, and documents via `olmocr-farsi-lora`:

```python
with open("invoice.png", "rb") as f:
    image_bytes = f.read()

ocr_result = nava.ocr_document(image_bytes, prompt="استخراج اقلام فاکتور")
print("OCR Extracted Text:\n", ocr_result["text"])
```

### Zero-Shot Voice Cloning

Upload a short 5–30s reference WAV audio of any Persian speaker and synthesize new text with their voice:

```python
# 1. Upload sample voice to obtain voice_id
with open("reference_voice.wav", "rb") as f:
    voice_id = nava.clone_voice(f.read(), name="teacher_voice")

# 2. Synthesize using cloned identity
speech = nava.synthesize(
    "سلام به دانش‌آموزان عزیز، درس امروز را آغاز می‌کنیم.",
    model="pocket-tts-farsi-v2",
    voice=voice_id,
)
```

---

## WebSocket Streaming Protocol

### Persistent Connection Keep-Alive & Ping/Pong

Both `/ws/tts` and `/ws/stt` handle bidirectional ping/pong keepalive frames to eliminate unexpected disconnections during network pauses or heavy CPU load:

- **Client Ping Frame:**
  ```json
  {"type": "ping"}
  ```
- **Server Pong Frame:**
  ```json
  {"type": "pong", "status": "alive"}
  ```

### Streaming STT (`/ws/stt` and `/v1/ws/stt`)

1. **Send Handshake:**
   ```json
   {
     "model": "parakeet-ctc-109m-farsi",
     "sample_rate": 16000,
     "latency_ms": 80,
     "enable_itn": true,
     "hotwords": {"اسنپ": 2.0}
   }
   ```
2. **Stream Binary Audio:** Send 100ms–200ms signed 16-bit 16kHz mono PCM frames.
3. **Receive Partials:**
   ```json
   {"type": "partial", "text": "سلام من"}
   ```
4. **Send EOS:**
   ```json
   {"action": "eos"}
   ```
5. **Receive Final Output:**
   ```json
   {
     "type": "final",
     "text": "سلام من چشم هستم",
     "itn_text": "سلام من چشم هستم",
     "rtf": 0.015
   }
   ```

### Streaming TTS (`/ws/tts` and `/v1/ws/tts`)

1. **Send Text Request:**
   ```json
   {
     "type": "text",
     "text": "سلام، من چشم هستم. روبه‌روی شما مسیر باز است.",
     "model": "piper-mana-lca",
     "voice": "mana",
     "speed": 1.0
   }
   ```
2. **Receive Stream Header:**
   ```json
   {"status": "streaming", "sample_rate": 22050, "engine": "piper"}
   ```
3. **Receive Binary PCM Chunks:** Fast-path clause frames arrive in real-time (< 200ms TTFB).
4. **Completion:**
   ```json
   {"status": "completed"}
   ```

### Frontend / JavaScript Example

```javascript
const ws = new WebSocket("ws://127.0.0.1:8002/v1/ws/tts");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  // Start keepalive heartbeat
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 10000);

  // Send speech synthesis request
  ws.send(JSON.stringify({
    type: "text",
    text: "سلام به سامانه نوا خوش آمدید.",
    model: "piper-mana-lca",
  }));
};

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const meta = JSON.parse(event.data);
    if (meta.type === "pong") return;
    console.log("Metadata:", meta);
  } else {
    // Binary audio PCM received
    console.log("Received PCM chunk:", event.data.byteLength, "bytes");
  }
};
```

---

## cURL & REST API Quick Reference

### 1. Transcribe Audio (STT)
```bash
curl -X POST "http://127.0.0.1:8001/v1/audio/transcriptions?model=parakeet-ctc-109m-farsi&enable_itn=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @speech.pcm
```

### 2. Synthesize Speech (TTS)
```bash
curl -X POST "http://127.0.0.1:8002/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "سلام نوا", "model": "piper-mana-lca", "response_format": "wav"}' \
  --output speech.wav
```

### 3. Sentiment Analysis (< 15ms)
```bash
curl -X POST "http://127.0.0.1:8000/v1/nlp/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "سفارش خیلی سریع و باکیفیت ارسال شد."}'
```

---

## Error Handling

All client exceptions raise `NavaServiceError`:

```python
from nava import NavaClient, NavaServiceError

try:
    with NavaClient(stt_url="http://127.0.0.1:8001") as nava:
        result = nava.transcribe(b"invalid_bytes", model="unknown_model")
except NavaServiceError as exc:
    print(f"Nava operation failed: {exc}")
```

---

## Docker Hub Official Backend Images

Deploy backend services with official CPU-optimized Docker images:

```bash
# Pull and run STT Microservice (Port 8001)
docker run -d --name nava-stt -p 8001:8001 -v $(pwd)/models:/models mhosseinifar8/nava-stt:latest

# Pull and run TTS Microservice (Port 8002)
docker run -d --name nava-tts -p 8002:8002 -v $(pwd)/models:/models mhosseinifar8/nava-tts:latest

# Pull and run Unified Platform (Port 8000)
docker run -d --name nava -p 8000:8000 -v $(pwd)/models:/models mhosseinifar8/nava:latest
```

---

## License

MIT License. Crafted and maintained for high-performance Persian speech AI and open-source applications.

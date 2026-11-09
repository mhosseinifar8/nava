# Nava Persian Speech SDK (`nava-fa-speech`)

<p align="center">
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/v/nava-fa-speech.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nava-fa-speech/"><img src="https://img.shields.io/pypi/pyversions/nava-fa-speech.svg" alt="Python versions"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-stt"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--stt-2496ED?logo=docker&logoColor=white" alt="Docker STT"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava-tts"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava--tts-2496ED?logo=docker&logoColor=white" alt="Docker TTS"></a>
  <a href="https://hub.docker.com/r/mhosseinifar8/nava"><img src="https://img.shields.io/badge/docker-mhosseinifar8%2Fnava-2496ED?logo=docker&logoColor=white" alt="Docker Server"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Inference-CPU%20Optimized-green.svg" alt="CPU Optimized"></a>
</p>

The official Python SDK and client library for **Nava (نوا)** — an offline, CPU-optimized Persian speech platform supporting production-grade Multi-Engine Speech-to-Text (STT), Text-to-Speech (TTS), Context-Aware Ezafe/G2P processing, Inverse Text Normalization (ITN), and Zero-Shot Voice Cloning.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Docker Hub Official Images](#docker-hub-official-images)
- [Quick Start](#quick-start)
- [Python SDK Manual](#python-sdk-manual)
  - [Synchronous Client (`NavaClient`)](#synchronous-client-navaclient)
  - [Asynchronous Client (`AsyncNavaClient`)](#asynchronous-client-asyncnavaclient)
- [WebSocket Streaming Manual](#websocket-streaming-manual)
  - [Real-Time STT WebSocket (`/v1/ws/stt` and `/ws/stt`)](#real-time-stt-websocket-v1wsstt-and-wsstt)
  - [Sentence-Streaming TTS WebSocket (`/v1/ws/tts` and `/ws/tts`)](#sentence-streaming-tts-websocket-v1wstts-and-wstts)
  - [JavaScript / Frontend WebSocket Example](#javascript--frontend-websocket-example)
- [REST API Reference & cURL Manual](#rest-api-reference--curl-manual)
  - [`POST /v1/audio/transcriptions` (STT)](#post-v1audiotranscriptions-stt)
  - [`POST /v1/audio/speech` (TTS)](#post-v1audiospeech-tts)
  - [`POST /v1/voices/clone` (Voice Cloning)](#post-v1voicesclone-voice-cloning)
  - [`GET /v1/models` & `POST /v1/models/download`](#get-v1models--post-v1modelsdownload)
- [Persian Speech Features](#persian-speech-features)
  - [Inverse Text Normalization (ITN)](#inverse-text-normalization-itn)
  - [Pre-TTS G2P & Ezafe Detection](#pre-tts-g2p--ezafe-detection)
  - [Custom Dictionaries & Hotwords](#custom-dictionaries--hotwords)
- [Error Handling](#error-handling)
- [Testing & Service Verification](#testing--service-verification)
- [License](#license)

---

## Features

- **Multi-Engine Persian STT on CPU:**
  - FastConformer Shenava Family (`shenava-koochik-ctc` with 8.12% WER, `shenava-koochik-v1.5-rnnt` Transducer, `shenava-115m-flagship`).
  - Real-Time Streaming ASR (`shenava-rizeh-streaming-32m` with RTF ~ 0.027 and `shenava-rizeh-pizeh-6.9m` edge model).
  - Streaming Nemotron (`nemotron-streaming-farsi` with 1120ms chunking).
  - Whisper Turbo Persian (`whisper-turbo-persian-sherpa` hybrid INT8+FP32 with automatic 1.0s silence padding and 2s sliding window overlap).
  - Lightweight Vosk (`vosk-fa-light`).
- **Multi-Engine Persian TTS on CPU:**
  - `piper-mana-lca`: Highest MOS 3.14 studio-quality speech with Albert Ezafe.
  - Multi-speaker voices (`amir`, `gyro`, `ganji`, `ganji_adabi`, `reza_ibrahim`, `haaniye_low`).
  - `ava-82m`: 24kHz Kokoro-82M Persian architecture with warm singleton memory caching.
  - `matcha-khadijah-fa-en` & `matcha-musa-fa-en`: Bilingual Persian-English flow matching with HiFi-GAN vocoder.
  - `pocket-tts-farsi` & `nava-fa-clone`: 10 built-in characters + **Zero-Shot Voice Cloning on CPU** from 5–30s reference audio.
- **Persian Text Processing:**
  - Context-Aware Ezafe detector (`persian-ezafe-albert` ONNX, <40ms CPU latency).
  - Spoken number conversion with Persian ITN (`"هشتصد و پنجاه"` $\rightarrow$ `"850"` or `"۸۵۰"`).
  - Text & Number normalization (dates, times, currencies, percentages).
- **Protocols & Compatibility:**
  - REST (OpenAI-compatible) and WebSockets (real-time chunk and sentence streaming).
  - Flexible routing supporting `/v1/audio/...`, `/api/v1/...`, `/v1/v1/...`, and WebSocket control handshakes.

---

## Installation

```bash
pip install nava-fa-speech
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/mhosseinifar8/nava.git#subdirectory=nava_sdk
```

*Note: Both `import nava` and `import nava_fa_speech` work interchangeably.*

---

## Docker Hub Official Images

If you need to deploy the server backend, prebuilt images are available on Docker Hub:

| Service | Docker Image | Port | Description |
|---|---|---|---|
| **STT Microservice** | [`mhosseinifar8/nava-stt:latest`](https://hub.docker.com/r/mhosseinifar8/nava-stt) | `8001` | FastConformer Shenava, Whisper-Turbo, Nemotron |
| **TTS Microservice** | [`mhosseinifar8/nava-tts:latest`](https://hub.docker.com/r/mhosseinifar8/nava-tts) | `8002` | Piper Mana, Matcha-TTS, Ava-82M, Pocket-TTS & Voice Cloning |
| **Unified Platform** | [`mhosseinifar8/nava:latest`](https://hub.docker.com/r/mhosseinifar8/nava) | `8000` | Complete platform with all STT and TTS engines |

```bash
# Pull images
docker pull mhosseinifar8/nava-stt:latest
docker pull mhosseinifar8/nava-tts:latest
docker pull mhosseinifar8/nava:latest
```

---

## Quick Start

```python
from nava import NavaClient

# Connect to Nava services:
# Option A (Microservices): STT on 8001, TTS on 8002
with NavaClient(stt_url="http://127.0.0.1:8001", tts_url="http://127.0.0.1:8002") as nava:
# Option B (Unified Platform): Both on port 8000
# with NavaClient(stt_url="http://127.0.0.1:8000", tts_url="http://127.0.0.1:8000") as nava:

    # 1. Transcribe Persian speech audio
    with open("input.pcm", "rb") as f:
        pcm_bytes = f.read()

    transcript = nava.transcribe(
        pcm_bytes,
        model="shenava-koochik-ctc",
        enable_itn=True,                    # Converts "هشتصد و پنجاه" -> "850"
        hotwords={"اسنپ‌فود": 2.0},
        custom_dictionary={"اسنب فود": "اسنپ‌فود"},
    )
    print("Transcript:", transcript.text)
    print("With ITN:", transcript.itn_text)

    # 2. Synthesize Persian speech with Ezafe
    speech = nava.synthesize(
        "سلام، به سامانه پیشرفته نوا خوش آمدید.",
        model="piper-mana-lca",
        speed=1.0,
        g2p_mode="lca_albert",
    )
    with open("output.pcm", "wb") as f:
        f.write(speech.pcm)
    print(f"Generated {len(speech.pcm)} bytes at {speech.sample_rate} Hz via {speech.engine}")
```

---

## Python SDK Manual

Both `import nava` and `import nava_fa_speech` are fully supported.

### Synchronous Client (`NavaClient`)

```python
from nava import NavaClient, TranscriptionResult, SpeechResult

# Microservice deployment (default ports):
client = NavaClient(
    stt_url="http://127.0.0.1:8001",
    tts_url="http://127.0.0.1:8002",
    timeout=120.0,
)

# Unified server deployment (single port 8000):
# client = NavaClient(stt_url="http://127.0.0.1:8000", tts_url="http://127.0.0.1:8000")
```

#### 1. `client.transcribe(...)`
Transcribes raw signed 16-bit little-endian PCM audio (16kHz mono) or WAV bytes.

```python
result: TranscriptionResult = client.transcribe(
    pcm=audio_bytes,
    model="shenava-koochik-ctc",        # Available: shenava-koochik-ctc, shenava-koochik-v1.5-rnnt,
                                        #            whisper-turbo-persian-sherpa, vosk-fa-light, etc.
    enable_itn=True,                    # Convert spoken numbers to digits (True by default)
    hotwords={"دیجیکالا": 2.5},          # Boost domain keywords (transducer / whisper)
    custom_dictionary={"اسنب": "اسنپ"},  # Correct domain vocabulary
    initial_prompt=None,                # Context prompt for Whisper models
)

print(result.text)          # Raw transcribed text
print(result.itn_text)      # Numbers normalized to digits
print(result.segments)      # [{"start": 0.0, "end": 2.4, "text": "..."}]
print(result.language_prob) # 0.99
print(result.rtf)           # Real-Time Factor (e.g. 0.045)
```

#### 2. `client.synthesize(...)`
Synthesizes Persian text into raw 16-bit PCM or WAV audio.

```python
speech: SpeechResult = client.synthesize(
    text="کتاب خوب و خانه بزرگ در تاریخ 1403/08/18 ثبت شد.",
    model="piper-mana-lca",             # piper-mana-lca, ava-82m, matcha-musa-fa-en, matcha-khadijah-fa-en, pocket-tts-farsi
    voice="default",                    # Speaker name: mana, amir, gyro, ganji, ava, or a cloned voice_id
    speed=1.0,                          # 0.5 to 2.0
    g2p_mode="lca_albert",              # "lca_albert" | "homo_ge2pe" | "fast_espeak"
    custom_dictionary={"مرد": "مَرد"},  # Custom pronunciation / eraab dictionary
    reference_audio=None,               # Optional 5–30s reference WAV bytes for one-shot voice cloning
    response_format="pcm",              # "pcm" or "wav"
)

# Access result fields
raw_bytes = speech.pcm
rate = speech.sample_rate    # 22050 or 24000
engine = speech.engine       # e.g. "piper", "matcha", "ava"
voice = speech.voice         # e.g. "mana"
```

#### 3. `client.clone_voice(...)`
Uploads reference audio (5–30 seconds) and registers a reusable `voice_id` for CPU voice cloning.

```python
with open("my_voice_sample.wav", "rb") as f:
    ref_audio = f.read()

voice_id = client.clone_voice(ref_audio, name="my_speaker")
print("Registered voice:", voice_id)  # e.g. "voice_my_speaker"

# Use in subsequent synthesis calls
cloned = client.synthesize(
    "سلام، این متن با صدای من خوانده می‌شود.",
    model="pocket-tts-farsi",
    voice=voice_id,
)
```

#### 4. `client.list_models()` & `client.download_model(...)`
```python
# List models and check installation status
models = client.list_models()
for m in models:
    print(f"{m['id']:<32} Installed: {m['installed']} Size: {m['size_mb']}MB")

# Trigger download of a specific model or profile
client.download_model(model_id="shenava-koochik-ctc")
client.download_model(profile="minimal")
```

---

### Asynchronous Client (`AsyncNavaClient`)

`AsyncNavaClient` provides async coroutines for REST plus real-time bidirectional WebSocket streaming.

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient(
        stt_url="http://127.0.0.1:8001",
        tts_url="http://127.0.0.1:8002",
    ) as nava:
        # Async REST transcription
        res = await nava.transcribe(pcm_bytes, model="shenava-koochik-ctc")
        print("Text:", res.text)

        # Async REST synthesis
        speech = await nava.synthesize("سلام", model="piper-mana-lca")

        # Real-time WebSocket STT
        ws_res = await nava.transcribe_ws(pcm_bytes, model="shenava-rizeh-streaming-32m")
        print("WS result:", ws_res.text)

        # Sentence-by-sentence streaming TTS over WebSocket
        async for chunk in nava.stream_speech_ws(
            "جمله اول به سرعت تولید می‌شود. سپس جمله دوم در ادامه می‌آید.",
            model="piper-mana-lca",
        ):
            print(f"Received chunk: {len(chunk.pcm)} bytes at {chunk.sample_rate} Hz")

asyncio.run(main())
```

---

## WebSocket Streaming Manual

### Real-Time STT WebSocket (`/v1/ws/stt` and `/ws/stt`)

Low-latency real-time Persian speech recognition with live partials and endpoint detection.

#### Flow Protocol:
1. **Connect:** Open WebSocket to `ws://<host>:<port>/v1/ws/stt` (or `/ws/stt`)
2. **Handshake (JSON):** Send configuration object:
   ```json
   {
     "model": "shenava-rizeh-streaming-32m",
     "sample_rate": 16000,
     "latency_ms": 80,
     "enable_itn": true,
     "hotwords": {"چشم": 2.5}
   }
   ```
   *Server responds:* `{"status": "configured", "model": "shenava-rizeh-streaming-32m"}`
3. **Stream Audio (Binary):** Send raw signed 16-bit little-endian PCM chunks (e.g. 100ms – 250ms chunks = 3200 – 8000 bytes at 16kHz mono).
4. **Receive Partials (JSON):**
   ```json
   {"type": "partial", "text": "سلام به"}
   {"type": "partial", "text": "سلام به سامانه نوا"}
   ```
5. **Finish Stream (JSON):** Send EOF:
   ```json
   {"action": "eos"}
   ```
6. **Receive Final Result (JSON):**
   ```json
   {
     "type": "final",
     "status": "success",
     "text": "سلام به سامانه نوا هشتصد و پنجاه",
     "itn_text": "سلام به سامانه نوا 850",
     "rtf": 0.027,
     "segments": [{"start": 0.0, "end": 2.15, "text": "سلام به سامانه نوا 850"}]
   }
   ```

---

### Sentence-Streaming TTS WebSocket (`/v1/ws/tts` and `/ws/tts`)

Pipelined text-to-speech where sentence $N+1$ G2P runs while sentence $N$ audio is synthesized and streamed, achieving **<100ms Time-To-First-Audio (TTFA)** on CPU.

#### Supported Handshake & Config Frames:
Nava supports both initial configuration frames (without `text`) and direct text frames:

**Option 1: Initial Config Frame (e.g. from client libraries):**
```json
{
  "model": "matcha-musa-fa-en",
  "voice": "default",
  "speed": 1.0,
  "g2p_mode": "lca_albert"
}
```
*Server responds:* `{"status": "ok", "action": "configured", "model": "matcha-musa-fa-en"}`

**Option 2: Text Frame with configuration:**
```json
{
  "type": "text",
  "text": "این جمله اول است که بلافاصله پخش می‌شود. این هم جمله دوم است.",
  "model": "piper-mana-lca",
  "voice": "mana",
  "speed": 1.0,
  "g2p_mode": "lca_albert"
}
```

#### Stream Sequence:
1. **Receive Metadata (JSON):**
   ```json
   {
     "status": "streaming",
     "type": "streaming",
     "engine": "piper",
     "sample_rate": 22050
   }
   ```
2. **Receive Audio (Binary):** Raw 16-bit PCM binary frames for each sentence as it finishes.
3. **Cancellation (Optional JSON):** Send `{"type": "cancel"}` at any time to immediately abort remaining sentences.
4. **Receive Completion (JSON):**
   ```json
   {"status": "completed", "type": "done"}
   ```

---

### JavaScript / Frontend WebSocket Example

```javascript
// Browser / Node.js Streaming STT Client
const ws = new WebSocket("ws://127.0.0.1:8001/ws/stt");

ws.onopen = () => {
  // 1. Send handshake
  ws.send(JSON.stringify({
    model: "shenava-rizeh-streaming-32m",
    sample_rate: 16000,
    enable_itn: true
  }));
};

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "partial") {
      console.log("Live:", msg.text);
    } else if (msg.type === "final") {
      console.log("Final Transcript:", msg.itn_text || msg.text);
    }
  }
};

// Stream microphone PCM data:
function sendAudioChunk(pcmInt16Array) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(pcmInt16Array.buffer);
  }
}

// When user stops speaking:
function stopRecording() {
  ws.send(JSON.stringify({ action: "eos" }));
}
```

---

## REST API Reference & cURL Manual

Nava exposes OpenAI-compatible REST endpoints with Persian extensions.

### `POST /v1/audio/transcriptions` (STT)
*(Aliases: `/api/v1/transcriptions`, `/v1/v1/audio/transcriptions`)*

#### Option A: Multipart Form Data
```bash
curl -X POST "http://127.0.0.1:8001/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=shenava-koochik-ctc" \
  -F "enable_itn=true" \
  -F 'hotwords={"اسنپفود": 2.0}' \
  -F 'custom_dictionary={"اسنب": "اسنپ"}'
```

#### Option B: Raw 16-bit PCM Octet-Stream
```bash
curl -X POST "http://127.0.0.1:8001/v1/audio/transcriptions?model=shenava-koochik-ctc&enable_itn=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@audio.pcm"
```

**Response (JSON):**
```json
{
  "text": "قیمت آن هشتصد و پنجاه تومان است",
  "itn_text": "قیمت آن 850 تومان است",
  "segments": [
    {"start": 0.0, "end": 2.4, "text": "قیمت آن 850 تومان است"}
  ],
  "language_prob": 0.99,
  "rtf": 0.038,
  "model": "shenava-koochik-ctc"
}
```

---

### `POST /v1/audio/speech` (TTS)
*(Aliases: `/api/v1/speech`, `/v1/v1/audio/speech`)*

```bash
curl -X POST "http://127.0.0.1:8002/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "سلام به پروژه نوا، کیفیت استودیویی روی پردازنده معمولی.",
    "model": "matcha-musa-fa-en",
    "voice": "default",
    "speed": 1.0,
    "g2p_mode": "lca_albert",
    "response_format": "wav"
  }' \
  --output speech.wav
```

**Request Parameters:**
- `input` *(string, required)*: Persian text to synthesize (alias: `text`).
- `model` *(string)*: `piper-mana-lca` (default), `matcha-musa-fa-en`, `matcha-khadijah-fa-en`, `ava-82m`, `pocket-tts-farsi`, `piper-voices-fa`.
- `voice` *(string)*: Speaker name (`mana`, `amir`, `gyro`, `ganji`, `ava`) or a cloned `voice_id`.
- `speed` *(float)*: Speed factor from `0.5` to `2.0` (default: `1.0`).
- `g2p_mode` *(string)*: `"lca_albert"` (default), `"homo_ge2pe"`, or `"fast_espeak"`.
- `custom_dictionary` *(object)*: Map of custom words to pronunciations or eraab (`{"برند": "بِرَند"}`).
- `reference_audio_base64` *(string, optional)*: Base64-encoded 5–30s reference WAV for one-shot voice cloning.
- `response_format` *(string)*: `"pcm"` (default signed 16-bit mono) or `"wav"`.

---

### `POST /v1/voices/clone` (Voice Cloning)

Registers reference audio and caches conditioning embeddings on CPU.

```bash
curl -X POST "http://127.0.0.1:8002/v1/voices/clone" \
  -F "file=@my_reference_sample.wav" \
  -F "name=speaker_ali"
```

**Response (JSON):**
```json
{
  "voice_id": "voice_speaker_ali",
  "status": "ready",
  "duration_seconds": 6.4,
  "sample_rate": 24000,
  "message": "Voice profile registered. Pass voice='voice_speaker_ali' to TTS requests."
}
```

---

### `GET /v1/models` & `POST /v1/models/download`

```bash
# List all models
curl "http://127.0.0.1:8000/v1/models"

# Trigger download of minimal profile in background
curl -X POST "http://127.0.0.1:8000/v1/models/download" \
  -H "Content-Type: application/json" \
  -d '{"profile": "minimal", "async_mode": true}'
```

---

## Persian Speech Features

### Inverse Text Normalization (ITN)
Automatically converts spoken Persian numerals into digits in STT outputs:
- Compound numbers: `"هشتصد و پنجاه و سه"` $\rightarrow$ `"853"`
- Scales: `"دو میلیون و پانصد هزار ریال"` $\rightarrow$ `"2500000 ریال"`
- Ordinals: `"طبقه سوم یا چهارم"` $\rightarrow$ `"طبقه 3م یا 4م"`
- Dates & Years: `"سال هزار و چهارصد و سه"` $\rightarrow$ `"سال 1403"`
- Decimals & Negatives: `"سه ممیز پنج"` $\rightarrow$ `"3.5"`, `"منفی ده"` $\rightarrow$ `"-10"`
- Persian digits option (`to_persian_digits=True`): `"هشتصد و پنجاه"` $\rightarrow$ `"۸۵۰"`

### Pre-TTS G2P & Ezafe Detection
Before synthesis, text runs through a 3-step pipeline:
1. **Normalization:** Expands digits (`1403/08/18` $\rightarrow$ `هجدهم آبان هزار و چهارصد و سه`), times (`14:30` $\rightarrow$ `چهارده و نیم`), percentages (`۲۵٪` $\rightarrow$ `بیست و پنج درصد`).
2. **Lexicon Mapping:** Replaces acronyms, colloquialisms, and brand names from `custom_dictionary`.
3. **Ezafe Detector (`persian-ezafe-albert`):** Detects grammatical noun-adjective connections and appends kasreh (`ِ`) for consonant endings (e.g. `کتاب خوب` $\rightarrow$ `کتابِ خوب`) or `‌ی` for vowel endings (e.g. `خانه بزرگ` $\rightarrow$ `خانه‌ی بزرگ`).

---

## Error Handling

All client errors raise `NavaServiceError` with descriptive HTTP status codes and messages:

```python
from nava import NavaClient, NavaServiceError

with NavaClient() as nava:
    try:
        res = nava.transcribe(b"too_short")
    except NavaServiceError as exc:
        print("Caught Nava error:", exc)
        # e.g. "Nava returned HTTP 422: audio is shorter than 100 ms at 16 kHz"
```

---

## Testing & Service Verification

You can verify your live STT & TTS deployment (health, REST synthesis, WebSocket streaming, and transcription) with the included automated testing script:

```bash
# Test against your server host / IP (default ports 8001 STT & 8002 TTS):
python3 scripts/test_nava_services.py --host 192.168.14.207

# Or test the unified platform (port 8000):
python3 scripts/test_nava_services.py --host 192.168.14.207 --unified-port 8000
```

---

## License

MIT License. Designed and developed for the open-source Persian speech AI community.
Contributions and issues are welcome on [GitHub](https://github.com/mhosseinifar8/nava).

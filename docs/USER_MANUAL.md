# Nava — User Manual

Nava (نوا) is an offline Persian speech platform you run on your own machine:

- **STT** — Persian speech-to-text (Whisper large-v3, faster-whisper, CPU)
- **TTS** — Persian text-to-speech with 3 engines (pocket / mana / piper, CPU)
- **SDK** — a small Python client library

Everything runs locally in Docker. No internet is needed after the models are
downloaded, and nothing leaves your machine.

---

## 1. Requirements

| Item | Minimum | Notes |
|---|---|---|
| CPU | 2 cores | any machine works out of the box; more cores = more speed (see §2) |
| RAM | ~8 GB free | STT with `int8` ≈ 2.5–3 GB, TTS ≈ 3–4 GB |
| Disk | ~10 GB | mostly model files |
| Software | Docker + docker compose, Python ≥ 3.10, `huggingface-hub[cli]` (`pip install "huggingface-hub[cli]"`) | internet needed once, for the model download |

---

## 2. Installation & infrastructure profiles

```bash
# 1. download all models into ./models  (one time, needs internet)
./scripts/download_models.sh

# 2. generate .env tuned to THIS machine (cores/RAM auto-detected)
./scripts/configure_infra.sh auto

# 3. build and start both services
docker compose up --build -d

# 4. install the Python SDK (optional, for §7)
pip install -e ./nava_sdk
```

`configure_infra.sh` writes a `.env` that docker compose reads automatically:

| Profile | Target hardware | What it sets |
|---|---|---|
| `low` | 2–4 cores, 8 GB (laptop / edge / small VPS) | `int8` Whisper, 1 worker, 2 threads, no core pinning |
| `medium` | 8 cores, 16 GB (standard server) | `int8`, 2 workers, 4 threads, cores split 0-3 / 4-7 |
| `high` | 16–32+ cores, 32 GB+ (accuracy node) | `float32`, 4 workers, 8 threads, cores split in halves |
| `auto` | *(default)* | detects cores/RAM and picks the best fit |

You can also hand-edit `.env` (see `.env.example` for every knob) and restart
with `docker compose up -d`. The compose defaults (without a `.env`) are safe
on any machine: soft CPU quotas, no pinning.

Check that both services are up:

```bash
curl -s http://127.0.0.1:8001/api/v1/health
# {"status":"ok","service":"nava-stt"}

curl -s http://127.0.0.1:8002/api/v1/health
# {"status":"ok","service":"nava-tts","engines":["pocket","mana","piper"]}
```

`docker compose ps` also shows container health (healthchecks are built in).

---

## 3. Audio formats

**STT input** — raw PCM, **signed 16-bit little-endian, mono, 16 000 Hz**:

- minimum 100 ms (3200 bytes), maximum 15 minutes by default
  (`NAVA_MAX_STT_AUDIO_BYTES`). Oversized uploads are rejected with
  `413 Payload Too Large` immediately, without buffering.

Convert any file to the right format with ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le speech.pcm
```

**TTS output** — raw PCM, signed 16-bit little-endian, mono. The **sample rate
depends on the engine** — always read it from the response:

- REST: `X-Nava-Sample-Rate` response header
- WebSocket: the `{"status":"streaming","sample_rate":...}` message

Play or convert a result:

```bash
ffplay -f s16le -ar <rate> -ch_layout mono speech.pcm
ffmpeg -f s16le -ar <rate> -ac 1 -i speech.pcm speech.wav
```

---

## 4. Speech-to-text (STT, port 8001)

### 4.1 REST

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/transcriptions \
  -H "Content-Type: application/octet-stream" \
  --data-binary @speech.pcm
```

Response:

```json
{
  "text": "متن تشخیص داده شده",
  "segments": [{"start": 0.0, "end": 2.4, "text": "متن تشخیص داده شده"}],
  "language_prob": 0.99
}
```

Optional query parameters (URL-encode Persian text — easiest via the SDK):

- `hotwords=واژه مهم` — bias recognition toward these words
- `initial_prompt=...` — replace the built-in Persian style prompt

Errors: `413` too large, `422` malformed/short audio.

### 4.2 WebSocket (`/ws/stt`)

One connection can transcribe many utterances. Frame sequence:

```
client →  (optional) {"initial_prompt": "...", "hotwords": "..."}   # text frame, before audio
client →  binary PCM frames (any number — the SDK sends 1 MB frames)
client →  {"action": "eos"}                                          # text frame
server →  {"text": "...", "segments": [...], "language_prob": 0.99}
          # then the connection is ready for the next utterance
```

Errors come back as `{"status": "error", "error": "..."}`; audio above the
size limit closes the socket with code 1009.

---

## 5. Text-to-speech (TTS, port 8002)

### 5.1 Choose an engine

| Engine | Voice | `voice_prompt` | `speed` | Best for |
|---|---|---|---|---|
| `pocket` | cloned from a wav prompt | yes (default `default`) | not supported (422 if ≠ 1.0) | expressive speech, custom voices |
| `mana` | Mana (fixed) | no | 0.5–2.0 | general use |
| `piper` | Ganji (fixed) | no | 0.5–2.0 | general use, fastest |

### 5.2 REST

```bash
curl -s -X POST http://127.0.0.1:8002/api/v1/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"سلام دنیا","engine":"piper","speed":1.0}' \
  -o speech.pcm -D headers.txt

grep X-Nava-Sample-Rate headers.txt   # sample rate of the PCM you received
```

Request fields:

| Field | Default | Rules |
|---|---|---|
| `text` | required | 1–20 000 chars; unsupported symbols are stripped automatically |
| `engine` | `pocket` | `pocket`, `mana` or `piper` |
| `voice_prompt` | `default` | pocket only; name of a wav in `models/voices/`, letters/digits/`_`/`-` |
| `speed` | `1.0` | 0.5–2.0; mana/piper only — pocket rejects other values with 422 |

The response is a **stream**; input problems (empty after cleaning, unknown
voice, unsupported speed) return real `422` errors — never an empty 200.

### 5.3 WebSocket (`/ws/tts`)

```
client →  {"text":"سلام","engine":"pocket","voice_prompt":"default","speed":1.0}
server →  {"status":"streaming","engine":"pocket","sample_rate":24000}
server →  binary PCM frame(s)
server →  {"status":"completed"}
          # connection stays open for the next request
```

Errors: `{"status":"error","error":"..."}`.

---

## 6. Custom words & pronunciation (lexicon)

Create `models/lexicon.json` (or point `NAVA_LEXICON_PATH` elsewhere) to fix
colloquial spellings or override pronunciations:

```json
{
  "surface":  {"می خوام": "می خواهم", "چه جالب": "چه جالب"},
  "phonemes": {"@": "?"}
}
```

- `surface` — literal text replacements, longest match first, applied for
  **all** engines before sentence splitting.
- `phonemes` — token-wise overrides of the neural G2P output (**pocket** only).

Restart the TTS container after editing: `docker compose restart nava-tts`.

---

## 7. Python SDK

```bash
pip install -e ./nava_sdk
```

### 7.1 Synchronous (REST)

```python
from nava import NavaClient

with NavaClient() as nava:                      # stt 127.0.0.1:8001, tts 127.0.0.1:8002
    with open("speech.pcm", "rb") as f:
        result = nava.transcribe(f.read(), hotwords="نوا")
    print(result.text, result.language_prob)

    speech = nava.synthesize("سلام دنیا", engine="piper", speed=1.1)
    print(speech.sample_rate, len(speech.pcm))  # raw PCM bytes
    with open("out.pcm", "wb") as f:
        f.write(speech.pcm)
```

Custom URLs: `NavaClient(stt_url="http://10.0.0.5:8001", tts_url="http://10.0.0.5:8002")`.

### 7.2 Asynchronous, incl. WebSockets

```python
import asyncio
from nava import AsyncNavaClient

async def main():
    async with AsyncNavaClient() as nava:
        # REST
        result = await nava.transcribe(pcm_bytes)
        speech = await nava.synthesize("سلام", engine="mana")

        # WebSocket STT — long audio is sent in 1 MB frames automatically
        result = await nava.transcribe_ws(pcm_bytes, hotwords="نوا")

        # WebSocket TTS — streams chunks as they are produced
        async for chunk in nava.stream_speech_ws("این یک آزمایش است", engine="pocket"):
            play(chunk.pcm)                      # your audio sink here

asyncio.run(main())
```

All HTTP/WS failures raise `NavaServiceError` with the server's message.

---

## 8. Voice prompts (pocket engine)

Any wav file placed in `models/voices/` becomes a `voice_prompt` — the file
name without `.wav` is the prompt name (only `A-Z a-z 0-9 _ -` allowed):

```
models/voices/my_voice.wav   →   "voice_prompt": "my_voice"
```

- The first ~5 seconds of the prompt are used.
- Prompt states are cached per voice after the first use.
- `default` is a symlink created by the download script.

---

## 9. Configuration reference

### Host-level (`.env`, read by docker compose)

| Variable | Default | Applies to |
|---|---|---|
| `WHISPER_COMPUTE_TYPE` | `int8` | STT — `int8` (fast) / `int8_float32` / `float32` (most accurate) |
| `STT_WORKERS` / `TTS_WORKERS` | `2` | inference thread pool; STT workers = parallel transcriptions |
| `STT_OMP_THREADS` / `TTS_OMP_THREADS` | `4` | BLAS/CTranslate2/torch threads |
| `STT_CPUS` / `TTS_CPUS` | `2.0` | soft CPU quota per container |
| `STT_CPUSET` / `TTS_CPUSET` | *(empty)* | hard core pinning, e.g. `0-3` / `4-7`; empty = off |
| `STT_MEM` / `TTS_MEM` | `8g` | container memory limit |

### Service-level (inside the containers)

| Variable | Default | Service |
|---|---|---|
| `NAVA_WHISPER_MODEL_DIR` | `/models/whisper-large-v3` | STT |
| `NAVA_WHISPER_COMPUTE_TYPE` | `int8` | STT |
| `NAVA_STT_WORKERS` | `2` | STT |
| `NAVA_STT_OMP_THREADS` | `4` | STT |
| `NAVA_MAX_STT_AUDIO_BYTES` | `28800000` (15 min) | STT |
| `NAVA_MODELS_DIR` | `/models` | TTS |
| `NAVA_TTS_WORKERS` | `2` | TTS |
| `NAVA_TTS_OMP_THREADS` | `4` | TTS |
| `NAVA_MAX_TTS_TEXT_CHARS` | `20000` | TTS |
| `NAVA_LEXICON_PATH` | `/models/lexicon.json` | TTS |

---

## 10. Tests

```bash
python -m pip install -r tests/requirements.txt

python -m pytest tests/unit -q     # anywhere — no models or services needed
python -m pytest tests -q          # + integration; auto-skips if services are down
```

---

## 11. Shutdown / update

```bash
docker compose down             # stop
./scripts/download_models.sh    # refresh models, then
docker compose up --build -d    # restart with new models
```

---

## 12. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Container restarts forever, log says `Whisper model missing` / `Pocket model missing` | models not downloaded → run `./scripts/download_models.sh`, check `models/` |
| STT slow | use a bigger profile (`./scripts/configure_infra.sh medium` or `high`) or more OMP threads |
| STT returns `413` | clip longer than 15 min — split it, or raise `NAVA_MAX_STT_AUDIO_BYTES` |
| STT returns `422` `audio is shorter than 100 ms` | PCM too small — check it is mono 16 kHz s16le |
| TTS returns `422` for `engine:"pocket"` with `speed` | pocket doesn't support speed — omit it or use `mana`/`piper` |
| TTS returns `422` `text has no speakable content` | text had no supported characters after cleaning (e.g. only symbols) |
| WebSocket STT closes with code 1009 | total audio above the 15-min limit |
| High RAM use / OOM | use the `low` profile (`int8`, 1 worker) or raise `STT_MEM`/`TTS_MEM` |
| Container shows `unhealthy` | see logs: `docker compose logs nava-stt` — usually missing models or OOM |
| `pip install -e ./nava_sdk` fails | Python ≥ 3.10 required |

---

## 13. Good to know

- Services are bound to `127.0.0.1` only — nothing outside your machine can
  reach them. There is **no authentication**; don't expose the ports without
  adding auth.
- STT output follows standard Persian orthography (ZWNJ, punctuation) via a
  built-in prompt; `hotwords` improves names/jargon.
- TTS normalises Arabic characters to Persian and strips unsupported symbols
  automatically; no cleanup needed before sending text.
- `mana` and `piper` never touch the neural G2P model — they are the fastest
  engines. `pocket` pipelines G2P with synthesis for expressive custom voices.

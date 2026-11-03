# Nava

Nava (نوا) is an offline, CPU-optimized Persian speech platform. Its two
isolated FastAPI services expose raw-binary WebSockets and REST APIs, with a
Python SDK (`pip install -e ./nava_sdk`).

## Quick start

```bash
./scripts/download_models.sh
./scripts/configure_infra.sh auto   # writes .env tuned to THIS machine
docker compose up --build -d
pip install -e ./nava_sdk
```

```python
from nava import NavaClient

with NavaClient() as nava:
    transcript = nava.transcribe(pcm_16khz)
    speech = nava.synthesize("سلام", engine="piper")
```

STT: `POST /api/v1/transcriptions` and `/ws/stt` on 8001. TTS:
`POST /api/v1/speech` and `/ws/tts` on 8002.

## Infrastructure profiles

`docker-compose.yml` is fully env-driven with safe defaults (soft CPU quotas,
no core pinning) that work on any machine. Tune per host:

```bash
./scripts/configure_infra.sh auto     # detect cores/RAM and pick low|medium|high
./scripts/configure_infra.sh low      # edge / laptop / small VPS
./scripts/configure_infra.sh high     # dedicated accuracy node (float32, pinned)
```

See `.env.example` for every knob.

## Tests

```bash
python -m pip install -r tests/requirements.txt
python -m pytest tests/unit -q        # anywhere — no models or services needed
python -m pytest tests -q             # + integration (auto-skips if services down)
```

## Docs

- [Architecture](docs/ARCHITECTURE.md) — components, concurrency model, data flow, tuning
- [User manual](docs/USER_MANUAL.md) — install, APIs, WebSocket protocols, SDK, troubleshooting

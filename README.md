# Nava

Nava (نوا) is an offline, CPU-optimized Persian speech platform. Its two
isolated FastAPI services expose both raw binary WebSockets and REST APIs.

```bash
./scripts/download_models.sh
docker compose up --build
pip install -e ./nava_sdk
```

```python
from nava import NavaClient, AsyncNavaClient

with NavaClient() as nava:
    transcript = nava.transcribe(pcm_16khz)
    speech = nava.synthesize("سلام", engine="piper")
```

STT: `POST /api/v1/transcriptions` and `/ws/stt` on 8001. TTS:
`POST /api/v1/speech` and `/ws/tts` on 8002.

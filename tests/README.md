# Nava test suite

```bash
# unit tests — run anywhere, no services or models needed
python -m pip install -r tests/requirements.txt
python -m pytest tests/unit -q

# integration tests — need running services (docker compose up) and downloaded models
python -m pytest tests -q
```

Integration tests are skipped automatically when `nava-stt` (8001) or
`nava-tts` (8002) is not reachable. Point them elsewhere with:

```bash
NAVA_TEST_STT_URL=http://10.0.0.5:8001 NAVA_TEST_TTS_URL=http://10.0.0.5:8002 python -m pytest tests
```

Coverage:

- `unit/` — Persian text pipeline (normalisation, pocket chunking ≤ token
  budget, lexicon surface/phoneme overrides), stream-safe 413 body limiting,
  service settings env precedence, SDK helpers and HTTP mocking (incl. the
  chunked WS send fix).
- `integration/` — health endpoints, REST STT/TTS (incl. regressions for the
  fixed bugs: 413 oversize, 422 empty text, 422 pocket+speed), raw WebSocket
  framing protocol, and SDK end-to-end round trips.

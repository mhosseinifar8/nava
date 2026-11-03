"""End-to-end SDK integration tests (auto-skipped when services are down)."""
from __future__ import annotations

import pytest

from conftest import STT_URL, TTS_URL, make_pcm
from nava import AsyncNavaClient, NavaClient, NavaServiceError, TranscriptionResult


@pytest.mark.stt_service
def test_sync_client_end_to_end():
    with NavaClient(stt_url=STT_URL, tts_url=TTS_URL, timeout=300) as nava:
        result = nava.transcribe(make_pcm(1.0))
        assert isinstance(result, TranscriptionResult)


@pytest.mark.stt_service
def test_sync_client_maps_service_errors():
    with NavaClient(stt_url=STT_URL, tts_url=TTS_URL, timeout=60) as nava:
        with pytest.raises(NavaServiceError, match="422|413"):
            nava.transcribe(b"\x00" * 10)


@pytest.mark.stt_service
@pytest.mark.tts_service
async def test_async_client_end_to_end():
    async with AsyncNavaClient(stt_url=STT_URL, tts_url=TTS_URL, timeout=300) as nava:
        result = await nava.transcribe(make_pcm(1.0))
        speech = await nava.synthesize("سلام دنیا", engine="piper")
        assert result.language_prob >= 0
        assert speech.pcm and speech.sample_rate > 8000

"""Stream-safe request-body size limiting (STT 413 guard)."""
from __future__ import annotations

import pytest


class FakeRequest:
    def __init__(self, chunks, content_length=None):
        self.headers = {"content-length": str(content_length)} if content_length is not None else {}
        self._chunks = chunks
        self.consumed_chunks = 0

    def stream(self):
        async def gen():
            for chunk in self._chunks:
                self.consumed_chunks += 1
                yield chunk
        return gen()


async def test_rejects_via_content_length_without_reading(bodylimit):
    request = FakeRequest(chunks=[b"x" * 100], content_length=11)
    with pytest.raises(bodylimit.TooLargeError):
        await bodylimit.read_limited(request, max_bytes=10)
    assert request.consumed_chunks == 0  # rejected before buffering anything


async def test_aborts_mid_stream_on_chunked_bodies(bodylimit):
    request = FakeRequest(chunks=[b"a" * 6, b"b" * 6, b"c" * 6])
    with pytest.raises(bodylimit.TooLargeError):
        await bodylimit.read_limited(request, max_bytes=10)
    assert request.consumed_chunks == 2  # stopped reading as soon as the limit broke


async def test_exact_limit_passes(bodylimit):
    request = FakeRequest(chunks=[b"a" * 5, b"b" * 5])
    assert await bodylimit.read_limited(request, max_bytes=10) == b"a" * 5 + b"b" * 5


async def test_under_limit_passes(bodylimit):
    request = FakeRequest(chunks=[b"pcm"], content_length=3)
    assert await bodylimit.read_limited(request, max_bytes=100) == b"pcm"


async def test_empty_body(bodylimit):
    request = FakeRequest(chunks=[])
    assert await bodylimit.read_limited(request, max_bytes=10) == b""


async def test_non_numeric_content_length_ignored(bodylimit):
    request = FakeRequest(chunks=[b"a"], content_length=None)
    request.headers["content-length"] = "not-a-number"
    assert await bodylimit.read_limited(request, max_bytes=10) == b"a"

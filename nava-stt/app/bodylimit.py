"""Stream-safe request-body size limiting.

Pure stdlib so it is unit-testable without FastAPI: ``request`` only needs
``headers.get()`` and ``stream()`` (an async iterable of bytes).
"""
from __future__ import annotations

from collections.abc import AsyncIterator


class TooLargeError(Exception):
    """Request body exceeds the configured byte limit."""


async def read_limited(request, max_bytes: int) -> bytes:
    """Read a request body while enforcing ``max_bytes``.

    Oversized payloads are rejected early via Content-Length and aborted
    mid-stream while reading chunked bodies, so they are never fully buffered.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise TooLargeError(f"request body of {declared} bytes exceeds limit of {max_bytes} bytes")
    body = bytearray()
    stream: AsyncIterator[bytes] = request.stream()
    async for chunk in stream:
        body.extend(chunk)
        if len(body) > max_bytes:
            raise TooLargeError(f"request body exceeds limit of {max_bytes} bytes")
    return bytes(body)

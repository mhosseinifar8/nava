"""Nava Persian Speech Platform.

An offline, CPU-optimized Persian speech platform and Python SDK.
"""
from typing import TYPE_CHECKING

__version__ = "2.0.0"

_CLIENT_SYMBOLS = {
    "AsyncNavaClient",
    "NavaClient",
    "NavaServiceError",
    "SpeechResult",
    "TranscriptionResult",
}

if TYPE_CHECKING:
    from .client import (AsyncNavaClient, NavaClient, NavaServiceError,
                         SpeechResult, TranscriptionResult)


def __getattr__(name: str):
    if name in _CLIENT_SYMBOLS:
        from . import client
        val = getattr(client, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncNavaClient",
    "NavaClient",
    "NavaServiceError",
    "SpeechResult",
    "TranscriptionResult",
    "__version__",
]

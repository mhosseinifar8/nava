"""Nava Persian Speech Platform.

An offline, CPU-optimized Persian speech platform and Python SDK.
"""
from .client import (AsyncNavaClient, NavaClient, NavaServiceError,
                     SpeechResult, TranscriptionResult)

__version__ = "2.0.0"

__all__ = [
    "AsyncNavaClient",
    "NavaClient",
    "NavaServiceError",
    "SpeechResult",
    "TranscriptionResult",
    "__version__",
]


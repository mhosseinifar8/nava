"""nava_fa_speech SDK package alias."""
from nava import (AsyncNavaClient, NavaClient, NavaServiceError, SpeechResult,
                  TranscriptionResult, __version__)

__all__ = [
    "AsyncNavaClient",
    "NavaClient",
    "NavaServiceError",
    "SpeechResult",
    "TranscriptionResult",
    "__version__",
]


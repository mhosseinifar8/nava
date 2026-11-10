"""Nava Unified Persian Speech Server (REST & WebSockets)."""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..runtime.cpu_tuning import configure_cpu_environment, configure_event_loop
from .routes_models import router as models_router
from .routes_nlp_ocr import router as nlp_ocr_router
from .routes_stt import router as stt_router
from .routes_tts import router as tts_router
from .routes_voices import router as voices_router
from .ws_stt import router as ws_stt_router
from .ws_tts import router as ws_tts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure CPU threads, OpenMP, MKL, and ONNX Runtime options
    configure_cpu_environment()
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    # Ensure optimal event loop policy
    configure_event_loop()

    app = FastAPI(
        title="Nava Persian Speech Platform",
        description="Offline, CPU-Optimized Persian STT, TTS, Voice Cloning, NLP, and OCR Server",
        version="2.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoints
    @app.get("/healthz", tags=["Health"])
    @app.get("/api/v1/health", tags=["Health"])
    @app.get("/v1/health", tags=["Health"])
    async def health_check() -> dict[str, Any]:
        from ..models.registry import MODELS
        return {
            "status": "ok",
            "service": "nava-unified-speech",
            "version": "2.1.0",
            "total_models": len(MODELS),
            "cpu_optimized": True,
        }

    # Include routers
    app.include_router(models_router)
    app.include_router(voices_router)
    app.include_router(stt_router)
    app.include_router(tts_router)
    app.include_router(nlp_ocr_router)
    app.include_router(ws_stt_router)
    app.include_router(ws_tts_router)

    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Nava Unified Persian Speech Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("nava.server.app:app", host=args.host, port=args.port, workers=args.workers, loop="uvloop")
    return 0


if __name__ == "__main__":
    sys.exit(main())

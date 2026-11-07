"""Nava Unified Persian Speech Server (REST & WebSockets)."""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes_models import router as models_router
from .routes_stt import router as stt_router
from .routes_tts import router as tts_router
from .routes_voices import router as voices_router
from .ws_stt import router as ws_stt_router
from .ws_tts import router as ws_tts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Nava Persian Speech Platform",
        description="Offline, CPU-Optimized Persian STT, TTS, and Voice Cloning Server",
        version="2.0.0",
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
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "nava-unified-speech", "version": "2.0.0"}

    # Include routers
    app.include_router(models_router)
    app.include_router(voices_router)
    app.include_router(stt_router)
    app.include_router(tts_router)
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
    uvicorn.run("nava.server.app:app", host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())


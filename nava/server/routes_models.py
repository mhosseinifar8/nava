"""REST endpoints for Model Registry and Downloads."""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..models.downloader import download_model, download_profile
from ..models.registry import list_models

router = APIRouter(tags=["Models"])


class DownloadRequest(BaseModel):
    model: str | None = Field(None, description="Specific model ID to download")
    profile: Literal["minimal", "realtime", "high-accuracy", "all"] | None = Field(
        None, description="Model profile to download"
    )
    async_mode: bool = Field(False, description="Run download in background task")


@router.get("/v1/models")
async def get_models(type: str | None = None) -> list[dict[str, Any]]:
    """List all registered models, installation status, RAM footprint, and features."""
    return list_models(model_type=type)  # type: ignore


@router.post("/v1/models/download")
async def post_download_model(
    payload: DownloadRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Trigger synchronous or background download of models by ID or profile."""
    if not payload.model and not payload.profile:
        raise HTTPException(400, "Must specify either 'model' or 'profile'")

    if payload.async_mode:
        if payload.model:
            background_tasks.add_task(download_model, payload.model, quiet=True)
            return {"status": "queued", "model": payload.model}
        elif payload.profile:
            background_tasks.add_task(download_profile, payload.profile, quiet=True)
            return {"status": "queued", "profile": payload.profile}

    # Synchronous mode
    loop = asyncio.get_running_loop()
    if payload.model:
        success = await loop.run_in_executor(None, download_model, payload.model, None, False, True)
        if not success:
            raise HTTPException(500, f"Failed to download model {payload.model}")
        return {"status": "completed", "model": payload.model}
    elif payload.profile:
        results = await loop.run_in_executor(None, download_profile, payload.profile, None, False, True)
        return {"status": "completed", "profile": payload.profile, "results": results}

    raise HTTPException(400, "Invalid request")


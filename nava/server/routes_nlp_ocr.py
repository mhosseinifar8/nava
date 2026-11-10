"""REST Endpoints for Persian NLP Sentiment Analysis & OlmOCR."""
from __future__ import annotations

import base64
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..engines.factory import get_document_ocr_engine, get_nlp_sentiment_engine
from ..runtime.cpu_tuning import run_in_cpu_pool

router = APIRouter(tags=["NLP & OCR"])


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Persian text to analyze for sentiment")


@router.post("/v1/nlp/sentiment")
@router.post("/api/v1/nlp/sentiment")
async def analyze_sentiment(payload: SentimentRequest) -> dict[str, Any]:
    """Classify sentiment of Persian text (< 15ms CPU execution)."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(422, "Text is required and cannot be empty")

    t0 = time.perf_counter()
    engine = get_nlp_sentiment_engine()
    result = await run_in_cpu_pool(engine.analyze, payload.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms
    return result


@router.post("/v1/ocr/farsi")
@router.post("/api/v1/ocr/farsi")
async def perform_farsi_ocr(request: Request) -> dict[str, Any]:
    """Perform Persian document OCR via olmOCR-Farsi LoRA (Qwen2.5-VL-7B)."""
    content_type = request.headers.get("content-type", "")
    image_bytes: bytes | None = None
    custom_prompt: str | None = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            b64_str = body.get("image_base64", "")
            if not b64_str:
                raise HTTPException(422, "Field 'image_base64' is required")
            image_bytes = base64.b64decode(b64_str)
            custom_prompt = body.get("prompt")
        except Exception as e:
            raise HTTPException(400, f"Invalid JSON / base64 payload: {e}")
    else:
        try:
            form = await request.form()
            file = form.get("file")
            custom_prompt = form.get("prompt")  # type: ignore
            if file and hasattr(file, "read"):
                image_bytes = await file.read()
        except Exception as e:
            raise HTTPException(400, f"Invalid multipart form payload: {e}")

    if not image_bytes:
        raise HTTPException(400, "Image must be provided via file upload or base64 JSON")

    engine = get_document_ocr_engine()
    # Offload to heavy task thread pool so real-time STT/TTS is never starved
    result = await run_in_cpu_pool(engine.ocr_image, image_bytes, custom_prompt, heavy=True)
    return result

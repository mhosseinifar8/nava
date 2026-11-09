#!/usr/bin/env python3
"""Comprehensive Test Script for Nava Persian Speech Platform (STT & TTS).

Tests REST APIs and WebSockets on Docker Compose services (default: 192.168.14.207):
1. Health checks & Engine / Model Discovery
2. Text-to-Speech (REST API & WebSocket API)
3. Speech-to-Text (REST API & WebSocket API)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests
import websockets

DEFAULT_HOST = "192.168.14.207"
SAMPLE_PERSIAN_TEXT = "سلام، سامانه صوتی هوشمند نوا با موفقیت راه‌اندازی شد."


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_ok(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def print_warn(msg: str) -> None:
    print(f"  \033[93m⚠\033[0m {msg}")


def print_err(msg: str) -> None:
    print(f"  \033[91m✗\033[0m {msg}")


def resample_pcm16(pcm_data: bytes, src_rate: int, dst_rate: int = 16000) -> bytes:
    """Resample 16-bit mono PCM to target sample rate using linear interpolation."""
    if src_rate == dst_rate or not pcm_data:
        return pcm_data
    samples = np.frombuffer(pcm_data, dtype="<i2")
    num_dst_samples = int(len(samples) * dst_rate / src_rate)
    resampled = np.interp(
        np.linspace(0, len(samples), num_dst_samples, endpoint=False),
        np.arange(len(samples)),
        samples,
    ).astype("<i2")
    return resampled.tobytes()


def save_wav(filename: Path | str, pcm_data: bytes, sample_rate: int) -> None:
    """Save raw 16-bit PCM bytes to a WAV file."""
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ==============================================================================
# Phase 1: Health & Models Check
# ==============================================================================
def check_health_and_models(host: str, stt_port: int, tts_port: int, unified_port: int) -> dict[str, Any]:
    print_header("PHASE 1: Health Checks & Model Discovery")
    info: dict[str, Any] = {"stt": False, "tts": False, "unified": False, "tts_engines": []}

    # 1. Check STT service
    stt_url = f"http://{host}:{stt_port}/healthz"
    try:
        t0 = time.perf_counter()
        r = requests.get(stt_url, timeout=5)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            print_ok(f"STT Service (: {stt_port}) is HEALTHY [{dt:.1f}ms]")
            print(f"    Payload: {data}")
            info["stt"] = True
        else:
            print_warn(f"STT Service (: {stt_port}) returned HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print_err(f"Cannot reach STT Service at {stt_url}: {e}")

    # 2. Check TTS service
    tts_url = f"http://{host}:{tts_port}/healthz"
    try:
        t0 = time.perf_counter()
        r = requests.get(tts_url, timeout=5)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            engines = data.get("engines", [])
            info["tts_engines"] = engines
            print_ok(f"TTS Service (: {tts_port}) is HEALTHY [{dt:.1f}ms]")
            print(f"    Payload: {data}")
            print(f"    Enabled TTS Engines: {', '.join(engines) if engines else 'N/A'}")
            info["tts"] = True
        else:
            print_warn(f"TTS Service (: {tts_port}) returned HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print_err(f"Cannot reach TTS Service at {tts_url}: {e}")

    # 3. Check Unified Server (port 8000 if running)
    unified_url = f"http://{host}:{unified_port}/healthz"
    try:
        r = requests.get(unified_url, timeout=3)
        if r.status_code == 200:
            print_ok(f"Unified Platform (: {unified_port}) is running!")
            info["unified"] = True

            # Query model catalog
            m_res = requests.get(f"http://{host}:{unified_port}/v1/models", timeout=5)
            if m_res.status_code == 200:
                models = m_res.json()
                print("\n  Registered Models Catalog on Unified Server:")
                print(f"    {'Model ID':<30} {'Type':<6} {'Installed'}")
                print(f"    {'-'*48}")
                for m in models:
                    inst = "\033[92mYES\033[0m" if m.get("installed") else "\033[90mNO\033[0m"
                    print(f"    {m.get('id', ''):<30} {m.get('type', ''):<6} {inst}")
    except Exception:
        print(f"  Note: Unified server (: {unified_port}) is optional (microservices STT: {stt_port} & TTS: {tts_port} active).")

    return info


# ==============================================================================
# Phase 2: TTS REST Test
# ==============================================================================
def test_tts_rest(host: str, port: int, text: str, engine: str = "mana") -> tuple[bytes, int] | None:
    print_header("PHASE 2: Text-to-Speech (REST API)")
    url = f"http://{host}:{port}/api/v1/speech"
    payload = {
        "text": text,
        "engine": engine,
        "voice_prompt": "default",
        "speed": 1.0,
    }
    print(f"  Target:  {url}")
    print(f"  Engine:  {engine}")
    print(f"  Input:   \"{text}\"")

    try:
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=30)
        dt = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            print_err(f"TTS REST failed with HTTP {resp.status_code}: {resp.text}")
            return None

        pcm_bytes = resp.content
        rate_header = resp.headers.get("X-Nava-Sample-Rate", "22050")
        sample_rate = int(rate_header)

        out_path = Path("tts_rest_output.wav")
        save_wav(out_path, pcm_bytes, sample_rate)

        duration = (len(pcm_bytes) / 2) / sample_rate
        print_ok(f"TTS REST Success [{dt:.1f}ms]!")
        print(f"    Audio Size:   {len(pcm_bytes):,} bytes (~{duration:.2f} seconds)")
        print(f"    Sample Rate:  {sample_rate} Hz")
        print(f"    Saved WAV to: {out_path.resolve()}")
        return pcm_bytes, sample_rate
    except Exception as e:
        print_err(f"TTS REST request failed: {e}")
        return None


# ==============================================================================
# Phase 3: TTS WebSocket Test
# ==============================================================================
async def test_tts_ws(host: str, port: int, text: str, engine: str = "mana") -> tuple[bytes, int] | None:
    print_header("PHASE 3: Text-to-Speech (WebSocket Streaming)")
    ws_url = f"ws://{host}:{port}/ws/tts"
    payload = {
        "text": text,
        "engine": engine,
        "voice_prompt": "default",
        "speed": 1.0,
    }
    print(f"  Target: {ws_url}")
    print(f"  Engine: {engine}")
    print(f"  Input:  \"{text}\"")

    pcm_chunks: list[bytes] = []
    sample_rate = 22050

    try:
        t0 = time.perf_counter()
        async with websockets.connect(ws_url) as ws:
            # Send text synthesis request
            await ws.send(json.dumps(payload, ensure_ascii=False))

            first_chunk_time = None
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    if first_chunk_time is None:
                        first_chunk_time = (time.perf_counter() - t0) * 1000
                    pcm_chunks.append(msg)
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    status = data.get("status")
                    if status == "streaming":
                        sample_rate = int(data.get("sample_rate", sample_rate))
                        print(f"    Stream started (Sample Rate: {sample_rate} Hz)")
                    elif status == "completed":
                        break
                    elif status == "error":
                        print_err(f"TTS WebSocket error: {data.get('error')}")
                        return None

        total_time = (time.perf_counter() - t0) * 1000
        full_pcm = b"".join(pcm_chunks)
        duration = (len(full_pcm) / 2) / sample_rate
        out_path = Path("tts_ws_output.wav")
        save_wav(out_path, full_pcm, sample_rate)

        print_ok(f"TTS WebSocket Success!")
        print(f"    Time-to-First-Audio: {first_chunk_time:.1f} ms" if first_chunk_time else "    TTFA: N/A")
        print(f"    Total Stream Time:   {total_time:.1f} ms")
        print(f"    Total Audio:         {len(full_pcm):,} bytes (~{duration:.2f} seconds)")
        print(f"    Saved WAV to:        {out_path.resolve()}")
        return full_pcm, sample_rate
    except Exception as e:
        print_err(f"TTS WebSocket test failed: {e}")
        return None


# ==============================================================================
# Phase 4: STT REST Test
# ==============================================================================
def test_stt_rest(host: str, port: int, pcm_16k: bytes) -> None:
    print_header("PHASE 4: Speech-to-Text (REST API)")
    url = f"http://{host}:{port}/api/v1/transcriptions"
    print(f"  Target:    {url}")
    print(f"  Audio PCM: {len(pcm_16k):,} bytes (16 kHz, 16-bit mono)")

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            url,
            data=pcm_16k,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        dt = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            print_err(f"STT REST failed with HTTP {resp.status_code}: {resp.text}")
            return

        result = resp.json()
        print_ok(f"STT REST Success [{dt:.1f}ms]!")
        print(f"    Recognized Text:  \033[92m\"{result.get('text', '')}\"\033[0m")
        print(f"    Confidence/Prob:  {result.get('language_prob', 1.0)}")
        segments = result.get("segments", [])
        if segments:
            print(f"    Segments ({len(segments)}):")
            for s in segments:
                print(f"      - [{s.get('start', 0.0):.2f}s -> {s.get('end', 0.0):.2f}s]: {s.get('text', '')}")
    except Exception as e:
        print_err(f"STT REST test failed: {e}")


# ==============================================================================
# Phase 5: STT WebSocket Test
# ==============================================================================
async def test_stt_ws(host: str, port: int, pcm_16k: bytes) -> None:
    print_header("PHASE 5: Speech-to-Text (WebSocket API)")
    ws_url = f"ws://{host}:{port}/ws/stt"
    print(f"  Target:    {ws_url}")
    print(f"  Streaming: {len(pcm_16k):,} bytes in 4KB chunks")

    try:
        t0 = time.perf_counter()
        async with websockets.connect(ws_url) as ws:
            # Send initial config frame
            await ws.send(json.dumps({"initial_prompt": "فارسی معیار"}))

            # Stream PCM in chunks
            chunk_size = 4096
            for i in range(0, len(pcm_16k), chunk_size):
                chunk = pcm_16k[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.01)

            # Send End-of-Stream control frame
            await ws.send(json.dumps({"action": "eos"}))

            # Receive transcription result
            resp = await ws.recv()
            dt = (time.perf_counter() - t0) * 1000

            result = json.loads(resp)
            if result.get("status") == "error":
                print_err(f"STT WebSocket returned error: {result.get('error')}")
                return

            print_ok(f"STT WebSocket Success [{dt:.1f}ms]!")
            print(f"    Recognized Text: \033[92m\"{result.get('text', '')}\"\033[0m")
            print(f"    Language Prob:   {result.get('language_prob', 1.0)}")
    except Exception as e:
        print_err(f"STT WebSocket test failed: {e}")


# ==============================================================================
# Main Runner
# ==============================================================================
def create_synthetic_beep(duration_sec: float = 1.5, sample_rate: int = 16000) -> bytes:
    """Generate a clean synthetic 440Hz tone as fallback audio if TTS is offline."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t) * 0.3
    return (tone * 32767).astype("<i2").tobytes()


async def run_all_tests(args: argparse.Namespace) -> None:
    host = args.host
    stt_port = args.stt_port
    tts_port = args.tts_port
    unified_port = args.unified_port
    text = args.text

    print_header(f"Nava Speech Platform E2E Tester -> Host: {host}")

    # Phase 1: Health & Discovery
    info = check_health_and_models(host, stt_port, tts_port, unified_port)

    # Determine TTS engine to test (default: mana if available, else first available)
    tts_engines = info.get("tts_engines", [])
    engine = "mana" if "mana" in tts_engines else (tts_engines[0] if tts_engines else "mana")

    # Phase 2: TTS REST
    tts_audio: bytes | None = None
    tts_rate = 22050
    if info["tts"]:
        res = test_tts_rest(host, tts_port, text, engine=engine)
        if res:
            tts_audio, tts_rate = res

    # Phase 3: TTS WebSocket
    if info["tts"]:
        res_ws = await test_tts_ws(host, tts_port, text, engine=engine)
        if not tts_audio and res_ws:
            tts_audio, tts_rate = res_ws

    # Prepare 16kHz audio for STT (use TTS output if available, else generate fallback)
    if tts_audio:
        print(f"\n  Resampling TTS output ({tts_rate} Hz) to 16,000 Hz for STT inference...")
        pcm_16k = resample_pcm16(tts_audio, tts_rate, 16000)
    else:
        print_warn("Using synthetic audio test sample for STT because TTS did not return audio.")
        pcm_16k = create_synthetic_beep(2.0, 16000)

    # Phase 4: STT REST
    if info["stt"]:
        test_stt_rest(host, stt_port, pcm_16k)

    # Phase 5: STT WebSocket
    if info["stt"]:
        await test_stt_ws(host, stt_port, pcm_16k)

    print_header("TEST SUMMARY COMPLETE")
    print("  Artifacts created in current directory:")
    for f in ["tts_rest_output.wav", "tts_ws_output.wav"]:
        p = Path(f)
        if p.exists():
            print(f"    - {p.resolve()} ({p.stat().st_size:,} bytes)")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Nava STT & TTS REST and WebSocket APIs")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server IP or hostname (default: {DEFAULT_HOST})")
    parser.add_argument("--stt-port", type=int, default=8001, help="STT service port (default: 8001)")
    parser.add_argument("--tts-port", type=int, default=8002, help="TTS service port (default: 8002)")
    parser.add_argument("--unified-port", type=int, default=8000, help="Unified server port (default: 8000)")
    parser.add_argument("--text", default=SAMPLE_PERSIAN_TEXT, help="Persian text prompt to synthesize and transcribe")

    args = parser.parse_args()
    asyncio.run(run_all_tests(args))


if __name__ == "__main__":
    main()


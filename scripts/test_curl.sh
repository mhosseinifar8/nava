#!/usr/bin/env bash
# Quick curl test for Nava STT and TTS services
set -Eeuo pipefail

HOST="${1:-192.168.14.207}"
STT_PORT="${2:-8001}"
TTS_PORT="${3:-8002}"

echo "======================================================"
echo " Nava Quick cURL Health & API Test -> $HOST"
echo "======================================================"

echo ""
echo "1. Checking STT Health (http://$HOST:$STT_PORT/healthz):"
curl -s "http://$HOST:$STT_PORT/healthz" || echo "STT unreachable"
echo ""

echo ""
echo "2. Checking TTS Health (http://$HOST:$TTS_PORT/healthz):"
curl -s "http://$HOST:$TTS_PORT/healthz" || echo "TTS unreachable"
echo ""

echo ""
echo "3. Testing TTS Synthesis (sending Persian text)..."
curl -s -X POST "http://$HOST:$TTS_PORT/api/v1/speech" \
     -H "Content-Type: application/json" \
     -d '{"text": "سلام، این یک پیام آزمایشی برای سامانه نوا است.", "engine": "mana", "speed": 1.0}' \
     --output tts_sample.pcm

if [[ -f tts_sample.pcm && -s tts_sample.pcm ]]; then
    echo "✓ TTS synthesized $(wc -c < tts_sample.pcm) bytes of audio (saved to tts_sample.pcm)"
    
    echo ""
    echo "4. Testing STT Transcription (sending synthesized audio back to STT)..."
    curl -s -X POST "http://$HOST:$STT_PORT/api/v1/transcriptions" \
         -H "Content-Type: application/octet-stream" \
         --data-binary @tts_sample.pcm
    echo ""
else
    echo "✗ TTS did not return audio."
fi

echo ""
echo "Done."


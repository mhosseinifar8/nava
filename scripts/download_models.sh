#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${NAVA_MODELS_DIR:-$ROOT/models}"
command -v hf >/dev/null || { echo "Install huggingface-hub[cli] first." >&2; exit 1; }
mkdir -p "$MODELS_DIR"/{whisper-large-v3,g2p,pocket,mana,piper,voices}
hf download Systran/faster-whisper-large-v3 --local-dir "$MODELS_DIR/whisper-large-v3"
hf download mehdi-hf/Homo-GE2PE-Persian-HF --local-dir "$MODELS_DIR/g2p"
hf download mehdi-hf/pocket-tts-farsi-v2 --local-dir "$MODELS_DIR/pocket"
hf download MahtaFetrat/Mana-Persian-Piper --local-dir "$MODELS_DIR/mana"
hf download rhasspy/piper-voices --local-dir "$MODELS_DIR/piper" fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx fa/fa_IR/ganji/medium/fa_IR-ganji-medium.onnx.json
hf download kyutai/tts-voices --local-dir "$MODELS_DIR/voices" alba-mackenna/casual.wav
[[ -e "$MODELS_DIR/voices/default.wav" ]] || ln -s alba-mackenna/casual.wav "$MODELS_DIR/voices/default.wav"

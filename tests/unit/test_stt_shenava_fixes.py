"""Unit tests for Shenava FastConformer metadata injection and resilient streaming."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nava.engines.stt_shenava import ShenavaCTCEngine, ShenavaStreamingEngine, ensure_nemo_ctc_metadata
from nava.server.app import app


def test_ensure_nemo_ctc_metadata_tokens_rewrite_and_meta():
    """Verify tokens.txt formatting and onnx metadata injection."""
    onnx = pytest.importorskip("onnx")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tokens_file = tmp_path / "tokens.txt"
        model_file = tmp_path / "model.onnx"

        # Create unindexed tokens.txt without blank
        tokens_file.write_text("سلام\nخوبید\nتست\n", encoding="utf-8")

        # Create a mock ONNX model
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, [1, 80, 100])
        Y = helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [1, 13, 4])
        length = helper.make_tensor_value_info("length", TensorProto.INT64, [1])
        node = helper.make_node("Constant", [], ["log_probs"], value=helper.make_tensor("v", TensorProto.FLOAT, [1, 13, 4], [0.0] * 52))
        graph = helper.make_graph([node], "ctc_graph", [X, length], [Y])
        opset = helper.make_opsetid("", 17)
        model = helper.make_model(graph, opset_imports=[opset], ir_version=8)
        onnx.save(model, str(model_file))

        # Run ensure_nemo_ctc_metadata
        ensure_nemo_ctc_metadata(model_file, tokens_file, subsampling_factor=8, normalize_type="")

        # Verify tokens.txt has IDs and <blk>
        tokens_text = tokens_file.read_text(encoding="utf-8")
        assert "<blk>" in tokens_text
        lines = [l for l in tokens_text.splitlines() if l.strip()]
        assert len(lines) == 4  # 3 original + 1 blank
        assert lines[-1].startswith("<blk>")

        # Verify ONNX model has metadata
        saved_m = onnx.load(str(model_file), load_external_data=False)
        meta = {p.key: p.value for p in saved_m.metadata_props}
        assert meta["vocab_size"] == "4"
        assert meta["subsampling_factor"] == "8"
        assert meta["normalize_type"] == ""
        assert meta["model_type"] == "EncDecCTCModelBPE"


def test_shenava_streaming_buffered_ctc_fallback():
    """Verify ShenavaStreamingEngine uses buffered CTC when causal streaming states are absent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        engine = ShenavaStreamingEngine("shenava-rizeh-streaming-32m", tmp_path)

        stream_obj = engine.create_stream()
        assert stream_obj is not None
        # Feed dummy PCM
        dummy_pcm = b"\x00\x00" * 1600
        engine.feed_audio(stream_obj, dummy_pcm)
        partial = engine.get_partial(stream_obj)
        assert isinstance(partial, str)
        raw, itn = engine.get_final(stream_obj)
        assert isinstance(raw, str)
        assert isinstance(itn, str)


def test_stt_websocket_aliases_and_eos():
    """Test STT WebSocket handles query parameters and various EOS messages."""
    client = TestClient(app)
    endpoints = [
        "/ws/stt",
        "/v1/audio/transcriptions",
        "/api/v1/transcriptions",
    ]

    for ep in endpoints:
        with client.websocket_connect(f"{ep}?model=shenava-koochik-ctc&enable_itn=true") as ws:
            # Send small audio chunk
            pcm_chunk = b"\x00\x00" * 1600
            ws.send_bytes(pcm_chunk)

            # Send empty byte frame as EOS
            ws.send_bytes(b"")
            resp = ws.receive_json()
            assert "text" in resp or "status" in resp

"""Nava CPU Runtime Package."""
from .cpu_tuning import (
    configure_cpu_environment,
    configure_event_loop,
    create_ort_session_options,
    detect_cpu_avx512_bf16,
    get_heavy_executor,
    get_realtime_executor,
    patch_global_onnxruntime,
    run_in_cpu_pool,
)

__all__ = [
    "configure_cpu_environment",
    "configure_event_loop",
    "create_ort_session_options",
    "detect_cpu_avx512_bf16",
    "get_heavy_executor",
    "get_realtime_executor",
    "patch_global_onnxruntime",
    "run_in_cpu_pool",
]


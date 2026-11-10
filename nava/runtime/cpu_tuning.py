"""Nava High-Performance CPU Runtime, Thread Pool, and ONNX Runtime Tuning.

Designed for 16-to-24 core, 32GB-to-64GB RAM CPU servers (no GPU):
- Sets OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, ORT_TENSORRT_UNAVAILABLE=1
- Configures ONNX Runtime SessionOptions with ORT_SEQUENTIAL, ORT_ENABLE_ALL,
  enable_cpu_mem_arena=True, enable_mem_pattern=True, intra_op_num_threads=4, inter_op_num_threads=1
- Installs uvloop when available
- Provides pre-warmed ThreadPoolExecutors for real-time STT/TTS vs. heavy OCR/LLM tasks
"""
from __future__ import annotations

import asyncio
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# 1. Configure environment variables immediately on import to prevent OpenMP thread thrashing
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("ORT_TENSORRT_UNAVAILABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ORT_PATCHED = False
_EXECUTOR_LOCK = threading.Lock()
_REALTIME_EXECUTOR: ThreadPoolExecutor | None = None
_HEAVY_EXECUTOR: ThreadPoolExecutor | None = None


def detect_cpu_avx512_bf16() -> bool:
    """Detect whether the host CPU supports AVX512_BF16 instructions."""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                info = f.read().lower()
                return "avx512_bf16" in info
    except Exception:
        pass
    return False


def create_ort_session_options(
    intra_op_threads: int | None = None,
    inter_op_threads: int = 1,
    model_tier: str = "realtime",
) -> Any:
    """Create a CPU-tuned onnxruntime.SessionOptions instance.

    For fast real-time models (parakeet-ctc-109m, pocket-tts, matcha, piper, snappfood):
      intra_op_num_threads = 4 (or 6), inter_op_num_threads = 1
      ExecutionMode.ORT_SEQUENTIAL, GraphOptimizationLevel.ORT_ENABLE_ALL
      enable_cpu_mem_arena = True, enable_mem_pattern = True
    """
    import onnxruntime as ort

    if intra_op_threads is None:
        if model_tier == "heavy":
            intra_op_threads = int(os.getenv("NAVA_HEAVY_ORT_THREADS", "8"))
        else:
            intra_op_threads = int(os.getenv("NAVA_ORT_INTRA_THREADS", os.getenv("OMP_NUM_THREADS", "4")))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, intra_op_threads)
    opts.inter_op_num_threads = max(1, inter_op_threads)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    return opts


def patch_global_onnxruntime() -> None:
    """Patch onnxruntime.SessionOptions so internal libraries (e.g. sherpa-onnx) inherit CPU tuning."""
    global _ORT_PATCHED
    if _ORT_PATCHED:
        return
    try:
        import onnxruntime as ort

        orig_session_options = ort.SessionOptions

        class TunedSessionOptions(orig_session_options):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                intra = int(os.getenv("NAVA_ORT_INTRA_THREADS", os.getenv("OMP_NUM_THREADS", "4")))
                self.intra_op_num_threads = max(1, intra)
                self.inter_op_num_threads = 1
                self.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.enable_cpu_mem_arena = True
                self.enable_mem_pattern = True

        ort.SessionOptions = TunedSessionOptions  # type: ignore[misc]
        _ORT_PATCHED = True
    except Exception:
        pass


def configure_cpu_environment(
    omp_threads: int = 4,
    torch_threads: int = 8,
) -> dict[str, Any]:
    """Configure OpenMP, MKL, ONNX Runtime, and PyTorch CPU threading."""
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"

    patch_global_onnxruntime()

    torch_configured = False
    try:
        import sys
        if "torch" in sys.modules:
            import torch
            torch.set_num_threads(max(1, torch_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            torch_configured = True
    except Exception:
        pass

    return {
        "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
        "mkl_num_threads": int(os.environ["MKL_NUM_THREADS"]),
        "ort_tensorrt_unavailable": os.environ["ORT_TENSORRT_UNAVAILABLE"],
        "ort_patched": _ORT_PATCHED,
        "torch_configured": torch_configured,
    }


def configure_event_loop() -> str:
    """Install uvloop as the default asyncio event loop policy if available."""
    try:
        import uvloop

        if not isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy):
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        return "uvloop"
    except ImportError:
        return "asyncio"


def get_realtime_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a pre-warmed ThreadPoolExecutor dedicated to real-time STT/TTS inference."""
    global _REALTIME_EXECUTOR
    if _REALTIME_EXECUTOR is not None:
        return _REALTIME_EXECUTOR

    with _EXECUTOR_LOCK:
        if _REALTIME_EXECUTOR is None:
            if max_workers is None:
                cpu_count = os.cpu_count() or 16
                # On a 16-24 core machine with 4 OMP threads per worker, 4-8 workers is optimal
                default_workers = max(4, min(8, cpu_count // 2))
                max_workers = int(os.getenv("NAVA_INFERENCE_WORKERS", str(default_workers)))
            pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nava-rt-cpu")
            # Pre-warm worker threads so first inference call has zero OS thread spawn latency
            futures = [pool.submit(lambda x: x, idx) for idx in range(max_workers)]
            for f in futures:
                try:
                    f.result(timeout=1.0)
                except Exception:
                    pass
            _REALTIME_EXECUTOR = pool
    return _REALTIME_EXECUTOR


def get_heavy_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    """Return a separate ThreadPoolExecutor for heavy tasks (OCR, LLM) to avoid starving STT/TTS."""
    global _HEAVY_EXECUTOR
    if _HEAVY_EXECUTOR is not None:
        return _HEAVY_EXECUTOR

    with _EXECUTOR_LOCK:
        if _HEAVY_EXECUTOR is None:
            workers = int(os.getenv("NAVA_HEAVY_WORKERS", str(max_workers)))
            _HEAVY_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="nava-heavy-cpu")
    return _HEAVY_EXECUTOR


async def run_in_cpu_pool(
    fn: Callable[..., T],
    *args: Any,
    heavy: bool = False,
    **kwargs: Any,
) -> T:
    """Offload a synchronous CPU-bound function to the pre-warmed ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    executor = get_heavy_executor() if heavy else get_realtime_executor()
    if kwargs:
        bound = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(executor, bound)
    return await loop.run_in_executor(executor, fn, *args)


# Apply global ORT session patch on module import
patch_global_onnxruntime()


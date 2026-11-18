"""Nava Server module."""
from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    # Lazy import so microservices can mount routes_models without loading the full app.
    if name in {"app", "create_app"}:
        from .app import app, create_app

        return app if name == "app" else create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

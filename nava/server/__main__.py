"""Entrypoint for running nava server via python -m nava.server."""
import sys
from .app import main

if __name__ == "__main__":
    sys.exit(main())


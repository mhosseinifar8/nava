#!/usr/bin/env python3
"""CLI script to download and cache Nava Persian Speech models."""
import sys
from pathlib import Path

# Add project root to sys.path so nava can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nava.models.downloader import main

if __name__ == "__main__":
    sys.exit(main())


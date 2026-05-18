"""Import helpers for the existing hybrid detector package."""

import sys
from pathlib import Path

from .config_harmonic_guard import HYBRID_SRC_DIR


def add_hybrid_path():
    path = Path(HYBRID_SRC_DIR)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

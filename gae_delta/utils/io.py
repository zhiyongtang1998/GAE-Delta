"""File I/O utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(results: Dict[str, Any], output_path: str | Path) -> None:
    """Save results dict to JSON file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {k: _convert(v) for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

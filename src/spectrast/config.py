"""Path resolution for the spectrast project.

All filesystem paths are resolved here, honoring the ``QST_DATA_ROOT``
environment variable so identical code runs locally (MPS) and on the HPC (CUDA).
Never hardcode paths elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/spectrast/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root for data inputs. Override with QST_DATA_ROOT (e.g. an HPC mount)."""
    env = os.environ.get("QST_DATA_ROOT")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


def raw_dir() -> Path:
    """Immutable inputs."""
    return data_root() / "raw"


def processed_dir() -> Path:
    """Derived shared objects (quaternion matrices, embeddings) — inputs to analyses."""
    return data_root() / "processed"


def external_dir() -> Path:
    """Atlases, references, downloaded example datasets."""
    return data_root() / "external"


def results_dir(analysis: str, ensure: bool = True) -> Path:
    """Reportable outputs for a given analysis type (figures + CSV + legend, METHODS_/RESULTS_).

    Creates the directory by default; pass ``ensure=False`` for a pure path resolve with no filesystem side
    effect (e.g. enumerating paths on a dry run or a read-only mount)."""
    p = REPO_ROOT / "results" / analysis
    if ensure:
        p.mkdir(parents=True, exist_ok=True)
    return p

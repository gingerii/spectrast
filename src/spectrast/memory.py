"""Automatic memory management for large / whole-transcriptome spatial statistics.

Two structural things let this toolkit scale to whole-transcriptome panels (~18k genes × hundreds of
thousands of cells) without the user tuning anything:

* the SPARSE-AWARE SVD (``spectrast.spatial_sc.svd_embeddings``) — auto-selected whenever the input is a
  scipy sparse matrix; it never densifies the N×G matrix (implicit-centred randomised SVD);
* memory-sized CHUNKING — this module picks per-gene chunk sizes from the *live* free-RAM budget, so
  functions that reconstruct per-cell (N × genes) maps stay within memory automatically.

Everything here is cheap/pure and imported lazily; heavy work stays in the callers.
"""
from __future__ import annotations
import numpy as np


def available_bytes(frac: float = 0.6, fallback_gb: float = 4.0) -> int:
    """Usable memory budget = ``frac`` of currently-available RAM (via psutil), or of a safe fallback if
    psutil is unavailable. ``frac`` is applied in BOTH cases — otherwise the fallback would ignore a caller's
    requested fraction (e.g. a tiny probe budget) and report a phantom fixed size."""
    try:
        import psutil  # lazy
        total = psutil.virtual_memory().available
    except Exception:
        total = int(fallback_gb * 1024 ** 3)
    return int(total * frac)


def gene_chunk(n_rows: int, *, n_buffers: int = 8, dtype_bytes: int = 8,
               floor: int = 64, cap: int = 128, budget: int | None = None) -> int:
    """Largest number of gene columns ``m`` such that an (``n_rows`` × ``m``) array — with
    ``n_buffers`` transient working copies — fits the memory budget. Clamped to [floor, cap].

    Used to stream per-cell reconstructions over genes: for small data ``m`` covers all genes in one
    pass (no overhead), for whole-transcriptome data it shrinks automatically to stay in RAM.

    The defaults are deliberately CONSERVATIVE (``n_buffers=8``, ``cap=128``): the per-cell local-map
    functions (focality / local Moran / Getis / local Geary) hold ~6 transient N×m arrays, and on
    hundreds-of-thousands of cells a budget-filling block spikes to tens of GB and trips the OS memory
    killer under real load. A per-chunk FFT dominates the cost, so a modest column block is essentially
    free in time while keeping the peak bounded (~n_rows·cap·n_buffers·8 bytes). Chunking is exact —
    the block size never changes a returned value.

    ``floor`` is a soft minimum: it applies only when it FITS the budget. Under real memory pressure the
    result can drop below ``floor`` (down to 1) rather than force a block larger than RAM allows (M6).
    """
    budget = available_bytes() if budget is None else int(budget)
    per_col = max(int(n_rows) * int(dtype_bytes) * int(n_buffers), 1)
    raw = int(budget // per_col)
    lo = floor if floor <= raw else max(raw, 1)                 # never force above what the budget can fit
    return int(np.clip(raw, lo, cap))


def densify_gb(X, dtype_bytes: int = 8) -> float:
    """GB an array / sparse matrix would occupy if made fully dense."""
    shp = getattr(X, "shape", None)
    if not shp:
        return 0.0
    return float(np.prod(shp)) * dtype_bytes / 1024 ** 3


def is_large(X, threshold_gb: float = 1.0) -> bool:
    """True if densifying ``X`` would exceed ``threshold_gb`` — a signal to stay sparse / stream."""
    return densify_gb(X) > threshold_gb


def guard_shape(shape, *, frac: float = 0.6, dtype_bytes: int = 8,
                hint: str = "pass fewer genes or select a gene subset") -> float:
    """Raise ``MemoryError`` if an array of ``shape`` would exceed the LIVE free-RAM budget — for an OUTPUT
    array whose size is set by the request (e.g. an (N × n_genes) local-map matrix), turning a silent OOM
    into an explicit early signal. Returns the size in GB. """
    need = float(np.prod(shape)) * dtype_bytes / 1024 ** 3
    budget = available_bytes(frac) / 1024 ** 3
    if need > budget:
        raise MemoryError(f"allocating an array of shape {tuple(shape)} ≈ {need:.1f} GB exceeds the "
                          f"~{budget:.1f} GB free-RAM budget; {hint}")
    return need


def guard_densify(X, *, frac: float = 0.6, hint: str = "use reduced_rank=True or pass fewer genes"):
    """Raise ``MemoryError`` if densifying ``X`` would exceed the LIVE free-RAM budget — turning a silent
    OOM into an explicit early signal. Call on the ALREADY column-sliced subset, so the check sees the
    genes actually requested (not the whole panel). Budget-relative (not a fixed cap) so it never trips a
    request that genuinely fits. Returns the dense size in GB (for logging). """
    need = densify_gb(X)
    budget = available_bytes(frac) / 1024 ** 3
    if need > budget:
        raise MemoryError(f"densifying {getattr(X, 'shape', None)} ≈ {need:.1f} GB exceeds the "
                          f"~{budget:.1f} GB free-RAM budget; {hint}")
    return need

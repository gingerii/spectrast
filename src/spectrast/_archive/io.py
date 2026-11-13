"""I/O: map spatial data (Visium, Visium HD, binned Xenium) to a quaternion grid.

Everything funnels through :func:`to_quaternion_grid`, which is resolution-agnostic
-- a Visium HD section is just a larger grid than a Visium one. Heavy deps
(anndata/scanpy) are lazy-imported so importing spectrast stays cheap.
"""
from __future__ import annotations

import numpy as np

from .quaternion import map_svd


def bin_coords_to_grid(x: np.ndarray, y: np.ndarray, n_rows: int | None = None,
                       n_cols: int | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Map continuous coords to integer grid indices.

    Returns (row_idx, col_idx, n_rows, n_cols). If n_rows/n_cols are given, the
    coordinates are scaled into that grid; otherwise the grid resolution is the
    number of unique coordinate values (natural for Visium / Visium HD, which are
    already lattice-based).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if n_rows is None or n_cols is None:
        ux = np.unique(x); uy = np.unique(y)
        n_cols = len(ux) if n_cols is None else n_cols
        n_rows = len(uy) if n_rows is None else n_rows
    col = np.clip(((x - x.min()) / max(np.ptp(x), 1e-9) * (n_cols - 1)).round().astype(int), 0, n_cols - 1)
    row = np.clip(((y - y.min()) / max(np.ptp(y), 1e-9) * (n_rows - 1)).round().astype(int), 0, n_rows - 1)
    return row, col, n_rows, n_cols


def lattice_indices(coords: np.ndarray, spacing=None):
    """Snap regular-lattice spatial coords (e.g. Visium HD square bins) to integer
    (row, col) indices. Returns (row, col, n_rows, n_cols, (sx, sy)).

    Bin spacing is inferred per axis as the median positive gap between sorted unique
    coordinates (exact for a true lattice), unless `spacing` is given.
    """
    x = np.asarray(coords[:, 0], float); y = np.asarray(coords[:, 1], float)

    def _spacing(v):
        u = np.unique(np.round(v, 6))
        d = np.diff(u)
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0

    if spacing is None:
        sx, sy = _spacing(x), _spacing(y)
    else:
        sx = sy = float(spacing)
    col = np.rint((x - x.min()) / sx).astype(int)
    row = np.rint((y - y.min()) / sy).astype(int)
    return row, col, int(row.max() + 1), int(col.max() + 1), (sx, sy)


def place_on_grid(quats: np.ndarray, row: np.ndarray, col: np.ndarray,
                  n_rows: int, n_cols: int, fill_iters: int = 0) -> np.ndarray:
    """Scatter per-location hypercomplex values (N, C) onto an (n_rows, n_cols, C) grid.

    C is inferred from `quats` (4 for quaternions, 8 for octonions, etc.). Collisions
    average. Empty interior cells optionally filled with the mean of non-empty
    8-neighbors (`fill_iters` passes), mirroring QSC's fill-in.
    """
    C = quats.shape[1]
    grid = np.zeros((n_rows, n_cols, C))
    count = np.zeros((n_rows, n_cols))
    np.add.at(grid, (row, col), quats)
    np.add.at(count, (row, col), 1.0)
    nz = count > 0
    grid[nz] /= count[nz][:, None]

    for _ in range(fill_iters):
        empty = count == 0
        if not empty.any():
            break
        neigh_sum = np.zeros_like(grid); neigh_cnt = np.zeros((n_rows, n_cols))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(grid, dr, 0), dc, 1)
                scount = np.roll(np.roll(count > 0, dr, 0), dc, 1)
                neigh_sum += shifted * scount[..., None]
                neigh_cnt += scount
        fillable = empty & (neigh_cnt > 0)
        grid[fillable] = neigh_sum[fillable] / neigh_cnt[fillable][:, None]
        count[fillable] = 1.0
    return grid


def to_quaternion_grid(adata=None, counts=None, coords=None, n_rows=None, n_cols=None,
                       fill_iters: int = 2, layer: str | None = None):
    """Spatial data -> (grid, per-location quaternions, (row, col)).

    Provide either an AnnData (`adata`, uses `obsm['spatial']` and counts from
    `layer` or `.X`) or raw `counts` (N x genes) + `coords` (N x 2). Counts must be
    UNNORMALIZED (the SVD mapping expects raw counts). Works for Visium, Visium HD
    (finer lattice -> bigger grid), and binned Xenium identically.
    """
    from scipy.sparse import issparse  # lazy
    if adata is not None:
        X = adata.layers[layer] if layer else adata.X
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    else:
        X = counts
    if X is None or coords is None:
        raise ValueError("provide adata, or counts + coords")
    quats = map_svd(X.toarray() if issparse(X) else np.asarray(X))
    x, y = coords[:, 0], coords[:, 1]
    row, col, n_rows, n_cols = bin_coords_to_grid(x, y, n_rows, n_cols)
    grid = place_on_grid(quats, row, col, n_rows, n_cols, fill_iters=fill_iters)
    return grid, quats, (row, col)

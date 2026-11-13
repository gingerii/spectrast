"""Adaptive rasterization for single-cell ST on the FFT engine.

Grid platforms (Visium, Visium HD) are physical lattices; imaging-based platforms (Xenium,
MERFISH) are point clouds that must be RASTERIZED (snap cells to a grid, average within-bin)
before the FFT spatial-statistics engine applies. Rasterization error in LOCAL statistics is
governed by the ratio r/b (neighbourhood radius / bin size) plus per-bin occupancy. This module turns that characterization into two tools:

* :func:`autotune_grid` — DATA-DRIVEN choice of bin size that balances FFT cost (grows as the
  grid, ~1/b^2) against graph-fusion cost (grows with the flagged fraction f(b)), using only
  cheap FFTs (no exact graph) to estimate f(b) via a bin-vs-half-bin consistency proxy.
* :func:`adaptive_local_moran` — run the FFT local statistic at a chosen/auto bin, then
  automatically recompute the highest-error cells with the exact neighbourhood graph and
  splice them in (the "fusion" fallback). Restores exact-grade local values for the sensitive
  minority at a fraction of the full-graph cost.

Heavy deps (scipy) are imported lazily so importing spectrast stays light.
"""
from __future__ import annotations
import numpy as np

from . import _fft  # pluggable FFT backend (cpu-threaded / cuda / mps)


# ----------------------------------------------------------------- rasterization primitives
def grid_index(xy, bin_size):
    """Snap continuous coords to an (H, W) lattice with pitch ~bin_size (production rule:
    round to nearest node). Returns (row, col, H, W, pitch_x).

    ``pitch`` is the x-axis pitch; a near-isotropic grid is assumed (H, W round independently, so pitch_y can
    differ by up to one rounding step — negligible except at extreme tissue aspect ratios). Downstream disc
    radii use this single pitch for both axes."""
    xy = np.asarray(xy, float); x, y = xy[:, 0], xy[:, 1]
    W = max(int(round(np.ptp(x) / bin_size)) + 1, 2)
    H = max(int(round(np.ptp(y) / bin_size)) + 1, 2)
    col = np.clip(((x - x.min()) / max(np.ptp(x), 1e-9) * (W - 1)).round().astype(int), 0, W - 1)
    row = np.clip(((y - y.min()) / max(np.ptp(y), 1e-9) * (H - 1)).round().astype(int), 0, H - 1)
    pitch = np.ptp(x) / (W - 1) if W > 1 else float(bin_size)
    return row, col, H, W, pitch


def occupancy(xy, bin_size):
    """Per-cell number of cells sharing its bin (1 = alone). Diagnostic for per-cell resolution."""
    row, col, H, W, _ = grid_index(xy, bin_size)
    cnt = np.zeros((H, W)); np.add.at(cnt, (row, col), 1.0)
    return cnt[row, col]


def _disc(r_bins):
    r = int(round(r_bins)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (yy ** 2 + xx ** 2 <= r ** 2).astype(float); k[r, r] = 0.0
    return k


_CONV_FFT_MIN_SIDE = 9    # kernels this wide+ go through FFT (radius-independent); smaller stay direct.
# Matches spatial._CONV_FFT_MIN_SIDE. The autotuned single-cell bin targets r/b≈5 → disc side 11, which
# MUST take the FFT branch (direct is 2–3× slower here and reintroduces radius-dependence into the path
# the toolkit advertises as radius-independent). FFT==direct to machine precision for symmetric kernels.

_SPARSE_DIRECT_CONST = 8   # constant-factor guard for the small-kernel branch: use the cell-only sparse-direct
# convolution when nnz·CONST < H·W (grid genuinely sparse, e.g. rasterized Xenium/MERFISH — sparse-direct is
# 2–3× faster than dense ndimage there), else fall back to the optimized-C dense ndimage (faster on dense grids
# / Visium HD, where sparse-direct's numpy orchestration loses on constant factors). Calibrated so a sparse
# 3µm-bin grid (S≈0.04) routes to sparse-direct and a denser 6µm-bin grid (S≈0.14) routes to dense.


def _sparse_direct_conv2d(stack, kernel):
    """Cell-only SPARSE-DIRECT 'same'-size zero-boundary convolution: scatter each NONZERO input
    bin over the kernel footprint (convolution semantics — matches ``ndimage.convolve(mode=
    'constant')`` for any kernel, symmetric or not). Cost is O(M · nnz(kernel)) with M = occupied
    bins, versus dense ndimage O(H·W · K²): empty bins (the vast majority of a rasterized single-cell
    grid) cost nothing. This is the sparsity-AWARE fallback for small kernels, where a tight
    O(M·K²) direct convolution beats the FFT. Batched over the
    trailing axis of a (H,W,K) stack. Zero-valued occupied bins are correctly skipped — they
    contribute 0 to the convolution regardless."""
    is2d = stack.ndim == 2
    s3 = stack[:, :, None] if is2d else stack
    H, W, Kc = s3.shape
    kh, kw = kernel.shape
    r0, c0 = (kh - 1) // 2, (kw - 1) // 2
    sr, sc = np.nonzero(np.any(s3 != 0.0, axis=2))       # occupied (nonzero) source bins only
    out = np.zeros((H * W, Kc))
    if sr.size:
        vals = s3[sr, sc, :]                              # (M, Kc) source values
        ky, kx = np.nonzero(kernel)                       # kernel taps
        kv = kernel[ky, kx]
        idx_list, val_list = [], []
        for w, dy, dx in zip(kv, ky - r0, kx - c0):       # convolution: out[sr+dy, sc+dx] += w * v
            tr, tc = sr + dy, sc + dx
            m = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)   # zero boundary = drop out-of-grid taps
            if m.any():
                idx_list.append(tr[m] * W + tc[m]); val_list.append(w * vals[m])
        if idx_list:
            idx = np.concatenate(idx_list); val = np.concatenate(val_list, axis=0)
            for c in range(Kc):                           # bincount scatter-add (fast C path)
                out[:, c] += np.bincount(idx, weights=val[:, c], minlength=H * W)
    out = out.reshape(H, W, Kc)
    return out[:, :, 0] if is2d else out


def _conv2d(stack, kernel):
    """'Same'-size zero-boundary convolution of (H,W) or (H,W,K) with a 2D kernel, batched over K.
    Two-stage dispatch. Large kernels use the FFT (cost independent of kernel radius) — so the shipped
    single-cell path is genuinely radius-independent, not O(r²). Small kernels choose between the
    CELL-ONLY sparse-direct convolution (:func:`_sparse_direct_conv2d`, O(M · K²), sparsity-aware,
    2–3× faster on the sparse rasterized grid) and the optimized-C dense ``ndimage`` (faster on dense
    grids, where sparse-direct's numpy orchestration loses on constant factors), via a cheap nnz-vs-grid
    guard. All three paths match ``ndimage.convolve(mode='constant')`` (zero padding) to machine
    precision (symmetric disc kernels ⇒ convolution == correlation)."""
    kh, kw = kernel.shape
    if max(kh, kw) < _CONV_FFT_MIN_SIDE:
        H, W = stack.shape[:2]
        occ2d = stack if stack.ndim == 2 else np.any(stack != 0.0, axis=2)
        if int(np.count_nonzero(occ2d)) * _SPARSE_DIRECT_CONST < H * W:   # grid genuinely sparse
            return _sparse_direct_conv2d(stack, kernel)
        from scipy import ndimage  # lazy — dense grid: optimized C wins on constant factors
        k = kernel[:, :, None] if stack.ndim == 3 else kernel
        return ndimage.convolve(stack, k, mode="constant")
    H, W = stack.shape[:2]
    fh, fw = H + kh - 1, W + kw - 1
    s3 = stack if stack.ndim == 3 else stack[:, :, None]
    Fz = _fft.rfft2(s3, s=(fh, fw), axes=(0, 1))
    Fk = _fft.rfft2(kernel, s=(fh, fw))[:, :, None]
    full = _fft.irfft2(Fz * Fk, s=(fh, fw), axes=(0, 1))
    r0, c0 = (kh - 1) // 2, (kw - 1) // 2
    out = full[r0:r0 + H, c0:c0 + W]
    return out if stack.ndim == 3 else out[:, :, 0]


def fft_local_mean_neighbor(xy, z, radius_um, bin_size):
    """Mask-normalized mean of z over the radius-``radius_um`` neighbourhood, via one FFT/direct
    convolution on the rasterized grid, read back PER CELL. Same estimand as the exact graph.

    Bin-weighted (averages over the grid); the shipped fusion path uses the cell-weighted
    :func:`_fft_neighbor_reduce` + a ``combine_*`` instead. Kept for the diagnostic proxy display; prefer
    :func:`adaptive_local_moran` for analysis (the two can differ on occupancy>1 grids)."""
    row, col, H, W, pitch = grid_index(xy, bin_size)
    r_bins = max(int(round(radius_um / pitch)), 1)
    val = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(val, (row, col), z); np.add.at(cnt, (row, col), 1.0)
    occ = cnt > 0
    mean_grid = np.zeros((H, W)); mean_grid[occ] = val[occ] / cnt[occ]
    k = _disc(r_bins)
    num = _conv2d(mean_grid, k)
    den = _conv2d(occ.astype(float), k)
    loc = np.full((H, W), np.nan); nz = den > 0; loc[nz] = num[nz] / den[nz]   # NaN where no neighbour, not 0
    return loc[row, col]


def fft_local_moran(xy, values, radius_um, bin_size):
    """Per-cell local Moran's I on the rasterized grid (I_i = z_i * mean_neighbour z_j)."""
    v = np.asarray(values, float); z = (v - v.mean()) / (v.std() + 1e-9)
    return z * fft_local_mean_neighbor(xy, z, radius_um, bin_size)


def rasterization_proxy(xy, values, radius_um, bin_size):
    """GRAPH-FREE per-cell error estimate: |I(b) - I(b/2)| / SD(I(b)). Cells whose statistic shifts when the
    grid is refined are the ones rasterization is unreliable for. Validated to track true error (Spearman ~0.75
    vs exact-graph); needs no neighbourhood graph.

    M15: computed through the SAME cell-weighted reduction the shipped fusion uses
    (:func:`_fft_neighbor_reduce` + :func:`combine_moran`), not the bin-weighted ``fft_local_mean_neighbor`` —
    so ``autotune_grid``'s flagged fraction matches the fraction real fusion actually flags."""
    v = np.asarray(values, float); z = (v - v.mean()) / (v.std() + 1e-9)
    ns_b, nc_b = _fft_neighbor_reduce(xy, z, radius_um, bin_size)
    ns_h, nc_h = _fft_neighbor_reduce(xy, z, radius_um, bin_size / 2.0)
    I_b = combine_moran(ns_b, nc_b, z)
    I_h = combine_moran(ns_h, nc_h, z)
    return np.abs(I_b - I_h) / (np.nanstd(I_b) + 1e-12)


# ----------------------------------------------------------------- exact graph (fusion patch)
def _exact_local_mean_neighbor(xy, z, radius_um, sub_idx=None, tree=None):
    """Exact mean of z over true-coordinate neighbours within radius (self excluded). If
    ``sub_idx`` is given, compute only for that subset of cells (the fusion set)."""
    from scipy.spatial import cKDTree  # lazy
    tree = tree or cKDTree(xy)
    q = xy if sub_idx is None else xy[sub_idx]
    ids = np.arange(len(xy)) if sub_idx is None else np.asarray(sub_idx)
    lists = tree.query_ball_point(q, radius_um)
    out = np.zeros(len(ids))
    for i, nb in enumerate(lists):
        nb = [j for j in nb if j != ids[i]]      # self excluded by GLOBAL index
        if nb:
            out[i] = z[nb].mean()
    return out


# ----------------------------------------------------------------- data-driven bin selection
def autotune_grid(xy, r_min_um, candidate_bins=None, probe_values=None,
                  max_fuse_frac=0.25, occ_cap=2.0, tol=0.1, seed=0):
    """Choose a bin size that balances FFT speed against fusion cost, data-drivenly.

    Rationale: FFT cost grows as the grid (~ pixels ~ 1/b^2); the graph-fusion fallback costs
    in proportion to the flagged fraction f(b), which grows as b coarsens. We want the LARGEST
    bin (fastest FFT / smallest grid) that (a) fusion can still clean up within budget
    [f(b) <= ``max_fuse_frac``] and (b) preserves ~per-cell resolution [median occupancy <=
    ``occ_cap``]. f(b) is estimated with the graph-free proxy at the smallest analysis radius
    ``r_min_um`` (the binding case) on a probe field — no exact graph needed.

    Parameters
    ----------
    xy : (N,2) cell coordinates (microns).
    r_min_um : smallest neighbourhood radius you intend to analyse (sets the binding r/b).
    candidate_bins : bins to evaluate (microns); default = geometric sweep around r_min/5.
    probe_values : (N,) a representative gene/score to estimate f(b); default = a synthetic
        random field of scale r_min (structure at the relevant scale).
    max_fuse_frac : max flagged fraction fusion is allowed to carry.
    occ_cap : max median cells/bin (keep per-cell resolution).
    tol : proxy tolerance defining a flagged cell.

    Returns
    -------
    dict with recommended ``bin_um``, its ``r_over_b``/``flagged_frac``/``median_occupancy``,
    the fusion-free bin (``r_min``/5), and a per-candidate ``table`` (for plotting the tradeoff).
    """
    xy = np.asarray(xy, float)
    if candidate_bins is None:
        base = r_min_um / 5.0
        candidate_bins = np.round(base * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]), 2)
    if probe_values is None:
        probe_values = _synthetic_probe(xy, r_min_um, seed)

    # Evaluate candidates COARSEST-first, occupancy (cheap) before the proxy (expensive at fine bins):
    # the first bin that passes both is the coarsest feasible = the answer, so we never pay the fine-bin
    # rasterization proxy (b/2 grid ∝ 1/b², the dominant cost) for bins finer than the pick. Same result.
    rows = []
    feasible = []
    for b in sorted((float(x) for x in candidate_bins), reverse=True):
        H, W = grid_index(xy, b)[2:4]
        med_occ = float(np.median(occupancy(xy, b)))
        row = dict(bin_um=b, r_over_b=float(r_min_um / b), pixels=int(H * W),
                   median_occupancy=med_occ, flagged_frac=np.nan)
        if med_occ <= occ_cap:                              # per-cell resolution ok → check fusion cost
            row["flagged_frac"] = float((rasterization_proxy(xy, probe_values, r_min_um, b) > tol).mean())
            if row["flagged_frac"] <= max_fuse_frac:
                rows.append(row); feasible.append(row); break   # coarsest feasible → stop (skip finer bins)
        rows.append(row)
    if not feasible and not any(r["flagged_frac"] == r["flagged_frac"] for r in rows):
        # m26: every candidate exceeded occ_cap so no proxy was computed — compute it on the finest bin so the
        # fallback pick has a real flagged_frac (not a silent NaN row with max_fuse_frac never enforced).
        finest = min(rows, key=lambda r: r["bin_um"])
        finest["flagged_frac"] = float((rasterization_proxy(xy, probe_values, r_min_um, finest["bin_um"]) > tol).mean())
    table = rows
    pick = (max(feasible, key=lambda r: r["bin_um"]) if feasible
            else min((r for r in rows if r["flagged_frac"] == r["flagged_frac"]),
                     key=lambda r: r["flagged_frac"], default=rows[-1]))
    return dict(bin_um=pick["bin_um"], r_over_b=pick["r_over_b"],
                flagged_frac=pick["flagged_frac"], median_occupancy=pick["median_occupancy"],
                fusion_free_bin_um=float(r_min_um / 5.0), r_min_um=float(r_min_um),
                feasible=bool(feasible), table=table)


def _synthetic_probe(xy, scale_um, seed=0):
    """A random field with structure at ``scale_um``, sampled at cell coords (probe for f(b))."""
    from scipy import ndimage  # lazy
    fine = max(scale_um / 8.0, 1.0)
    row, col, H, W, _ = grid_index(xy, fine)
    white = np.random.default_rng(seed).standard_normal((H, W))
    field = ndimage.gaussian_filter(white, (scale_um / fine) / 2.0, mode="wrap")
    return field[row, col]


# ================================================================= GENERIC local-statistic fusion
# Every per-cell local statistic in the toolkit is a pointwise function of NEIGHBOURHOOD REDUCTIONS
# (sums/counts of one or more fields over the radius-r disc) plus the cell's own value. Rasterization
# error enters only through those reductions. So one reduce primitive (FFT + exact) + a per-tool
# `combine` gives the SAME fusion for every tool: local Moran, Getis-Ord Gi*, local bivariate,
# local diversity, per-cell colocalization, ...
#
# combine(neigh_sum, neigh_count, self_fields) -> per-cell statistic, where all three arrays are
# aligned to the SAME cells (all cells on the FFT pass; the flagged subset on the fusion pass), so an
# identical combine serves both — guaranteeing the FFT and exact estimands match and the gap is pure
# rasterization.

def _fft_neighbor_reduce(xy, fields, radius_um, bin_size):
    """Per-cell (neighbour SUM per field, neighbour COUNT) via one masked convolution on the grid.
    ``fields`` is (N, K). Self/home-bin excluded (disc kernel excludes the centre bin)."""
    fields = np.asarray(fields, float)
    if fields.ndim == 1:
        fields = fields[:, None]
    row, col, H, W, pitch = grid_index(xy, bin_size)
    r_bins = max(int(round(radius_um / pitch)), 1)
    K = fields.shape[1]
    sumg = np.zeros((H, W, K)); cntg = np.zeros((H, W))
    np.add.at(sumg, (row, col), fields); np.add.at(cntg, (row, col), 1.0)
    k = _disc(r_bins)
    nsum = _conv2d(sumg, k)                       # FFT for large kernels (radius-independent)
    # Counts are integers by construction (0/1 disc x integer bin counts). The FFT branch (disc side >= 9, e.g.
    # the autotuned r/b~5) leaves ~1e-15 round-off, which every `nc > 0` guard downstream read as a neighbour: a
    # cell with none got noise/noise instead of NaN (2026-10-28). Round, and zero the sums where nothing is there.
    ncnt = np.rint(_conv2d(cntg, k))
    nsum[ncnt == 0] = 0.0
    return nsum[row, col], ncnt[row, col]


def _exact_neighbor_reduce(xy, fields, radius_um, sub_idx=None, tree=None):
    """Exact (neighbour SUM per field, neighbour COUNT) on true coordinates, self excluded. If
    ``sub_idx`` given, only for that subset (the fusion set) — cost scales with the subset size."""
    from scipy.spatial import cKDTree  # lazy
    fields = np.asarray(fields, float)
    if fields.ndim == 1:
        fields = fields[:, None]
    tree = tree or cKDTree(xy)
    q = xy if sub_idx is None else xy[sub_idx]
    ids = np.arange(len(xy)) if sub_idx is None else np.asarray(sub_idx)
    lists = tree.query_ball_point(q, radius_um)
    K = fields.shape[1]
    nsum = np.zeros((len(ids), K)); ncnt = np.zeros(len(ids))
    for i, nb in enumerate(lists):
        nb = [j for j in nb if j != ids[i]]      # self excluded by GLOBAL index
        ncnt[i] = len(nb)
        if nb:
            nsum[i] = fields[nb].sum(0)
    return nsum, ncnt


def adaptive_local_stat(xy, fields, combine, radius_um, bin_size=None, tol=0.1,
                        r_min_um=None, tree=None, fuse=True, return_meta=False):
    """Generic adaptive local statistic: FFT everywhere, exact-graph patch for the flagged minority.

    ``fields`` : (N,) or (N,K) per-cell field(s) the statistic aggregates.
    ``combine(neigh_sum, neigh_count, self_fields) -> (M,) statistic`` : the pointwise tool formula.

    Steps: pick bin (autotune if None) -> FFT reductions -> combine -> graph-free proxy (bin b vs b/2)
    -> flag cells above ``tol`` -> recompute ONLY those with the exact graph -> splice. Set
    ``fuse=False`` for the plain FFT statistic (autotuned bin, no graph fallback).
    """
    xy = np.asarray(xy, float)
    fields = np.asarray(fields, float)
    f0 = fields[:, 0] if fields.ndim == 2 else fields
    if bin_size is None:
        bin_size = autotune_grid(xy, r_min_um or radius_um, probe_values=f0, tol=tol)["bin_um"]
    ns, nc = _fft_neighbor_reduce(xy, fields, radius_um, bin_size)
    stat = np.asarray(combine(ns, nc, fields), float)
    flagged = 0
    if fuse:
        ns2, nc2 = _fft_neighbor_reduce(xy, fields, radius_um, bin_size / 2.0)
        stat2 = np.asarray(combine(ns2, nc2, fields), float)
        proxy = np.abs(stat - stat2) / (np.nanstd(stat) + 1e-12)
        fidx = np.where(proxy > tol)[0]
        # B2: cells that SHARE a bin (occupancy>1) are dropped by the home-bin-excluded FFT disc AND are
        # invisible to the b-vs-b/2 proxy (they stay co-binned when the grid is refined), so they can carry a
        # silently wrong — even sign-flipped — value. Force every occupancy>1 cell into the exact fusion set,
        # where _exact_neighbor_reduce (self-only exclusion by global index) counts the co-binned partners.
        co = np.where(occupancy(xy, bin_size) > 1)[0]
        fidx = np.union1d(fidx, co)
        flagged = fidx.size
        if fidx.size:
            nsx, ncx = _exact_neighbor_reduce(xy, fields, radius_um, sub_idx=fidx, tree=tree)
            stat[fidx] = np.asarray(combine(nsx, ncx, fields[fidx]), float)
    if return_meta:
        pitch = grid_index(xy, bin_size)[4]
        r_bins = max(int(round(radius_um / pitch)), 1)             # M13: surface the quantized effective radius
        return stat, dict(bin_um=float(bin_size), flagged_frac=float(flagged / len(xy)),
                          radius_um=float(radius_um), eff_radius_um=float(r_bins * pitch),
                          tol=float(tol), n_flagged=int(flagged), fused=bool(fuse))
    return stat


# ----------------------------------------------------------------- per-tool `combine` builders
def combine_moran(ns, nc, sf):
    """Local Moran's I: z_i * (mean neighbour z). ``fields`` = standardized z (1 col).

    A cell with NO neighbour in radius (``nc == 0``) has an UNDEFINED neighbour mean → returns NaN, not 0,
    matching the grid path — a 0 would be silently read as 'no autocorrelation' in downstream summaries.
    Downstream aggregations are nan-aware (use ``np.nanmean`` etc.)."""
    z = sf[:, 0] if sf.ndim == 2 else sf
    mean = np.divide(ns[:, 0], nc, out=np.full_like(nc, np.nan), where=nc > 0)
    return z * mean


def combine_getis(xbar, s, n):
    """Getis-Ord Gi* local z of the neighbourhood sum, INCLUDING the focal cell (fields = raw x, 1 col).
    NaN where the standardization is undefined (den==0, a constant field)."""
    def _c(ns, nc, sf):
        xself = sf[:, 0] if sf.ndim == 2 else sf     # Gi* includes cell i itself
        W = nc + 1.0
        num = (ns[:, 0] + xself) - xbar * W
        den = s * np.sqrt(np.maximum((n * W - W ** 2) / max(n - 1, 1), 1e-9))
        return np.divide(num, den, out=np.full_like(nc, np.nan), where=den > 0)
    return _c


def combine_bivariate(ns, nc, sf):
    """Local bivariate Moran: zx_i * (mean neighbour zy). ``fields`` = [zx, zy] (2 cols). NaN where the cell
    has no neighbour (undefined neighbour mean), not 0."""
    zx = sf[:, 0]
    mean_zy = np.divide(ns[:, 1], nc, out=np.full_like(nc, np.nan), where=nc > 0)
    return zx * mean_zy


def combine_diversity(ns, nc, sf):
    """Local Shannon diversity of neighbourhood cell-type composition. ``fields`` = one-hot (K cols)."""
    tot = ns.sum(1, keepdims=True)
    p = np.divide(ns, tot, out=np.zeros_like(ns), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    ent[tot[:, 0] <= 0] = np.nan          # empty neighbourhood → diversity undefined (NaN, not 0)
    return ent


def combine_count(col):
    """Per-cell count of a target type in the neighbourhood (per-cell colocalization). fields = one-hot."""
    return lambda ns, nc, sf: ns[:, col]


# ----------------------------------------------------------------- convenience wrappers
def adaptive_local_moran(xy, values, radius_um, bin_size=None, tol=0.1,
                         r_min_um=None, fuse=True, return_meta=False):
    """Local Moran's I for single-cell ST (thin wrapper over :func:`adaptive_local_stat`)."""
    v = np.asarray(values, float); z = (v - v.mean()) / (v.std() + 1e-9)
    return adaptive_local_stat(xy, z, combine_moran, radius_um, bin_size=bin_size, tol=tol,
                               r_min_um=r_min_um, fuse=fuse, return_meta=return_meta)


def adaptive_getis_ord(xy, values, radius_um, bin_size=None, tol=0.1, r_min_um=None,
                       fuse=True, return_meta=False):
    """Getis-Ord Gi* hotspots for single-cell ST (adaptive fusion)."""
    x = np.asarray(values, float)
    return adaptive_local_stat(xy, x, combine_getis(x.mean(), x.std() + 1e-9, len(x)),
                               radius_um, bin_size=bin_size, tol=tol, r_min_um=r_min_um,
                               fuse=fuse, return_meta=return_meta)


def adaptive_local_bivariate(xy, values_x, values_y, radius_um, bin_size=None, tol=0.1,
                             r_min_um=None, fuse=True, return_meta=False):
    """Local bivariate Moran (zx_i * mean neighbour zy) for single-cell ST (adaptive fusion)."""
    x = np.asarray(values_x, float); y = np.asarray(values_y, float)
    zx = (x - x.mean()) / (x.std() + 1e-9); zy = (y - y.mean()) / (y.std() + 1e-9)
    return adaptive_local_stat(xy, np.c_[zx, zy], combine_bivariate, radius_um, bin_size=bin_size,
                               tol=tol, r_min_um=r_min_um, fuse=fuse, return_meta=return_meta)


def adaptive_local_diversity(xy, labels, radius_um, bin_size=None, tol=0.1, r_min_um=None,
                             fuse=True, return_meta=False):
    """Local Shannon diversity of neighbourhood cell-type composition (adaptive fusion)."""
    labels = np.asarray(labels)
    types = np.unique(labels)
    oh = np.stack([(labels == t).astype(float) for t in types], 1)
    return adaptive_local_stat(xy, oh, combine_diversity, radius_um, bin_size=bin_size, tol=tol,
                               r_min_um=r_min_um, fuse=fuse, return_meta=return_meta)


def adaptive_colocalization(xy, labels, target, radius_um, bin_size=None, tol=0.1, r_min_um=None,
                            fuse=True, return_meta=False):
    """Per-cell colocalization: count of ``target`` cell-type in each cell's neighbourhood (adaptive)."""
    labels = np.asarray(labels)
    types = list(np.unique(labels))
    if target not in types:
        raise ValueError(f"target {target!r} not in labels ({types})")
    oh = np.stack([(labels == t).astype(float) for t in types], 1)
    return adaptive_local_stat(xy, oh, combine_count(types.index(target)), radius_um,
                               bin_size=bin_size, tol=tol, r_min_um=r_min_um, fuse=fuse,
                               return_meta=return_meta)

"""FFT-accelerated spatial statistics for grid spatial transcriptomics.

On a regular lattice (Visium, Visium HD, binned Xenium) the spatial-weight matrix
for a fixed-radius (or Gaussian) neighborhood is translation-invariant, so the core
quantity z'^T W z' is a convolution and can be computed for ALL genes at once via a
single batched FFT -- O(N log N) and independent of neighborhood radius. This is a
fast drop-in for graph-based spatial autocorrelation (e.g. squidpy) in the grid case.

Functions take a gene grid (H, W, G) + a boolean `mask` of valid cells.
Only numpy/scipy are required.
"""
from __future__ import annotations

import functools
import hashlib
import os
import threading
import warnings
from collections import OrderedDict

import numpy as np

from . import _fft  # pluggable FFT backend (cpu-threaded / cuda / mps); fftfreq/fftshift stay on np.fft

# ---------------------------------------------------------------------------
# Gridding + kernels
# ---------------------------------------------------------------------------

def expression_to_grid(expr, row, col):
    """Rasterize (N_spots, G) expression to a (H, W, G) grid using integer lattice
    indices `row`, `col` (e.g. Visium array_row/array_col). Collisions average.

    Returns (grid, mask). Counts are expected library-size-normalized/log already if
    that is how you want the statistic computed (matches squidpy on adata.X).
    """
    from scipy.sparse import issparse  # lazy
    X = expr.toarray() if issparse(expr) else np.asarray(expr, dtype=float)
    row = np.asarray(row).astype(int); col = np.asarray(col).astype(int)
    row -= row.min(); col -= col.min()
    H, W, G = row.max() + 1, col.max() + 1, X.shape[1]
    grid = np.zeros((H, W, G)); cnt = np.zeros((H, W))
    np.add.at(grid, (row, col), X)
    np.add.at(cnt, (row, col), 1.0)
    nz = cnt > 0
    grid[nz] /= cnt[nz][:, None]
    return grid, nz


def neighbor_kernel(radius: int = 1, kind: str = "disc") -> np.ndarray:
    """Translation-invariant neighborhood weights (center excluded).

    ``kind='gaussian'`` is a TRUNCATED Gaussian: sigma = radius/2, support clipped to the
    ``(2·radius+1)`` box (i.e. truncated at 2·sigma, ~5% of the radial mass beyond). This is intentional
    (a compact neighbourhood matched to the disc/queen kernels), not a full Gaussian; widen ``radius`` for
    a longer tail.
    """
    r = radius
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    if kind == "disc":
        k = (yy ** 2 + xx ** 2 <= r ** 2).astype(float)
    elif kind == "gaussian":
        k = np.exp(-(yy ** 2 + xx ** 2) / (2 * (r / 2.0) ** 2))     # sigma=r/2, truncated at the r-box (2 sigma)
    elif kind == "queen":
        k = np.ones((2 * r + 1, 2 * r + 1))
    else:
        raise ValueError(kind)
    k[r, r] = 0.0
    return k


# Kernels with side <= this use direct (scipy.ndimage) convolution; larger use FFT.
# From batched benchmarks (280x280 grid, 1-200 genes): direct wins for <=7x7 (radius<=3),
# FFT wins from ~9-11x11 up. FFT is radius-independent, direct is O(N*kernel_area).
_CONV_FFT_MIN_SIDE = 9


def _fftconv_same(stack: np.ndarray, kernel: np.ndarray, method: str = "auto") -> np.ndarray:
    """'Same'-size convolution of (H, W, G) with a 2D (symmetric) kernel, batched over G.

    ``method="auto"`` (default) dispatches by kernel size: small kernels go through direct
    spatial convolution (faster + lower overhead), large kernels through FFT (radius-independent).
    Force with ``method="fft"`` or ``method="direct"``. Kernels here are symmetric (disc/Gaussian),
    so convolution == correlation and the two paths agree to machine precision.
    """
    kh, kw = kernel.shape
    use_fft = method == "fft" or (method == "auto" and max(kh, kw) >= _CONV_FFT_MIN_SIDE)
    if not use_fft:
        from scipy import ndimage  # lazy
        k = kernel[:, :, None] if stack.ndim == 3 else kernel
        return ndimage.convolve(stack, k, mode="constant")
    from scipy.fft import next_fast_len  # lazy
    H, W = stack.shape[:2]
    # round the linear-conv length up to a fast (5-smooth) FFT size: the extra samples are zeros beyond
    # the valid H+kh-1 region, so the crop below is byte-identical — just a faster transform.
    fh, fw = next_fast_len(H + kh - 1), next_fast_len(W + kw - 1)
    # host float64 (M2): the single-radius neighbour SUM here is paired everywhere with a scipy float64
    # `fftconvolve` degree/S0 — running the numerator on a GPU float32 backend would mix precision within one
    # statistic. This is a single-shot conv (not the batched multiscale loop), so float64 costs nothing material.
    Fz = _fft.rfft2_hp(stack, s=(fh, fw), axes=(0, 1))
    Fk = _fft.rfft2_hp(kernel, s=(fh, fw))[:, :, None]
    full = _fft.irfft2_hp(Fz * Fk, s=(fh, fw), axes=(0, 1))
    r0, c0 = (kh - 1) // 2, (kw - 1) // 2
    return full[r0:r0 + H, c0:c0 + W]


# ---------------------------------------------------------------------------
# Global spatial autocorrelation (batched over genes)
# ---------------------------------------------------------------------------

def morans_i(grid: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
             radius: int = 1) -> np.ndarray:
    """Moran's I per gene for a (H, W, G) grid. Returns (G,). FFT, radius-independent."""
    if grid.ndim == 2:
        grid = grid[..., None]
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    m = mask.astype(float)
    msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum
    z = (grid - zbar) * m[..., None]
    W1 = fftconvolve(m, kernel, mode="same")           # neighbour count per cell (degree)
    inv = np.where(W1 > 0, 1.0 / np.maximum(W1, 1e-12), 0.0)[..., None]
    meanN = _fftconv_same(z, kernel) * inv             # ROW-STANDARDIZED neighbour mean (D^-1 W z)
    num = (z * meanN).sum((0, 1))                       # Σ_i z_i·mean_{j∈N(i)} z_j
    den = (z * z).sum((0, 1))
    return num / np.where(den == 0, np.nan, den)        # row-standardized Moran (S0=N ⇒ N/S0=1); C ≈ 1−I


def spatial_smooth(grid: np.ndarray, mask: np.ndarray, radius: int = 5,
                   kind: str = "gaussian") -> np.ndarray:
    """Large-neighborhood spatial smoothing of a (H, W, K) grid via FFT convolution.

    Mask-aware (averages only over occupied neighbors). Cost is O(N log N) and
    independent of radius -- a tissue-scale neighborhood op that an explicit neighbor
    graph cannot build at large radius on dense HD-scale grids.
    """
    from scipy.signal import fftconvolve  # lazy
    if grid.ndim == 2:
        grid = grid[..., None]
    k = neighbor_kernel(radius, kind); k[radius, radius] = 1.0  # include center
    m = mask.astype(float)
    num = _fftconv_same(grid * m[..., None], k)
    den = fftconvolve(m, k, mode="same")[..., None]
    out = num / np.where(den == 0, np.nan, den)
    return out


def morans_i_multiscale(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc") -> np.ndarray:
    """Moran's I per gene at MANY neighborhood radii, sharing one forward FFT.

    Returns (n_radii, G). The data is transformed once; each scale costs only a
    kernel FFT + product + inverse. This is the FFT amortization win: spatial
    autocorrelation across a whole range of scales (a "spatial scalogram") far
    cheaper than rebuilding a neighbor graph per scale -- especially at large radii.
    """
    if _fft.get_backend() != "cpu":
        return _morans_i_multiscale_torch(grid, mask, radii, kind)   # GPU: push stack once, sweep on-device
    # CPU: the shared implementation (_ms_cpu), bit-identical to the per-function code it replaced. Row-
    # standardized weights sum to N over occupied cells, so N/S0 = 1: this matches the single-radius morans_i
    # primitive (mean(LISA) = global I) on masked/bordered tissue. A binary-weight (Wz, msum/S0) form drifts
    # from morans_i at every tissue boundary and breaks that identity -- the Methods declare row-standardized
    # weights throughout.
    return _ms_cpu(grid, mask, radii, kind, ("moran",))["moran"]


def _gene_block_gpu(fh: int, fw: int, dev) -> int:
    """Gene-tile size so the (fh, fw, block) complex spectra + real fields fit device memory.
    CUDA sizes from free VRAM; MPS (unified memory) from a fraction of the recommended budget."""
    import torch
    per_gene = fh * fw * 8 * 6                       # bytes/gene: complex spectrum + real fields + temporaries
    if dev.type == "cuda":
        free = torch.cuda.mem_get_info(dev)[0]; budget = int(free * 0.5)
    else:                                            # mps / unified memory
        try:
            budget = int(torch.mps.recommended_max_memory() * 0.3)
        except Exception:
            budget = 2 * 1024 ** 3
    return max(1, budget // per_gene)


def _reduce_hw_f64(t):
    """Sum an ``(H, W, G)`` device tensor over its spatial axes -> ``(G,)`` float64 host array.

    Accumulating ~10^5 occupied bins in float32 is measurably lossy: on a real 401x666 section it
    costs ~2e-5 relative on the grid mean and ~3e-7 on the sum of squares. Accumulating in float64
    instead recovers ~85x on the mean and ~5x on the sum of squares, for no measurable time.

    Worth being precise about what this does NOT fix. The end-to-end float32 error on Moran's I is
    ~1e-4..1e-3 and is dominated by a different term — ``num = sum(z * Wz * invW)``, where
    cancellation amplifies the ~2e-7 float32 TRANSFORM error by ~400x. Fixing the reductions leaves
    that untouched (measured: 2.87e-3 -> 2.78e-3 end to end). This is cheap insurance on the terms
    it does control, not a cure for float32 Moran.

    Not a GPU defect either: torch on the CPU in float32 is just as lossy. The float64 numpy path
    never showed any of this because ``mask.astype(float)`` promotes the whole expression to float64.

    CUDA and torch-CPU accumulate in float64 directly. MPS has no float64 at all, so the short axis
    is reduced on-device — a few hundred float32 terms — and the ``(W, G)`` partials finish on the
    host in float64. That copy is ``H`` times smaller than the field stack, so it is free next to
    the transforms.
    """
    import torch
    if t.device.type != "mps":
        return t.sum((0, 1), dtype=torch.float64).detach().to("cpu").numpy()
    return t.sum(0).detach().to("cpu").numpy().astype(np.float64).sum(0)


def _reduce_n_f64(t, chunk=2048):
    """Sum an ``(N, G)`` device tensor over axis 0 -> ``(G,)`` float64 host array.

    As :func:`_reduce_hw_f64`, but for the flat per-cell reductions. The focality Gini sorts over
    every occupied bin and weights by ``2i - n - 1``, so ``N`` is ~10^5 and the weighted sum spans
    many orders — the worst float32 accumulation in this module if left on-device. On MPS the sum is
    chunked so each on-device partial covers a bounded number of terms.
    """
    import torch
    if t.device.type != "mps":
        return t.sum(0, dtype=torch.float64).detach().to("cpu").numpy()
    acc = np.zeros(t.shape[1], np.float64)
    for i in range(0, t.shape[0], chunk):
        acc += t[i:i + chunk].sum(0).detach().to("cpu").numpy().astype(np.float64)
    return acc


def _morans_i_multiscale_torch(grid, mask, radii, kind="disc", dev=None):
    """Device (CUDA/MPS) multiscale Moran, mirroring ``morans_i_multiscale`` in float32/complex64.

    The field stack is pushed to the device ONCE per gene-block and the whole radius sweep runs
    on-device; only the reduced (nR, block) output returns to the host. Kernel + mask spectra are
    precomputed once. Numerically matches the float64 CPU path to a validated float32 tolerance.
    ``dev`` overrides the active-backend device (used by the fidelity benchmark to run a torch-CPU
    float32 control that isolates MPS-specific error from generic float32 round-off).
    """
    import torch, warnings
    from scipy.fft import next_fast_len  # lazy
    warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
    if dev is None:
        dev = _fft._TORCH_DEVICE
    if grid.ndim == 2:
        grid = grid[..., None]
    radii = list(radii); rmax = max(radii)
    H, W, G = grid.shape
    m = torch.as_tensor(np.asarray(mask, np.float32), device=dev)
    msum = float(m.sum().item())                    # 0/1 sum: exact in float32 below 2**24
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fm = torch.fft.rfft2(m, s=(fh, fw))
    Fk, Wm_off, invW = [], [], []
    for r in radii:
        k = neighbor_kernel(r, kind); off = (k.shape[0] - 1) // 2
        fk = torch.fft.rfft2(torch.as_tensor(k.astype(np.float32), device=dev), s=(fh, fw))
        Wm = torch.fft.irfft2(Fm * fk, s=(fh, fw))[off:off + H, off:off + W]
        Fk.append(fk); Wm_off.append(off)
        # A bin has a neighbour when its degree is at least the kernel's smallest weight. In float32 a bin
        # with none gets ~1e-7 of FFT round-off instead of 0, so the float64 path's 1e-12 gate would divide
        # by round-off; half the smallest weight (0.5 for disc/queen, ~0.009 for the gaussian) separates them.
        gate = 0.5 * float(k[k > 0].min())
        invW.append(torch.where(Wm > gate, 1.0 / torch.clamp(Wm, min=1e-12), torch.zeros_like(Wm)))
    out = np.empty((len(radii), G))
    block = _gene_block_gpu(fh, fw, dev)
    for g0 in range(0, G, block):
        gsl = slice(g0, min(g0 + block, G))
        gz = torch.as_tensor(np.ascontiguousarray(grid[:, :, gsl], np.float32), device=dev)
        # every (H, W) reduction accumulates in float64 — see _reduce_hw_f64
        zbar_h = _reduce_hw_f64(gz * m[..., None]) / msum
        zbar = torch.as_tensor(zbar_h.astype(np.float32), device=dev)
        z = (gz - zbar) * m[..., None]
        den = _reduce_hw_f64(z * z)
        Fz = torch.fft.rfft2(z, dim=(0, 1), s=(fh, fw))              # shared forward transform (this block)
        for si in range(len(radii)):
            off = Wm_off[si]
            Wz = torch.fft.irfft2(Fz * Fk[si][:, :, None], dim=(0, 1), s=(fh, fw))[off:off + H, off:off + W]
            num = _reduce_hw_f64(z * (Wz * invW[si][..., None]))     # ROW-STANDARDIZED (matches CPU/ morans_i)
            out[si, gsl] = num / np.where(den == 0, np.nan, den)     # final divide on the host, float64
        del gz, z, Fz
    return out


def _gearys_c_multiscale_torch(grid, mask, radii, kind="disc", dev=None):
    """Device (CUDA/MPS) multiscale Geary's C, mirroring ``gearys_c_multiscale`` in float32.
    Mask-only per-radius quantities (degree, gate, S0) are precomputed once; x/x² spectra are
    formed per gene-block and swept on-device. ``dev`` overrides the active-backend device."""
    import torch, warnings
    from scipy.fft import next_fast_len  # lazy
    warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
    if dev is None:
        dev = _fft._TORCH_DEVICE
    if grid.ndim == 2:
        grid = grid[..., None]
    radii = list(radii); rmax = max(radii)
    H, W, G = grid.shape
    mb = torch.as_tensor(np.asarray(mask, bool), device=dev)
    m = mb.to(torch.float32); msum = float(m.sum().item())
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fm = torch.fft.rfft2(m, s=(fh, fw))
    perR = []
    for r in radii:
        k = neighbor_kernel(r, kind); off = (k.shape[0] - 1) // 2
        fk = torch.fft.rfft2(torch.as_tensor(k.astype(np.float32), device=dev), s=(fh, fw))
        W1 = torch.fft.irfft2(Fm * fk, s=(fh, fw))[off:off + H, off:off + W]
        has_nb = W1 > 0.5
        inv = torch.where(has_nb, 1.0 / torch.clamp(W1, min=1e-12), torch.zeros_like(W1))
        gate = (has_nb & mb).to(torch.float32)
        perR.append((fk, off, inv[..., None], gate[..., None], float(gate.sum().item())))
    out = np.empty((len(radii), G))
    block = _gene_block_gpu(fh, fw, dev)
    for g0 in range(0, G, block):
        gsl = slice(g0, min(g0 + block, G))
        gt = torch.as_tensor(np.ascontiguousarray(grid[:, :, gsl], np.float32), device=dev)
        zbar_h = _reduce_hw_f64(gt * m[..., None]) / msum            # float64 reductions throughout
        zbar = torch.as_tensor(zbar_h.astype(np.float32), device=dev)
        z = (gt - zbar) * m[..., None]
        x = gt * m[..., None]; x2 = x * x
        den = _reduce_hw_f64(z * z)
        Fx = torch.fft.rfft2(x, dim=(0, 1), s=(fh, fw))
        Fx2 = torch.fft.rfft2(x2, dim=(0, 1), s=(fh, fw))
        for si, (fk, off, inv, gate, S0) in enumerate(perR):
            Wx = torch.fft.irfft2(Fx * fk[:, :, None], dim=(0, 1), s=(fh, fw))[off:off + H, off:off + W]
            Wx2 = torch.fft.irfft2(Fx2 * fk[:, :, None], dim=(0, 1), s=(fh, fw))[off:off + H, off:off + W]
            num = _reduce_hw_f64((x * x - 2.0 * x * (Wx * inv) + (Wx2 * inv)) * gate)
            # S0 = 0: no occupied bin has a neighbour at this radius -> undefined (NaN), not 1/0
            out[si, gsl] = (((msum - 1) / (2.0 * S0)) if S0 > 0 else np.nan) * (num / np.where(den == 0, np.nan, den))
        del gt, z, x, x2, Fx, Fx2
    return out


def _focality_scalogram_torch(grid, mask, radii, kind="disc", dev=None):
    """Device (CUDA/MPS) focality (positive-LISA Gini) scalogram, mirroring ``focality_scalogram``.
    The per-radius LISA map is reduced to a per-gene positive-Gini on-device (sort over occupied
    cells). ``dev`` overrides the active-backend device."""
    import torch, warnings
    from scipy.fft import next_fast_len  # lazy
    warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
    if dev is None:
        dev = _fft._TORCH_DEVICE
    if grid.ndim == 2:
        grid = grid[..., None]
    radii = list(radii); rmax = max(radii)
    H, W, G = grid.shape
    mb = torch.as_tensor(np.asarray(mask, bool), device=dev)
    m = mb.to(torch.float32); msum = float(m.sum().item())
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fm = torch.fft.rfft2(m, s=(fh, fw))
    nocc = int(mb.sum().item())
    wt = (2 * torch.arange(1, nocc + 1, device=dev, dtype=torch.float32) - nocc - 1)[:, None]
    perR = []
    for r in radii:
        k = neighbor_kernel(r, kind); off = (k.shape[0] - 1) // 2
        fk = torch.fft.rfft2(torch.as_tensor(k.astype(np.float32), device=dev), s=(fh, fw))
        W1 = torch.fft.irfft2(Fm * fk, s=(fh, fw))[off:off + H, off:off + W]
        inv = torch.where(W1 > 0.5, 1.0 / torch.clamp(W1, min=1e-12), torch.zeros_like(W1))
        perR.append((fk, off, inv[..., None]))
    out = np.empty((len(radii), G))
    block = _gene_block_gpu(fh, fw, dev)
    for g0 in range(0, G, block):
        gsl = slice(g0, min(g0 + block, G))
        gt = torch.as_tensor(np.ascontiguousarray(grid[:, :, gsl], np.float32), device=dev)
        zbar_h = _reduce_hw_f64(gt * m[..., None]) / msum                   # float64 reductions
        zbar = torch.as_tensor(zbar_h.astype(np.float32), device=dev)
        z = (gt - zbar) * m[..., None]
        m2_h = _reduce_hw_f64(z * z) / msum
        m2 = torch.as_tensor(m2_h.astype(np.float32), device=dev)
        Fz = torch.fft.rfft2(z, dim=(0, 1), s=(fh, fw))
        for si, (fk, off, inv) in enumerate(perR):
            meanN = torch.fft.irfft2(Fz * fk[:, :, None], dim=(0, 1), s=(fh, fw))[off:off + H, off:off + W] * inv
            lisa = z * meanN / torch.where(m2 == 0, torch.tensor(float("nan"), device=dev), m2)
            v = lisa[mb]                                                    # (nocc, block)
            v = torch.where(torch.isfinite(v), v, torch.zeros_like(v)).clamp(min=0.0)
            pos, _ = torch.sort(v, dim=0)
            # the Gini weights run to +-nocc, so these two sums span ~10 orders over ~10^5 terms —
            # the worst float32 accumulation in the file if left on-device
            s = _reduce_n_f64(pos)
            wsum = _reduce_n_f64(wt * pos)
            out[si, gsl] = np.nan_to_num(wsum / np.where(s > 0, nocc * s, np.nan), nan=0.0)
        del gt, z, Fz
    return out


def celltype_colocalization_multiscale(xy, labels, radii, bin_size: float,
                                       n_perm: int = 100, seed: int = 0,
                                       kind: str = "disc", return_null: bool = False):
    """Multi-scale cell-type colocalization (the CRAWDAD question, on the FFT engine).

    A third analysis mode alongside per-gene (Moran/focality) and gene-pair (CCI): for each
    ordered cell-type pair (A, B) and neighborhood radius r, ``coloc[r, A, B]`` = average number
    of type-B cells within radius r of a type-A cell, computed as a CONVOLUTION of the B
    count-field with a radius-r kernel (summed over A locations, normalized by the A count).
    Every scale reuses the shared grid; significance is a batched LABEL-PERMUTATION null (shuffle
    cell types among fixed positions) -> a z per pair per scale (positive=colocalize, negative=segregate).

    Parameters
    ----------
    xy : (N, 2) cell coordinates (same units as ``bin_size``).
    labels : (N,) cell-type labels.
    radii : iterable of neighborhood radii in GRID BINS.
    bin_size : grid bin size (coordinate units per bin).
    n_perm : number of label permutations for the null.

    Returns
    -------
    types : list of cell-type names (row/col order of the K x K matrices).
    obs : (n_radii, K, K) observed colocalization.
    z   : (n_radii, K, K) permutation z-scores.
    """
    xy = np.asarray(xy, float)
    labels = np.asarray(labels).astype(str)
    types = sorted(np.unique(labels).tolist())
    K = len(types)
    tidx = {t: i for i, t in enumerate(types)}
    lab = np.array([tidx[x] for x in labels])
    g = xy - xy.min(0)
    col = (g[:, 0] / bin_size).astype(int)
    row = (g[:, 1] / bin_size).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    radii = list(radii)
    # include the focal (home) bin so SAME-BIN partners count as neighbours (m17): a type-l cell in the focal
    # cell's own bin is at distance ~0 and should count. Self-pairing (a cell with itself) is removed on the
    # A–A diagonal below. A home-excluded kernel dropped every co-binned partner, biasing the coloc count low.
    kernels = [neighbor_kernel(r, kind) for r in radii]
    for ker in kernels:
        ker[ker.shape[0] // 2, ker.shape[1] // 2] = 1.0

    def grid_counts(lv):
        G = np.zeros((H, W, K), "float32")
        np.add.at(G, (row, col, lv), 1.0)
        return G

    def stat(G):
        tot = G.reshape(-1, K).sum(0)                        # cells per type (= self-pairs to remove on diagonal)
        di = np.arange(K)
        out = np.empty((len(radii), K, K), "float32")
        for ri, ker in enumerate(kernels):
            sm = _fftconv_same(G, ker)                       # local per-type density fields (home bin INCLUDED)
            num = np.einsum("xyk,xyl->kl", G, sm)            # (k,l) pairs within radius, incl. same bin
            num[di, di] -= tot                               # drop each cell paired with ITSELF (diagonal only)
            out[ri] = num / np.where(tot == 0, np.nan, tot)[:, None]
        return out

    obs = stat(grid_counts(lab))
    rng = np.random.default_rng(seed)
    # pre-draw label permutations (sequential RNG -> identical to the serial loop), then fan the
    # per-perm FFT stat out across threads and reduce in submission order (Phase 2).
    perms = [rng.permutation(lab) for _ in range(n_perm)]
    from ._parallel import parallel_map
    stats = parallel_map(lambda pl: stat(grid_counts(pl)), perms)
    acc = np.zeros_like(obs); acc2 = np.zeros_like(obs)
    for s in stats:
        acc += s; acc2 += s * s
    mu = acc / n_perm
    sd = np.sqrt(np.maximum(acc2 / n_perm - mu * mu, 0.0))
    floor = np.nanmedian(sd) * 1e-3                          # robust variance floor (avoid /~0 blowups)
    z = np.clip((obs - mu) / np.maximum(sd, floor), -500, 500)
    if return_null:
        return types, obs, z, mu, sd                         # mu = permutation-expected coloc (for fold-enrichment)
    return types, obs, z


def npc_colocalization_multiscale(xy, labels, label_set, radii, bin_size: float, *,
                                  abundance_normalize: bool = True, n_perm: int = 0,
                                  seed: int = 0, kind: str = "disc", eps: float = 1e-9):
    """Multiscale, per-cell Neighborhood Product Co-localization (NPC) on the FFT engine.

    NPC (Gingerich & Frost) scores a niche as the PRODUCT over a label SET of each label's local
    neighbourhood proportion -- a per-cell, multivariate co-localization readout. The published score
    is single-scale (one kNN graph); here every radius is computed from the SAME shared FFT, giving a
    per-cell NPC *profile* over scale (and the multivariate product stays free). With
    ``abundance_normalize`` the score is divided by the product of global label fractions, i.e. the
    expected product under spatial independence, so ``NPC > 1`` = the niche co-occurs more than chance.

    Each label's local proportion is ``p_c(r) = (count_c * disc_r) / (count_total * disc_r)`` (a
    convolution), and ``NPC_i(C, r) = prod_{c in C} p_{i,c}(r)`` sampled at each cell's grid bin. The
    permutation null shuffles labels among fixed positions (the total-count field is label-invariant, so
    only the per-label fields are re-convolved), giving a per-cell one-sided p via ``(#>=obs + 1)/(n+1)``.

    Parameters
    ----------
    xy : (N, 2) cell coordinates (same units as ``bin_size``).
    labels : (N,) cell-type labels.
    label_set : iterable of >=2 labels defining the niche (the multivariate microenvironment).
    radii : iterable of neighbourhood radii in GRID BINS.
    bin_size : grid bin size (coordinate units per bin); use ~cell spacing for a per-cell readout.
    abundance_normalize : divide by prod(global label fractions) (expected product under independence).
    n_perm : label permutations for the null; if >0 also returns per-cell one-sided p-values.

    Returns
    -------
    npc : (N, n_radii) per-cell NPC score (columns = radii).
    pval : (N, n_radii) one-sided permutation p (ONLY if ``n_perm > 0``).
    """
    xy = np.asarray(xy, float)
    labels = np.asarray(labels).astype(str)
    cset = [str(c) for c in label_set]
    if len(cset) < 2:
        raise ValueError("label_set needs >=2 labels (NPC is a product over a label set)")
    missing = [c for c in cset if c not in set(labels)]
    if missing:
        raise ValueError(f"label_set labels not in labels: {missing}")
    radii = list(radii)
    g = xy - xy.min(0)
    col = (g[:, 0] / bin_size).astype(int); row = (g[:, 1] / bin_size).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    N = len(xy)

    def count_field(sel):                                    # (H, W) count of selected cells
        f = np.zeros((H, W), "float32"); np.add.at(f, (row[sel], col[sel]), 1.0); return f

    total = count_field(np.ones(N, bool))                    # label-INVARIANT (positions fixed) -> convolve once/radius
    frac = np.array([(labels == c).mean() for c in cset])    # global label fractions
    denom = np.prod(frac) if abundance_normalize else 1.0
    # NPC keeps the focal (home) bin EXCLUDED by design (unlike the celltype coloc MATRIX, m17): NPC is a
    # per-cell PROPORTION of each label in the neighbourhood, so including the home bin would fold the focal
    # cell's OWN label into its own neighbourhood proportion (a self-bias the bin-level FFT can't subtract per
    # cell). The trade-off — co-binned partners are not counted — matters only at coarse bins.
    kernels = [neighbor_kernel(r, kind) for r in radii]
    # clip FFT roundoff (empty regions can go slightly negative) so proportions stay in [0, 1]
    tot_sm = [np.clip(_fftconv_same(total[..., None], ker)[..., 0], 0, None) for ker in kernels]

    def npc_at_cells(lab):                                   # -> (N, n_radii)
        cfields = np.stack([count_field(lab == c) for c in cset], -1)   # (H, W, len(C))
        out = np.empty((N, len(radii)), "float32")
        for ri, ker in enumerate(kernels):
            sm = np.clip(_fftconv_same(cfields, ker), 0, None)          # (H, W, len(C)) local per-label counts
            p = np.clip(sm / (tot_sm[ri][..., None] + eps), 0.0, 1.0)   # local proportions in [0, 1]
            field = np.prod(p, axis=-1) / denom              # NPC product (abundance-normalized)
            out[:, ri] = field[row, col]                     # sample at each cell's bin
        return out

    npc = npc_at_cells(labels)
    if n_perm <= 0:
        return npc
    rng = np.random.default_rng(seed)
    ge = np.zeros_like(npc)
    pl = [rng.permutation(labels) for _ in range(n_perm)]      # pre-draw (identical to serial)
    from ._parallel import parallel_map
    for hit in parallel_map(lambda lb: (npc_at_cells(lb) >= npc), pl):   # FFT-bound null, threaded (Phase 2)
        ge += hit
    return npc, (ge + 1.0) / (n_perm + 1.0)


def correlation_length(scalogram: np.ndarray, radii, frac: float = 0.5,
                       interp: bool = True) -> np.ndarray:
    """Per-gene spatial correlation length from a Moran scalogram.

    The neighbourhood radius at which Moran's I falls to ``frac`` of its value at the smallest
    radius — i.e. how far spatial autocorrelation persists. Short = fine/cellular-scale structure;
    long = tissue/regional-scale. More discriminative than ``argmax(I)`` (usually the smallest
    radius). Genes whose I never drops below the threshold are assigned the largest radius.

    Parameters
    ----------
    scalogram : numpy.ndarray
        Moran scalogram of shape ``(n_radii, n_genes)`` (e.g. from
        ``adata.uns["spectrast_scalogram"]["scalogram"]`` or :func:`morans_i_multiscale`).
    radii : array-like
        The radii ``(n_radii,)`` the scalogram was evaluated at, in physical units.
    frac : float, optional
        Fraction of the smallest-radius I defining the crossing. Default ``0.5``.
    interp : bool, optional
        If ``True`` (default) log-interpolate the crossing radius between bracketing radii for a
        continuous scale; if ``False`` snap to the radii grid.

    Returns
    -------
    numpy.ndarray
        Per-gene correlation length ``(n_genes,)`` in the units of ``radii``.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> radii = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    >>> scal = np.array([[1.0, 1.0], [0.7, 0.9], [0.4, 0.8], [0.2, 0.6], [0.1, 0.5]])
    >>> L = spectrast.spatial.correlation_length(scal, radii)
    >>> L.shape
    (2,)
    """
    radii = np.asarray(radii, float)
    I = np.asarray(scalogram, float)
    I0 = I[0]
    thr = frac * I0
    out = np.full(I.shape[1], radii[-1])
    for g in range(I.shape[1]):
        if not np.isfinite(I0[g]) or I0[g] <= 0:
            out[g] = np.nan; continue
        below = np.where(I[:, g] <= thr[g])[0]
        if not below.size:
            continue
        j = below[0]
        if j == 0 or not interp:
            out[g] = radii[j]; continue
        y0, y1 = I[j - 1, g], I[j, g]
        r0, r1 = np.log(radii[j - 1]), np.log(radii[j])
        t = 0.5 if y0 == y1 else (y0 - thr[g]) / (y0 - y1)
        t = min(max(t, 0.0), 1.0)                              # never extrapolate outside the bracketing radii
        out[g] = np.exp(r0 + t * (r1 - r0))
    return out


def adaptive_smooth(grid: np.ndarray, mask: np.ndarray, radius_per_gene: np.ndarray,
                    kind: str = "gaussian") -> np.ndarray:
    """Smooth each gene at its OWN spatial scale (e.g. its correlation length).

    Genes are grouped by radius so each unique radius costs one batched FFT.
    Denoises fine-scale genes lightly and coarse-scale genes heavily -- an
    adaptive alternative to a single fixed smoothing radius.
    """
    if grid.ndim == 2:
        grid = grid[..., None]
    out = np.zeros_like(grid, dtype=float)
    rad = np.rint(np.nan_to_num(radius_per_gene, nan=1)).astype(int)
    for r in np.unique(rad):
        cols = np.where(rad == r)[0]
        if r <= 0:
            out[..., cols] = grid[..., cols]
        else:
            out[..., cols] = spatial_smooth(grid[..., cols], mask, radius=int(r), kind=kind)
    return out


def scale_signature_clusters(scalogram: np.ndarray, n_clusters: int = 5, seed: int = 0):
    """Cluster genes by their (normalized) multi-scale autocorrelation profile.

    Groups genes that share a spatial-scale signature. Returns (labels (G,),
    cluster_mean_profiles (n_clusters, n_radii)). Profiles are normalized to their
    own max so clustering is by scale *shape*, not overall autocorrelation strength.
    """
    from sklearn.cluster import KMeans  # lazy
    I = np.asarray(scalogram, float).T            # (G, n_radii)
    I = np.nan_to_num(I)
    norm = I / (np.abs(I).max(1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(norm)
    means = np.array([norm[km.labels_ == c].mean(0) for c in range(n_clusters)])
    return km.labels_, means


def bandpass(grid: np.ndarray, mask: np.ndarray, r_lo: int, r_hi: int,
             kind: str = "gaussian") -> np.ndarray:
    """Scale band-pass: structure at scales between r_lo and r_hi (difference of
    low-pass smooths, i.e. a difference-of-Gaussians / Laplacian-pyramid band).
    Isolates e.g. fine cellular texture (small band) vs tissue gradients (large band).
    """
    return spatial_smooth(grid, mask, r_lo, kind) - spatial_smooth(grid, mask, r_hi, kind)


def spectral_bandpass(field: np.ndarray, mask: np.ndarray, lam_lo: float = 0.0,
                      lam_hi: float = np.inf, bin_size: float = 1.0, apodize: float = 0.0) -> np.ndarray:
    """Ideal (or raised-cosine apodized) band-pass in the 2-D frequency domain: keep only spatial
    WAVELENGTHS in [lam_lo, lam_hi] (physical units; wavelength lambda = 1/|k|). Native to the FFT and
    with no spatial-graph equivalent. Two headline uses:
      * lam_hi < inf  -> LOW-PASS: drop sub-lam_hi wavelengths (per-cell dropout / technical chatter).
      * lam_lo > 0    -> remove the DC + macro gradients (illumination, coverslip, zonation drift) while
                         leaving local cell-cell structure untouched.
    Combined -> isolate a scale band (e.g. 50-200 um immune clustering, independent of macro zonation).

    Hard cutoffs cause Gibbs ringing near sharp boundaries; ``apodize`` in (0,1] replaces each edge with a
    raised-cosine taper spanning that fraction of the pass-band (in 1/lambda), suppressing the ringing at
    the cost of a softer scale cut. Returns the filtered field (mean removed within the mask, zero outside).
    """
    f = np.asarray(field, float); m = np.asarray(mask, bool)
    f = np.where(m, f - f[m].mean(), 0.0)
    H, W = f.shape
    ky = np.fft.fftfreq(H, d=bin_size); kx = np.fft.fftfreq(W, d=bin_size)
    Kr = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)            # spatial frequency |k| (1/length)
    k_lo = 0.0 if lam_hi == np.inf else 1.0 / lam_hi             # keep |k| in [k_lo, k_hi]
    k_hi = np.inf if lam_lo == 0.0 else 1.0 / lam_lo
    Hk = ((Kr >= k_lo) & (Kr <= k_hi)).astype(float)
    if apodize > 0:                                             # raised-cosine tapers rolling INWARD (soften cut)
        width = (k_hi if np.isfinite(k_hi) else Kr.max()) - k_lo + 1e-12
        span = apodize * width
        if k_lo > 0:                                            # high-pass edge: roll 0->1 over [k_lo, k_lo+span]
            lo = (Kr >= k_lo) & (Kr < k_lo + span)
            Hk[lo] = 0.5 * (1 - np.cos(np.pi * (Kr[lo] - k_lo) / span))
        if np.isfinite(k_hi):                                   # low-pass edge: roll 1->0 over [k_hi-span, k_hi]
            hi = (Kr > k_hi - span) & (Kr <= k_hi)
            Hk[hi] = 0.5 * (1 + np.cos(np.pi * (Kr[hi] - (k_hi - span)) / span))
    out = _fft.ifft2_hp(_fft.fft2_hp(f) * Hk).real
    return out * m


def scale_space_blobs(field: np.ndarray, mask: np.ndarray, scales_um, bin_size: float = 1.0,
                      threshold_rel: float = 0.3, exclude_border: int = 2) -> dict:
    """Detect, localize AND SIZE transcriptional structures via a Laplacian-of-Gaussian scale-space cube
    (the wavelet/scale-space capability on top of the radius sweep). At each scale sigma we convolve the
    field with the scale-normalized LoG (sigma^2 * grad^2 G_sigma); a bright blob of radius r peaks at
    sigma = r/sqrt(2), so a local maximum in the 3-D (sigma, y, x) volume marks a structure AND reads off
    its diameter in one pass — e.g. distinguishing small follicles from large germinal centres.

    Returns dict(x_um, y_um, diameter_um, strength) arrays, one entry per detected blob (strongest first).
    ``scales_um`` = candidate structure RADII to probe. Overlaps classic blob detection (skimage.blob_log);
    the point here is the FFT scale-space integrates with the same rasterized engine as the rest of spectrast.
    """
    from scipy import ndimage  # lazy
    m = np.asarray(mask, bool); f = np.where(m, np.asarray(field, float), 0.0)
    if m.any():
        f = np.where(m, f - f[m].mean(), 0.0)
    scales_px = np.array([max(s / bin_size / np.sqrt(2.0), 0.5) for s in scales_um])   # sigma per radius
    cube = np.stack([-(sp ** 2) * ndimage.gaussian_laplace(f, sp) for sp in scales_px], 0)  # bright-blob positive
    mx = ndimage.maximum_filter(cube, size=(3, 3, 3))                # 3-D local maxima
    peak = (cube == mx) & (cube > threshold_rel * cube.max())
    si, yi, xi = np.where(peak)
    H, W = f.shape
    keep = m[yi, xi] & (yi >= exclude_border) & (yi < H - exclude_border) & (xi >= exclude_border) & (xi < W - exclude_border)
    si, yi, xi = si[keep], yi[keep], xi[keep]
    strength = cube[si, yi, xi]; order = np.argsort(-strength)
    si, yi, xi, strength = si[order], yi[order], xi[order], strength[order]
    rad = scales_px[si] * np.sqrt(2.0)                              # blob radius (pixels)
    keep2 = np.ones(len(si), bool)                                  # greedy overlap suppression (strongest first)
    for i in range(len(si)):
        if not keep2[i]:
            continue
        for j in range(i + 1, len(si)):
            if keep2[j] and (yi[i] - yi[j]) ** 2 + (xi[i] - xi[j]) ** 2 < max(rad[i], rad[j]) ** 2:
                keep2[j] = False
    si, yi, xi, strength, rad = si[keep2], yi[keep2], xi[keep2], strength[keep2], rad[keep2]
    return dict(x_um=xi * bin_size, y_um=yi * bin_size, diameter_um=2.0 * rad * bin_size, strength=strength)


def cross_spectral_density(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                           n_bins: int = 24, bin_size: float = 1.0, kmax_frac: float = 0.7) -> dict:
    """Scale-resolved co-expression of two spatial fields via their cross-spectral density.

    Bivariate Moran's I collapses spatial covariance to a single scalar, but two genes can be
    positively co-localized at the tissue scale yet mutually exclusive at the cellular scale — a
    sign flip a scalar averages away. This computes the cross-spectrum ``G_AB(k) = F(A)·conj(F(B))``,
    radially bins it, and reports a signed correlation per wavelength (FFT-native; no neighbour-graph
    equivalent).

    Parameters
    ----------
    a, b : numpy.ndarray
        Two ``(H, W)`` gridded fields on a shared lattice (e.g. two genes rasterized to the same bins).
    mask : numpy.ndarray
        Boolean ``(H, W)`` mask of occupied bins; fields are demeaned within it.
    n_bins : int, optional
        Number of radial wavelength bins. Default ``24``.
    bin_size : float, optional
        Physical size of a grid bin (sets the wavelength units). Default ``1.0``.
    kmax_frac : float, optional
        Exclude the noisy high-frequency corner above this fraction of Nyquist. Default ``0.7``.

    Returns
    -------
    dict
        ``wavelength_um``, ``coexpr`` (signed correlation in ``[-1, 1]``; ``>0`` co-vary, ``<0``
        anti-localize), ``coherence`` (phase consistency in ``[0, 1]``), and ``k`` — each ``(n_bins,)``.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:64, 0:64]
    >>> a = np.sin(2 * np.pi * xx / 16.0)
    >>> b = np.sin(2 * np.pi * xx / 16.0 + 0.1)
    >>> mask = np.ones((64, 64), bool)
    >>> out = spectrast.spatial.cross_spectral_density(a, b, mask)
    >>> out["coexpr"].shape
    (24,)
    """
    m = np.asarray(mask, bool)
    fa = np.asarray(a, float); fb = np.asarray(b, float)
    fa = np.where(m, fa - fa[m].mean(), 0.0); fb = np.where(m, fb - fb[m].mean(), 0.0)
    H, W = fa.shape
    FA = _fft.fft2_hp(fa); FB = _fft.fft2_hp(fb)
    G = FA * np.conj(FB); Paa = np.abs(FA) ** 2; Pbb = np.abs(FB) ** 2
    ky = np.fft.fftfreq(H, d=bin_size); kx = np.fft.fftfreq(W, d=bin_size)
    Kr = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    knyq = Kr.max(); edges = np.linspace(0, kmax_frac * knyq, n_bins + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(Kr.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins) & (Kr.ravel() > 1e-12)          # exclude DC + out-of-band
    idx = idx[valid]
    reG = np.real(G).ravel()[valid]; pa = Paa.ravel()[valid]; pb = Pbb.ravel()[valid]
    norm = np.sqrt(pa * pb) + 1e-12
    coexpr = np.bincount(idx, weights=reG / norm, minlength=n_bins) / np.maximum(np.bincount(idx, minlength=n_bins), 1)
    reG_sum = np.bincount(idx, weights=reG, minlength=n_bins)
    imG_sum = np.bincount(idx, weights=np.imag(G).ravel()[valid], minlength=n_bins)
    pa_sum = np.bincount(idx, weights=pa, minlength=n_bins); pb_sum = np.bincount(idx, weights=pb, minlength=n_bins)
    coherence = np.sqrt(reG_sum ** 2 + imG_sum ** 2) / (np.sqrt(pa_sum * pb_sum) + 1e-12)
    return dict(wavelength_um=1.0 / np.maximum(ctr, 1e-12), coexpr=coexpr, coherence=coherence, k=ctr)


# ---------------------------------------------------------------------------
# Bivariate mark cross-correlation — the real-space dual of the cross-spectrum
# ---------------------------------------------------------------------------

MCC_MIN_SHIFT_OVERLAP = 0.25   # a torus shift must retain this fraction of the tissue to count


def _mcc_annuli(bin_size, n_bins, r_max, H, W):
    """Annulus edges in physical units.

    The first edge is one bin, not zero: on a lattice the smallest non-zero displacement is
    exactly one bin, so an annulus [0, bin_size) would always be empty. ``r_max`` defaults to a
    quarter of the shorter side — past that the overlap between the tissue and its own translate
    shrinks fast, and the per-annulus pair count gets too small to normalize against.
    """
    if r_max is None:
        r_max = 0.25 * min(H, W) * bin_size
    r_max = max(float(r_max), 2.0 * bin_size)
    return np.linspace(bin_size, r_max, int(n_bins) + 1)


def _mcc_binner(edges, bin_size, fh, fw, include_self):
    """Displacement lattice -> annulus index, built once per (grid, annuli).

    Which annulus a displacement falls in depends only on the padded transform size, so it is the
    same for every field pair on that grid. Hoisting it out of the per-pair work is half of what
    makes the batched path cheap; the other half is memoizing the forward transforms.
    """
    dy = np.fft.fftfreq(fh, d=1.0 / fh)
    dx = np.fft.fftfreq(fw, d=1.0 / fw)
    dist = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2) * bin_size
    n_bins = len(edges) - 1
    idx = np.digitize(dist.ravel(), edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    if not include_self:
        sp_ = np.zeros(dist.shape, bool)
        sp_[0, 0] = True                       # d = (0,0): a bin paired with itself
        keep &= ~sp_.ravel()
    return dict(idx=idx[keep], keep=keep, n_bins=n_bins, fh=fh, fw=fw)


class _MccCache:
    """Memoized forward transforms, keyed by a caller-supplied name.

    Over a list of P pairs drawn from K distinct fields, each field is transformed once rather than
    once per pair it appears in — the saving the batched path exists for. The inverse transform is
    genuinely per pair and stays.
    """

    def __init__(self, fh, fw):
        self.fh, self.fw = fh, fw
        self._f = {}

    def __call__(self, key, arr):
        F = self._f.get(key)
        if F is None:
            F = self._f[key] = _fft.rfft2_hp(arr, s=(self.fh, self.fw))
        return F


def _mcc_bin(FA, FB, bn):
    """Per-annulus SUM of the displacement cross-correlation, from two forward transforms.

    ``S(d) = sum_x fa(x)·fb(x+d)`` from one inverse transform, summed over the lattice displacements
    falling in each annulus. Every quantity the statistic needs — mark numerator, density
    denominator, mask pair count — is this same reduction on a different pair of fields, so the
    normalizers below are ratios of calls to this.
    """
    S = _fft.irfft2_hp(np.conj(FA) * FB, s=(bn["fh"], bn["fw"]))
    return np.bincount(bn["idx"], weights=S.ravel()[bn["keep"]], minlength=bn["n_bins"])


def _mcc_radial(fa, fb, bn):
    """``_mcc_bin`` for two loose fields — the uncached form, for one-off use."""
    return _mcc_bin(_fft.rfft2_hp(fa, s=(bn["fh"], bn["fw"])),
                    _fft.rfft2_hp(fb, s=(bn["fh"], bn["fw"])), bn)


def _mcc_curve(mA, mB, nA, nB, m, bn, normalize, cache=None, keys=("a", "b")):
    """One curve. Returns (value, n_pairs).

    ``normalize`` picks which question is being asked:

    ``csr``       density numerator over the CSR expectation -> the cross pair-correlation
                  function. "Are A-cells enriched near B-cells", geometry only, marks ignored.
    ``density``   mark numerator over the DENSITY numerator -> Stoyan's mark correlation. The mean
                  product of marks CONDITIONAL on a cell pair existing at that distance, so tissue
                  geometry divides out and what is left is expression. This is the one that answers
                  "is a ligand-expressing sender enriched around a receptor-expressing receiver
                  BEYOND where cells simply happen to sit", and its peak is the interaction range.
    ``coherence`` mark fields against the CSR expectation. Descriptive only: it conflates "A and B
                  are co-expressed" with "many cells are here", which is exactly why it must not be
                  used to read an interaction distance.

    ``cache`` and ``keys`` are the batching hooks: pass a shared :class:`_MccCache` and a per-field
    identifier for each side, and a field appearing in several pairs is transformed once.
    """
    F = cache if cache is not None else _MccCache(bn["fh"], bn["fw"])
    ka, kb = keys
    Fm = F("mask", m.astype(float))
    n_pairs = _mcc_bin(Fm, Fm, bn)
    occ = float(m.sum())
    geom = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if normalize == "csr":
            num = _mcc_bin(F(("n", ka), nA), F(("n", kb), nB), bn)
            mu = (nA.sum() / max(occ, 1.0)) * (nB.sum() / max(occ, 1.0))
            val = np.where(n_pairs > 0, num / np.maximum(n_pairs, 1e-30), np.nan) / mu
        elif normalize == "density":
            WA, WB = nA * mA, nB * mB
            num = _mcc_bin(F(("W", ka), WA), F(("W", kb), WB), bn)
            den = _mcc_bin(F(("n", ka), nA), F(("n", kb), nB), bn)
            # density-WEIGHTED mark means, so the ratio is 1 when marks are independent of position
            mu = (WA.sum() / max(nA.sum(), 1e-30)) * (WB.sum() / max(nB.sum(), 1e-30))
            val = np.where(np.abs(den) > 0, num / np.where(np.abs(den) > 0, den, np.nan),
                           np.nan) / mu
            # The GEOMETRY companion, free: `den` IS the density cross-correlation, so normalizing it
            # by the CSR expectation gives the cross-PCF for this same pair at no extra transform.
            # Reporting it beside k_mm separates the two questions a single curve conflates — are the
            # cells carrying A near the cells carrying B at all, and is the MARK coupled beyond that.
            mu_g = (nA.sum() / max(occ, 1.0)) * (nB.sum() / max(occ, 1.0))
            geom = np.where(n_pairs > 0, den / np.maximum(n_pairs, 1e-30), np.nan) / mu_g
        elif normalize == "coherence":
            num = _mcc_bin(F(("m", ka), mA), F(("m", kb), mB), bn)
            mu = (mA.sum() / max(occ, 1.0)) * (mB.sum() / max(occ, 1.0))
            val = np.where(n_pairs > 0, num / np.maximum(n_pairs, 1e-30), np.nan) / mu
        else:
            raise ValueError("normalize must be 'csr', 'density', 'density_robust' or "
                             "'coherence'")
    return val, n_pairs, geom


def _mcc_fields(a, mask, density, *, name):
    """Coerce one side's mark and density onto the mask, refusing sign-changing input."""
    a = np.asarray(a, float)
    if a.shape != mask.shape:
        raise ValueError(f"{name} and mask must share a shape, got {a.shape} and {mask.shape}")
    f = np.where(mask, a, 0.0)
    n = mask.astype(float) if density is None else np.where(mask, np.asarray(density, float), 0.0)
    for nm, v in ((name, f), (f"density of {name}", n)):
        if v[mask].min() < 0:
            raise ValueError(
                f"{nm} has negative values; this statistic is a ratio to the product of the "
                "marginal means and is not interpretable for sign-changing input. Shift it "
                "non-negative, or use cross_spectral_density for a signed measure.")
    return f, n


def _mcc_homogenize(dens, m, bandwidth, bin_size):
    """Divide a density field by its own LOCAL MEAN — the density-robust pre-step.

    Our statistic is a global RATIO OF SUMS, so every location enters weighted by how many partners
    it has: a handful of dense regions can carry the tissue-level number. LIANA's estimator instead
    averages per-receiver ratios, which weights locations equally — that is one of the two reasons
    the two disagree at r~0.87. Rescaling each density by the local mean density recovers most of
    that behaviour for one convolution per field, without giving up the FFT (a faithful
    mean-of-ratios needs a per-receiver denominator inside the sum, i.e. one convolution per
    receiver, which forfeits the whole speedup).

    Because ``W = n * m``, dividing the DENSITY alone gives both homogenized fields at once:
    ``n/rho`` and ``W/rho = (n/rho) * m``. The mark is untouched.

    ``bandwidth`` trades two errors against each other and MEASURED, neither is free (planted
    hotspot vs planted co-located coupling):

    ======================  ==========================  ============================
    bandwidth               hotspot artefact remaining  real co-located signal kept
    ======================  ==========================  ============================
    0.25 x r_max            0.009  (of 0.196)           37%
    0.50 x r_max (default)  0.041                       68%
    1.00 x r_max            0.097                       159% (AMPLIFIES)
    ======================  ==========================  ============================

    rho has to TRACK the density heterogeneity to divide it out, so a kernel wider than the analysis
    range smooths straight over a nest and can even up-weight it (the >100% row). Narrow it and the
    correction bites harder — including into coupling that genuinely lives in a dense region, which
    it cannot tell apart from an artefact because the two are the same signal geometrically. The
    default sits at the balanced point.
    """
    from scipy.signal import fftconvolve  # lazy
    r = max(1, int(round(float(bandwidth) / float(bin_size))))
    yy, xx = np.mgrid[-2 * r:2 * r + 1, -2 * r:2 * r + 1]
    g = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * r ** 2))
    g /= g.sum()
    mf = m.astype(float)
    num = fftconvolve(dens, g, mode="same")
    den = fftconvolve(mf, g, mode="same")          # occupancy-aware: local MEAN, not local total
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 1e-12, num / np.where(den > 1e-12, den, np.nan), np.nan)
    # spatial_smooth-style ratios are garbage outside the mask (see the toolkit skill); mask first,
    # and floor rho by a tiny fraction of its in-mask mean so an empty pocket cannot explode.
    rho = np.where(m & np.isfinite(rho), rho, 0.0)
    scale = float(rho[m].mean()) if m.any() else 1.0
    rho = np.maximum(rho, 1e-6 * max(scale, 1e-30))
    return np.where(m, dens / rho, 0.0)


def _mcc_curves(marks, dens, m, pairs, bn, normalize, cache=None):
    """Observed curves for every pair on one grid. Returns ``(g, n_pairs, geom)``.

    ``geom`` is ``None`` unless the normalizer produces the geometry companion. Shared by the
    observed pass of :func:`_mcc_run` and by nulls that rebuild the fields per draw (the
    density-stratified label permutation in :mod:`spectrast.null`), so both take the same path.
    """
    cache = cache if cache is not None else _MccCache(bn["fh"], bn["fw"])
    P, nb = len(pairs), bn["n_bins"]
    g = np.empty((P, nb))
    geom = np.full((P, nb), np.nan)
    n_pairs, have_geom = None, False
    for p, (i, j) in enumerate(pairs):
        g[p], n_pairs, gm = _mcc_curve(marks[i], marks[j], dens[i], dens[j], m, bn, normalize,
                                       cache=cache, keys=(i, j))
        if gm is not None:
            geom[p], have_geom = gm, True
    return g, n_pairs, (geom if have_geom else None)


def _mcc_score(g, draws, r):
    """z, per-annulus p, and the max-over-annuli statistic from a stack of null draws.

    ``draws`` is ``(n_perm, P, nb)``; a rejected draw is a row of NaN. The max statistic
    standardizes each annulus by its own null spread and takes the maximum over annuli in the
    observed curve AND every draw, so choosing the peak by looking is priced in — the same
    construction as the differential's max-over-scale. One implementation for every null.
    """
    n_valid = np.isfinite(draws).any(axis=2).sum(axis=0)               # (P,)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(draws, axis=0)
        sd = np.nanstd(draws, axis=0)
    med = np.nanmedian(sd, axis=1)
    floor = np.where(np.isfinite(med), med * 1e-3, 1e-12)
    sd_f = np.maximum(sd, np.maximum(floor, 1e-12)[:, None])
    ge = np.nansum(np.abs(draws - mu) >= np.abs(g - mu), axis=0)
    # A per-annulus p read at the annulus where the curve happens to peak is optimistic — the peak
    # was chosen by looking. The max-statistic standardizes each annulus by its own null spread and
    # takes the maximum over annuli in BOTH the observed curve and every draw, so selecting the peak
    # is priced in. Same construction as the differential's max-over-scale statistic.
    with np.errstate(invalid="ignore"):
        dev = np.abs(g - mu) / sd_f                            # (P, nb)
        null_dev = np.abs(draws - mu) / sd_f[None]             # (n_perm, P, nb)
    # A fully degenerate null (every draw rejected for want of overlap) leaves these all-NaN. Report
    # NaN rather than raise — the caller, _mcc_run, is the one that knows n_valid against n_perm and
    # warns there when usable draws run short; raising here would take down a whole batched screen
    # because one pair's geometry was awkward.
    finite = np.isfinite(dev)
    scored = np.where(finite, dev, -np.inf)
    usable = finite.any(axis=1)
    T_obs = np.where(usable, scored.max(axis=1), np.nan)
    r_at = np.where(usable, np.asarray(r)[scored.argmax(axis=1)], np.nan)
    T_null = np.where(np.isfinite(null_dev), null_dev, -np.inf).max(axis=2)
    # A NaN observed value compares as less extreme than every draw, which put its p at the floor,
    # 1 / (n_valid + 1): the differential's false positive (fixed in 0.13.1). An annulus with no observed
    # value or no null, and a curve with no finite annulus, have no test: p = 1.
    with np.errstate(invalid="ignore"):
        pval = np.where(np.isfinite(g - mu), (ge + 1.0) / (n_valid[:, None] + 1.0), 1.0)
        p_max = np.where(np.isfinite(T_obs), (np.nansum(T_null >= T_obs[None], axis=0) + 1.0) / (n_valid + 1.0), 1.0)
    return dict(z=(g - mu) / sd_f, pval=pval, null_mean=mu, null_sd=sd, T_max=T_obs, p_max=p_max,
                r_at_max_um=r_at, n_valid=n_valid)


def _mcc_run(marks, dens, m, pairs, edges, bin_size, include_self, normalize,
             null, n_perm, seed, density_bw=None):
    """The engine behind both the single-pair and the batched entry points.

    Routing both through one function is what keeps them from drifting: the batch is the same code
    with the forward transforms memoized across pairs and each null draw's geometry built once and
    reused for every pair. The draws come off a generator seeded only by ``seed`` and advanced by the
    mask, never by a mark, so a batch reproduces looped single calls bit for bit.
    """
    H, W = m.shape
    if normalize == "density_robust":
        # one convolution per FIELD, not per pair — the cost is amortized over the pair list
        bw = float(density_bw) if density_bw else 0.5 * float(edges[-1])
        dens = [_mcc_homogenize(d, m, bw, bin_size) for d in dens]
        normalize = "density"
    # Zero-pad past the largest lag: an unpadded transform wraps pairs from one tissue edge to the
    # other, inventing pairs that do not exist, in both the numerator and the pair count.
    from scipy.fft import next_fast_len  # lazy
    lag = int(np.ceil(edges[-1] / bin_size)) + 1
    fh, fw = next_fast_len(H + lag), next_fast_len(W + lag)
    bn = _mcc_binner(edges, bin_size, fh, fw, include_self)

    P, nb = len(pairs), bn["n_bins"]
    g, n_pairs, geom = _mcc_curves(marks, dens, m, pairs, bn, normalize)
    out = dict(r=0.5 * (edges[:-1] + edges[1:]), g=g, n_pairs=n_pairs, edges=edges,
               normalize=normalize)
    if geom is not None:
        out["g_geometry"] = geom
    if null is None or not n_perm:
        return out
    if null not in ("shift", "reshuffle"):
        raise ValueError("null must be None, 'shift' or 'reshuffle'")

    rng = np.random.default_rng(seed)
    draws = np.full((int(n_perm), P, nb), np.nan)
    for d in range(int(n_perm)):
        dcache = _MccCache(fh, fw)                     # the draw's fields, shared across pairs
        if null == "reshuffle":
            perm = rng.permutation(int(m.sum()))       # one shuffle, applied to every B mark
            for p, (i, j) in enumerate(pairs):
                pm = np.zeros_like(marks[j])
                pm[m] = marks[j][m][perm]              # permute MARKS at fixed geometry
                draws[d, p] = _mcc_curve(marks[i], pm, dens[i], dens[j], m, bn, normalize,
                                         cache=dcache, keys=(("a", i), ("perm", j)))[0]
            continue
        ms = sy = sx = None
        for _ in range(32):
            cy = int(rng.integers(1, H)); cx = int(rng.integers(1, W))
            cand = np.roll(np.roll(m, cy, 0), cx, 1) & m
            if cand.sum() >= MCC_MIN_SHIFT_OVERLAP * m.sum():
                ms, sy, sx = cand, cy, cx
                break
        if ms is None:
            continue

        def roll(f, sy=sy, sx=sx):
            return np.roll(np.roll(f, sy, 0), sx, 1)
        for p, (i, j) in enumerate(pairs):
            draws[d, p] = _mcc_curve(np.where(ms, marks[i], 0.0), np.where(ms, roll(marks[j]), 0.0),
                                     np.where(ms, dens[i], 0.0), np.where(ms, roll(dens[j]), 0.0),
                                     ms, bn, normalize, cache=dcache,
                                     keys=(("a", i), ("b", j)))[0]

    sc = _mcc_score(g, draws, np.asarray(out["r"]))
    n_valid = sc["n_valid"]
    if n_valid.min() < max(min(int(n_perm), 10), 0.5 * int(n_perm)):
        import warnings as _w
        _w.warn(f"mark_cross_correlation: only {int(n_valid.min())}/{n_perm} {null} draws were "
                f"usable (the tissue overlaps its own translate too little). Treat z and p as "
                f"indicative.", RuntimeWarning, stacklevel=3)
    out.update(**sc, null=null, n_perm=int(n_perm))
    return out


def mark_cross_correlation(a, b, mask, *, density_a=None, density_b=None,
                           normalize: str = "density", bin_size: float = 1.0,
                           n_bins: int = 12, r_max=None, annuli=None,
                           null: str | None = None, n_perm: int = 0, seed: int = 0,
                           include_self: bool = False, density_bw=None) -> dict:
    """Distance-resolved co-organization of two marked fields — the real-space cross-statistic.

    The real-space dual of :func:`cross_spectral_density`. Where that reports co-expression per
    WAVELENGTH, this reports it per DISPLACEMENT, and — unlike a plain field cross-correlation — it
    keeps GEOMETRY and VALUE apart. Each bin carries a density ``n(x)`` (how many cells sit there)
    and a mark ``m(x)`` (what they express); tracking the two separately is what lets the normalizer
    choose which question is being asked:

    ``normalize="csr"``
        Density against the random-placement expectation: the cross pair-correlation function.
        Marks are ignored. ``g > 1`` means A-cells are enriched near B-cells.
    ``normalize="density"`` (default)
        Mark product CONDITIONED on a cell pair existing at that distance — Stoyan's mark
        correlation. Geometry divides out, so what remains is expression, and the peak is the
        interaction range. This is the one to read a distance off.
    ``normalize="coherence"``
        Mark fields against the random-placement expectation. Descriptive only: it conflates "A and
        B are co-expressed" with "many cells are here", which is why it must not be used to read an
        interaction distance.
    ``normalize="density_robust"``
        As ``density``, but each density field is first divided by its own local mean (bandwidth
        ``density_bw``, default half the largest annulus). A deliberately CONSERVATIVE robustness
        check, not a better answer, and it reads in one direction only: if a result survives, it is
        not being carried by a few dense regions; if it does NOT survive, that is ambiguous, because
        real coupling inside a dense region is attenuated too (measured: 68% retained at the default
        bandwidth). Exactly a no-op on homogeneous density. One convolution per field.

    One statistic covers the named ones: ligand and receptor marks give the ligand-receptor
    interaction curve (``density``); two cell-type densities give the cross-PCF (``csr``); two
    program or fingerprint fields give co-organization (``density``).

    Parameters
    ----------
    a, b : numpy.ndarray
        ``(H, W)`` MARK fields — mean mark per bin. Non-negative: the statistic is a ratio to the
        product of the marginal means and is not interpretable for a sign-changing mark.
    mask : numpy.ndarray
        Boolean ``(H, W)`` tissue support. Its autocorrelation gives the exact pair count per
        displacement, so tissue shape and edges are handled exactly rather than approximated.
    density_a, density_b : numpy.ndarray, optional
        ``(H, W)`` cell-density fields. Default: the mask itself (one unit of density per occupied
        bin), which is right when the marks are already per-bin means.
    null : {None, "shift", "reshuffle"}, optional
        ``"shift"`` rolls B toroidally and intersects the masks, PRESERVING each field's own
        autocorrelation. ``"reshuffle"`` permutes B's marks over occupied bins at fixed geometry.
        For smooth marks reshuffle is badly anti-conservative — 88% false positives against 4% for
        shift — because it destroys the
        autocorrelation the observed statistic still has.

    Returns
    -------
    dict
        ``r``, ``g``, ``n_pairs``, ``edges``, ``normalize``; plus ``z``, ``pval``, ``null_mean``,
        ``null_sd``, ``n_valid`` and the peak-selection-corrected ``T_max`` / ``p_max`` /
        ``r_at_max_um`` when a null is run. Under ``normalize="density"`` also ``g_geometry`` — the
        cross-PCF of the same pair, obtained free from the denominator, which says whether the cells
        carrying the two marks are near each other at all. A mark signal without geometric
        co-location is coupling unexplained by where cells sit; geometric co-location without a mark
        signal is cells that are adjacent but not co-organized. It is a REFERENCE curve: no null is
        run on it, so it carries no p-value.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:64, 0:64]
    >>> blob = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 8.0 ** 2))
    >>> mask = np.ones((64, 64), bool)
    >>> out = spectrast.spatial.mark_cross_correlation(blob, blob, mask, n_bins=6)
    >>> bool(out["g"][0] > out["g"][-1])
    True
    """
    m = np.asarray(mask, bool)
    if not m.any():
        raise ValueError("mask selects no bins")
    if np.asarray(a).shape != m.shape or np.asarray(b).shape != m.shape:
        raise ValueError("a, b and mask must share a shape")
    mA, nA = _mcc_fields(a, m, density_a, name="a")
    mB, nB = _mcc_fields(b, m, density_b, name="b")
    H, W = m.shape
    edges = np.asarray(annuli, float) if annuli is not None else \
        _mcc_annuli(bin_size, n_bins, r_max, H, W)

    out = _mcc_run([mA, mB], [nA, nB], m, [(0, 1)], edges, bin_size, include_self, normalize,
                   null, n_perm, seed, density_bw)
    for k in ("g", "g_geometry", "z", "pval", "null_mean", "null_sd"):   # one pair -> drop axis
        if k in out:
            out[k] = out[k][0]
    for k in ("T_max", "p_max", "r_at_max_um"):
        if k in out:
            out[k] = float(out[k][0])
    if "n_valid" in out:
        out["n_valid"] = int(out["n_valid"][0])
    return out


def _mcc_annulus_kernel(r_lo, r_hi, bin_size, include_self=False):
    """0/1 kernel selecting the displacements that fall in one annulus.

    Built with the SAME membership test the curve's binner uses, so the map below is a decomposition
    of the curve rather than a lookalike computed a second way. Symmetric under ``d -> -d``, so
    correlation and convolution coincide and one ``fftconvolve`` suffices.
    """
    L = int(np.ceil(float(r_hi) / float(bin_size)))
    dy, dx = np.mgrid[-L:L + 1, -L:L + 1]
    dist = np.sqrt(dy ** 2 + dx ** 2) * bin_size
    k = ((dist >= r_lo) & (dist < r_hi)).astype(float)
    if not include_self:
        k[L, L] = 0.0
    return k


def mark_cross_map(a, b, mask, *, r_lo, r_hi, density_a=None, density_b=None,
                   normalize: str = "density", bin_size: float = 1.0,
                   kind: str = "contribution", include_self: bool = False) -> dict:
    """WHERE the curve's signal at one annulus comes from — the first-order map behind ``k(r)``.

    A distance-resolved curve says at what RANGE two marks are coupled; it cannot say WHERE in the
    tissue that coupling lives, and a curve without its map is not an interpretable result. This
    returns the per-location integrand of the very sum the curve reports:

        num(r) = sum_x  A(x) · (B * K_r)(x)

    for the annulus kernel ``K_r``, so ``contribution.sum()`` reproduces the curve's numerator at
    that annulus to floating-point. The map is therefore a DECOMPOSITION of the statistic, not a
    separate heuristic that merely resembles it.

    Parameters
    ----------
    r_lo, r_hi : float
        The annulus, in the same units as ``bin_size``. Take these from a curve's ``edges`` at the
        peak index to get "where does the interaction at the optimal distance happen".
    kind : {"contribution", "local"}, optional
        ``contribution`` is the raw integrand; its sum IS the numerator, so it answers "which
        locations produced this number" and is the right thing to overlay on tissue.
        ``local`` divides by the matching density term and the global mean product, giving a local
        analogue of ``k`` that is ~1 where coupling matches the tissue-wide expectation. Use it to
        compare regions of different cell density, where the raw contribution is dominated by
        wherever the cells are.

    Returns
    -------
    dict
        ``map`` ``(H, W)``, ``numerator`` (its sum), ``r_lo``, ``r_hi``, ``kind``, ``normalize``.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:48, 0:48]
    >>> blob = np.exp(-((yy - 24) ** 2 + (xx - 24) ** 2) / 32.0)
    >>> out = spectrast.spatial.mark_cross_map(blob, blob, np.ones((48, 48), bool),
    ...                                        r_lo=1.0, r_hi=4.0)
    >>> bool(np.unravel_index(np.argmax(out["map"]), out["map"].shape) == (24, 24))
    True
    """
    m = np.asarray(mask, bool)
    fa, na = _mcc_fields(a, m, density_a, name="a")
    fb, nb = _mcc_fields(b, m, density_b, name="b")
    if kind not in ("contribution", "local"):
        raise ValueError("kind must be 'contribution' or 'local'")
    if normalize == "csr":
        A, B = na, nb
    elif normalize == "density":
        A, B = na * fa, nb * fb
    elif normalize == "coherence":
        A, B = fa, fb
    else:
        raise ValueError("normalize must be 'csr', 'density' or 'coherence'")

    k = _mcc_annulus_kernel(r_lo, r_hi, bin_size, include_self)
    fld = A * _fftconv_same(B[..., None], k)[..., 0]
    out = dict(map=fld, numerator=float(fld.sum()), r_lo=float(r_lo), r_hi=float(r_hi),
               kind=kind, normalize=normalize)
    if kind == "contribution":
        return out
    # local: divide by the same-annulus density term, then by the global mean product, so the value
    # is on the curve's own scale (1 = as expected tissue-wide) rather than in units of expression.
    den = na * _fftconv_same(nb[..., None], k)[..., 0]
    if normalize == "density":
        mu = (A.sum() / max(na.sum(), 1e-30)) * (B.sum() / max(nb.sum(), 1e-30))
    else:
        occ = max(float(m.sum()), 1.0)
        mu = (A.sum() / occ) * (B.sum() / occ)
        den = m.astype(float) * _fftconv_same(m.astype(float)[..., None], k)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["map"] = np.where(den > 0, fld / np.where(den > 0, den, np.nan) / max(mu, 1e-30), np.nan)
    return out


def mark_cross_correlation_batch(marks, mask, pairs, *, densities=None,
                                 normalize: str = "density", bin_size: float = 1.0,
                                 n_bins: int = 12, r_max=None, annuli=None,
                                 null: str | None = None, n_perm: int = 0, seed: int = 0,
                                 include_self: bool = False, density_bw=None) -> dict:
    """:func:`mark_cross_correlation` over a whole pair list, transforming each field once.

    A pair list of length ``P`` drawn from ``K`` distinct fields needs only ``K`` forward transforms,
    not ``4P``, and one displacement-to-annulus map instead of ``P``. Under a null the draw geometry
    is built once per permutation and reused across pairs, so the saving compounds. Results are
    bit-identical to calling :func:`mark_cross_correlation` once per pair with the same ``seed`` —
    both route through the same engine.

    Parameters
    ----------
    marks : sequence of numpy.ndarray
        ``K`` mark fields, each ``(H, W)``.
    mask : numpy.ndarray
        Boolean ``(H, W)`` tissue support, shared by every field.
    pairs : sequence of tuple(int, int)
        Index pairs ``(i, j)`` into ``marks``; ``i`` is the A side, ``j`` the B side.
    densities : sequence of numpy.ndarray, optional
        Matching density fields. Default: the mask for every field.

    Returns
    -------
    dict
        As :func:`mark_cross_correlation`, but ``g`` (and ``z``, ``pval``, ``null_mean``,
        ``null_sd``) are ``(P, n_bins)``, plus ``pairs``.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:48, 0:48]
    >>> f = [np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / 128.0) for c in (16, 24, 32)]
    >>> out = spectrast.spatial.mark_cross_correlation_batch(
    ...     f, np.ones((48, 48), bool), [(0, 1), (0, 2)], n_bins=6)
    >>> out["g"].shape
    (2, 6)
    """
    m = np.asarray(mask, bool)
    if not m.any():
        raise ValueError("mask selects no bins")
    marks = list(marks)
    densities = [None] * len(marks) if densities is None else list(densities)
    if len(densities) != len(marks):
        raise ValueError("densities must be None or one field per mark")
    pairs = [(int(i), int(j)) for i, j in pairs]
    if not pairs:
        raise ValueError("pairs is empty")
    for i, j in pairs:
        if not (0 <= i < len(marks) and 0 <= j < len(marks)):
            raise IndexError(f"pair ({i}, {j}) is out of range for {len(marks)} marks")
    fields = [_mcc_fields(v, m, d, name=f"marks[{k}]")
              for k, (v, d) in enumerate(zip(marks, densities))]
    H, W = m.shape
    edges = np.asarray(annuli, float) if annuli is not None else \
        _mcc_annuli(bin_size, n_bins, r_max, H, W)
    out = _mcc_run([f for f, _ in fields], [n for _, n in fields], m, pairs, edges, bin_size,
                   include_self, normalize, null, n_perm, seed, density_bw)
    out["pairs"] = pairs
    return out


def variance_decomposition(grid: np.ndarray, mask: np.ndarray, radii,
                           kind: str = "gaussian"):
    """Decompose each gene's SPATIAL variance across scale bands (Laplacian pyramid).

    Given increasing smoothing radii ``radii = [r1 < r2 < ... < rK]``, build low-pass
    smooths ``S_k = spatial_smooth(grid, r_k)`` and split the signal into bands::

        band_0 = grid - S_1        # detail finer than r1 (cellular texture)
        band_k = S_k - S_{k+1}     # structure between r_k and r_{k+1}
        band_K = S_K               # coarse residual (>= rK; regional/tissue field)

    The per-band spatial variance (over occupied cells; the mean field drops out) is
    normalized to a fraction per gene. So a single FFT pyramid tells you WHERE a gene's
    spatial variance lives: fine-band-dominated = cell-type/cellular markers,
    coarse-band-dominated = regional/anatomical gradients. Complements
    :func:`correlation_length` (a threshold crossing) with a full scale spectrum.

    Returns (frac, var, band_labels):
      frac        (n_bands, G) fraction of summed band variance per gene (columns sum to 1)
      var         (n_bands, G) raw per-band spatial variance
      band_labels list[str] human-readable scale band edges
    """
    if grid.ndim == 2:
        grid = grid[..., None]
    radii = sorted(int(r) for r in radii)
    smooths = [spatial_smooth(grid, mask, radius=r, kind=kind) for r in radii]
    bands = [grid - smooths[0]]
    for k in range(len(radii) - 1):
        bands.append(smooths[k] - smooths[k + 1])
    bands.append(smooths[-1])
    m = mask
    var = np.empty((len(bands), grid.shape[-1]))
    for i, b in enumerate(bands):
        bm = b[m]                                   # (n_occupied, G)
        var[i] = bm.var(axis=0)
    frac = var / np.where(var.sum(0, keepdims=True) == 0, np.nan, var.sum(0, keepdims=True))
    edges = [f"<{radii[0]}"] + [f"{radii[k]}-{radii[k+1]}" for k in range(len(radii) - 1)] + [f">{radii[-1]}"]
    return frac, var, edges


def neighborhood_enrichment(label_grid: np.ndarray, mask: np.ndarray, radius: int = 5,
                            n_perm: int = 50, seed: int = 0, kind: str = "disc"):
    """FFT cell-type neighborhood co-occurrence enrichment at ANY radius.

    label_grid: (H, W) integer label codes, -1 where empty. For each label pair (a,b),
    observed co-occurrence = sum_i [label_i=a] * (count of b within `radius` of i),
    computed by convolving one-hot fields (O(N log N), radius-independent). Compared to
    label permutations among occupied cells.

    Returns (enrichment, z, labels):
      enrichment = log2(observed / expected) -- the headline, interpretable AND stable
                   at HD N (a z-score is not: huge N -> tiny null variance -> z blows up).
      z          = (obs - mean_perm) / sd_perm  (kept for reference; inflated at large N).
    """
    from scipy.signal import fftconvolve  # lazy
    occ = mask & (label_grid >= 0)
    idx = np.argwhere(occ)
    codes = label_grid[occ]
    labels = np.unique(codes)
    L = len(labels)
    k = neighbor_kernel(radius, kind)

    def cooccurrence(code_vec):
        fields = np.zeros((L, *label_grid.shape))
        for li, lab in enumerate(labels):
            fields[li][idx[code_vec == lab, 0], idx[code_vec == lab, 1]] = 1.0
        conv = np.stack([fftconvolve(f, k, mode="same") for f in fields])
        return np.array([[(fields[a] * conv[b]).sum() for b in range(L)] for a in range(L)])

    obs = cooccurrence(codes)
    rng = np.random.default_rng(seed)
    pl = [rng.permutation(codes) for _ in range(n_perm)]       # pre-draw (identical to serial)
    from ._parallel import parallel_map
    perms = np.array(parallel_map(cooccurrence, pl))           # FFT-bound null, threaded (Phase 2)
    mu, sd = perms.mean(0), perms.std(0) + 1e-9
    enrichment = np.log2((obs + 1.0) / (mu + 1.0))
    return enrichment, (obs - mu) / sd, labels


def bivariate_morans(field_a: np.ndarray, field_b: np.ndarray, mask: np.ndarray,
                     kernel: np.ndarray | None = None, radius: int = 1) -> float:
    """Bivariate Moran's I: does field_a at a location predict field_b in its
    neighborhood? = (N/S0) * sum_i z_a,i (W z_b)_i / sqrt(sum z_a^2 sum z_b^2).
    Captures spatial cross-organization (e.g. ligand-receptor, co-localized programs).
    """
    from scipy.signal import fftconvolve  # lazy
    if kernel is None:
        kernel = neighbor_kernel(radius)
    m = mask.astype(float); msum = m.sum()
    za = (field_a - field_a[mask].mean()) * m
    zb = (field_b - field_b[mask].mean()) * m
    Wzb = _fftconv_same(zb[..., None], kernel)[..., 0]
    S0 = (fftconvolve(m, kernel, mode="same") * m).sum()
    denom = np.sqrt((za * za).sum() * (zb * zb).sum())
    return float((msum / S0) * (za * Wzb).sum() / denom) if denom > 0 and S0 > 0 else np.nan


def spectral_anisotropy(field: np.ndarray, mask: np.ndarray, low_freq_cut: int = 2):
    """Directional anisotropy of a spatial pattern from its 2D POWER SPECTRUM (inertia tensor of P(k)).

    Returns a (index, orientation) TUPLE. (The autocorrelation-tensor variant ``anisotropy`` below
    returns a dict and previously SHADOWED this function under the same name — renamed to disambiguate.)
    index ~0 = isotropic (no preferred
    direction); ~1 = strongly oriented/striped (e.g. layered or fibrous structure). The
    orientation is the dominant axis of the SPATIAL pattern (perpendicular to the power
    spectrum's major axis). Uses the inertia tensor (2nd moment) of the power spectrum.
    """
    z = (field - field[mask].mean()) * mask
    H, W = field.shape
    P = np.abs(np.fft.fftshift(_fft.fft2_hp(z))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H))[:, None] * np.ones((1, W))
    kx = np.fft.fftshift(np.fft.fftfreq(W))[None, :] * np.ones((H, 1))
    keep = (kx ** 2 + ky ** 2) > (low_freq_cut / max(H, W)) ** 2  # drop DC / lowest freqs
    P = P * keep
    tot = P.sum()
    if tot <= 0:
        return 0.0, 0.0
    Cxx = (P * kx * kx).sum() / tot
    Cyy = (P * ky * ky).sum() / tot
    Cxy = (P * kx * ky).sum() / tot
    cov = np.array([[Cxx, Cxy], [Cxy, Cyy]])
    w, v = np.linalg.eigh(cov)               # w[1] >= w[0]
    index = float(1.0 - np.sqrt(max(w[0], 0) / max(w[1], 1e-12)))
    major = v[:, 1]                          # power-spectrum major axis
    orient = float(np.arctan2(major[0], major[1]) + np.pi / 2)  # spatial axis ⟂ to it
    return index, orient


def multivariate_morans_i(grid: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
                          radius: int = 1) -> float:
    """Wartenberg multivariate Moran's I on a (H, W, K) feature grid (any K).

    I = (N/S0) * sum_ij w_ij (z_i . z_j) / sum_i (z_i . z_i), with z the centered
    multivariate vector. A single statistic for spatial coordination ACROSS features
    -- aggregates shared weak spatial signal that per-feature Moran's I can miss.
    """
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    m = mask.astype(float)
    msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum
    z = (grid - zbar) * m[..., None]
    Wz = _fftconv_same(z, kernel)
    S0 = (fftconvolve(m, kernel, mode="same") * m).sum()
    num = (z * Wz).sum()
    den = (z * z).sum()
    if S0 <= 0:  # kernel hits no occupied neighbors (e.g. wrong kernel for a hex grid)
        return np.nan
    return float((msum / S0) * (num / den)) if den > 0 else np.nan


def transition_field(qgrid: np.ndarray, mask: np.ndarray, radius: int = 1,
                     kernel: np.ndarray | None = None) -> np.ndarray:
    """Quaternion-native spatial transition map: angle (rad) between each location's
    expression direction and its neighborhood-mean direction.

    ~0 inside homogeneous domains, large at boundaries/transitions. A boundary
    detector defined directly on the relative expression profile (vector part).
    Pass `kernel` (e.g. a hex kernel) when the lattice is not a simple square grid.
    """
    from scipy.signal import fftconvolve  # lazy
    if kernel is None:
        kernel = neighbor_kernel(radius)
    m = mask.astype(float)
    v = qgrid[..., 1:4]
    nbr = _fftconv_same(v * m[..., None], kernel)        # neighbor-summed directions
    cnt = fftconvolve(m, kernel, mode="same")[..., None]
    nbr = nbr / np.where(cnt == 0, np.nan, cnt)
    nn = nbr / (np.linalg.norm(nbr, axis=-1, keepdims=True) + 1e-12)
    cos = np.clip((v * nn).sum(-1), -1, 1)
    ang = np.arccos(cos)
    ang[~mask] = np.nan
    return ang


def gearys_c(grid: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
             radius: int = 1) -> np.ndarray:
    """Geary's C per gene (complement of Moran's I; ~0 strong autocorr, ~1 none).

    ROW-STANDARDIZED weights (each cell's neighbours weighted 1/deg), consistent with the mean-neighbour
    Moran here and with squidpy (so C ≈ 1 − I at any radius, not just uniform-degree). Uses the identity
    Σ_j w_rs_ij(x_i−x_j)² = x_i² − 2 x_i·meanN(x_i) + meanN(x_i²); with row weights Σ_ij w_rs = N."""
    if grid.ndim == 2:
        grid = grid[..., None]
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    mb = np.asarray(mask, bool)
    m = mb.astype(float)
    msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum
    z = (grid - zbar) * m[..., None]
    x = grid * m[..., None]
    W1 = fftconvolve(m, kernel, mode="same")          # neighbour count per cell (degree)
    has_nb = W1 > 0.5                                  # >=1 neighbour (counts are integers; 0.5 beats fft roundoff)
    inv = np.where(has_nb, 1.0 / np.maximum(W1, 1e-12), 0.0)[..., None]
    meanN_x = _fftconv_same(x, kernel) * inv          # D^-1 W x
    meanN_x2 = _fftconv_same(x ** 2, kernel) * inv     # D^-1 W x^2
    # a neighbourless occupied cell has zero neighbour-pairs -> it must contribute 0, not the raw x^2
    # left when inv=0 zeroes the meanN terms. Gate the per-cell bracket on having >=1 neighbour.
    num = ((x ** 2 - 2.0 * x * meanN_x + meanN_x2) * (has_nb[..., None] & mb[..., None])).sum((0, 1))
    den = (z * z).sum((0, 1))
    S0 = float((has_nb & mb).sum())                   # row-standardized total weight = occupied cells with a neighbour
    if S0 == 0:                                       # no occupied cell has a neighbour: undefined, not 1/0
        return np.full(np.shape(num), np.nan)
    return ((msum - 1) / (2.0 * S0)) * (num / np.where(den == 0, np.nan, den))


def gearys_c_multiscale(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc") -> np.ndarray:
    """Geary's C per gene at MANY radii, sharing forward FFTs (the same amortization as morans_i_multiscale).

    Row-standardized (matches :func:`gearys_c` at each radius to machine precision). Geary needs three
    convolutions — conv(x, k_r), conv(x², k_r), conv(mask, k_r) — so we transform x, x², and the mask ONCE and
    each radius costs only a kernel FFT + product + inverse. Returns (n_radii, G). Far cheaper than looping the
    single-radius primitive (which re-transforms the field at every scale)."""
    if _fft.get_backend() != "cpu":
        return _gearys_c_multiscale_torch(grid, mask, radii, kind)
    # CPU: the shared implementation (_ms_cpu), bit-identical to the per-function code it replaced: conv(x),
    # conv(x^2) and conv(mask) per radius from transforms taken once; a neighbourless occupied cell is gated out.
    return _ms_cpu(grid, mask, radii, kind, ("geary",))["geary"]


def focality_scalogram(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc") -> np.ndarray:
    """Focality (Gini of the positive local-Moran map) per gene at MANY radii, sharing forward FFTs.

    Reduces each radius's LISA map to one scalar (Gini over occupied cells) on the fly, so it needs only the
    shared forward FFT of z and the mask — no (n_radii, H, W, G) map is ever materialized. Returns (n_radii, G):
    the diffuse (0) -> punctate (1) transition across scale. Matches :func:`local_morans_i_batch` + a per-gene
    positive Gini at each radius to machine precision."""
    if _fft.get_backend() != "cpu":
        return _focality_scalogram_torch(grid, mask, radii, kind)
    # CPU: the shared implementation (_ms_cpu), bit-identical to the per-function code it replaced. The local
    # Moran map is formed on occupied bins only (the Gini never read anything else) and the sort is threaded.
    return _ms_cpu(grid, mask, radii, kind, ("focality",))["focality"]


def texture_bands(HW, radii):
    """Radial-frequency bands placed AT the wavelengths matching ``radii`` (radius r <-> wavelength 2r).

    This is the curve bank's binning, lifted out of the driver so one definition serves every caller.
    It is NOT the binning behind :func:`spectrast.gr.texture`, which spreads 24 linear bins to 0.7 of
    the Nyquist radius and reduces them to a single spectral centroid. Both are called "texture" and
    they are different quantities: that one is a scalar fingerprint axis, this one is a curve on the
    same length axis as the other five statistics. Mixing them silently gives curves that agree only
    to about rho 0.93.

    Out-of-band modes are DROPPED rather than clipped into the end bins. Clipping made the coarsest
    bin a catch-all that swallowed DC and reported the tissue mean instead of its texture. A band
    holding fewer than three Fourier modes is marked invalid: at long wavelengths a geometric band can
    be narrower than the section's own frequency resolution, which is physics rather than a defect.
    """
    H, W = HW
    if len(radii) < 2:
        # A band's edges are the geometric midpoints to its NEIGHBOURS, so one radius has no band. The
        # edge array used to be read before it was written (np.empty), which silently turned the "band"
        # into every non-DC mode -- a value, but not texture at that radius.
        raise ValueError("texture bands need at least two radii: each band is bounded by the "
                         "midpoints to its neighbours")
    ky = np.fft.fftfreq(H); kx = np.fft.fftfreq(W)
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2).ravel()
    ka = (1.0 / (2.0 * np.asarray(radii, float)))[::-1]           # ascending frequency
    edges = np.empty(len(ka) + 1)
    edges[1:-1] = np.sqrt(ka[:-1] * ka[1:])
    edges[0] = ka[0] ** 2 / edges[1]
    edges[-1] = ka[-1] ** 2 / edges[-2]
    raw = np.digitize(K, edges) - 1
    inband = (raw >= 0) & (raw < len(ka)) & (K > 0)
    which = np.where(inband, raw, -1)
    ck = np.bincount(which[inband], minlength=len(ka)).astype(float)
    return dict(which=which, inband=inband, ck=ck, valid=(ck >= 3)[::-1])


def texture_multiscale(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc") -> np.ndarray:
    """Spectral TEXTURE per gene in the bands matching ``radii``. Returns (n_radii, G).

    The sixth curve, and the only one that was not a function. It lived inline in the curve bank and
    was copied into every driver that wanted it; this is that code, bit-for-bit, behind the shared
    ``(grid, mask, radii, kind)`` signature so a caller dispatching over statistics needs no special
    case. Validated against the curve bank's inline version at machine precision.

    **It is not a disc statistic**, and ``kind`` is accepted and ignored for signature compatibility
    only. The other five sum over a neighbourhood of radius *r*; this reads the radial power spectrum
    in the band matching wavelength 2*r*, which places it on the same length axis without measuring
    the same thing. A radius here is a band centre, not a disc.

    Bands come from :func:`texture_bands` -- see its note on the OTHER texture in this package.

    **It needs a grid to work on, unlike the other five.** A band must contain enough Fourier modes
    to estimate power, so separation degrades on a small lattice where a disc statistic is still
    fine. Measured on a synthetic structured field against its own shuffle, the ratio of structured
    to unstructured |z| runs 2.4x on a 26x26 grid, 6.1x on 48x48 and 11.7x on 72x72 -- the last
    matching what a real section gives. Below roughly 48x48, prefer the disc statistics.
    """
    if grid.ndim == 2:
        grid = grid[..., None]
    mb = np.asarray(mask, bool)
    H, W, G = grid.shape
    radii = list(radii)
    tb = texture_bands((H, W), radii)
    which, inband, ck = tb["which"], tb["inband"], tb["ck"]
    denom = np.where(ck > 0, ck, np.nan)
    band = which[inband]
    out = np.empty((len(radii), G))

    def _gene(j):
        f = np.where(mb, grid[..., j] - grid[..., j][mb].mean(), 0.0)
        P = (np.abs(np.fft.fft2(f)) ** 2).ravel()
        prof = np.bincount(band, weights=P[inband], minlength=len(radii)) / denom
        out[:, j] = prof[::-1]

    # Genes are independent and numpy's FFT releases the GIL, so the per-gene loop runs on threads -- each gene's
    # arithmetic is exactly the serial loop's (numpy builds its FFT plan per call; nothing is shared), so the
    # result does not depend on the thread count. Batching instead (one fft2 over an (H, W, G) stack) is also
    # bit-identical but no faster: numpy's FFT is single-threaded, and scipy's threaded FFT differs from numpy's
    # in the last bits (~5e-12 on a real section), so it cannot stand in. Threads follow the FFT worker setting
    # (as _sort_columns does), capped so the per-thread (H, W) temporaries stay within ~512 MB.
    nt = _texture_threads(G, H * W)
    if nt <= 1:
        for j in range(G):
            _gene(j)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=nt) as ex:
            list(ex.map(_gene, range(G)))
    return out


def _texture_threads(G: int, n_bins: int) -> int:
    """Threads for texture_multiscale's per-gene loop: the FFT worker setting, at most one per gene, and few
    enough that every thread's temporaries (~48 bytes per bin: the field, its spectrum, |F|^2) fit ~512 MB."""
    w = _fft.get_workers()
    nt = (os.cpu_count() or 1) if (w is None or w < 1) else int(w)
    return max(1, min(nt, G, (512 << 20) // max(1, 48 * int(n_bins))))


def general_g_multiscale(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc") -> np.ndarray:
    """Getis-Ord GENERAL G per gene at MANY radii, sharing one forward FFT. Returns (n_radii, G).

    .. math:: G(d) = \\frac{\\sum_i \\sum_{j \\neq i} w_{ij}(d)\\, x_i x_j}{\\sum_i \\sum_{j \\neq i} x_i x_j}

    Getis & Ord 1992. This is the third canonical ESDA question and it is NOT autocorrelation: Moran
    asks whether neighbours deviate from the mean together, Geary how roughly values change between
    neighbours, and General G whether HIGH VALUES cluster — a field of cold spots and a field of hot
    spots can share a Moran's I and differ here. G > its expectation means high values sit together.

    The field is deliberately NOT demeaned. G is defined on the values themselves and is only
    interpretable for a non-negative variable (expression, after normalisation, qualifies);
    subtracting the mean would make the products change sign and destroy the reading. This is the one
    scalogram in the module that does not centre its input, which is why it is a separate function
    rather than a flag on :func:`morans_i_multiscale`.

    Cost is one shared forward FFT of the field plus, per radius, a kernel FFT, a product and an
    inverse — the same amortisation as :func:`morans_i_multiscale`.
    """
    # The shared implementation (_ms_cpu), bit-identical to the per-function code it replaced: the masked field,
    # NOT centred; the denominator sums over ORDERED pairs i != j; the kernel is centre-excluded, so j != i.
    return _ms_cpu(grid, mask, radii, kind, ("general_g",))["general_g"]


#: LOSH is undefined (NaN) where its mean residual h1 is at rounding level: the field has no residual variance on the
#: bins it is measured on -- an absent or constant field, or one constant within each tissue fragment at radii below
#: the gaps between fragments -- so H = 0/0. The local mean comes from a convolution, so those residuals are rounding,
#: not exactly 0 (measured <= 2e-15 of the field's magnitude on 2000 x 2000 grids up to radius 80), and before 0.13.2
#: they read as noise (a Gini near 0.5, 0.50 on arm64 against 0.47 on x86) or, for an absent field, as 0. Residuals
#: below LOSH_RTOL x max|x| count as rounding, at least 1000x above it; a field varying by more is tested as before.
LOSH_RTOL = 1e-11


def losh_multiscale(grid: np.ndarray, mask: np.ndarray, radii, kind: str = "disc",
                    a: float = 2.0, reduce: str = "gini") -> np.ndarray:
    """LOSH concentration per gene at MANY radii. Returns (n_radii, G).

    The fourth canonical ESDA question — variance heterogeneity (:func:`losh`) — reduced to one number
    per radius so it can sit beside the Moran, Geary and General G scalograms on a shared grid.

    WHICH number is a judgement call, and it is declared rather than hidden. ``H_i`` is normalised so
    that its tissue mean is ~1 by construction, which makes the mean useless as a summary: what
    carries information is how UNEVENLY the local variance is distributed. ``reduce="gini"`` (default)
    takes the Gini coefficient of the H map, exactly as :func:`focality_scalogram` does for the local
    Moran map. That is a defensible COMPOSITION of two standard tools, not a canonical named
    statistic, and must be described that way — unlike Moran's I, Geary's C and General G, LOSH has no
    established global form. ``reduce="sd"`` gives the standard deviation of H instead.

    Unlike the other scalograms this one cannot share a single forward FFT across radii: the residual
    ``e = |x - mean_r(x)|^a`` is itself a function of the radius, so its transform must be rebuilt at
    each scale. The forward transform of the field and of the mask are still shared, so the cost is
    about twice :func:`morans_i_multiscale`, not n_radii times the single-radius primitive.

    A radius at which the field has no residual variance (an absent or constant field, or one constant within
    each fragment below the gaps between them) reads NaN, as Moran does on a constant field: H is 0/0 there
    (:data:`LOSH_RTOL`; changed in 0.13.2 from rounding noise, or 0 for an absent field).
    """
    if reduce not in ("gini", "sd"):
        raise ValueError(reduce)
    # The shared implementation (_ms_cpu), bit-identical to the per-function code it replaced: the residual
    # e = |x - mean_r(x)|^a is taken against the local mean AT j and its transform rebuilt per radius; the H map
    # is formed on occupied bins only and reduced by `reduce`.
    return _ms_cpu(grid, mask, radii, kind, ("losh",), a=a, reduce=reduce)["losh"]


# ---------------------------------------------------------------------------
# One implementation behind the five disc scalograms, and the fused entry point (2026-10-12)
# ---------------------------------------------------------------------------
#: Statistics :func:`multiscale_statistics` computes, in the curve bank's order.
MULTISCALE_STATS = ("moran", "geary", "general_g", "losh", "focality", "texture")
#: Statistics that read the ``centred`` input when one is given (diff feeds them its in-mask z-stack).
_CENTRED_STATS = ("moran", "geary", "focality")
_SORT_MIN_COLS = 8          # fewer columns than this: one in-place sort beats a thread pool
#: A CPU :func:`multiscale_statistics` call whose single pass would hold more than this many bytes of field
#: spectra and field-sized stacks (:func:`_ms_pass_bytes`) runs its sharing groups one after another instead
#: (see its docstring). 2 GiB sits between the two measured sections: 0.4 M bins x 32 genes needs 0.94 GiB
#: and runs fastest as one pass; 2.05 M bins x 32 genes needs 3.9 GiB and runs best in groups.
_FUSE_MAX_BYTES = 2 * 2 ** 30


def _ms_passes(stats, centred_given: bool) -> list:
    """The sharing groups of ``stats``, one tuple per pass: the statistics that read the same field transform."""
    want = set(stats)
    groups = (("moran", "focality"),
              ("general_g", "losh") + (() if centred_given else ("geary",)),
              ("geary",) if centred_given else (),
              ("texture",))
    return [p for p in (tuple(s for s in g if s in want) for g in groups) if p]


def _ms_pass_bytes(shape, radii, stats, centred_given: bool) -> int:
    """Bytes of field spectra and field-sized stacks ONE :func:`_ms_cpu` pass over ``stats`` keeps alive for its
    whole radius loop: a padded half-spectrum per shared transform (z; x; Geary's x unless it reads x's; x^2)
    and a grid-sized stack per field read (z, x, Geary's x). A size estimate for scheduling, nothing more."""
    from scipy.fft import next_fast_len  # lazy
    H, W, G = shape
    rmax = max(radii)
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    want = set(stats)
    raw = bool(want & {"general_g", "losh"})
    geary = "geary" in want
    shared = geary and raw and not centred_given
    n_spec = bool(want & {"moran", "focality"}) + raw + geary * (2 - shared)
    n_grid = bool(want & set(_CENTRED_STATS)) + raw + (geary and not shared)
    return int(n_spec * fh * (fw // 2 + 1) * G * 16 + n_grid * H * W * G * 8)


class _MsWork:
    """Scratch that ONE :func:`_ms_cpu` call reuses across its radii and statistics.

    * :meth:`zeros` -- one zero grid for :func:`_scatter_sum`'s scatter path. Every use writes ALL the occupied
      bins and nothing else, so the unoccupied bins stay exactly zero and the reused grid holds, bit for bit,
      what a fresh ``np.zeros`` plus the scatter would.
    * :meth:`buf` -- named arrays the Gini reduction refills at every radius (its positive-part map and mask).
    * :meth:`pool` -- one thread pool for the Gini's column sorts, instead of a new pool per call.

    :meth:`close` drops the arrays and shuts the pool down."""

    def __init__(self, nocc: int):
        self.nocc = int(nocc)
        self._zeros = None
        self._bufs = {}
        self._pool, self._pool_n = None, 0

    def zeros(self, shape) -> np.ndarray:
        if self._zeros is None or self._zeros.shape != tuple(shape):
            self._zeros = np.zeros(shape)
        return self._zeros

    def buf(self, name: str, shape, dtype=np.float64) -> np.ndarray:
        b = self._bufs.get(name)
        if b is None or b.shape != tuple(shape) or b.dtype != dtype:
            b = self._bufs[name] = np.empty(shape, dtype)
        return b

    def pool(self, n: int):
        if self._pool is None or self._pool_n != n:
            from concurrent.futures import ThreadPoolExecutor  # lazy
            if self._pool is not None:
                self._pool.shutdown()
            self._pool, self._pool_n = ThreadPoolExecutor(max_workers=n), n
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool, self._pool_n = None, 0
        self._zeros = None
        self._bufs.clear()


def _sort_threads(G: int) -> int:
    """Threads for the Gini's column sorts: the FFT worker setting (:func:`spectrast._fft.set_workers`, so a
    process pool that divides FFT threads divides these too), at most one per TWO columns (see
    :func:`_sort_sum_columns` for why a block never holds a single column)."""
    import os
    w = _fft.get_workers()
    nt = (os.cpu_count() or 1) if (w is None or w < 1) else int(w)
    return min(nt, G // 2)


def _sort_sum_columns(pos: np.ndarray, wt: np.ndarray, work: _MsWork | None = None):
    """Sort every column of ``pos`` IN PLACE; return ``(pos.sum(0), (wt * pos).sum(0))`` of the sorted map.

    Bit-identical to ``pos.sort(axis=0)`` followed by those two sums: a column's sort and its sums never read
    another column. With enough columns the work runs on threads in column blocks (sorting, ``multiply`` and
    ``add.reduce`` release the GIL), and each thread forms its block's two sums while the block is in cache.
    A block always keeps at least TWO columns: numpy sums a multi-column array one row at a time -- the order
    :func:`_scatter_sum` documents, and the order the whole map is summed in -- but a single column pairwise,
    so a one-column block would move the last bits."""
    G = pos.shape[1]
    nt = _sort_threads(G)
    if nt <= 1 or G < _SORT_MIN_COLS:
        pos.sort(axis=0)
        return pos.sum(0), (wt * pos).sum(0)
    edges = [k * G // nt for k in range(nt + 1)]            # every width >= G // nt >= 2
    s, ws = np.empty(G), np.empty(G)

    def _block(j0, j1):
        b = pos[:, j0:j1]
        b.sort(axis=0)
        s[j0:j1] = b.sum(0)
        ws[j0:j1] = (wt * b).sum(0)

    if work is None:
        from concurrent.futures import ThreadPoolExecutor  # lazy
        with ThreadPoolExecutor(max_workers=nt) as ex:
            list(ex.map(_block, edges[:-1], edges[1:]))
    else:
        list(work.pool(nt).map(_block, edges[:-1], edges[1:]))
    return s, ws


def _gini_occupied(v: np.ndarray, wt: np.ndarray, nocc: int, work: _MsWork | None = None) -> np.ndarray:
    """Gini of the positive part of an (n_occupied, G) map -- the reduction focality and LOSH share.

    The same arithmetic as their original inline code: ``pos = clip(where(isfinite(v), v, 0), 0, None)``, every
    column sorted, ``(wt * pos).sum(0) / (nocc * pos.sum(0))``. Two things changed, neither moves a bit. The
    positive part is built in ONE reused buffer instead of two fresh temporaries: ``maximum(v, 0)`` -- exactly
    what that ``clip`` calls -- and then +0.0 wherever the result is not finite (a finite value goes through
    the identical ``maximum``; NaN and +-inf end as +0.0 either way). The sort and both column sums run on
    threads (:func:`_sort_sum_columns`). ``v`` itself is not modified."""
    if v.dtype != np.float64 or v.ndim != 2:                 # the original temporaries, for any other input
        pos = np.clip(np.where(np.isfinite(v), v, 0.0), 0.0, None)
        s, ws = _sort_sum_columns(pos, wt, work)
        return np.nan_to_num(ws / np.where(s > 0, nocc * s, np.nan), nan=0.0)
    if work is None:
        pos, bad = np.empty(v.shape), np.empty(v.shape, bool)
    else:
        pos, bad = work.buf("gini_pos", v.shape), work.buf("gini_bad", v.shape, np.bool_)
    np.maximum(v, 0.0, out=pos)                             # the clip, on every value
    np.isfinite(pos, out=bad)
    np.logical_not(bad, out=bad)
    if bad.any():
        np.copyto(pos, 0.0, where=bad)                      # NaN, +inf -> +0.0 (-inf is already +0.0)
    s, ws = _sort_sum_columns(pos, wt, work)
    return np.nan_to_num(ws / np.where(s > 0, nocc * s, np.nan), nan=0.0)


def _scatter_sum(term_occ: np.ndarray, mb: np.ndarray, shape, work: _MsWork | None = None) -> np.ndarray:
    """Sum over the whole grid of a quantity computed only on occupied bins, in the ORIGINAL order.

    The full-grid code multiplied by a masked field, so every unoccupied bin held an exact zero, and summed
    the C-ordered (H, W, G) array over axes (0, 1). Two routes reproduce that sum to the bit.

    **G >= 2: the occupied rows alone.** numpy's ``add.reduce`` (``PyUFunc_ReduceWrapper``,
    ``numpy/_core/src/umath/reduction.c``) iterates in memory order (``NPY_KEEPORDER``) with the gene axis
    innermost, because it has the smallest stride; the reduced axes have output stride 0, so they cannot be
    coalesced with it (``npyiter_coalesce_axes``) and the inner loop is that axis alone
    (``npyiter_find_buffering_setup`` makes it the "core" of a reduce-mode iteration, or a one-row buffer).
    Every inner-loop call therefore adds ONE row of G values elementwise into the G running sums --
    ``DOUBLE_add`` (``loops_arithm_fp.dispatch.c.src``) takes its pairwise-summation branch only when the
    output stride is 0, i.e. when the inner axis is itself reduced -- so each column is summed strictly row
    after row, ``((0 + t[0]) + t[1]) + ...``. The full grid is summed by the same rule, its occupied rows in
    the same (C) order as ``full[mb]`` gathers them, with exact +0.0 rows in between; adding +0.0 never
    changes a running sum, which starts at +0.0 and so is never -0.0. Checked bit for bit (uint64 views)
    over G = 2..64, grids 1x1 to 1378x1488, random and structured masks, and normal, heavy-tailed,
    cancelling, signed-zero, NaN and inf contents, on numpy 2.4.4 and 2.4.6;
    ``tests/test_multiscale_reductions.py`` keeps checking it on whatever numpy is installed.

    **G == 1: the scatter.** The gene axis has length 1 and is dropped, the inner loop becomes the reduced
    axis and numpy sums it PAIRWISE, in an order that depends on where the zeros sit -- so the occupied values
    go back into a zero grid of the original shape and that grid is summed. ``work`` lends one zero grid for
    the whole call (only occupied bins are ever written, all of them every time). Anything else that does
    not fit the row route (another dtype, a non-contiguous or odd-shaped term) takes the scatter as well."""
    G = shape[-1]
    nocc = work.nocc if work is not None else int(np.count_nonzero(mb))
    if (G >= 2 and type(term_occ) is np.ndarray and term_occ.dtype == np.float64
            and term_occ.shape == (nocc, G) and term_occ.flags.c_contiguous):
        return term_occ.sum(0)
    full = np.zeros(shape) if work is None else work.zeros(shape)
    full[mb] = term_occ
    return full.sum((0, 1))


# ---------------------------------------------------------------------------
# Transform plumbing behind _ms_cpu (2026-10-12): the padded 2-D transforms run as their two 1-D passes so that
# rows nobody reads are never transformed, large arrays are allocated once per call, and what depends only on
# the mask is cached across calls
# ---------------------------------------------------------------------------
# scipy runs rfft2(x, s=(fh, fw), axes=(0, 1)) as pocketfft's r2c: a real transform along axis 1 of every row
# of the zero-padded input, then an unscaled complex transform along axis 0. irfft2 is the mirror image: an
# unscaled complex transform along axis 0, a complex-to-real transform along axis 1 of every row, and one
# final multiplication by T(1 / ldbl_t(fh * fw)). Each 1-D line goes through the same pocketfft plan whichever
# lines it is batched with -- the reason the worker count never changes a result -- so running the two passes
# as separate calls reproduces the 2-D transform to the bit; tests/test_multiscale_transforms.py holds every
# helper here to exact equality with the scipy call it replaces, bit patterns and signed zeros included. What
# the split buys: a forward transform skips the fh - h rows the padding adds (an all-zero row always transforms
# to the same fixed pattern, which is copied in), and the inverse transforms only the H rows the 'same' crop
# keeps, in row blocks small enough to stay in cache while the occupied bins are copied out. The row passes
# write into reused block buffers through scipy's pocketfft binding (the public calls take no output array and
# allocate on every call); the binding is used only once verified against the public calls (_pocketfft).
# Without it a forward transform is scipy's plain 2-D call and the inverse the same split through the public
# calls. On a GPU backend, or for non-float64 data, every helper is the plain _fft call it replaces.

#: Bytes of real data per row block of a split transform (rows of fw x G doubles).
_XF_BLOCK_BYTES = 8 << 20
#: Cross-call cache of the mask-only quantities (_MsGeom): at most this many (mask, fh, fw, kind) geometries and
#: this many bytes in total, least recently used evicted first. 0 entries turns the cache off.
_MS_CACHE_MAX_ENTRIES = 4
_MS_CACHE_MAX_BYTES = 256 << 20
#: Fewer complex elements than this and the spectrum product runs on the calling thread (threads cost more).
_PRODUCT_MIN_THREADED = 1 << 18
_PFFT = None                    # scipy's pocketfft binding once verified, False if absent or not bit-identical


def _pocketfft():
    """scipy's own pocketfft binding (``scipy.fft._pocketfft.pypocketfft``) if it is importable AND its 1-D row
    and column passes reproduce the public scipy.fft calls bit for bit on a probe; otherwise False. Checked once
    per process. It is private API, so this is what keeps a future scipy from changing a result silently --
    every use of the binding in this module goes through here."""
    global _PFFT
    if _PFFT is None:
        try:
            from scipy import fft as sfft
            from scipy.fft._pocketfft import pypocketfft as p
            rng = np.random.default_rng(20261012)
            x = rng.standard_normal((7, 9, 3))
            ok = True
            for n in (16, 21, 22):                                       # even, odd, and radix-11 row lengths
                o = np.empty((7, n // 2 + 1, 3), complex)
                xp = np.zeros((7, n, 3)); xp[:, :9] = x
                p.r2c(xp, axes=(1,), forward=True, inorm=0, out=o, nthreads=1)
                ok &= _same_bits(o, sfft.rfft(x, n=n, axis=1))
                y = np.empty((7, n, 3))
                p.c2r(o, axes=(1,), lastsize=n, forward=False, inorm=0, out=y, nthreads=1)
                ok &= _same_bits(y, sfft.irfft(o, n=n, axis=1, norm="forward"))
                for fwd in (True, False):
                    c = o.copy()
                    p.c2c(c, axes=(0,), forward=fwd, inorm=0, out=c, nthreads=1)
                    ref = sfft.fft(o, axis=0) if fwd else sfft.ifft(o, axis=0, norm="forward")
                    ok &= _same_bits(c, ref)
            _PFFT = p if ok else False
        except Exception:                                                # absent or changed: use the public calls
            _PFFT = False
    return _PFFT


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality including the sign of zero (the bit patterns)."""
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _xf_threads() -> int:
    """The thread count scipy.fft resolves from the FFT worker setting (-1 = every core), refusing the same
    settings scipy refuses, so the binding route fails where the public calls would."""
    w, n = int(_fft.get_workers()), os.cpu_count() or 1
    if w < 0:
        if w < -n:
            raise ValueError(f"workers value out of range; got {w}, must not be less than {-n}")
        return w + 1 + n
    if w == 0:
        raise ValueError("workers must not be zero")
    return w


def _xf_block_rows(fw: int, batch: int) -> int:
    """Rows per block of a split transform whose rows hold ``fw * batch`` doubles."""
    return max(1, _XF_BLOCK_BYTES // (8 * int(fw) * max(1, int(batch))))


def _xf_split(a: np.ndarray, dtype) -> bool:
    """The split transforms stand in for scipy's 2-D ones only on the CPU backend and in double precision."""
    return _fft.get_backend() == "cpu" and a.dtype == dtype


@functools.lru_cache(maxsize=64)
def _zero_row_rfft(fw: int) -> np.ndarray:
    """pocketfft's unscaled real transform of one all-zero row of length fw, signed zeros and all (read-only)."""
    z = _pocketfft().r2c(np.zeros((1, int(fw))), axes=(1,), forward=True, inorm=0, nthreads=1)[0]
    z.flags.writeable = False
    return z


def _irfft2_scale(fh: int, fw: int) -> np.float64:
    """The factor pocketfft's 2-D c2r multiplies by under norm='backward': T(1 / ldbl_t(fh * fw))."""
    return np.float64(1 / np.longdouble(int(fh) * int(fw)))


def _rfft2_pad(x: np.ndarray, fh: int, fw: int, pad: np.ndarray | None = None,
               out: np.ndarray | None = None) -> np.ndarray:
    """``rfft2(x, s=(fh, fw), axes=(0, 1))`` bit for bit, without transforming the rows the padding adds.

    ``x`` is (h, w) or (h, w, ...) with h <= fh, w <= fw. Its h rows take the axis-1 transform block by block,
    each block zero-padded to fw columns in ``pad`` -- a (rows, fw, ...) buffer whose columns >= w are zero,
    reusable for inputs of the same width (allocated here if None) -- and written straight into the spectrum
    (``out`` if given, a C-contiguous complex128 (fh, fw//2 + 1, ...) array that is overwritten and returned);
    the remaining fh - h rows get the row transform of an all-zero row, and the axis-0 transform runs in place.
    Needs the verified pocketfft binding; without it this is scipy's plain 2-D transform."""
    p = _pocketfft() if _xf_split(x, np.float64) and x.ndim >= 2 else False
    if not p or x.shape[0] > fh or x.shape[1] > fw:
        return _fft.rfft2(x, s=(fh, fw)) if x.ndim == 2 else _fft.rfft2(x, s=(fh, fw), axes=(0, 1))

    def fill(i0, i1, dst):
        dst[...] = x[i0:i1]

    return _rfft2_rows(fill, x.shape[:2], x.shape[2:], fh, fw, pad, out)


def _rfft2_rows(fill, hw, trail, fh: int, fw: int, pad: np.ndarray | None = None,
                out: np.ndarray | None = None) -> np.ndarray:
    """The split forward transform of :func:`_rfft2_pad` for an (h, w) + ``trail`` float64 field that is supplied
    one row block at a time: ``fill(i0, i1, dst)`` writes rows i0..i1 of the field into ``dst``, the (i1 - i0, w)
    + trail data part of the zero-padded block buffer -- so a field that is an elementwise expression need never
    exist as a whole grid. Needs the verified pocketfft binding (callers check)."""
    p = _pocketfft()
    nt = _xf_threads()
    h, w = hw
    trail = tuple(trail)
    shape = (fh, fw // 2 + 1) + trail
    if out is None or out.shape != shape or out.dtype != np.complex128 or not out.flags.c_contiguous:
        out = np.empty(shape, complex)
    # the padding rows hold exactly what scipy's row pass makes of an all-zero row -- not +0 everywhere: that
    # pass leaves -0.0 in a few places, and a -0 vs +0 there can survive into an output that is exactly zero
    out[h:] = _zero_row_rfft(fw).reshape((-1,) + (1,) * len(trail))
    if pad is None or pad.shape[0] < 1 or pad.shape[1:] != (fw,) + trail:
        rows = max(1, min(h, _xf_block_rows(fw, int(np.prod(trail, dtype=np.int64)))))
        pad = np.zeros((rows, fw) + trail)
    step = pad.shape[0]
    for i0 in range(0, h, step):
        i1 = min(h, i0 + step)
        blk = pad[:i1 - i0]
        fill(i0, i1, blk[:, :w])                           # columns >= w are never written: they stay zero
        p.r2c(blk, axes=(1,), forward=True, inorm=0, out=out[i0:i1], nthreads=nt)
    return p.c2c(out, axes=(0,), forward=True, inorm=0, out=out, nthreads=nt)


class _OccGather:
    """Where the occupied bins of a mask sit in row blocks of a cropped inverse transform (:func:`_irfft2_occ`):
    per block of ``step`` crop rows, its slice of the (n_occupied, G) output and the bins' flat positions in a
    (rows, fw, G) block. Blocks of empty rows are left out."""
    __slots__ = ("nocc", "step", "blocks", "nbytes")

    def __init__(self, mb: np.ndarray, fw: int, step: int):
        ii, jj = np.nonzero(mb)                                  # row-major: the order boolean indexing returns
        self.nocc, self.step, self.blocks = int(ii.size), int(step), []
        rows = np.flatnonzero(mb.any(1))
        if rows.size:
            starts = np.searchsorted(ii, np.arange(mb.shape[0] + 1))
            for i0 in range(int(rows[0]), int(rows[-1]) + 1, self.step):
                i1 = min(int(rows[-1]) + 1, i0 + self.step)
                s0, s1 = int(starts[i0]), int(starts[i1])
                if s1 > s0:                                      # a block of empty rows is never transformed
                    self.blocks.append((i0, i1, s0, s1, (ii[s0:s1] - i0) * int(fw) + jj[s0:s1]))
        self.nbytes = sum(b[4].nbytes for b in self.blocks)


def _irfft2_occ(X: np.ndarray, fh: int, fw: int, off: int, crop, mb: np.ndarray, gather,
                buf: np.ndarray | None = None) -> np.ndarray:
    """``irfft2(X, s=(fh, fw), axes=(0, 1))[crop][mb]`` of an (fh, fw//2 + 1, G) spectrum, bit for bit, as an
    (n_occupied, G) array. Overwrites ``X``. The crop rows take the axis-1 pass in the blocks of ``gather`` (an
    :class:`_OccGather` of ``mb`` at this ``fw``), each into ``buf`` (a (gather.step, fw, G) buffer, allocated
    here if None), from which the occupied bins are copied; the scale multiplies the occupied values only."""
    if gather is None or X.ndim != 3 or not _xf_split(X, np.complex128):
        return _fft.irfft2(X, s=(fh, fw), axes=(0, 1))[crop][mb]
    nt = _xf_threads()
    G = X.shape[2]
    out = np.empty((gather.nocc, G))
    p = _pocketfft()
    if p:
        p.c2c(X, axes=(0,), forward=False, inorm=0, out=X, nthreads=nt)          # unscaled, as inside c2r
        if buf is None or buf.shape[0] < gather.step or buf.shape[1:] != (fw, G):
            buf = np.empty((gather.step, fw, G))
        for i0, i1, s0, s1, flat in gather.blocks:
            y = buf[:i1 - i0]
            p.c2r(X[off + i0:off + i1], axes=(1,), lastsize=fw, forward=False, inorm=0, out=y, nthreads=nt)
            np.take(y.reshape(-1, G)[off:], flat, axis=0, out=out[s0:s1], mode="clip")
    else:
        from scipy import fft as sfft  # lazy
        X = sfft.ifft(X, axis=0, norm="forward", overwrite_x=True, workers=nt)
        for i0, i1, s0, s1, flat in gather.blocks:
            y = sfft.irfft(X[off + i0:off + i1], n=fw, axis=1, norm="forward", workers=nt)
            np.take(y.reshape(-1, G)[off:], flat, axis=0, out=out[s0:s1], mode="clip")
    out *= _irfft2_scale(fh, fw)
    return out


class _MsGeom:
    """What every call on one (mask, padded size, kernel kind) recomputes and nothing else changes: per radius,
    the degree-derived arrays ``(has_nb, inv05, invW)`` on occupied bins exactly as _ms_cpu forms them, and the
    row blocks of the occupied bins. Stored arrays are read-only."""
    __slots__ = ("mb", "nb", "gathers", "nbytes")

    def __init__(self, mb: np.ndarray):
        self.mb = mb
        self.nb = {}
        self.gathers = {}
        self.nbytes = int(mb.nbytes)


_ms_cache = OrderedDict()                 # key -> _MsGeom, least recently used first
_ms_cache_lock = threading.Lock()


def _ms_cache_clear() -> None:
    """Empty the cross-call cache of mask-only quantities."""
    with _ms_cache_lock:
        _ms_cache.clear()


def _ms_cache_after_fork() -> None:
    """A forked child (multiprocessing's default start on Linux) gets a fresh lock -- the parent's could have
    been held by another thread at the fork -- and an empty cache of its own."""
    global _ms_cache_lock
    _ms_cache_lock = threading.Lock()
    _ms_cache.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_ms_cache_after_fork)


def _ms_cache_trim(keep) -> None:
    """Evict least-recently-used geometries (never ``keep``) until both bounds hold. Caller holds the lock."""
    while len(_ms_cache) > 1 and (len(_ms_cache) > _MS_CACHE_MAX_ENTRIES or
                                  sum(g.nbytes for g in _ms_cache.values()) > _MS_CACHE_MAX_BYTES):
        oldest = next(iter(_ms_cache))
        if oldest == keep:
            break
        del _ms_cache[oldest]


def _ms_geom(mb: np.ndarray, fh: int, fw: int, kind) -> "_MsGeom | None":
    """The cached geometry of this mask at this padded size and kernel kind, created empty on a miss. A key is
    the mask's shape and a digest of its bytes plus (fh, fw, kind), and a hit must also match the stored mask
    exactly, so a different mask or geometry never reads another's entries. None when the cache is off or the
    backend is not the CPU (a GPU transform's float32 results must not reach a float64 call)."""
    if _MS_CACHE_MAX_ENTRIES <= 0 or _fft.get_backend() != "cpu":
        return None
    mbc = np.ascontiguousarray(mb, dtype=bool)
    key = (mbc.shape, int(fh), int(fw), str(kind), hashlib.blake2b(mbc.tobytes(), digest_size=16).digest())
    with _ms_cache_lock:
        g = _ms_cache.get(key)
        if g is not None and np.array_equal(g.mb, mbc):
            _ms_cache.move_to_end(key)
            return g
        g = _MsGeom(mbc.copy())
        g.mb.flags.writeable = False
        _ms_cache[key] = g
        _ms_cache_trim(key)
        return g


def _ms_cache_put(g: "_MsGeom", table: dict, key, value, nbytes: int):
    """Store ``value`` in ``table`` (a dict of ``g``) unless that would break the byte bound. Returns the stored
    value (an earlier, identical one if another thread got there first)."""
    with _ms_cache_lock:
        if key in table:
            return table[key]
        if g.nbytes + nbytes <= _MS_CACHE_MAX_BYTES:
            table[key] = value
            g.nbytes += nbytes
            _ms_cache_trim(next((k for k, v in _ms_cache.items() if v is g), None))
    return value


class _NeighbourSums:
    """Per-call transform state behind :func:`_ms_cpu`: the field spectra, the neighbour sums
    ``irfft2(F * Fk[:, :, None], s=(fh, fw), axes=(0, 1))[crop][mb]``, LOSH's residual spectrum and the
    degree-derived weights -- every value the plain expression's, bit for bit. On a GPU backend (general_g and
    losh reach _ms_cpu there) every method evaluates the plain expression unchanged.

    * **Product.** ``F * Fk[:, :, None]`` goes into ONE buffer reused for every field and radius of the call, on
      threads over row blocks: each element is the same numpy complex multiply with the same operands, strides
      and innermost (gene) axis -- only the outermost axis is split. Threads follow the FFT worker setting.
    * **Inverse** (:meth:`inverse`, :func:`_irfft2_occ`). The column pass runs in place on that buffer; the row
      pass runs on only the H rows the 'same' crop keeps, a block of rows at a time into one small reused
      buffer, from which the occupied bins are taken (the elements ``[crop][mb]`` selects, in its C order);
      1/(fh fw) multiplies the kept values.
    * **Forward** (:meth:`forward`, :func:`_rfft2_pad`). Only the data rows take the row pass, a block at a time
      through one small zero-padded buffer; LOSH's residual, nonzero only on occupied bins, is scattered into
      its own zeroed block buffer and transformed straight into the product buffer (:meth:`residual_spectrum`).
    * **Degree** (:meth:`neighbours`). The neighbour count and the weights derived from it depend only on the
      mask and the kernel, so they come from the cross-call cache (:func:`_ms_geom`) when present.

    Large arrays are allocated once per call (the product buffer; the field spectra the caller keeps), never per
    radius. :meth:`close` drops the buffers and the product's thread pool."""

    def __init__(self, mb, fh, fw, G, kind="disc", m=None):
        self.mb, self.fh, self.fw, self.G = mb, int(fh), int(fw), int(G)
        self.H, self.W = mb.shape
        self.m = mb.astype(float) if m is None else m
        self.cpu = _fft.get_backend() == "cpu"
        self.geom = _ms_geom(mb, fh, fw, kind) if self.cpu else None
        self._Fm = None
        self._gathers = {}
        self._pad = self._rpad = self._blk = self._blk1 = None
        self.prod = None                              # (fh, fw // 2 + 1, G) complex, allocated on first use
        self.pool = None
        if self.cpu:
            w = _fft.get_workers()
            nt = min((os.cpu_count() or 1) if (w is None or w < 1) else int(w), self.fh)
            if nt > 1 and self.fh * (self.fw // 2 + 1) * self.G >= _PRODUCT_MIN_THREADED:
                from concurrent.futures import ThreadPoolExecutor
                self.pool = ThreadPoolExecutor(max_workers=nt)
                self.edges = np.linspace(0, self.fh, nt + 1).astype(int)

    def _crop(self, off):
        H, W = self.H, self.W
        if off + H > self.fh or off + W > self.fw:         # a flat index would wrap where the crop would not fit
            raise ValueError(f"crop at offset {off} does not fit the ({self.fh}, {self.fw}) padded grid")
        return slice(off, off + H), slice(off, off + W)

    def gather(self, G):
        """The occupied-bin row blocks for G columns (cached across calls with the geometry)."""
        step = _xf_block_rows(self.fw, G)
        gp = self._gathers.get(step)
        if gp is None:
            g = self.geom
            gp = None if g is None else g.gathers.get(step)
            if gp is None:
                gp = _OccGather(self.mb, self.fw, step)
                if g is not None:
                    gp = _ms_cache_put(g, g.gathers, step, gp, gp.nbytes)
            self._gathers[step] = gp
        return gp

    def _buffer(self, shape, dtype=np.complex128):
        """The call's product buffer, (re)allocated only when its shape or dtype changes."""
        P = self.prod
        if P is None or P.shape != tuple(shape) or P.dtype != dtype:
            P = self.prod = np.empty(shape, dtype)
        return P

    def scratch(self, shape):
        """A C-ordered float64 grid of ``shape`` laid over the product buffer's memory (a view; never None on the
        CPU path, where an (H, W, G) grid always fits in the (fh, fw // 2 + 1, G) complex buffer). For grid-sized
        temporaries that are finished with before the first product overwrites them: one large allocation per
        call instead of two."""
        n = int(np.prod(shape, dtype=np.int64))
        if not self.cpu or 2 * self.fh * (self.fw // 2 + 1) * self.G < n:
            return None
        P = self._buffer((self.fh, self.fw // 2 + 1, self.G))
        return P.reshape(-1).view(np.float64)[:n].reshape(shape)

    def product(self, F, Fk):
        """``F * Fk[:, :, None]`` into the reused buffer (row blocks on threads when the pool exists). ``F`` may
        be the buffer itself (LOSH's residual spectrum): the product is then taken in place."""
        P = self._buffer(F.shape, np.result_type(F, Fk))
        if self.pool is None:
            np.multiply(F, Fk[:, :, None], out=P)
        else:
            e = self.edges
            list(self.pool.map(lambda a, b: np.multiply(F[a:b], Fk[a:b, :, None], out=P[a:b]), e[:-1], e[1:]))
        return P

    def inverse(self, P, off):
        """``irfft2(P, s=(fh, fw), axes=(0, 1))[crop][mb]`` as an (n_occupied, G) array, OVERWRITING ``P``."""
        crop = self._crop(off)
        if not (self.cpu and P.ndim == 3 and P.dtype == np.complex128):
            return _fft.irfft2(P, s=(self.fh, self.fw), axes=(0, 1))[crop][self.mb]
        gp = self.gather(P.shape[2])
        if self._blk is None or self._blk.shape != (gp.step, self.fw, P.shape[2]):
            self._blk = np.empty((gp.step, self.fw, P.shape[2]))
        return _irfft2_occ(P, self.fh, self.fw, off, crop, self.mb, gp, self._blk)

    def __call__(self, F, Fk, off):
        """Neighbour sums at the occupied bins of the fields whose spectra are ``F`` (fh, fw // 2 + 1, G)."""
        if not self.cpu:
            return _fft.irfft2(F * Fk[:, :, None], s=(self.fh, self.fw), axes=(0, 1))[self._crop(off)][self.mb]
        return self.inverse(self.product(F, Fk), off)

    def _pad_for(self, G):
        """The padded row-block buffer of the forward transforms (columns >= W only ever hold zeros)."""
        if self._pad is None or self._pad.shape[2:] != (G,):
            rows = max(1, min(self.H, _xf_block_rows(self.fw, G)))
            self._pad = np.zeros((rows, self.fw, G))
        return self._pad

    def forward(self, x):
        """``rfft2(x, s=(fh, fw), axes=(0, 1))`` of an (H, W, G) field, as a new array (see :func:`_rfft2_pad`)."""
        if not (self.cpu and x.ndim == 3 and x.dtype == np.float64 and _pocketfft()):
            return _fft.rfft2(x, s=(self.fh, self.fw), axes=(0, 1))
        return _rfft2_pad(x, self.fh, self.fw, self._pad_for(x.shape[2]))

    def forward_masked(self, Zc, m, square=False):
        """``rfft2(xg, s=(fh, fw), axes=(0, 1))`` of ``xg = Zc * m[..., None]`` -- or of ``xg ** 2`` -- as a new
        array, without forming xg: each row block is formed in the padded block buffer by the same elementwise
        operations (``np.square`` is what ``** 2`` computes) and transformed there. Geary's two fields."""
        if not (self.cpu and Zc.ndim == 3 and np.result_type(Zc.dtype, m.dtype) == np.float64 and _pocketfft()):
            xg = Zc * m[..., None]
            return self.forward(xg ** 2 if square else xg)

        def fill(i0, i1, dst):
            np.multiply(Zc[i0:i1], m[i0:i1, :, None], out=dst)
            if square:
                np.square(dst, out=dst)

        return _rfft2_rows(fill, (self.H, self.W), Zc.shape[2:], self.fh, self.fw, self._pad_for(Zc.shape[2]))

    def residual_spectrum(self, eo):
        """``rfft2(e, s=(fh, fw), axes=(0, 1))`` of the field ``e`` that is ``eo`` (n_occupied, G) on the occupied
        bins and +0.0 everywhere else -- LOSH's residual -- written into the product buffer and returned.

        A block of e's rows at a time is built in one small buffer that is all +0.0 between uses: ``eo`` goes
        onto the block's occupied bins, the block takes the row pass straight into the spectrum, and those bins
        are reset to +0.0. Rows without an occupied bin, and the rows the padding adds, get the row transform of
        a zero row, which is what that pass makes of them; the column pass then runs in place."""
        fh, fw = self.fh, self.fw
        p = _pocketfft() if (self.cpu and eo.dtype == np.float64 and eo.ndim == 2) else False
        if not p:
            e = np.zeros((self.H, self.W, eo.shape[1])); e[self.mb] = eo
            return _fft.rfft2(e, s=(fh, fw), axes=(0, 1))
        G = eo.shape[1]
        gp = self.gather(G)
        P = self._buffer((fh, fw // 2 + 1, G))
        if self._rpad is None or self._rpad.shape != (gp.step, fw, G):
            self._rpad = np.zeros((gp.step, fw, G))
        rows = self._rpad.reshape(-1, G)
        zrow = _zero_row_rfft(fw)[:, None]
        nt = _xf_threads()
        done = 0
        for i0, i1, s0, s1, flat in gp.blocks:
            if i0 > done:
                P[done:i0] = zrow                              # rows holding no occupied bin
            rows[flat] = eo[s0:s1]
            p.r2c(self._rpad[:i1 - i0], axes=(1,), forward=True, inorm=0, out=P[i0:i1], nthreads=nt)
            rows[flat] = 0.0
            done = i1
        P[done:] = zrow                                        # the rest of the grid and the padding rows
        return p.c2c(P, axes=(0,), forward=True, inorm=0, out=P, nthreads=nt)

    def neighbours(self, r, Fk, off):
        """``(has_nb, inv05, invW)`` at radius ``r`` on occupied bins: the degree W1 = irfft2(Fm * Fk)[crop][mb]
        compared with 0.5 (0.5 beats fft roundoff), and the row-standardising weights at the 0.5 (focality,
        Geary, LOSH) and 1e-12 (Moran) gates. From the cross-call cache when present; read-only there."""
        g = self.geom
        hit = None if g is None else g.nb.get(r)
        if hit is not None:
            return hit
        if self._Fm is None:
            self._Fm = _rfft2_pad(self.m, self.fh, self.fw)
        X = self._Fm * Fk
        crop = self._crop(off)
        if self.cpu and X.dtype == np.complex128:
            if self._blk1 is None:
                self._blk1 = np.empty((self.gather(1).step, self.fw, 1))
            W1 = _irfft2_occ(X[:, :, None], self.fh, self.fw, off, crop, self.mb, self.gather(1), self._blk1)[:, 0]
        else:
            W1 = _fft.irfft2(X, s=(self.fh, self.fw))[crop][self.mb]
        has_nb = W1 > 0.5
        inv05 = np.where(has_nb, 1.0 / np.maximum(W1, 1e-12), 0.0)[:, None]
        invW = np.where(W1 > 1e-12, 1.0 / np.maximum(W1, 1e-12), 0.0)[:, None]
        val = (has_nb, inv05, invW)
        if g is None:
            return val
        for v in val:
            v.flags.writeable = False
        return _ms_cache_put(g, g.nb, r, val, has_nb.nbytes + inv05.nbytes + invW.nbytes)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)
            self.pool = None
        self.prod = self._pad = self._rpad = self._blk = self._blk1 = self._Fm = None


def _c_grid(a: np.ndarray, shape) -> bool:
    """True if ``a`` is a C-ordered ``shape`` grid whose products with float64 are float64 -- the case in which a
    reused C-ordered float64 scratch holds exactly the temporaries numpy would allocate for it."""
    return (a.shape == tuple(shape) and a.flags.c_contiguous
            and np.result_type(a.dtype, np.float64) == np.float64)


def _ms_cpu(grid, mask, radii, kind="disc", stats=MULTISCALE_STATS, centred=None, a=2.0, reduce="gini"):
    """The CPU implementation of every disc scalogram, for any subset of statistics, sharing transforms.

    Each single-statistic function on the CPU path is this function with one statistic requested, so a fused
    call and five separate calls run the same arithmetic on the same arrays. Per element the operations and
    their order are those of the original per-function code (tests/data/multiscale_reference.npz holds that
    code's outputs and the current code must reproduce them exactly); what changed is WHERE they run:
    neighbour sums are cut to the occupied bins straight off the inverse transform, so the per-radius
    elementwise work scales with the tissue, not the bounding box (9 % of it on a Pipelle biopsy section),
    full-grid sums go through :func:`_scatter_sum` to keep their summation order, and the transforms run
    through :class:`_NeighbourSums`."""
    from scipy.fft import next_fast_len  # lazy
    if grid.ndim == 2:
        grid = grid[..., None]
    Zc = grid if centred is None else (centred[..., None] if centred.ndim == 2 else centred)
    want = set(stats)
    radii = list(radii); rmax = max(radii)
    mb = np.asarray(mask, bool); m = mb.astype(float); msum = m.sum()
    H, W, G = grid.shape
    shape = (H, W, G)
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    nocc = int(mb.sum())
    work = _MsWork(nocc)                    # one zero grid + one sort pool for the whole call
    out = {s: np.empty((len(radii), G)) for s in stats if s != "texture"}
    nb = _NeighbourSums(mb, fh, fw, G, kind, m)   # transforms, neighbour sums and degree weights for the call

    disc = want & {"moran", "geary", "general_g", "losh", "focality"}
    need_nb = bool(want & {"moran", "geary", "losh", "focality"})
    if want & {"focality", "losh"}:
        idx = np.arange(1, nocc + 1); wt = (2 * idx - nocc - 1)[:, None]        # Gini weights
    # Grid-sized temporaries of the setup share ONE scratch grid per call whenever the grid they come from is
    # C-ordered with float64 products: numpy would give each temporary that same C layout, so every full-grid sum
    # below reduces the same values in the same order. Otherwise the original expressions run. The scratch is laid
    # over the product buffer (nb.scratch), which the radius loop only starts writing after the setup is done --
    # one large allocation per call for both (on the largest section a fresh (H, W, G) grid is 525 MB of
    # first-touch page faults).
    def grid_scratch():
        g_ = nb.scratch(shape)
        return work.buf("grid", shape) if g_ is None else g_

    # centred statistics: z as morans_i_multiscale / gearys_c_multiscale / focality_scalogram built it
    if want & set(_CENTRED_STATS):
        if _c_grid(Zc, shape):
            s = grid_scratch()
            np.multiply(Zc, m[..., None], out=s)
            zbar = s.sum((0, 1)) / msum
            np.subtract(Zc, zbar, out=s)
            np.multiply(s, m[..., None], out=s)                         # z = (Zc - zbar) * m, in the scratch
            z_occ = s[mb]
            if want & {"moran", "focality"}:
                Fz = nb.forward(s)
            np.multiply(s, s, out=s)
            zz = s.sum((0, 1))                                           # (z * z).sum((0, 1))
        else:
            zbar = (Zc * m[..., None]).sum((0, 1)) / msum
            z = (Zc - zbar) * m[..., None]
            z_occ = z[mb]
            if want & {"moran", "focality"}:
                Fz = nb.forward(z)
            zz = (z * z).sum((0, 1))
            del z
    if "moran" in want:
        den_I = zz
    if "focality" in want:
        m2 = zz / msum                            # (z ** 2).sum((0, 1)) / msum: z ** 2 is np.square(z), i.e. z * z
        m2s = np.where(m2 == 0, np.nan, m2)
    # uncentred statistics: the masked raw field, NOT demeaned (general_g_multiscale / losh_multiscale)
    if want & {"general_g", "losh"}:
        if grid.dtype == np.float64 and _c_grid(grid, shape):         # np.asarray(grid, float) is grid itself
            x = grid_scratch()
            np.multiply(grid, m[..., None], out=x)                       # x = grid * m, in the scratch
        else:
            x = np.asarray(grid, float) * m[..., None]
        if "general_g" in want and (x < -1e-9).any():
            warnings.warn("general_g_multiscale received negative values; General G is defined for a "
                          "non-negative variable and the result will not be interpretable.",
                          RuntimeWarning, stacklevel=3)
        Fx = nb.forward(x)
        x_occ = x[mb]
        if "losh" in want:                     # h1 at or below this is rounding of the local mean: 0/0 (LOSH_RTOL)
            losh_floor = (LOSH_RTOL * np.abs(x_occ).max(0)) ** a if nocc else np.zeros(G)
        if "general_g" in want:
            xsum = x.sum((0, 1))
            if x.flags.c_contiguous:                                     # (x ** 2) is x * x: square in the scratch
                s = grid_scratch()
                np.multiply(x, x, out=s)
                den_G = xsum ** 2 - s.sum((0, 1))
            else:
                den_G = xsum ** 2 - (x ** 2).sum((0, 1))
        del x
    if "geary" in want:
        share_x = centred is None and want & {"general_g", "losh"}       # same field -> same transform
        # xg = Zc * m[..., None] and xg ** 2 are never formed: each row block is formed where it is transformed
        Fxg = Fx if share_x else nb.forward_masked(Zc, m)
        Fxg2 = nb.forward_masked(Zc, m, square=True)
        den_C = zz
        xg_occ = x_occ if share_x else Zc[mb] * m[mb][:, None]            # = (Zc * m[..., None])[mb]
        xg_sq, xg_2 = xg_occ ** 2, 2.0 * xg_occ                          # radius-invariant terms of the numerator

    for si, r in enumerate(radii) if disc else ():
        k = neighbor_kernel(r, kind); off = (k.shape[0] - 1) // 2
        Fk = _rfft2_pad(k, fh, fw)
        if need_nb:
            # degree and what derives from it, on occupied bins (nothing else ever read them)
            has_nb, inv05, invW = nb.neighbours(r, Fk, off)
        if want & {"moran", "focality"}:
            Wz_occ = nb(Fz, Fk, off)
        if want & {"general_g", "losh"}:
            Wx_occ = nb(Fx, Fk, off)
        if "moran" in want:
            meanN = Wz_occ * invW                                            # row-standardised neighbour mean
            num = _scatter_sum(z_occ * meanN, mb, shape, work)
            out["moran"][si] = num / np.where(den_I == 0, np.nan, den_I)
        if "focality" in want:
            v = z_occ * (Wz_occ * inv05) / m2s                               # local Moran on occupied bins
            out["focality"][si] = _gini_occupied(v, wt, nocc, work)
        if "geary" in want:
            Wx_g = Wx_occ if share_x else nb(Fxg, Fk, off)
            Wx2_g = nb(Fxg2, Fk, off)
            meanN_x = Wx_g * inv05; meanN_x2 = Wx2_g * inv05
            gate = has_nb[:, None]
            num = _scatter_sum((xg_sq - xg_2 * meanN_x + meanN_x2) * gate, mb, shape, work)   # x^2 - 2x mean + mean2
            S0 = float(np.count_nonzero(has_nb))
            if S0 == 0:                     # no occupied bin has a neighbour at this radius: undefined, said
                out["geary"][si] = np.nan   # directly rather than reached through 1/0 and inf * 0
            else:
                out["geary"][si] = ((msum - 1) / (2.0 * S0)) * (num / np.where(den_C == 0, np.nan, den_C))
        if "general_g" in want:
            num = _scatter_sum(x_occ * Wx_occ, mb, shape, work)
            out["general_g"][si] = num / np.where(den_G == 0, np.nan, den_G)
        if "losh" in want:
            vo = has_nb[:, None]                                             # valid: occupied with a neighbour
            xbar = Wx_occ * inv05
            eo = np.abs(x_occ - xbar) ** a * vo                              # residual vs the local mean AT j
            Fe = nb.residual_spectrum(eo)                                    # rebuilt per radius: e depends on r
            Hm = nb(Fe, Fk, off) * inv05
            h1 = _scatter_sum(eo, mb, shape, work) / max(float(np.count_nonzero(has_nb)), 1.0)   # = e.sum((0,1))/..
            Hm = Hm / np.where(h1 == 0, np.nan, h1)
            v = np.where(vo, Hm, np.nan)
            if reduce == "sd":
                out["losh"][si] = np.nanstd(v, axis=0)
            elif reduce == "gini":
                out["losh"][si] = _gini_occupied(v, wt, nocc, work)
            else:
                raise ValueError(reduce)
            out["losh"][si][h1 <= losh_floor] = np.nan                     # no residual variance here: 0/0
    nb.close()
    work.close()
    if "texture" in want:
        out["texture"] = texture_multiscale(grid, mask, radii, kind)
    return {s: out[s] for s in stats}


def multiscale_statistics(grid: np.ndarray, mask: np.ndarray, radii, stats=MULTISCALE_STATS,
                          kind: str = "disc", centred: np.ndarray | None = None, a: float = 2.0,
                          reduce: str = "gini") -> dict:
    """Several multiscale statistics of the same fields in one pass. Returns ``{stat: (n_radii, G)}``.

    Every array is BIT-IDENTICAL to the single-statistic function on the same input --
    :func:`morans_i_multiscale`, :func:`gearys_c_multiscale`, :func:`general_g_multiscale`,
    :func:`losh_multiscale` (``a``, ``reduce``), :func:`focality_scalogram`, :func:`texture_multiscale` --
    because on the CPU path those functions ARE this code with one statistic requested; sharing a
    transform cannot change a value. What is shared:

    * per radius, the kernel transform and the neighbour-count map, across every disc statistic;
    * the centred field's transform and its neighbour sum, between Moran and focality;
    * the raw field's transform and its neighbour sum, between General G and LOSH (and Geary when no
      ``centred`` input is given, since it then reads the same field).

    **Speed (CPU), measured 2026-10-12** against the pre-refactor per-statistic code (e4753f8) on real
    Pipelle biopsy sections, 32 genes x the five disc statistics x 18 radii, all cores of an idle M3, every
    array identical to the bit: 0.4 M bins (9 % occupied) 12.4 s -> 2.2 s in one call (3.0 s as five
    separate calls); 2.05 M bins (5 % occupied) 59.4 s -> 9.5 s (12.7 s separate). Where it comes from:
    per-radius arithmetic on occupied bins only; transforms that skip the zero padding rows and keep only the
    rows the crop reads; reused, threaded buffers instead of fresh padded arrays (on macOS each allocation
    above ~400 MB re-faults all its pages: 12.7 M page faults per run before, 0.39 M after); full-grid sums
    taken over occupied rows where numpy's reduction order provably matches (:func:`_scatter_sum`).

    **Schedule (CPU).** The transforms worth sharing are the padded field spectra, and only the statistics
    within one group read the same one: {moran, focality} the centred field's, {general_g, losh} the raw
    field's (with geary when no ``centred`` is given), and geary on its own when it is. Across groups a single
    pass shares only the per-radius kernel transform and neighbour count, yet keeps every group's spectra and
    fields alive through the whole radius loop. So a call whose single pass would hold more than
    :data:`_FUSE_MAX_BYTES` of them runs its groups one after another, each freeing its stacks before the
    next starts: the same transforms and arithmetic in less memory. With the final buffers the two schedules
    take the same time (2.05 M bins: 9.5 s either way); grouping is kept above the threshold for its smaller
    peak (on a memory-loaded machine an earlier single pass ran 1.8x slower than the groups). Either way
    every array is the single-statistic function's, bit for bit: each pass is this function's own pass over
    a subset, and no statistic's value depends on which others share its pass.

    Parameters
    ----------
    grid : (H, W, G) or (H, W)
        The field every statistic reads -- and the only one, unless ``centred`` is given.
    centred : (H, W, G) or (H, W), optional
        A separate field for the CENTRED statistics (moran, geary, focality): e.g. the in-mask z-stack
        :mod:`spectrast.diff` feeds them while General G and LOSH read the raw binned field.
    stats : iterable of str
        Any of :data:`MULTISCALE_STATS`.

    On a non-CPU backend each statistic is computed by its own function (the GPU paths are separate code),
    so the result still equals the single-statistic calls, without the sharing.
    """
    stats = tuple(stats)
    unknown = sorted(set(stats) - set(MULTISCALE_STATS))
    if unknown:
        raise ValueError(f"unknown statistic(s) {unknown}; choose from {MULTISCALE_STATS}")
    if _fft.get_backend() != "cpu":
        fns = {"moran": morans_i_multiscale, "geary": gearys_c_multiscale, "general_g": general_g_multiscale,
               "focality": focality_scalogram, "texture": texture_multiscale}
        out = {}
        for s in stats:
            g = centred if (centred is not None and s in _CENTRED_STATS) else grid
            out[s] = (losh_multiscale(g, mask, radii, kind, a=a, reduce=reduce) if s == "losh"
                      else fns[s](g, mask, radii, kind))
        return out
    radii = list(radii)
    passes = _ms_passes(stats, centred is not None)
    if len(passes) > 1:
        try:
            shape = grid.shape if grid.ndim == 3 else grid.shape + (1,)
            big = _ms_pass_bytes(shape, radii, stats, centred is not None) > _FUSE_MAX_BYTES
        except Exception:                        # malformed input: let the single pass raise its own error
            big = False
        if big:
            out = {}
            for p in passes:                     # each pass frees its spectra before the next one starts
                out.update(_ms_cpu(grid, mask, radii, kind, p, centred=centred, a=a, reduce=reduce))
            return {s: out[s] for s in stats}
    return _ms_cpu(grid, mask, radii, kind, stats, centred=centred, a=a, reduce=reduce)


def local_morans_i(field: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
                   radius: int = 1) -> np.ndarray:
    """Local Moran's I (LISA) map for a single field (H, W) -> (H, W) hotspot scores."""
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    m = mask.astype(float)
    zbar = field[mask].mean()
    z = (field - zbar) * m
    W1 = fftconvolve(m, kernel, mode="same")
    inv = np.where(W1 > 0.5, 1.0 / np.maximum(W1, 1e-12), 0.0)  # 0.5 beats fft roundoff at neighbourless cells
    meanN = _fftconv_same(z[..., None], kernel)[..., 0] * inv   # ROW-STANDARDIZED ⇒ mean(LISA)=global I
    m2 = (z[mask] ** 2).mean()
    lisa = (z / m2) * meanN
    lisa[~mask] = np.nan
    return lisa


def local_morans_i_batch(grid: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
                         radius: int = 1) -> np.ndarray:
    """Batched local Moran's I (LISA) for a (H, W, G) grid -> (H, W, G). One FFT over all genes."""
    if grid.ndim == 2:
        grid = grid[..., None]
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    m = mask.astype(float); msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum
    z = (grid - zbar) * m[..., None]
    W1 = fftconvolve(m, kernel, mode="same")
    inv = np.where(W1 > 0.5, 1.0 / np.maximum(W1, 1e-12), 0.0)[..., None]  # 0.5 beats fft roundoff (cf gearys_c)
    meanN = _fftconv_same(z, kernel) * inv             # ROW-STANDARDIZED ⇒ mean(LISA) = global Moran
    m2 = (z ** 2).sum((0, 1)) / msum
    lisa = z * meanN / np.where(m2 == 0, np.nan, m2)
    lisa[~np.asarray(mask, bool)] = np.nan
    return lisa


def focality(local_map: np.ndarray, mask: np.ndarray) -> dict:
    """Global summaries of a LOCAL statistic map that capture spatial HETEROGENEITY (focal vs diffuse).

    Two fields with the same MEAN (the conventional global statistic, e.g. global Moran's I = mean of
    local) can differ entirely in whether the signal is spread uniformly across the tissue or
    concentrated in a few hotspots. Since the local map is already computed (one FFT), these come free.
    Returns dict:
      mean      tissue-mean of the local statistic (≈ the conventional global statistic)
      frac_pos  fraction of spots with a positive local value
      gini      Gini coefficient of the positive local values (0 = diffuse/uniform, 1 = focal)
      area50    fraction of tissue holding 50% of the total positive local signal (small = focal)
    """
    v = np.asarray(local_map, float)[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return dict(mean=np.nan, frac_pos=np.nan, gini=np.nan, area50=np.nan)
    pos = np.sort(np.clip(v, 0.0, None)); s = pos.sum()
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(pos) / (n * s)) if s > 0 else 0.0
    if s > 0:
        cum = np.cumsum(pos[::-1]) / s
        area50 = float((np.searchsorted(cum, 0.5) + 1) / n)
    else:
        area50 = 1.0
    return dict(mean=float(v.mean()), frac_pos=float((v > 0).mean()), gini=gini, area50=area50)


# ---------------------------------------------------------------------------
# Significance via batched permutation (same FFT machinery)
# ---------------------------------------------------------------------------

def _perm_pvalues(nulls, obs):
    """One-sided permutation p-values ``(p_up, p_down)``, counted over the finite draws.

    ``nulls`` is ``(P, ...)`` and ``obs`` has the shape of one draw. A draw that is not finite is an
    undefined statistic, not a value above or below ``obs``, so it leaves both the count and the
    denominator: ``p_up = (1 + #{finite draws >= obs}) / (#{finite draws} + 1)``, and ``p_down``
    counts ``<=``. Where ``obs`` is not finite there is nothing to test and both are NaN. With every
    draw finite this is the usual ``(1 + #{draws >= obs}) / (P + 1)``, to the bit.
    """
    nulls = np.asarray(nulls, float)
    obs = np.asarray(obs, float)
    fin = np.isfinite(nulls)
    n1 = fin.sum(0) + 1.0
    p_up = (1.0 + (fin & (nulls >= obs)).sum(0)) / n1
    p_dn = (1.0 + (fin & (nulls <= obs)).sum(0)) / n1
    undefined = ~np.isfinite(obs)
    return np.where(undefined, np.nan, p_up), np.where(undefined, np.nan, p_dn)


def permutation_pvalues(grid: np.ndarray, mask: np.ndarray, observed: np.ndarray,
                        kernel: np.ndarray | None = None, radius: int = 1,
                        n_perm: int = 100, seed: int = 0) -> np.ndarray:
    """One-sided permutation p-values for Moran's I per gene (greater).

    Each permutation reshuffles spot positions (same shuffle across genes) and
    recomputes I for all genes via one batched FFT, so n_perm permutations cost
    n_perm batched convolutions. Returns (G,) p-values,
    ``(1 + #{finite draws >= observed}) / (#{finite draws} + 1)``. A gene whose
    observed I is not finite (e.g. an all-zero gene) gets NaN, and a draw whose I
    is undefined is left out of the null.
    """
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    vals = grid[mask]                      # (n_spots, G)
    m = mask.astype(float)
    W1 = fftconvolve(m, kernel, mode="same")               # neighbour count — INVARIANT under permutation
    inv = np.where(W1 > 0, 1.0 / np.maximum(W1, 1e-12), 0.0)[..., None]
    zbar = vals.mean(0)                                     # value mean and Σz² are invariant (same value set)
    den = np.where((d := ((vals - zbar) ** 2).sum(0)) == 0, np.nan, d)
    observed = np.asarray(observed, float)
    ge = np.ones(grid.shape[-1])           # count >= observed (start at 1 for +obs)
    n1 = np.ones(grid.shape[-1])           # finite draws, +1 for obs
    perms = [rng.permutation(len(vals)) for _ in range(n_perm)]   # pre-draw (identical to serial)

    def _one(pm):
        buf = np.zeros_like(grid)          # own scatter buffer per perm (thread-safe)
        buf[idx[:, 0], idx[:, 1]] = vals[pm]
        z = (buf - zbar) * m[..., None]                    # only the numerator FFT per permutation
        Ip = (z * (_fftconv_same(z, kernel) * inv)).sum((0, 1)) / den   # row-standardized Moran (matches morans_i)
        fin = np.isfinite(Ip)
        return (fin & (Ip >= observed)).astype(float), fin.astype(float)

    from ._parallel import parallel_map
    for hit, fin in parallel_map(_one, perms):             # FFT-bound null, threaded
        ge += hit; n1 += fin
    # a NaN observed I compares False with every draw, which would give it the floor p
    return np.where(np.isfinite(observed), ge / n1, np.nan)


# ---------------------------------------------------------------------------
# Quaternion joint (multivariate) spatial autocorrelation -- the novel statistic
# ---------------------------------------------------------------------------

def joint_morans_i(qgrid: np.ndarray, mask: np.ndarray, kernel: np.ndarray | None = None,
                   radius: int = 1) -> float:
    """Multivariate Moran's I on the quaternion vector part (joint 3-feature profile).

    Uses the vector (i,j,k) channels jointly: numerator sums the neighbor dot-products
    of centered vectors. A single statistic for coordinated spatial structure across
    the three transcriptomic features, not available from per-gene Moran's I.
    """
    if kernel is None:
        kernel = neighbor_kernel(radius)
    from scipy.signal import fftconvolve  # lazy
    m = mask.astype(float)
    msum = m.sum()
    v = qgrid[..., 1:4]
    vbar = (v * m[..., None]).sum((0, 1)) / msum
    z = (v - vbar) * m[..., None]                      # (H, W, 3)
    Wz = _fftconv_same(z, kernel)                      # neighbor-summed vectors
    num = (z * Wz).sum()                               # sum over space AND 3 features
    den = (z * z).sum()
    S0 = (fftconvolve(m, kernel, mode="same") * m).sum()
    return float((msum / S0) * (num / den)) if den > 0 else np.nan


# ---------------------------------------------------------------------------
# Lag-space (full spatial autocorrelation / cross-correlation) via one FFT
# ---------------------------------------------------------------------------

def _lag_map(a, mask, b=None, maskb=None):
    """Linear (zero-padded) lag-space product map of fields a and b on a lattice.

    Returns (pairs, overlap, dist): all (2H, 2W) and fftshift-centered so lag (0,0) is
    the center. pairs[Δ] = Σ_x a(x) b(x+Δ); overlap[Δ] = Σ_x mask(x) maskb(x+Δ) (the
    number of in-window position pairs at lag Δ, for translation edge correction); dist =
    Euclidean lag distance. One FFT gives EVERY lag at once -- the basis of Ripley's K,
    the variogram and the spatial cross-correlogram below.
    """
    a = np.asarray(a, float); m = np.asarray(mask, float)
    b = a if b is None else np.asarray(b, float)
    mb = m if maskb is None else np.asarray(maskb, float)
    from scipy.fft import next_fast_len  # lazy
    H, W = a.shape; fh, fw = next_fast_len(2 * H), next_fast_len(2 * W)   # ≥2·H avoids lag wraparound; fast size
    Fa = _fft.fft2_hp(a, s=(fh, fw)); Fb = _fft.fft2_hp(b, s=(fh, fw))
    Fm = _fft.fft2_hp(m, s=(fh, fw)); Fmb = _fft.fft2_hp(mb, s=(fh, fw))
    # cross-correlation (a ⋆ b)[Δ] = Σ_x a(x) b(x+Δ) = ifft(conj(A)·B); this convention
    # puts the peak at the TRUE lag (a leads b by +Δ). (Symmetric for autocorrelation.)
    pairs = np.fft.fftshift(_fft.ifft2_hp(np.conj(Fa) * Fb).real)
    overlap = np.fft.fftshift(_fft.ifft2_hp(np.conj(Fm) * Fmb).real)
    cy, cx = fh // 2, fw // 2
    yy, xx = np.mgrid[0:fh, 0:fw]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return pairs, overlap, dist


def ripley_k(points: np.ndarray, mask: np.ndarray, radii, cross: np.ndarray | None = None):
    """Ripley's K, L and pair-correlation g(r) across MANY radii from ONE FFT.

    `points`: (H, W) per-bin counts of a cell type (or 0/1). `cross`: optional second count
    grid for the BIVARIATE cross-K_ab (cell-type co-localization). Translation edge-corrected
    (normalizes each lag by the in-window overlap = mask autocorrelation), so it is unbiased
    near tissue borders. Radius-independent: the whole K(r) curve costs one FFT, where a
    neighbor-graph estimator pays per radius.

    Returns dict(radii, K, L, g, csr):
      K(r)  Ripley's K; under complete spatial randomness (CSR) K(r)=πr².
      L(r)  = sqrt(K/π); CSR L(r)=r (variance-stabilized; L>r ⇒ clustering, <r ⇒ dispersion).
      g(r)  pair-correlation = K'(r)/(2πr); CSR g=1 (g>1 attraction, <1 inhibition at scale r).
      csr   πr² reference.
    """
    a = np.asarray(points, float); m = np.asarray(mask, float)
    b = a if cross is None else np.asarray(cross, float)
    pairs, overlap, dist = _lag_map(a, m, b, m)
    A = m.sum(); Na = a.sum(); Nb = b.sum()
    thr = overlap.max() * 1e-6
    ratio = np.where(overlap > thr, pairs / np.where(overlap > thr, overlap, 1.0), 0.0)
    radii = np.asarray(radii, float)
    K = np.empty(len(radii))
    for i, r in enumerate(radii):
        sel = dist <= r
        if cross is None:
            sel &= dist > 0                       # exclude self-pairs (Δ=0) for univariate
        K[i] = ratio[sel].sum()
    K *= A * A / (Na * Nb)
    L = np.sqrt(np.clip(K, 0, None) / np.pi)
    g = (np.gradient(K, radii) / (2 * np.pi * np.maximum(radii, 1e-9))
         if len(radii) > 1 else np.full_like(K, np.nan))   # g needs >=2 radii
    return dict(radii=radii, K=K, L=L, g=g, csr=np.pi * radii ** 2)


def cross_correlogram(a: np.ndarray, b: np.ndarray, mask: np.ndarray, max_lag: int = 20):
    """Lag-resolved spatial cross-correlation of two fields a, b -> directional coupling.

    Computes the FULL 2D cross-correlogram corr(Δ) = corr(a(x), b(x+Δ)) for every lag Δ
    (mean-centered, mask/overlap-corrected, normalized to a correlation), via one FFT. Unlike
    a symmetric-kernel bivariate Moran's I (which collapses to Pearson), the lag map is
    DIRECTIONAL: an off-center peak means "a leads b by that displacement" (e.g. a ligand
    gradient feeding a receptor downstream). Returns dict:
      lag2d   (2*max_lag+1, 2*max_lag+1) cropped correlogram (center = zero lag)
      radial  (max_lag+1,) isotropic mean correlation vs lag distance
      peak    (dy, dx) lag of maximum correlation (the lead-lag offset)
      peak_corr, zero_corr  correlation at the peak and at zero lag
      asymmetry  corr(peak) - corr(-peak) (directionality magnitude; ~0 = symmetric/co-located)
    """
    m = np.asarray(mask, float)
    inm = m > 0
    za = (np.asarray(a, float) - np.asarray(a, float)[inm].mean()) * m
    zb = (np.asarray(b, float) - np.asarray(b, float)[inm].mean()) * m
    pairs, overlap, dist = _lag_map(za, m, zb, m)
    thr = overlap.max() * 1e-3
    cov = np.where(overlap > thr, pairs / np.where(overlap > thr, overlap, 1.0), np.nan)
    sa = za[inm].std(); sb = zb[inm].std()
    corr = cov / (sa * sb + 1e-12)
    fh, fw = corr.shape; cy, cx = fh // 2, fw // 2
    lag2d = corr[cy - max_lag:cy + max_lag + 1, cx - max_lag:cx + max_lag + 1]
    # radial profile
    radial = np.empty(max_lag + 1)
    for r in range(max_lag + 1):
        sel = (dist >= r - 0.5) & (dist < r + 0.5) & np.isfinite(corr)
        radial[r] = corr[sel].mean() if sel.any() else np.nan
    # directional peak within the window
    win = lag2d.copy()
    fy, fx = np.unravel_index(np.nanargmax(win), win.shape)
    dy, dx = fy - max_lag, fx - max_lag
    peak_corr = float(win[fy, fx])
    opp = win[max_lag - dy, max_lag - dx]
    return dict(lag2d=lag2d, radial=radial, peak=(int(dy), int(dx)),
                peak_corr=peak_corr, zero_corr=float(win[max_lag, max_lag]),
                asymmetry=float(peak_corr - opp))


def anisotropy(field: np.ndarray, mask: np.ndarray, max_lag: int = 15) -> dict:
    """Directional anisotropy of a field's spatial autocorrelation (3rd spatial-fingerprint axis).

    Isotropic patterns (dispersed cells, blobs) have a circularly-symmetric autocorrelation;
    directional patterns (layers, gradients, fibres) have an elongated one. We take the 2D
    autocorrelation (one FFT, via :func:`cross_correlogram` of the field with itself), then the
    inertia (2nd-moment) tensor of its positive part. Returns dict:
      anisotropy   1 - sqrt(lambda_min/lambda_max) of the tensor (0 = isotropic, →1 = directional)
      orientation  angle (deg) of the major autocorrelation axis
    """
    r = cross_correlogram(field, field, mask, max_lag=max_lag)
    A = r["lag2d"]; c = max_lag
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1]
    w = np.clip(np.nan_to_num(A), 0.0, None).copy()
    w[c, c] = 0.0                               # drop the zero-lag spike
    s = w.sum()
    if s <= 0:
        return dict(anisotropy=np.nan, orientation=np.nan)
    Ixx = (w * xx * xx).sum() / s; Iyy = (w * yy * yy).sum() / s; Ixy = (w * xx * yy).sum() / s
    ev, evec = np.linalg.eigh(np.array([[Ixx, Ixy], [Ixy, Iyy]]))
    lam_min, lam_max = max(ev[0], 0.0), ev[1]
    aniso = 1.0 - np.sqrt(lam_min / lam_max) if lam_max > 0 else 0.0
    angle = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
    return dict(anisotropy=float(aniso), orientation=angle)


def polar_power_spectrum(field: np.ndarray, mask: np.ndarray, n_theta: int = 36,
                         kmin_frac: float = 0.03, kmax_frac: float = 0.7) -> dict:
    """2-D polar power spectrum -> directional anisotropy of a field (frequency-domain; the FFT-native
    complement to real-space :func:`anisotropy`). Projecting the field onto the orthogonal Fourier basis
    lets us read tissue ORIENTATION directly from where spectral power concentrates in angle — impossible
    to parametrize on a spatial neighbour graph without rebuilding a directed W per angle.

    We take ``P(kx,ky) = |FFT(demeaned masked field)|^2``, keep an annulus kmin..kmax of the Nyquist band
    (drop the DC/near-DC and the noisy high-frequency corner), and bin power into ``n_theta`` orientation
    wedges over [0, 180) deg (spatial frequency is axial: k and -k are the same orientation). Anisotropy
    and the dominant angle come from the doubled-angle circular resultant of P(theta) (order parameter),
    the standard axial statistic.

    IMPORTANT convention: power concentrates PERPENDICULAR to real-space elongation — a field of stripes
    running along theta varies fastest ACROSS them, so its spectral power peaks at theta+90. We therefore
    report both ``power_orientation`` (where spectral power peaks) and ``structure_orientation`` =
    power_orientation + 90 (the real-space elongation axis, comparable to :func:`anisotropy`'s orientation).

    Returns dict:
      theta_deg (n_theta), power (n_theta)      the angular power profile P(theta)
      anisotropy   0 (isotropic) -> 1 (a single dominant orientation)  [circular resultant length]
      power_orientation      deg, axis of maximum spectral power (mod 180)
      structure_orientation  deg, real-space elongation axis = power_orientation + 90 (mod 180)
    """
    f = np.asarray(field, float); m = np.asarray(mask, bool)
    f = np.where(m, f - f[m].mean(), 0.0)
    H, W = f.shape
    P = np.abs(np.fft.fftshift(_fft.fft2_hp(f))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H)); kx = np.fft.fftshift(np.fft.fftfreq(W))
    KX, KY = np.meshgrid(kx, ky); Kr = np.sqrt(KX ** 2 + KY ** 2)
    knyq = Kr.max()
    band = (Kr >= kmin_frac * knyq) & (Kr <= kmax_frac * knyq)
    ang = (np.degrees(np.arctan2(KY, KX)) % 180.0)              # axial: fold to [0,180)
    edges = np.linspace(0, 180, n_theta + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(ang[band], edges) - 1, 0, n_theta - 1)
    Pw = np.bincount(idx, weights=P[band], minlength=n_theta)
    cnt = np.maximum(np.bincount(idx, minlength=n_theta), 1)
    Pth = Pw / cnt                                              # mean power per orientation wedge
    # doubled-angle circular resultant (axial order parameter)
    th = np.radians(ctr); a = (Pth * np.cos(2 * th)).sum(); b = (Pth * np.sin(2 * th)).sum()
    tot = Pth.sum() + 1e-12
    aniso = float(np.sqrt(a ** 2 + b ** 2) / tot)
    power_orient = float((0.5 * np.degrees(np.arctan2(b, a))) % 180.0)
    return dict(theta_deg=ctr, power=Pth, anisotropy=aniso,
                power_orientation=power_orient,
                structure_orientation=float((power_orient + 90.0) % 180.0))


def variogram(grid: np.ndarray, mask: np.ndarray, radii):
    """Empirical semivariogram gamma(h) per gene across lag distances, via FFT.

    gamma(h) = 0.5 * mean over in-mask pairs at lag h of (z(x) - z(x+h))^2. Geary's C is
    gamma at a single lag; this returns the WHOLE geostatistical curve in one FFT, giving
    per-gene range (decorrelation distance), sill (plateau variance) and nugget (small-lag
    variance / noise). `grid`: (H, W) or (H, W, G). NOTE memory ~ 4*H*W*G; subset genes for HD.

    Returns dict(radii, gamma (n_radii[,G]), sill, nugget, rng) where rng = smallest h with
    gamma >= 0.95*sill (the variogram range).
    """
    g2 = grid.ndim == 2
    if g2:
        grid = grid[..., None]
    from scipy.fft import next_fast_len  # lazy
    m = np.asarray(mask, float); H, W, G = grid.shape
    z = grid * m[..., None]; z2 = (grid ** 2) * m[..., None]
    fh, fw = next_fast_len(2 * H), next_fast_len(2 * W)
    Fz = _fft.fft2_hp(z, s=(fh, fw), axes=(0, 1))
    Fz2 = _fft.fft2_hp(z2, s=(fh, fw), axes=(0, 1))
    Fm = _fft.fft2_hp(m, s=(fh, fw))
    sh = lambda a: np.fft.fftshift(a, axes=(0, 1))
    ac = sh(_fft.ifft2_hp(np.conj(Fz) * Fz, s=(fh, fw), axes=(0, 1)).real)          # Σ z(x)z(x+Δ)
    t1 = sh(_fft.ifft2_hp(np.conj(Fz2) * Fm[..., None], s=(fh, fw), axes=(0, 1)).real)  # Σ z(x)^2 over overlap
    t2 = sh(_fft.ifft2_hp(np.conj(Fm[..., None]) * Fz2, s=(fh, fw), axes=(0, 1)).real)  # Σ z(x+Δ)^2
    ov = np.fft.fftshift(_fft.ifft2_hp(np.conj(Fm) * Fm, s=(fh, fw)).real)          # pair count
    gam = (t1 + t2 - 2 * ac) / (2 * np.where(ov[..., None] > 1e-6, ov[..., None], np.nan))
    cy, cx = fh // 2, fw // 2
    yy, xx = np.mgrid[0:fh, 0:fw]; dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radii = np.asarray(radii, float)
    gamma = np.empty((len(radii), G))
    for i, r in enumerate(radii):
        sel = (dist >= r - 0.5) & (dist < r + 0.5)
        w = ov[sel]
        gamma[i] = np.nansum(gam[sel] * w[:, None], 0) / max(w.sum(), 1e-9)
    sill = np.nanmax(gamma, 0)
    nugget = gamma[0]
    rng = np.full(G, radii[-1])
    for gidx in range(G):
        hit = np.where(gamma[:, gidx] >= 0.95 * sill[gidx])[0]
        if hit.size:
            rng[gidx] = radii[hit[0]]
    out = dict(radii=radii, gamma=gamma[:, 0] if g2 else gamma,
               sill=float(sill[0]) if g2 else sill,
               nugget=float(nugget[0]) if g2 else nugget,
               rng=float(rng[0]) if g2 else rng)
    return out


def getis_ord(grid: np.ndarray, mask: np.ndarray, radius: int = 5, kind: str = "disc") -> np.ndarray:
    """Getis-Ord Gi* local hot/cold-spot z-scores per gene at ANY radius (FFT, radius-indep).

    Gi*(x) = (Σ_j w_xj x_j - X̄ Σ_j w_xj) / (S * sqrt[(n Σw² - (Σw)²)/(n-1)]), with w a disc
    of `radius` INCLUDING the focal cell. Large positive = hotspot (high-value cluster), large
    negative = coldspot. Returns (H, W) or (H, W, G) z-scores (NaN off-mask).
    """
    from scipy.signal import fftconvolve  # lazy
    g2 = grid.ndim == 2
    if g2:
        grid = grid[..., None]
    mb = np.asarray(mask, bool); m = mb.astype(float); n = m.sum()
    k = neighbor_kernel(radius, kind); k[radius, radius] = 1.0          # include focal cell
    x = grid * m[..., None]
    Wc = fftconvolve(m, k, mode="same")                                 # Σw  (local weight sum)
    Wc2 = fftconvolve(m, k ** 2, mode="same")                           # Σw² (=Σw only for a binary disc)
    local = _fftconv_same(x, k)                                         # local weighted sum
    xbar = x.sum((0, 1)) / n
    S = np.sqrt(np.clip((grid ** 2 * m[..., None]).sum((0, 1)) / n - xbar ** 2, 0, None))
    num = local - xbar * Wc[..., None]
    den = S * np.sqrt(np.clip((n * Wc2[..., None] - Wc[..., None] ** 2) / (n - 1), 0, None))
    Gi = num / np.where(den == 0, np.nan, den)
    Gi[~mb] = np.nan
    return Gi[..., 0] if g2 else Gi


def losh(grid: np.ndarray, mask: np.ndarray, radius: int = 5, kind: str = "disc",
         a: float = 2.0) -> np.ndarray:
    """LOSH — Local Spatial Heteroscedasticity, per gene at ANY radius (FFT, radius-independent).

    Ord & Getis, *Local spatial heteroscedasticity (LOSH)*, Annals of Regional Science 48:529-539
    (2012). Where local Moran asks whether a cell sits in a same-signed cluster, LOSH asks how
    VARIABLE the process is around it: ``H_i > 1`` marks a neighbourhood noisier than the tissue
    average, ``H_i < 1`` a locally homogeneous one. It is the variance companion to the local mean
    statistics, and is what distinguishes a smooth gradient from a speckled patch of the same mean.

    .. math::
        H_i = \\frac{\\sum_j w_{ij} |e_j|^a}{h_1 \\sum_j w_{ij}}, \\quad
        e_j = x_j - \\bar{x}_j, \\quad
        \\bar{x}_j = \\frac{\\sum_k w_{jk} x_k}{\\sum_k w_{jk}}, \\quad
        h_1 = \\frac{1}{n}\\sum_j |e_j|^a

    Note the residual is formed POINTWISE, against the local mean at *j*, before the second
    smoothing — so this is not the local variance ``conv(x²)/conv(1) - (conv(x)/conv(1))²`` about
    the focal cell, which is a different statistic with no χ² null. Two convolution passes on the
    same kernel; ``a=1`` (absolute deviation) therefore costs exactly what ``a=2`` (variance) does.

    Weights are the row-standardized disc of `radius` EXCLUDING the focal cell, matching the rest
    of the toolkit and libpysal's default ``W``. Cells with no in-mask neighbour have no local mean:
    they return NaN and contribute 0 to ``h_1``.

    Parameters
    ----------
    grid : numpy.ndarray
        ``(H, W)`` or ``(H, W, G)`` rasterized field(s); values off-mask are ignored.
    mask : numpy.ndarray
        ``(H, W)`` boolean tissue mask.
    radius : int
        Neighbourhood radius in grid cells.
    kind : {'disc', 'gaussian', 'queen'}
        Kernel shape, as in :func:`neighbor_kernel`.
    a : float
        Exponent on the residual. ``2`` (default) gives a variance measure, ``1`` an
        absolute-deviation measure that is far less sensitive to a single outlying cell.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` or ``(H, W, G)`` of ``H_i``, NaN off-mask. Invariant to any affine rescaling of
        the input field, so no standardization is applied or needed. A gene with no residual variance
        in the mask (absent, or constant) is NaN everywhere: H is 0/0 there (:data:`LOSH_RTOL`).
    """
    g2 = grid.ndim == 2
    if g2:
        grid = grid[..., None]
    from scipy.signal import fftconvolve  # lazy
    mb = np.asarray(mask, bool); m = mb.astype(float)
    k = neighbor_kernel(radius, kind)
    x = np.asarray(grid, float) * m[..., None]
    W1 = fftconvolve(m, k, mode="same")
    has_nb = W1 > 0.5                                   # 0.5 beats fft roundoff (cf gearys_c)
    inv = np.where(has_nb, 1.0 / np.maximum(W1, 1e-12), 0.0)[..., None]
    xbar = _fftconv_same(x, k) * inv                    # row-standardized local mean AT EACH CELL
    valid = mb & has_nb
    e = np.abs(x - xbar) ** a * valid[..., None]        # residuals; islands contribute nothing
    nval = float(valid.sum())
    h1 = e.sum((0, 1)) / nval if nval else np.full(x.shape[2], np.nan)
    H = (_fftconv_same(e, k) * inv) / np.where(h1 == 0, np.nan, h1)
    H[~valid] = np.nan
    if mb.any():                                        # no residual variance where measured: 0/0 (LOSH_RTOL)
        H[..., h1 <= (LOSH_RTOL * np.abs(x[mb]).max(0)) ** a] = np.nan
    return H[..., 0] if g2 else H


def local_diversity(label_grid: np.ndarray, mask: np.ndarray, radius: int = 5, kind: str = "disc"):
    """Local cell-type Shannon entropy and richness within `radius` (FFT, radius-independent).

    label_grid: (H, W) integer cell-type codes, -1 where empty. For each occupied cell, computes
    the local cell-type composition within `radius` (one-hot fields convolved with a disc) and its
    Shannon entropy -Σ p_k log p_k (high = mixed/interface, low = homogeneous domain) and richness
    (number of types present). Returns (entropy (H,W), richness (H,W)); NaN/0 off occupied cells.
    """
    from scipy.signal import fftconvolve  # lazy
    occ = np.asarray(mask, bool) & (label_grid >= 0)
    labels = np.unique(label_grid[occ])
    k = neighbor_kernel(radius, kind); k[radius, radius] = 1.0
    tot = np.zeros(label_grid.shape); ent = np.zeros(label_grid.shape); rich = np.zeros(label_grid.shape)
    cs = []
    for lab in labels:
        c = fftconvolve(((label_grid == lab) & occ).astype(float), k, mode="same")
        cs.append(c); tot += c
    totv = np.where(tot <= 0, np.nan, tot)
    for c in cs:
        p = c / totv
        ent -= np.where(p > 0, p * np.log(p), 0.0)
        rich += (c >= 0.5)
    ent[~occ] = np.nan; rich[~occ] = 0
    return ent, rich


# ---------------------------------------------------------------------------
# (8) Fast permutation null + Benjamini-Hochberg FDR (general, batched)
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values for a vector of p-values.

    A NaN p (a test that is undefined, e.g. a gene with no variance) stays NaN, and the other
    p-values are corrected over the finite ones only, as if the NaN entries were absent. This
    differs from :func:`spectrast.diff.bh_fdr`, which reads NaN as 1.0 and counts it; on input
    without NaN the two agree exactly.
    """
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    q = pv[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qv = np.empty(n); qv[order] = np.clip(q, 0, 1)
    out[ok] = qv
    return out


def permutation_test(stat_fn, grid: np.ndarray, mask: np.ndarray, n_perm: int = 200,
                     seed: int = 0, alternative: str = "greater"):
    """Generic fast spatial permutation null + BH-FDR for any per-gene spatial statistic.

    `stat_fn(grid, mask) -> (G,)` is any batched statistic here (e.g. `morans_i`, `gearys_c`).
    Each permutation reshuffles values among occupied cells (positions fixed; same shuffle across
    genes) and recomputes the whole gene vector via the statistic's own batched FFT, so n_perm
    permutations cost n_perm batched transforms. Destroys spatial structure while preserving each
    gene's marginal distribution. Returns dict(observed, pval, qval, z).

    ``pval`` is ``(1 + #{finite draws as or more extreme}) / (#{finite draws} + 1)``; the two-sided
    value is twice the smaller tail, capped at 1. A gene whose observed statistic is not finite
    (e.g. Moran's I of an all-zero gene) gets NaN ``pval`` and ``qval``, a draw whose statistic is
    undefined is left out of the null, and ``qval`` corrects over the finite p only
    (:func:`bh_fdr`).
    """
    if grid.ndim == 2:
        grid = grid[..., None]
    obs = np.asarray(stat_fn(grid, mask))
    idx = np.argwhere(mask); vals = grid[mask]                # (n_occ, G)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(vals)) for _ in range(n_perm)]   # pre-draw (identical to serial)

    def _one(pm):
        g = np.zeros_like(grid)                                   # own buffer per perm (thread-safe)
        g[idx[:, 0], idx[:, 1]] = vals[pm]
        return np.asarray(stat_fn(g, mask))

    from ._parallel import parallel_map
    nulls = np.array(parallel_map(_one, perms))                   # FFT-bound null, threaded
    p_up, p_dn = _perm_pvalues(nulls, obs)
    if alternative == "greater":
        p = p_up
    elif alternative == "less":
        p = p_dn
    else:
        p = np.minimum(1.0, 2 * np.minimum(p_up, p_dn))
    z = (obs - nulls.mean(0)) / (nulls.std(0) + 1e-12)
    return dict(observed=obs, pval=p, qval=bh_fdr(p), z=z)


# ---------------------------------------------------------------------------
# (7) Fourier (Moran-eigenvector) spatial filtering: spatially-aware regression
# ---------------------------------------------------------------------------

def fourier_modes(mask: np.ndarray, n_modes: int = 24) -> np.ndarray:
    """Low-frequency 2D Fourier spatial basis evaluated at occupied cells -> (n_occ, n_modes).

    On a regular lattice the eigenvectors of a translation-invariant spatial-weight matrix are the
    2D Fourier modes; the lowest-frequency ones are the Moran eigenvector maps (MEMs) that carry
    large-scale spatial trend. Used as covariates they absorb spatial autocorrelation. Returns the
    n_modes lowest-frequency cos/sin modes (DC excluded), sampled at the occupied cells.
    """
    m = np.asarray(mask, bool); H, W = m.shape
    ys, xs = np.nonzero(m)
    freqs = []
    kmax = int(np.ceil(np.sqrt(n_modes))) + 2
    for u in range(kmax):
        for v in range(kmax):
            if u == 0 and v == 0:
                continue
            freqs.append((u, v, u * u + v * v))
    freqs.sort(key=lambda t: t[2])
    cols = []
    for u, v, _ in freqs:
        ph = 2 * np.pi * (u * ys / H + v * xs / W)
        cols.append(np.cos(ph)); cols.append(np.sin(ph))
        if len(cols) >= n_modes:
            break
    return np.column_stack(cols[:n_modes])


def spatial_regression(y: np.ndarray, x: np.ndarray, mask: np.ndarray, n_modes: int = 24):
    """Association of fields y ~ x at occupied cells, CONTROLLING for spatial autocorrelation via
    low-frequency Fourier (Moran-eigenvector) covariates.

    Two spatially-autocorrelated fields can correlate strongly by chance (spatial confounding),
    inflating naive p-values. Including the spatial Fourier basis as covariates absorbs the shared
    large-scale trend, giving a calibrated test of genuine local association. Returns dict with the
    naive Pearson r/p and the spatially-adjusted coefficient + p-value.
    """
    from scipy import stats  # lazy
    m = np.asarray(mask, bool)
    yy = np.asarray(y, float)[m]; xx = np.asarray(x, float)[m]
    r, p_naive = stats.pearsonr(xx, yy)
    M = fourier_modes(m, n_modes)
    X = np.column_stack([np.ones(len(xx)), xx, M])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = len(yy) - X.shape[1]
    sigma2 = (resid @ resid) / max(dof, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * XtX_inv[1, 1])
    t = beta[1] / (se + 1e-12)
    p_adj = float(2 * stats.t.sf(abs(t), max(dof, 1)))
    return dict(r=float(r), p_naive=float(p_naive), beta_x=float(beta[1]),
                p_adjusted=p_adj, n_modes=n_modes)


def spatial_association(y: np.ndarray, x: np.ndarray, mask: np.ndarray, n_modes: int = 40,
                        n_shift: int = 199, seed: int = 0):
    """Calibrated test of spatial association between fields x and y.

    Two-stage: (1) remove the coarse spatial trend by residualizing BOTH fields on the
    low-frequency Fourier modes (Moran eigenvectors); (2) assess the residual correlation with a
    TOROIDAL-SHIFT permutation null, which preserves x's own spatial autocorrelation while breaking
    its alignment with y. This is properly calibrated under spatial confounding, where an OLS
    t-test is anti-conservative (it counts every cell as independent though residuals stay
    autocorrelated). Returns dict(r_resid, p_shift, r_naive).

    A shift counts only when the tissue overlaps its translate in more than 10 bins, and
    ``p_shift = (1 + n_ge) / (n_ok + 1)``, where ``n_ok`` is the number of accepted shifts and
    ``n_ge`` the number of those whose residual correlation is at least as large in absolute value
    as ``r_resid``. When fewer than ``max(min(n_shift, 10), n_shift / 2)`` shifts are accepted (a
    small or fragmented tissue), ``p_shift`` is NaN and a RuntimeWarning says how many were usable.
    """
    m = np.asarray(mask, bool); H, W = m.shape
    yv = np.asarray(y, float); xv = np.asarray(x, float)
    M = np.column_stack([np.ones(m.sum()), fourier_modes(m, n_modes)])

    def _resid(f):
        v = f[m]; b, *_ = np.linalg.lstsq(M, v, rcond=None); return v - M @ b
    rx, ry = _resid(xv), _resid(yv)
    rxg = np.zeros((H, W)); rxg[m] = rx
    ryg = np.zeros((H, W)); ryg[m] = ry

    def _corr(a, b, sel):
        a = a[sel] - a[sel].mean(); b = b[sel] - b[sel].mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0
    obs = _corr(rxg, ryg, m)
    rng = np.random.default_rng(seed)
    ge = 1
    n_ok = 0                               # accepted shifts: a rejected one is not a null draw
    for _ in range(n_shift):
        dy, dx = int(rng.integers(1, H)), int(rng.integers(1, W))
        shx = np.roll(np.roll(rxg, dy, 0), dx, 1)
        shm = np.roll(np.roll(m, dy, 0), dx, 1) & m
        if shm.sum() > 10:
            n_ok += 1
            if abs(_corr(shx, ryg, shm)) >= abs(obs):
                ge += 1
    # Too few accepted shifts is no test. The bar is the one mark_cross_correlation's shift null
    # uses to say its usable draws ran short.
    if n_ok < max(min(n_shift, 10), 0.5 * n_shift):
        warnings.warn(f"spatial_association: only {n_ok}/{n_shift} toroidal shifts were usable (the "
                      f"tissue overlaps its own translate in 10 bins or fewer); p_shift is NaN.",
                      RuntimeWarning, stacklevel=2)
        p_shift = np.nan
    else:
        p_shift = ge / (n_ok + 1)
    return dict(r_resid=obs, p_shift=p_shift,
                r_naive=_corr(xv, yv, m))


# ---------------------------------------------------------------------------
# (3) FFT spatial cell-cell communication
# ---------------------------------------------------------------------------

def cci_map(ligand: np.ndarray, receptor: np.ndarray, mask: np.ndarray, radius: int = 10,
            kind: str = "gaussian") -> np.ndarray:
    """Spatial signaling-potential map for one ligand-receptor pair via FFT.

    The ligand diffuses over `radius` (mask-aware smoothing = local availability); multiplying by
    receptor expression gives the per-location signaling potential (high where diffused ligand meets
    receptor). Radius-independent O(N log N). Returns (H, W) map (0 off-mask).
    """
    m = np.asarray(mask, bool)
    avail = spatial_smooth(np.asarray(ligand, float), m, radius=radius, kind=kind)
    if avail.ndim == 3:
        avail = avail[..., 0]
    pot = np.nan_to_num(avail) * np.asarray(receptor, float) * m
    return pot


def cci_scores(grid: np.ndarray, mask: np.ndarray, lr_pairs, radius: int = 10,
               kind: str = "gaussian"):
    """FFT cell-cell communication scores for MANY ligand-receptor pairs at once.

    grid: (H, W, G) expression. lr_pairs: list of (ligand_idx, receptor_idx). Each pair's score is
    the mean over tissue of [diffused-ligand × receptor] (the signaling-potential map of
    :func:`cci_map`). All distinct ligands are diffused in ONE batched FFT, so scoring a whole
    ligand-receptor database is radius-independent and fast. Returns scores (n_pairs,).
    """
    m = np.asarray(mask, bool)
    lig_idx = sorted({int(l) for l, _ in lr_pairs})
    sm = spatial_smooth(grid[..., lig_idx], m, radius=radius, kind=kind)   # batched diffusion
    sm = np.nan_to_num(sm)
    pos = {g: i for i, g in enumerate(lig_idx)}
    scores = np.empty(len(lr_pairs))
    for k, (l, r) in enumerate(lr_pairs):
        pot = sm[..., pos[int(l)]] * grid[..., int(r)]
        scores[k] = pot[m].mean()
    return scores


# ---------------------------------------------------------------------------
# Local bivariate CCI (FFT form of LIANA's local metrics) + MULTI-SCALE
# ---------------------------------------------------------------------------

# Validated FFT-convolution metrics (match liana-py). Rank-based spearman/masked_spearman are NOT
# included: local rank-variance collapses on sparse genes at small bandwidth (numerically unstable).
_BIV_METRICS = ("cosine", "pearson", "product", "morans", "jaccard")


def biv_kernel(radius: float, kind: str = "gaussian", cutoff: float = 0.1) -> np.ndarray:
    """Spatial-weight kernel for local bivariate CCI; center INCLUDED (self-weight=1), matching
    LIANA's ``set_diag=True``. Gaussian = exp(-d^2 / 2 r^2) truncated at `cutoff` (LIANA default)."""
    if kind == "gaussian":
        rad = int(np.ceil(radius * np.sqrt(-2.0 * np.log(cutoff))))
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * radius ** 2))
        k[k < cutoff] = 0.0
    elif kind == "disc":
        rad = int(round(radius)); yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        k = (xx ** 2 + yy ** 2 <= radius ** 2).astype(float)
    else:
        raise ValueError(kind)
    return k


def _biv_channels(x, y, m, metric):
    """Return (channel_stack (H,W,C), assemble(convs)->map) for a metric. All metrics reduce to a
    fixed set of neighborhood-weighted sums (convolutions) of transformed x, y."""
    fm = m.astype(float)
    if metric not in _BIV_METRICS:
        raise ValueError(f"metric must be one of {_BIV_METRICS} (rank-based metrics unsupported)")
    if metric == "morans":                # SpatialDM: standardize, row-normalised weights
        xs = np.zeros_like(x); ys = np.zeros_like(y)
        xs[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
        ys[m] = (y[m] - y[m].mean()) / (y[m].std() + 1e-12)
        ch = np.stack([xs, ys, fm], -1)
        def asm(c):
            Wxs, Wys, W1 = c[..., 0], c[..., 1], c[..., 2]
            return xs * (Wys / (W1 + 1e-12)) + ys * (Wxs / (W1 + 1e-12))
        return ch, asm
    if metric == "jaccard":
        xb = (x > 0).astype(float) * fm; yb = (y > 0).astype(float) * fm
        ch = np.stack([xb * yb, xb + yb - xb * yb], -1)
        def asm(c):
            return c[..., 0] / (c[..., 1] + 1e-12)
        return ch, asm
    x = x * fm; y = y * fm
    ch = np.stack([x * y, x * x, y * y, x, y, fm], -1)     # Wxy, Wx2, Wy2, Wx, Wy, W1
    rel = lambda a: np.clip(a, 0.0, None)              # guard tiny negative sums-of-squares (FFT roundoff)
    if metric == "product":
        asm = lambda c: c[..., 3] * c[..., 4]
    elif metric == "cosine":
        asm = lambda c: c[..., 0] / (np.sqrt(rel(c[..., 1])) * np.sqrt(rel(c[..., 2])) + 1e-12)
    elif metric == "pearson":
        def asm(c):
            Wxy, Wx2, Wy2, Wx, Wy, W1 = (c[..., i] for i in range(6))
            return (W1 * Wxy - Wx * Wy) / (np.sqrt(rel(W1 * Wx2 - Wx ** 2) * rel(W1 * Wy2 - Wy ** 2)) + 1e-12)
    else:
        raise ValueError(f"metric must be one of {_BIV_METRICS}")
    return ch, asm


def bivariate_local(ligand, receptor, mask, radius: float = 5, metric: str = "cosine",
                    kind: str = "gaussian") -> np.ndarray:
    """Local bivariate spatial CCI metric for one ligand-receptor pair, via FFT convolution.

    FFT form of LIANA+'s local bivariate metrics (validated: ``product`` matches liana-py exactly,
    ``cosine``/``pearson`` Pearson r>0.95). ``metric`` in cosine|pearson|spearman|product|morans|
    jaccard. O(N log N), radius-independent. Returns an (H, W) local-score map (0 off-mask)."""
    m = np.asarray(mask, bool)
    x = np.asarray(ligand, float); y = np.asarray(receptor, float)
    ch, asm = _biv_channels(x, y, m, metric)
    k = biv_kernel(radius, kind)
    conv = _fftconv_same(ch, k)
    return np.where(m, asm(conv), 0.0)


def bivariate_local_multiscale(ligand, receptor, mask, radii, metric: str = "cosine",
                               kind: str = "gaussian"):
    """Local bivariate CCI at MANY bandwidths from ONE shared forward FFT — multi-scale CCI with no
    a-priori bandwidth. Returns (len(radii), H, W) local-score maps. The per-scale tissue mean gives
    each interaction's SCALE PROFILE; its peak = the characteristic scale of the interaction."""
    m = np.asarray(mask, bool)
    x = np.asarray(ligand, float); y = np.asarray(receptor, float)
    ch, asm = _biv_channels(x, y, m, metric)            # (H, W, C)
    H, W, C = ch.shape
    radii = list(radii)
    kernels = [biv_kernel(r, kind) for r in radii]
    rmax = max(k.shape[0] // 2 for k in kernels)
    from scipy.fft import next_fast_len  # lazy
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fch = _fft.rfft2(ch, s=(fh, fw), axes=(0, 1))      # shared forward transform
    out = np.empty((len(radii), H, W))
    for s, k in enumerate(kernels):
        kh = k.shape[0]; off = (kh - 1) // 2
        Fk = _fft.rfft2(k, s=(fh, fw))[:, :, None]
        full = _fft.irfft2(Fch * Fk, s=(fh, fw), axes=(0, 1))
        conv = full[off:off + H, off:off + W]
        out[s] = np.where(m, asm(conv), 0.0)
    return out


def bivariate_local_pvals(ligand, receptor, mask, radius: float = 5, metric: str = "cosine",
                          kind: str = "gaussian", n_perms: int = 100, seed: int = 1337,
                          two_sided: bool = True):
    """Permutation p-values for a local bivariate CCI map (same null as LIANA+/SpatialDM).

    The (ligand, receptor) value-pairs are jointly relocated to random spots (spatial structure
    destroyed, x-y coupling preserved) and the metric recomputed; per-spot p = fraction of
    permutations with |perm| >= |observed|. FFT makes each permutation cheap. Returns
    (score_map (H,W), pval_map (H,W) with NaN off-mask)."""
    m = np.asarray(mask, bool)
    obs = bivariate_local(ligand, receptor, mask, radius, metric, kind)
    xv = np.asarray(ligand, float)[m]; yv = np.asarray(receptor, float)[m]
    pos = np.where(m)
    a_obs = np.abs(obs) if two_sided else obs
    rng = np.random.default_rng(seed)
    cnt = np.zeros(obs.shape)
    perms = [rng.permutation(len(xv)) for _ in range(n_perms)]    # pre-draw (identical to serial)

    def _one(p):
        xg = np.zeros_like(obs); yg = np.zeros_like(obs)          # own buffers per perm (thread-safe)
        xg[pos] = xv[p]; yg[pos] = yv[p]
        ps = bivariate_local(xg, yg, mask, radius, metric, kind)
        return ((np.abs(ps) if two_sided else ps) >= a_obs).astype(float)

    from ._parallel import parallel_map
    for hit in parallel_map(_one, perms):                         # FFT-bound null, threaded
        cnt += hit
    # (r+1)/(n+1): the observed value is one draw from the null, so p is never exactly 0
    # (matches permutation_pvalues and bivariate_multiscale_significance).
    return obs, np.where(m, (cnt + 1.0) / (n_perms + 1.0), np.nan)


def bivariate_multiscale_significance(ligand, receptor, mask, radii, metric: str = "morans",
                                      kind: str = "gaussian", n_perms: int = 200, seed: int = 1337):
    """Scale-resolved GLOBAL significance of a ligand-receptor interaction (the multi-scale CCI test).

    For each bandwidth: global statistic = tissue-mean local metric; permutation null jointly relocates
    the (x,y) pairs (shared shuffles across scales) and recomputes the per-scale global mean. Returns
    (radii, global_score[n_radii], pval[n_radii]). The characteristic scale = argmax|global_score|; a
    fixed-bandwidth analysis (e.g. LIANA's single bandwidth) only sees one column of this profile."""
    m = np.asarray(mask, bool)
    radii = list(radii)
    obs_ms = bivariate_local_multiscale(ligand, receptor, mask, radii, metric, kind)
    obs = np.array([obs_ms[s][m].mean() for s in range(len(radii))])
    xv = np.asarray(ligand, float)[m]; yv = np.asarray(receptor, float)[m]
    pos = np.where(m)
    rng = np.random.default_rng(seed)
    cnt = np.zeros(len(radii))
    perms = [rng.permutation(len(xv)) for _ in range(n_perms)]    # pre-draw (identical to serial)

    def _one(p):
        xg = np.zeros(m.shape); yg = np.zeros(m.shape)            # own buffers per perm (thread-safe)
        xg[pos] = xv[p]; yg[pos] = yv[p]
        nm = bivariate_local_multiscale(xg, yg, mask, radii, metric, kind)
        return np.array([nm[s][m].mean() for s in range(len(radii))])

    from ._parallel import parallel_map
    for null in parallel_map(_one, perms):                        # FFT-bound null, threaded
        cnt += (np.abs(null) >= np.abs(obs))
    return np.array(radii), obs, (cnt + 1) / (n_perms + 1)


# ===========================================================================
# Single-cell (point-cloud) entry points  -- Xenium / MERFISH
# ---------------------------------------------------------------------------
# The functions above operate on a GRID (correct + exact for physical lattices:
# Visium, Visium HD). Imaging-based platforms are point clouds; rasterizing them
# distorts LOCAL statistics unless the bin is fine relative to the radius.
# These entry points take raw cell coordinates
# and by DEFAULT auto-choose the bin and repair the highest-error cells with the
# exact neighbourhood graph (adaptive fusion) -- so the toolkit "just works" on
# single-cell data. Pass ``adaptive=False`` for the plain FFT (autotuned bin, no
# fallback) or an explicit ``bin_size`` to skip auto-tuning. ``radius_um`` (and,
# for multi-scale use, ``r_min_um`` = the smallest radius) are in microns.
# Aggregate statistics (global Moran, colocalization matrix, neighbourhood
# enrichment) average over cells and are robust to rasterization as-is, so they
# do not need fusion: rasterize once with ``rasterize_auto`` and call the grid fn.
# ===========================================================================

def local_morans_i_sc(xy, values, radius_um, *, bin_size=None, adaptive=True, tol=0.1,
                      r_min_um=None, return_meta=False):
    """Per-cell local Moran's I on single-cell coordinates (adaptive rasterization + fusion)."""
    from .adaptive import adaptive_local_moran  # lazy
    return adaptive_local_moran(xy, values, radius_um, bin_size=bin_size, tol=tol,
                                r_min_um=r_min_um, fuse=adaptive, return_meta=return_meta)


def getis_ord_sc(xy, values, radius_um, *, bin_size=None, adaptive=True, tol=0.1,
                 r_min_um=None, return_meta=False):
    """Per-cell Getis-Ord Gi* hotspots on single-cell coordinates (adaptive fusion)."""
    from .adaptive import adaptive_getis_ord  # lazy
    return adaptive_getis_ord(xy, values, radius_um, bin_size=bin_size, tol=tol,
                              r_min_um=r_min_um, fuse=adaptive, return_meta=return_meta)


def bivariate_morans_sc(xy, values_x, values_y, radius_um, *, bin_size=None, adaptive=True,
                        tol=0.1, r_min_um=None, return_meta=False):
    """Per-cell local bivariate Moran (x lead vs neighbour y) on coordinates (adaptive fusion)."""
    from .adaptive import adaptive_local_bivariate  # lazy
    return adaptive_local_bivariate(xy, values_x, values_y, radius_um, bin_size=bin_size, tol=tol,
                                    r_min_um=r_min_um, fuse=adaptive, return_meta=return_meta)


def local_diversity_sc(xy, labels, radius_um, *, bin_size=None, adaptive=True, tol=0.1,
                       r_min_um=None, return_meta=False):
    """Per-cell local Shannon diversity of neighbourhood composition on coordinates (adaptive fusion)."""
    from .adaptive import adaptive_local_diversity  # lazy
    return adaptive_local_diversity(xy, labels, radius_um, bin_size=bin_size, tol=tol,
                                    r_min_um=r_min_um, fuse=adaptive, return_meta=return_meta)


def colocalization_sc(xy, labels, target, radius_um, *, bin_size=None, adaptive=True, tol=0.1,
                      r_min_um=None, return_meta=False):
    """Per-cell colocalization: count of ``target`` cell-type near each cell (adaptive fusion)."""
    from .adaptive import adaptive_colocalization  # lazy
    return adaptive_colocalization(xy, labels, target, radius_um, bin_size=bin_size, tol=tol,
                                   r_min_um=r_min_um, fuse=adaptive, return_meta=return_meta)


def morans_global_genes(xy, X, radius_um, *, reduced_rank=False, rank=30, bin_size=None):
    """Global Moran's I per gene on single-cell coordinates. Multi-purpose: ``reduced_rank=False``
    computes each actual gene; ``reduced_rank=True`` runs an SVD (rank ``rank``) under the hood and
    reconstructs every gene's I exactly (fast + denoised). ``X`` is (N, G). Returns (G,)."""
    from .spatial_sc import morans_global  # lazy
    return morans_global(xy, X, radius_um, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def local_morans_i_genes(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None):
    """Per-cell local Moran maps for selected ``genes`` (column indices of X). ``reduced_rank=True``
    reconstructs each map from the rank-k (denoised) gene; ``False`` uses the actual gene. Returns
    (N, len(genes))."""
    from .spatial_sc import morans_local  # lazy
    return morans_local(xy, X, radius_um, genes, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def local_geary_genes(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None):
    """Per-cell local Geary's C for selected ``genes`` — a boundary/transition detector (high where a
    gene changes abruptly between neighbours; complements local Moran's cluster map). ``reduced_rank=True``
    reconstructs it from the rank-k gene via the k² embedding-product neighbour-sums. Returns (N, m)."""
    from .spatial_sc import local_geary  # lazy
    return local_geary(xy, X, radius_um, genes, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def getis_ord_genes(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None):
    """Per-cell Getis-Ord Gi* hotspots for selected ``genes``. ``reduced_rank`` toggles the
    denoised rank-k reconstruction vs the actual gene. Returns (N, len(genes))."""
    from .spatial_sc import getis_local  # lazy
    return getis_local(xy, X, radius_um, genes, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def losh_genes(xy, X, radius_um, genes, *, a=2.0, reduced_rank=False, rank=30, bin_size=None):
    """Per-cell LOSH (local spatial heteroscedasticity) for selected ``genes`` — the VARIANCE
    companion to local Moran/Getis: high where a gene's neighbourhood is noisier than the tissue
    average, low where it is locally smooth. ``a=2`` variance, ``a=1`` absolute deviation.
    ``reduced_rank=True`` computes it on the rank-k (denoised) gene. Returns (N, len(genes))."""
    from .spatial_sc import losh as _losh  # lazy
    return _losh(xy, X, radius_um, genes, a=a, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def reduced_rank_concordance(xy, X, radius_um, *, rank=None, bin_size=None, n_split=3, z=2.0,
                             seed=0, genes=None):
    """Trust layer: per-gene DIRECT vs reduced-rank Moran's I with a concordance flag (believe the
    reconstructed value where they agree; scrutinise where they don't). Returns a DataFrame. See
    :func:`spectrast.spatial_sc.concordance`."""
    from .spatial_sc import concordance  # lazy
    return concordance(xy, X, radius_um, rank=rank, bin_size=bin_size, n_split=n_split, z=z,
                       seed=seed, genes=genes)


def autotune_rank_genes(xy, X, radius_um, *, candidate_ranks=None, bin_size=None,
                        method="reproducibility", seed=0):
    """Data-driven reduced rank: split-half-reproducibility-optimal k (no ground truth) + a
    Gavish–Donoho heuristic. Use its ``rank`` for the ``reduced_rank`` tools. Returns a dict with
    ``rank``, ``candidate_ranks``, ``reproducibility``, ``gd_rank``."""
    from .spatial_sc import autotune_rank  # lazy
    return autotune_rank(xy, X, radius_um, candidate_ranks=candidate_ranks, bin_size=bin_size,
                         method=method, seed=seed)


def morans_scalogram_genes(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None):
    """Moran's I vs radius for every gene -> (n_radii, G). ``reduced_rank=True`` reconstructs the
    whole scalogram from k embedding convolutions per radius (fast + denoised)."""
    from .spatial_sc import morans_scalogram  # lazy
    return morans_scalogram(xy, X, radii, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def correlation_length_genes(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None, frac=0.5):
    """Per-gene characteristic spatial scale from the (optionally reduced-rank/denoised) scalogram -> (G,)."""
    from .spatial_sc import correlation_length  # lazy
    return correlation_length(xy, X, radii, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size, frac=frac)


def bivariate_spatial_matrix(xy, X, genes, radius_um, *, reduced_rank=False, rank=30, bin_size=None):
    """Symmetric all-pairs spatial cross-correlation (bivariate Moran) among ``genes`` -> (m, m).
    ``reduced_rank=True`` reconstructs the matrix from one k×k embedding matrix (the O(G²) collapse)."""
    from .spatial_sc import bivariate_matrix  # lazy
    return bivariate_matrix(xy, X, genes, radius_um, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def focality_genes(xy, X, radius_um, genes=None, *, reduced_rank=False, rank=30, bin_size=None):
    """Per-gene focality (Gini of the local Moran map) -> (m,). Streams over memory-sized gene chunks
    (auto), so it scales to whole-transcriptome panels; sparse X stays sparse. ``reduced_rank=False``
    computes each actual gene (default); ``reduced_rank=True`` runs the rank-k reconstruction."""
    from .spatial_sc import focality  # lazy
    return focality(xy, X, radius_um, genes, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def general_g_genes(xy, X, radius_um, *, bin_size=None):
    """Global Getis-Ord General G per gene at one physical radius, on cells -> (G,).

    The single-cell entry point beside :func:`morans_global_genes` and :func:`geary_c_genes`; the
    curve bank's lattice form is :func:`general_g_multiscale`. Binary weights within ``radius_um``,
    the focal cell excluded, so it is the statistic ``esda.G`` computes on a binary distance band.
    The field is not centred: X must be non-negative."""
    from .spatial_sc import general_g_global  # lazy
    return general_g_global(xy, X, radius_um, bin_size=bin_size)


def geary_c_genes(xy, X, radius_um, *, reduced_rank=False, rank=30, bin_size=None):
    """Global Geary's C per gene -> (G,). ``reduced_rank`` toggles the denoised rank-k reconstruction."""
    from .spatial_sc import geary_global  # lazy
    return geary_global(xy, X, radius_um, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def variogram_genes(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None):
    """Disc-based semivariance γ(r) per gene -> (n_radii, G). ``reduced_rank=True`` removes the noise
    nugget (rank-k gene sheds unstructured variance)."""
    from .spatial_sc import variogram  # lazy
    return variogram(xy, X, radii, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)


def power_threshold_filter(field, *, keep_frac=0.9, mask=None):
    """Hard spectral-power threshold on a 2-D field: keep the fewest frequency components carrying
    ``keep_frac`` of the total AC power (Parseval), zero the rest, invert. A compact frequency-domain
    denoiser/compressor — demonstrates the engine supports general spectral MANIPULATIONS, not just
    convolution. It is the spatial-frequency dual of reduced rank: reduced rank truncates in the
    gene/SVD basis (shared components across genes); this truncates in the spatial-frequency basis
    (dominant scales within one gene).

    Returns (filtered_field, frac_coeffs_kept). ``mask`` (bool) restricts the mean and output support.
    """
    field = np.asarray(field, float)
    m = np.ones_like(field, bool) if mask is None else np.asarray(mask, bool)
    mu = field[m].mean() if m.any() else 0.0
    f = np.where(m, field - mu, 0.0)
    F = _fft.fft2_hp(f)
    P = np.abs(F.ravel()) ** 2
    order = np.argsort(P)[::-1]                                    # high → low power
    cum = np.cumsum(P[order])
    n = int(np.searchsorted(cum, keep_frac * cum[-1]) + 1) if cum[-1] > 0 else 0
    keep = np.zeros(P.size, bool); keep[order[:n]] = True
    keep = keep.reshape(F.shape)
    H_, W_ = F.shape                                              # keep each mode's Hermitian partner too, so
    keep = keep | keep[np.ix_((-np.arange(H_)) % H_, (-np.arange(W_)) % W_)]   # np.real(ifft2) drops no power
    out = np.real(_fft.ifft2_hp(np.where(keep, F, 0))) + mu
    return np.where(m, out, 0.0), int(keep.sum()) / keep.size


def power_spectrum_scale_genes(xy, X, *, bin_size=None, chunk=128):
    """Per-gene characteristic spatial scale + anisotropy from the 2-D power spectrum (Wiener–Khinchin).

    Rasterize each gene, take |FFT2|² (broadband noise floor subtracted), and summarise the DC-excluded
    2-D spectrum: characteristic wavelength = 1 / (power-weighted mean radial frequency) [µm]; anisotropy =
    1 − λ_min/λ_max of the power-weighted frequency covariance. Returns (scale_um (G,), anisotropy (G,)).

    HONEST SCOPE (validated 2026-07-02): valid on CLEAN, full-support fields (synthetic GRF: scale is
    monotone in the true field scale — see tests). On REAL masked ST it is UNRELIABLE as an independent
    scale estimator: every gene's field shares the same irregular tissue support, so the mask's own
    spectrum dominates every periodogram identically and per-gene differences wash out (empirically,
    RAPSD scale vs the Moran-scalogram correlation length gave Spearman ρ≈0.10 on MERFISH brain). The
    mask-NORMALIZED neighbourhood statistics (Moran scalogram, variogram) are the correct scale estimators
    on real ST — this function is kept for clean/stationary fields and as documentation of the negative.
    """
    from .adaptive import grid_index, autotune_grid  # lazy
    xy = np.asarray(xy, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if bin_size is None:
        bin_size = autotune_grid(xy, 50.0)["bin_um"]
    row, col, H, W, pitch = grid_index(xy, bin_size)
    fy = np.fft.fftfreq(H, d=pitch); fx = np.fft.fftfreq(W, d=pitch)      # cycles / µm
    KX, KY = np.meshgrid(fx, fy)
    K = np.sqrt(KX ** 2 + KY ** 2).ravel()
    kx, ky = KX.ravel(), KY.ravel()
    pos = K > 0                                                          # exclude DC
    cnt = np.zeros((H, W)); np.add.at(cnt, (row, col), 1.0); occ = cnt > 0
    G = X.shape[1]; scale = np.empty(G); aniso = np.empty(G)
    for s in range(0, G, chunk):
        Xi = X[:, s:s + chunk]; m = Xi.shape[1]
        val = np.zeros((H, W, m)); np.add.at(val, (row, col), Xi)
        g = np.zeros_like(val); g[occ] = val[occ] / cnt[occ][:, None]
        g[occ] -= g[occ].mean(0)                                         # remove per-gene DC
        g[~occ] = 0.0
        P = (np.abs(_fft.fft2_hp(g, axes=(0, 1))) ** 2).reshape(-1, m)
        floor = P[K > 0.6 * K.max()].mean(0)                             # broadband noise floor per gene
        P = np.maximum(P - floor, 0.0)                                   # isolate SIGNAL power
        w = P[pos]; tot = w.sum(0) + 1e-12
        kc = (K[pos][:, None] * w).sum(0) / tot                          # power-weighted mean freq
        scale[s:s + m] = 1.0 / np.maximum(kc, 1e-9)
        cxx = (kx[pos][:, None] ** 2 * w).sum(0) / tot
        cyy = (ky[pos][:, None] ** 2 * w).sum(0) / tot
        cxy = (kx[pos][:, None] * ky[pos][:, None] * w).sum(0) / tot
        tr = cxx + cyy; disc = np.sqrt(np.maximum((tr / 2) ** 2 - (cxx * cyy - cxy ** 2), 0))
        l1 = tr / 2 + disc; l2 = tr / 2 - disc
        aniso[s:s + m] = 1.0 - l2 / np.maximum(l1, 1e-12)
    return scale, aniso


def rasterize_auto(xy, values, r_min_um, *, bin_size=None):
    """Auto-choose a bin (:func:`spectrast.adaptive.autotune_grid`) and rasterize ``values`` to a grid
    for the AGGREGATE/grid functions above (global Moran, colocalization matrix, neighbourhood
    enrichment, smoothing, ...), which are robust to rasterization and need no fusion.

    Returns (grid, mask, bin_um). ``values`` may be (N,) or (N, G).
    """
    from .adaptive import autotune_grid, grid_index  # lazy
    xy = np.asarray(xy, float)
    V = np.asarray(values, float)
    if V.ndim == 1:
        V = V[:, None]
    if bin_size is None:
        bin_size = autotune_grid(xy, r_min_um, probe_values=V[:, 0])["bin_um"]
    row, col, H, W, _ = grid_index(xy, bin_size)
    grid = np.zeros((H, W, V.shape[1])); cnt = np.zeros((H, W))
    np.add.at(grid, (row, col), V); np.add.at(cnt, (row, col), 1.0)
    occ = cnt > 0
    grid[occ] /= cnt[occ][:, None]
    return grid, occ, float(bin_size)

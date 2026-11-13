"""Regression tests for the global autocorrelation statistics (Moran's I, Geary's C).

Locks the convention fix (2026-07): Geary's C must use ROW-STANDARDIZED weights (consistent with the
mean-neighbour Moran here and with squidpy), so C ≈ 1 − I. The prior BINARY-weight Geary passed the
r=1 accuracy test (uniform degree, where binary ≈ row-standardized) but at a physical radius on real
coordinates gave off-scale values (~[4.8, 8.0]) and mis-ranked genes. These tests run at VARIABLE
degree (a finite point cloud → edge cells have fewer neighbours), where the two conventions diverge —
so they would have caught the bug. Dependency-free (no squidpy). Run: `pytest tests/test_autocorr.py -q`.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter

from spectrast.spatial import morans_global_genes, geary_c_genes

RADIUS = 50.0


def _dataset(n=3000, ngenes_per_scale=6, seed=0):
    """Point cloud (variable degree at edges) with genes spanning a range of spatial scales + noise,
    so Moran/Geary vary across genes and the C≈1−I relationship is non-degenerate."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 500.0, (n, 2))
    G = 96
    ij = ((xy - xy.min(0)) / np.ptp(xy, 0) * (G - 1)).astype(int)
    cols = []
    for sc in (1.0, 3.0, 8.0, 20.0):                     # smooth fields at increasing scale
        for k in range(ngenes_per_scale):
            field = gaussian_filter(rng.standard_normal((G, G)), sc, mode="wrap")
            cols.append(field[ij[:, 1], ij[:, 0]])
    for _ in range(ngenes_per_scale):                    # pure-noise genes (low autocorrelation)
        cols.append(rng.standard_normal(n))
    X = np.stack(cols, 1).astype(np.float32)
    return xy, X


def _hetero(n=4000, seed=0):
    """Heterogeneous-DENSITY cloud (dense clusters + sparse background) → strongly variable degree, the
    regime where binary vs row-standardized weights and Gi vs Gi* diverge (the review's bug regime)."""
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.uniform(0, 500, (n // 2, 2)),
                    rng.normal([150, 150], 22, (n // 4, 2)), rng.normal([370, 320], 18, (n // 4, 2))])
    G = 90
    ij = ((xy - xy.min(0)) / np.ptp(xy, 0) * (G - 1)).astype(int)
    X = np.stack([gaussian_filter(rng.standard_normal((G, G)), s, mode="wrap")[ij[:, 1], ij[:, 0]]
                  for s in np.r_[np.repeat([1.5, 4, 10], 8), np.full(6, 0.01)]], 1).astype(np.float32)
    return xy, X


def _radius_graph_reduce(xy, r):
    """Independent row-standardized neighbour reduction on the exact radius graph (brute force, no spectrast)."""
    from scipy.spatial import cKDTree
    t = cKDTree(xy); nb = t.query_ball_point(xy, r)
    deg = np.array([max(len(x) - 1, 0) for x in nb])          # neighbours excluding self
    return nb, deg


def test_geary_scale_is_physical():
    """Geary's C lies in ~[0, 2] (row-standardized). The binary-weight bug produced ~[4.8, 8.0]."""
    xy, X = _dataset()
    C = geary_c_genes(xy, X, RADIUS, reduced_rank=False)
    assert np.all(np.isfinite(C))
    assert C.min() > 0.0 and C.max() < 2.0, f"Geary out of [0,2]: [{C.min():.2f}, {C.max():.2f}]"


def test_grid_moran_lisa_row_standardized_at_variable_degree():
    """Grid Moran & LISA use ROW-standardized weights: mean(LISA) == global Moran (was ~mean-degree× off
    under binary weights). Checked on a masked grid with holes ⇒ variable degree (interior + boundary)."""
    from spectrast.spatial import morans_i, local_morans_i_batch, neighbor_kernel
    rng = np.random.default_rng(0)
    grid = np.stack([gaussian_filter(rng.standard_normal((80, 80)), s) for s in (2.0, 5.0, 12.0)], -1)
    mask = np.ones((80, 80), bool); mask[20:30, 20:60] = False; mask[55:70, 10:25] = False  # holes → var. degree
    gI = morans_i(grid, mask, radius=3)
    lisa = local_morans_i_batch(grid, mask, radius=3)
    lmean = np.array([lisa[..., g][mask].mean() for g in range(grid.shape[-1])])
    assert np.allclose(lmean, gI, atol=1e-6), f"mean(LISA) != global Moran: {lmean} vs {gI}"


def test_getis_is_gistar_includes_self_vs_bruteforce():
    """Single-cell Getis is Gi* (INCLUDES the focal cell), matching an independent brute-force Gi* on the
    exact radius graph, at heterogeneous degree. The prior Gi (self-excluded) mismatched at hotspots."""
    from spectrast.spatial import getis_ord_genes
    xy, X = _hetero()
    z = (X - X.mean(0)) / (X.std(0) + 1e-9); N = len(xy)
    nb, _ = _radius_graph_reduce(xy, RADIUS)
    gs = [0, 10, 20, 29]
    Wi = np.array([len(x) for x in nb])                       # INCLUDES self (Gi*)
    bf = np.stack([np.array([z[x, g].sum() for x in nb]) /
                   np.sqrt(np.maximum((N * Wi - Wi ** 2) / (N - 1), 1e-9)) for g in gs], 1)
    q = getis_ord_genes(xy, X, RADIUS, gs, reduced_rank=False)
    assert pearsonr(q.ravel(), bf.ravel())[0] > 0.95, "single-cell Getis is not Gi* (self-inclusion)"


@pytest.mark.parametrize("tool", ["moran", "geary", "getis"])
def test_reduced_rank_exact_at_full_rank_tall(tool):
    """Reduced rank reconstructs the direct statistic EXACTLY at full rank, for a TALL panel (N>G, the
    Xenium/MERFISH regime) at the shipped default bin (autotune). Locks the rank-cap + field-independent
    bin fixes — this differed by ~1e-2 to 0.2 before."""
    from spectrast.spatial import getis_ord_genes
    rng = np.random.default_rng(1)
    N, G = 260, 14                                            # tall
    xy = rng.uniform(0, 260, (N, 2)); gg = 44
    ij = ((xy - xy.min(0)) / np.ptp(xy, 0) * (gg - 1)).astype(int)
    X = np.stack([gaussian_filter(rng.standard_normal((gg, gg)), s, mode="wrap")[ij[:, 1], ij[:, 0]]
                  for s in np.linspace(1, 9, G)], 1).astype(np.float32)
    if tool == "moran":
        d = morans_global_genes(xy, X, RADIUS, reduced_rank=False); r = morans_global_genes(xy, X, RADIUS, reduced_rank=True, rank=G)
    elif tool == "geary":
        d = geary_c_genes(xy, X, RADIUS, reduced_rank=False); r = geary_c_genes(xy, X, RADIUS, reduced_rank=True, rank=G)
    else:
        d = getis_ord_genes(xy, X, RADIUS, [0, 6, 13], reduced_rank=False); r = getis_ord_genes(xy, X, RADIUS, [0, 6, 13], reduced_rank=True, rank=G)
    assert np.abs(d - r).max() < 1e-4, f"{tool} reduced-rank not exact at full rank (tall): {np.abs(d-r).max():.1e}"


def test_geary_complements_moran():
    """Row-standardized Geary satisfies C ≈ 1 − I, so it strongly anti-correlates with Moran across
    genes (at variable degree). The binary bug broke this (rank corr collapsed)."""
    xy, X = _dataset()
    I = morans_global_genes(xy, X, RADIUS, reduced_rank=False)
    C = geary_c_genes(xy, X, RADIUS, reduced_rank=False)
    rho = spearmanr(I, 1.0 - C).correlation
    assert rho > 0.9, f"Geary not complementary to Moran (rho={rho:.3f}) — weight convention regressed"


def test_reduced_rank_geary_identity():
    """The reduced-rank Geary reconstruction (k×k forms) must also satisfy C ≈ 1 − I against the
    reduced-rank Moran — a check on the embedding-based formula."""
    xy, X = _dataset()
    I = morans_global_genes(xy, X, RADIUS, reduced_rank=True, rank=20)
    C = geary_c_genes(xy, X, RADIUS, reduced_rank=True, rank=20)
    assert C.min() > 0.0 and C.max() < 2.0
    rho = spearmanr(I, 1.0 - C).correlation
    assert rho > 0.9, f"reduced-rank Geary inconsistent with reduced-rank Moran (rho={rho:.3f})"


@pytest.mark.parametrize("tool", [morans_global_genes, geary_c_genes])
def test_degenerate_inputs_are_finite(tool):
    """All-zero and constant genes (undefined autocorrelation) must return finite values, not NaN/inf —
    a common crash/regression site (_standardize_cols divides by std)."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (800, 2))
    X = np.zeros((800, 3), np.float32)
    X[:, 0] = 5.0                                             # constant gene
    X[:, 1] = gaussian_filter(rng.standard_normal((40, 40)), 3, mode="wrap")[
        ((xy[:, 1] - xy[:, 1].min()) / np.ptp(xy[:, 1]) * 39).astype(int),
        ((xy[:, 0] - xy[:, 0].min()) / np.ptp(xy[:, 0]) * 39).astype(int)]   # one real gene
    # X[:,2] left all-zero
    out = tool(xy, X, RADIUS, reduced_rank=False)
    assert np.all(np.isfinite(out)), f"{tool.__name__} not finite on constant/zero genes: {out}"


def test_single_gene_shape():
    """A single-gene (N,1) panel must not break the batched machinery."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 200, (600, 2))
    X = rng.standard_normal((600, 1)).astype(np.float32)
    assert morans_global_genes(xy, X, RADIUS, reduced_rank=False).shape == (1,)


def test_moran_positive_for_smooth_negligible_for_noise():
    """Sanity: smooth (large-scale) genes have clearly higher Moran's I than pure-noise genes."""
    xy, X = _dataset()
    I = morans_global_genes(xy, X, RADIUS, reduced_rank=False)
    smooth = I[18:24].mean()      # scale-20 block
    noise = I[24:30].mean()       # noise block
    assert smooth > noise + 0.1, f"Moran did not separate smooth ({smooth:.3f}) from noise ({noise:.3f})"


# --- memory pass (2026-07): the direct paths must NOT densify the full
#     matrix; sparse and dense input must return identical values (the Atera densification bug class) --------

def test_direct_paths_identical_sparse_vs_dense():
    """M1/M2: the reduced_rank=False selected-gene paths (local Moran/Getis/local Geary/bivariate) now slice
    columns while sparse, and the whole-transcriptome paths (global Moran/Geary/variogram) stream 256-gene
    blocks — none densifies the full (N×G) matrix. All must be byte-identical on sparse vs dense input (the
    refactor is memory-only). Guards against a regression of the Atera 'densified the expression matrix' bug."""
    import scipy.sparse as sp
    from spectrast.spatial import (morans_global_genes, geary_c_genes, variogram_genes,
                             local_morans_i_genes, getis_ord_genes, local_geary_genes, bivariate_spatial_matrix)
    xy, Xd = _hetero(seed=1)
    Xs = sp.csr_matrix(Xd)
    gs = [0, 1, 5, 12, 20]
    pairs = {
        "morans_global": (morans_global_genes(xy, Xd, RADIUS), morans_global_genes(xy, Xs, RADIUS)),
        "geary_global":  (geary_c_genes(xy, Xd, RADIUS), geary_c_genes(xy, Xs, RADIUS)),
        "variogram":     (variogram_genes(xy, Xd, [20., 50., 100.]), variogram_genes(xy, Xs, [20., 50., 100.])),
        "local_moran":   (local_morans_i_genes(xy, Xd, RADIUS, gs), local_morans_i_genes(xy, Xs, RADIUS, gs)),
        "getis":         (getis_ord_genes(xy, Xd, RADIUS, gs), getis_ord_genes(xy, Xs, RADIUS, gs)),
        "local_geary":   (local_geary_genes(xy, Xd, RADIUS, gs), local_geary_genes(xy, Xs, RADIUS, gs)),
        "bivariate":     (bivariate_spatial_matrix(xy, Xd, gs, RADIUS), bivariate_spatial_matrix(xy, Xs, gs, RADIUS)),
    }
    for name, (a, b) in pairs.items():
        m = np.abs(np.asarray(a) - np.asarray(b)).max()
        assert np.allclose(a, b, atol=1e-8, rtol=0), f"{name}: sparse != dense (max |Δ|={m:.2e})"


def test_guard_densify_raises_only_over_budget():
    """M4: the densify guard turns a silent OOM into an explicit MemoryError when the (already column-sliced)
    subset would exceed the live free-RAM budget, and is silent for a request that fits."""
    import scipy.sparse as sp
    from spectrast.memory import guard_densify
    X = sp.random(4000, 300, density=0.1, format="csc")
    assert guard_densify(X) >= 0.0                            # comfortably fits -> returns GB, no raise
    with pytest.raises(MemoryError):
        guard_densify(X, frac=1e-12)                          # forced ~0 budget -> explicit raise


# --- review-2 correctness fixes (2026-07): grid stats on porous masks / non-default kernels ----------

def _brute_geary_grid(grid, mask, r=1):
    """Textbook row-standardized Geary's C on a grid: C = (N-1)/(2*S0) * Σ_ij w_ij(x_i-x_j)² / Σ z²,
    S0 = #occupied cells with ≥1 neighbour (a neighbourless cell contributes 0)."""
    H, W = mask.shape; x = grid
    zbar = x[mask].mean(); den = ((x[mask] - zbar) ** 2).sum()
    num = 0.0; S0 = 0
    for (i, j) in np.argwhere(mask):
        nb = [(i + di, j + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
              if (di or dj) and di * di + dj * dj <= r * r and 0 <= i + di < H and 0 <= j + dj < W and mask[i + di, j + dj]]
        if not nb:
            continue
        S0 += 1; w = 1.0 / len(nb)
        num += sum(w * (x[i, j] - x[ii, jj]) ** 2 for (ii, jj) in nb)
    return ((mask.sum() - 1) / (2.0 * S0)) * (num / den)


def test_geary_porous_mask_matches_bruteforce():
    """Geary's C on a mask with ISOLATED (neighbourless) occupied cells matches textbook brute-force.
    The bug (2026-07) left a spurious raw x_i² in the numerator for neighbourless cells (W1=0 zeroed the
    mean-neighbour terms) and normalized by N instead of S0 — inflating C on porous/holey tissue."""
    from spectrast.spatial import gearys_c
    rng = np.random.default_rng(0); H = W = 34
    occ = rng.random((H, W)) < 0.55                          # ~porous → many isolated cells at r=1
    val = rng.random((H, W)) * np.arange(1, W + 1)[None, :]  # heterogeneous magnitude (amplifies the bug)
    grid = (val * occ).astype(float)
    n_iso = sum(1 for (i, j) in np.argwhere(occ)
                if not any(occ[i + di, j + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                           if (di or dj) and di * di + dj * dj <= 1 and 0 <= i + di < H and 0 <= j + dj < W))
    assert n_iso > 0, "test fixture must contain isolated cells"
    code = gearys_c(grid[..., None], occ, radius=1)[0]
    assert np.isclose(code, _brute_geary_grid(grid, occ, 1), atol=1e-6), "Geary wrong on porous mask"


def _brute_gistar_grid(grid, mask, radius, kind):
    from spectrast.spatial import neighbor_kernel
    H, W = mask.shape; x = grid; n = mask.sum()
    xbar = x[mask].mean(); S = np.sqrt((x[mask] ** 2).mean() - xbar ** 2)
    k = neighbor_kernel(radius, kind).copy(); k[radius, radius] = 1.0
    out = np.full((H, W), np.nan)
    for (i, j) in np.argwhere(mask):
        sw = sw2 = lw = 0.0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                w = k[radius + di, radius + dj]
                if w == 0:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < H and 0 <= jj < W and mask[ii, jj]:
                    sw += w; sw2 += w * w; lw += w * x[ii, jj]
        d = S * np.sqrt(max((n * sw2 - sw ** 2) / (n - 1), 0.0))
        out[i, j] = (lw - xbar * sw) / d if d > 0 else np.nan
    return out


@pytest.mark.parametrize("kind", ["disc", "gaussian"])
def test_getis_gistar_variance_both_kernels(kind):
    """Gi* denominator uses the true Σw² (=Σw only for a binary disc). The bug substituted Σw for Σw²,
    silently wrong for kind='gaussian' (continuous weights) by ~2 z-units; disc must stay exact."""
    from spectrast.spatial import getis_ord
    rng = np.random.default_rng(3); H = W = 24
    mask = np.ones((H, W), bool)
    grid = rng.random((H, W)) * np.arange(1, W + 1)[None, :]
    code = getis_ord(grid, mask, radius=3, kind=kind)
    bf = _brute_gistar_grid(grid, mask, 3, kind)
    assert np.nanmax(np.abs(code - bf)) < 1e-6, f"Gi* ({kind}) variance term wrong"


def test_bivariate_local_pvals_never_zero():
    """Permutation p-values use (r+1)/(n+1), so p is never exactly 0 (a p=0 → q=0 'infinitely
    significant' CCI hotspot). The bug used r/n, flooring at 0."""
    from spectrast.spatial import bivariate_local_pvals
    rng = np.random.default_rng(0); H = W = 28
    mask = np.ones((H, W), bool)
    lig = rng.random((H, W)); rec = 0.9 * lig + 0.1 * rng.random((H, W))   # strong coupling → obs beats null
    n = 50
    _, pv = bivariate_local_pvals(lig, rec, mask, radius=4, metric="cosine", n_perms=n)
    assert np.nansum(pv == 0.0) == 0, "p-values hit exactly 0 (missing (r+1)/(n+1))"
    assert np.nanmin(pv) >= 1.0 / (n + 1) - 1e-12, "p-value below the 1/(n+1) floor"


def _brute_moran_grid(field, mask, r=1):
    """Textbook row-standardized global Moran on a grid: I = Σ_i z_i·meanN(z_i) / Σ_i z_i²,
    meanN = mean of z over occupied neighbours (neighbourless cells contribute 0)."""
    H, W = mask.shape; x = field
    zbar = x[mask].mean(); z = (x - zbar) * mask
    num = 0.0
    for (i, j) in np.argwhere(mask):
        nb = [(i + di, j + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
              if (di or dj) and di * di + dj * dj <= r * r and 0 <= i + di < H and 0 <= j + dj < W and mask[i + di, j + dj]]
        if nb:
            num += z[i, j] * np.mean([z[ii, jj] for (ii, jj) in nb])
    return num / (z[mask] ** 2).sum()


def test_grid_moran_absolute_vs_bruteforce():
    """Global grid Moran has an INDEPENDENT absolute reference (not just the mean(LISA)==I identity, which
    holds for any consistent weighting). A common normalization error would move I here but not the identity."""
    from spectrast.spatial import morans_i
    rng = np.random.default_rng(0); H = W = 30
    field = np.stack([gaussian_filter(rng.standard_normal((H, W)), 4.0)], -1)[..., 0]
    mask = np.ones((H, W), bool); mask[8:14, 6:20] = False        # hole → variable degree
    code = morans_i(field[..., None], mask, radius=1)[0]
    assert np.isclose(code, _brute_moran_grid(field, mask, 1), atol=1e-6), "grid Moran != brute row-standardized"


def test_multiscale_moran_matches_single_radius_on_porous_mask():
    """B1 (review 2026-07): morans_i_multiscale must be ROW-STANDARDIZED like morans_i — equal to the
    single-radius primitive (and the brute-force) at every radius on a MASKED/porous grid. The prior
    binary-weight (msum/S0) multiscale form drifted from morans_i at every tissue boundary and broke the
    mean(LISA)=global-Moran identity the Methods declare, silently changing the scalogram / correlation_length
    shape on real (bordered) tissue. Interior-only fields would NOT catch it (binary≈row-standardized there),
    so this runs on holey tissue with variable degree."""
    from spectrast.spatial import morans_i, morans_i_multiscale
    rng = np.random.default_rng(0); H = W = 40
    grid = np.stack([gaussian_filter(rng.standard_normal((H, W)), s) for s in (2.0, 5.0, 10.0)], -1)
    mask = np.ones((H, W), bool); mask[10:16, 8:28] = False; mask[28:36, 6:14] = False   # holes → variable degree
    radii = [1, 2, 4]
    ms = morans_i_multiscale(grid, mask, radii)                    # (n_radii, G)
    for si, r in enumerate(radii):
        single = morans_i(grid, mask, radius=r)
        assert np.allclose(ms[si], single, atol=1e-9), f"multiscale r={r} != morans_i (row-standardization)"
    for g in range(grid.shape[-1]):                                # independent brute-force reference at r=1
        assert np.isclose(ms[0, g], _brute_moran_grid(grid[..., g], mask, 1), atol=1e-6)


@pytest.mark.parametrize("tool", ["moran", "geary", "getis"])
def test_reduced_rank_rank_above_cap_is_clamped(tool):
    """Passing rank > min(N-1, G) must be CLAMPED (not error / not diverge): RR still equals direct. The
    review found removing the rank cap left every test green — this exercises rank beyond the cap."""
    from spectrast.spatial import getis_ord_genes
    rng = np.random.default_rng(2)
    N, G = 90, 8                                                 # small so min(N-1,G)=G=8
    xy = rng.uniform(0, 120, (N, 2)); gg = 30
    ij = ((xy - xy.min(0)) / np.ptp(xy, 0) * (gg - 1)).astype(int)
    X = np.stack([gaussian_filter(rng.standard_normal((gg, gg)), s, mode="wrap")[ij[:, 1], ij[:, 0]]
                  for s in np.linspace(1, 8, G)], 1).astype(np.float32)
    over = G + 25                                                # far above the min(N-1,G)=8 cap
    if tool == "moran":
        d = morans_global_genes(xy, X, RADIUS, reduced_rank=False); r = morans_global_genes(xy, X, RADIUS, reduced_rank=True, rank=over)
    elif tool == "geary":
        d = geary_c_genes(xy, X, RADIUS, reduced_rank=False); r = geary_c_genes(xy, X, RADIUS, reduced_rank=True, rank=over)
    else:
        d = getis_ord_genes(xy, X, RADIUS, [0, 4, 7], reduced_rank=False); r = getis_ord_genes(xy, X, RADIUS, [0, 4, 7], reduced_rank=True, rank=over)
    assert np.abs(d - r).max() < 1e-4, f"{tool} RR!=direct with rank>cap (cap not clamping): {np.abs(d-r).max():.1e}"


@pytest.mark.parametrize("rr", [False, True])
def test_point_cloud_geary_gates_isolated_cells(rr):
    """The SHIPPED point-cloud Geary (geary_global, direct AND reduced-rank) must gate neighbourless cells
    out of the numerator and normalize by S0 — like the grid gearys_c fix. The original summed the raw z_i²
    of isolated cells and divided by N, inflating C where a rasterized bin has no neighbour within radius."""
    from spectrast.spatial_sc import geary_global, _neighbor_sum_count, _standardize_cols
    from spectrast.adaptive import autotune_grid
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal([50, 50], 7, (700, 2)), rng.uniform(0, 400, (120, 2))])   # scattered lone cells
    X = np.column_stack([np.sin(xy[:, 0] / 50) + 0.3 * rng.standard_normal(len(xy)) for _ in range(4)])
    r = 30.0; N = len(xy); b = float(autotune_grid(xy, r)["bin_um"])
    Z = _standardize_cols(X); K = Z.shape[1]
    nsum, ncnt, _ = _neighbor_sum_count(xy, np.concatenate([Z, Z ** 2], 1), r, b)
    has = ncnt > 0; inv = (1.0 / np.maximum(ncnt, 1))[:, None]
    br = Z ** 2 - 2.0 * Z * (nsum[:, :K] * inv) + (nsum[:, K:] * inv)
    den = (Z ** 2).sum(0)
    gated = (N - 1) / (2.0 * has.sum()) * (br * has[:, None]).sum(0) / den            # CORRECT
    ungated = (N - 1) / (2.0 * N) * br.sum(0) / den                                  # the OLD bug
    assert int((~has).sum()) >= 3, "fixture must have isolated cells (else the fix is untested)"
    assert not np.allclose(gated, ungated, atol=1e-3), "fixture must make gated != ungated (non-vacuous)"
    out = geary_global(xy, X, r, reduced_rank=rr, rank=K)
    assert np.allclose(out, gated, atol=1e-6), f"point-cloud Geary (rr={rr}) not gating isolated cells"


def test_autotune_rank_accepts_sparse():
    """autotune_rank (and gavish_donoho_rank) must accept sparse X without densifying the whole matrix —
    the same np.asarray(sparse,float) crash fixed in concordance, in a sibling on the whole-transcriptome
    path. On STRUCTURED data (where a signal rank exists) sparse matches dense closely."""
    import scipy.sparse as sp
    from spectrast.spatial import autotune_rank_genes
    from spectrast.simulate import simulate_st
    xy = np.random.default_rng(0).uniform(0, 700, (3000, 2))
    Xd = np.clip(simulate_st(xy, n_factors=8, n_genes=150, noise=1.0, dropout=0.3, seed=1)["X_noisy"], 0, None)
    rd = autotune_rank_genes(xy, Xd, 40.0, bin_size=8.0)["rank"]
    rs = autotune_rank_genes(xy, sp.csr_matrix(Xd.astype(np.float32)), 40.0, bin_size=8.0)["rank"]  # no crash
    assert isinstance(rs, int) and rs >= 1, "sparse autotune_rank did not return a valid rank"
    assert abs(rd - rs) <= 8, f"sparse rank {rs} far from dense {rd} on structured data"


def test_concordance_accepts_sparse():
    """concordance() must accept a sparse X (whole-transcriptome/Xenium path) and match dense — the bug
    did np.asarray(sparse,float) → ValueError while every sibling guards sp.issparse."""
    import scipy.sparse as sp
    from spectrast.spatial_sc import concordance
    rng = np.random.default_rng(0); N, G = 350, 6
    xy = rng.random((N, 2)) * 200
    Xd = (rng.random((N, G)) * (rng.random((N, G)) < 0.3)).astype(float)
    d_sparse = concordance(xy, sp.csr_matrix(Xd), 40.0)
    d_dense = concordance(xy, Xd, 40.0)
    assert np.allclose(d_sparse["I_direct"].values, d_dense["I_direct"].values), "sparse≠dense concordance"

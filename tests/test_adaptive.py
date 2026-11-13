"""Tests for the single-cell adaptive rasterization + fusion (src/spectrast/adaptive.py + the
spectrast.spatial.*_sc entry points). Run: `pytest` (or `pytest tests/test_adaptive.py -q`).

Guarantees locked here:
  * every per-cell tool routes through the generic path and returns one value per cell,
  * at a deliberately coarse bin, adaptive FUSION improves concordance with the exact graph
    over the plain FFT and reaches high agreement,
  * adaptive=False returns the plain FFT (no cells altered),
  * the graph-free proxy predicts true per-cell error,
  * outputs are deterministic.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

from spectrast.adaptive import (adaptive_local_stat, combine_moran, combine_getis, combine_bivariate,
                          combine_diversity, combine_count, _exact_neighbor_reduce,
                          _fft_neighbor_reduce, autotune_grid)
from spectrast import spatial

RADIUS = 40.0
COARSE_BIN = 20.0      # r/b = 2 -> real error for fusion to fix


def test_coincident_cells_fused_to_exact():
    """B2 (review 2026-07): a tight cluster of near-COINCIDENT cells (closer than bin/2, so they co-bin at
    both b and b/2) is dropped by the home-bin-excluded FFT disc and is invisible to the b-vs-b/2 proxy — the
    plain FFT can return a sign-flipped local Moran there. The fusion must force every occupancy>1 cell onto
    the exact graph, so the fused value matches the exact-graph local Moran in sign and magnitude."""
    rng = np.random.default_rng(0)
    bg = rng.uniform(0, 400, (1500, 2))                                   # background
    clump = np.array([200.0, 200.0]) + rng.normal(0, 0.02, (6, 2))       # 6 cells ~0.02µm apart -> co-bin
    xy = np.vstack([bg, clump])
    z = np.r_[rng.standard_normal(1500), np.full(6, 5.0)]                 # strong shared signal in the clump
    zc = (z - z.mean()) / (z.std() + 1e-9)
    ci = np.arange(1500, 1506)                                           # the clump cells
    # exact-graph local Moran (self-only exclusion) = ground truth for the clump cells
    nsx, ncx = _exact_neighbor_reduce(xy, zc, RADIUS, sub_idx=ci)
    exact = combine_moran(nsx, ncx, zc[ci])
    from spectrast.adaptive import adaptive_local_moran
    fused = adaptive_local_moran(xy, z, RADIUS, bin_size=5.0)[ci]         # bin=5µm -> clump occupancy=6
    plain = adaptive_local_moran(xy, z, RADIUS, bin_size=5.0, fuse=False)[ci]
    assert np.all(np.sign(fused) == np.sign(exact)), f"fused sign != exact: {fused} vs {exact}"
    assert np.allclose(fused, exact, atol=1e-6), f"fused != exact on co-binned cells: {fused} vs {exact}"
    # and this genuinely mattered: the plain FFT was wrong (else the test is vacuous)
    assert not np.allclose(plain, exact, atol=1e-3), "fixture non-vacuous: plain FFT must differ from exact"


@pytest.fixture(scope="module")
def data():
    """Random point cloud with a smooth spatial field + cell-type labels from it."""
    rng = np.random.default_rng(0)
    n = 3000
    xy = rng.uniform(0, 1000, (n, 2))
    # smooth field: sum of a few 2D sinusoids -> genuine spatial autocorrelation
    f = (np.sin(xy[:, 0] / 130) + np.cos(xy[:, 1] / 110) + 0.5 * np.sin((xy[:, 0] + xy[:, 1]) / 90))
    x = np.clip(f + rng.normal(0, 0.3, n) + 2, 0, None)     # non-negative "expression"
    y = np.clip(np.cos(xy[:, 0] / 100) + rng.normal(0, 0.3, n) + 2, 0, None)
    labels = np.array(["A", "B", "C"])[np.clip(((f + 2) / 1.5).astype(int), 0, 2)]
    return dict(xy=xy, x=x, y=y, labels=labels, tree=cKDTree(xy))


def _exact(xy, fields, combine, tree):
    ns, nc = _exact_neighbor_reduce(xy, fields, RADIUS, tree=tree)
    return np.asarray(combine(ns, nc, np.asarray(fields, float)), float)


def _rho(a, b):
    # nan-aware: isolated cells (no neighbour in radius) are NaN in BOTH exact and FFT/fused; compare the
    # defined cells (M14: undefined local statistics are NaN, not a spurious 0).
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).correlation


# --------------------------------------------------------------------- per-tool fusion
@pytest.mark.parametrize("tool", ["moran", "getis", "bivariate", "diversity", "coloc"])
def test_fusion_improves_and_recovers(data, tool):
    xy, tree = data["xy"], data["tree"]
    if tool == "moran":
        v = data["x"]; z = (v - v.mean()) / v.std()
        fields, combine = z, combine_moran
    elif tool == "getis":
        v = data["x"]; fields = v
        combine = combine_getis(v.mean(), v.std() + 1e-9, len(v))
    elif tool == "bivariate":
        zx = (data["x"] - data["x"].mean()) / data["x"].std()
        zy = (data["y"] - data["y"].mean()) / data["y"].std()
        fields, combine = np.c_[zx, zy], combine_bivariate
    elif tool == "diversity":
        types = np.unique(data["labels"])
        fields = np.stack([(data["labels"] == t).astype(float) for t in types], 1)
        combine = combine_diversity
    else:  # coloc
        types = list(np.unique(data["labels"]))
        fields = np.stack([(data["labels"] == t).astype(float) for t in types], 1)
        combine = combine_count(types.index("B"))

    exact = _exact(xy, fields, combine, tree)
    fft_only = adaptive_local_stat(xy, fields, combine, RADIUS, bin_size=COARSE_BIN, fuse=False)
    hybrid, meta = adaptive_local_stat(xy, fields, combine, RADIUS, bin_size=COARSE_BIN,
                                       tree=tree, fuse=True, return_meta=True)
    assert hybrid.shape == (len(xy),)
    rho_fft = _rho(fft_only, exact)
    rho_hyb = _rho(hybrid, exact)
    assert rho_hyb >= rho_fft - 1e-9, f"{tool}: fusion hurt ({rho_hyb:.3f} < {rho_fft:.3f})"
    assert rho_hyb > 0.9, f"{tool}: fusion did not recover exact ({rho_hyb:.3f})"
    assert 0.0 <= meta["flagged_frac"] <= 1.0 and meta["fused"] is True


@pytest.mark.parametrize("side", [3, 5, 7])            # small-kernel branch (< _CONV_FFT_MIN_SIDE=9)
@pytest.mark.parametrize("ndim", [2, 3])
def test_sparse_direct_conv_matches_ndimage(side, ndim):
    """The cell-only sparse-direct small-kernel path must equal dense ndimage.convolve(mode='constant')
    to machine precision — for a symmetric disc AND an asymmetric random kernel, on a sparse grid, for
    2D and batched 3D stacks. This is the sparsity-aware fallback (O(M·K²) not O(N²·K²))."""
    from scipy import ndimage
    from spectrast.adaptive import _sparse_direct_conv2d, _conv2d, _disc
    rng = np.random.default_rng(side + 10 * ndim)
    H, W = 60, 48
    shape = (H, W) if ndim == 2 else (H, W, 4)
    grid = np.zeros(shape)
    occ = rng.random((H, W)) < 0.08                    # ~8% occupied -> genuinely sparse
    if ndim == 2:
        grid[occ] = rng.normal(size=occ.sum())
    else:
        grid[occ] = rng.normal(size=(occ.sum(), 4))
    for kernel in (_disc(side // 2), rng.normal(size=(side, side))):   # symmetric disc + asymmetric
        k = kernel[:, :, None] if ndim == 3 else kernel
        ref = ndimage.convolve(grid, k, mode="constant")
        got = _sparse_direct_conv2d(grid, kernel)
        assert got.shape == grid.shape
        assert np.max(np.abs(got - ref)) < 1e-10, f"sparse-direct != ndimage (side={side}, ndim={ndim})"
        assert np.max(np.abs(_conv2d(grid, kernel) - ref)) < 1e-10   # dispatcher routes small kernels here
    # dense grid: the guard must route to ndimage and still match exactly
    dense = rng.normal(size=shape)
    kd = _disc(side // 2)
    ref_d = ndimage.convolve(dense, kd[:, :, None] if ndim == 3 else kd, mode="constant")
    assert np.max(np.abs(_conv2d(dense, kd) - ref_d)) < 1e-10


@pytest.mark.parametrize("theta0", [0.0, 30.0, 60.0, 90.0, 135.0])
def test_polar_power_spectrum_recovers_orientation(theta0):
    """polar_power_spectrum recovers a known elongation angle (structure_orientation ≈ theta0) and gives
    near-zero anisotropy on an isotropic field. The frequency-domain directional readout (E7)."""
    from scipy import ndimage
    from spectrast.spatial import polar_power_spectrum
    w = np.random.default_rng(int(theta0) + 1).standard_normal((256, 256))
    f = ndimage.gaussian_filter(w, sigma=(2.0, 12.0), mode="wrap")          # elongated along axis 1
    f = ndimage.rotate(f, -theta0, reshape=False, mode="reflect", order=1)  # sign matches arctan2(KY,KX)
    m = np.ones_like(f, bool)
    r = polar_power_spectrum(f, m)
    d = abs((r["structure_orientation"] - theta0) % 180.0); d = min(d, 180.0 - d)
    assert d < 8.0, f"orientation {r['structure_orientation']:.1f} vs true {theta0} (err {d:.1f})"
    assert r["anisotropy"] > 0.3, "elongated field must read as anisotropic"
    iso = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((256, 256)), 3.0, mode="wrap")
    assert polar_power_spectrum(iso, m)["anisotropy"] < 0.15, "isotropic field must read ~0 anisotropy"


def test_spectral_bandpass_filters_and_apodizes():
    """spectral_bandpass: (i) removing low frequencies strips a macro gradient, (ii) a band isolates a
    band-limited field and rejects an out-of-band one, (iii) apodize reduces Gibbs overshoot on a step."""
    from spectrast.spatial import spectral_bandpass
    N, BIN = 200, 10.0
    yy, xx = np.mgrid[0:N, 0:N].astype(float); xx *= BIN; yy *= BIN
    m = np.ones((N, N), bool)
    # band-limited plane-wave fields
    def band(lmin, lmax, seed):
        r = np.random.default_rng(seed); f = np.zeros((N, N))
        for _ in range(50):
            lam = r.uniform(lmin, lmax); th = r.uniform(0, np.pi); ph = r.uniform(0, 2 * np.pi)
            f += np.cos(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / lam + ph)
        return (f - f.mean()) / f.std()
    fine = band(25, 55, 1); grad = 4.0 * xx / xx.max()
    obs = fine + grad
    degrad = spectral_bandpass(obs, m, lam_lo=0.0, lam_hi=300.0, bin_size=BIN)
    from scipy.stats import pearsonr
    assert pearsonr(degrad.ravel(), fine.ravel())[0] > pearsonr(obs.ravel(), fine.ravel())[0]  # gradient removed
    meso = band(140, 260, 2)
    iso = spectral_bandpass(fine + meso, m, lam_lo=15.0, lam_hi=90.0, bin_size=BIN)
    assert abs(pearsonr(iso.ravel(), fine.ravel())[0]) > 0.8       # in-band recovered
    assert abs(pearsonr(iso.ravel(), meso.ravel())[0]) < 0.2       # out-of-band rejected
    step = np.zeros((N, N)); step[:, N // 2:] = 1.0
    oi = spectral_bandpass(step, m, lam_lo=300.0, lam_hi=np.inf, bin_size=BIN, apodize=0.0).max()
    oa = spectral_bandpass(step, m, lam_lo=300.0, lam_hi=np.inf, bin_size=BIN, apodize=0.8).max()
    assert oa < oi                                                 # apodization suppresses Gibbs overshoot


def test_scale_space_blobs_recovers_size_and_position():
    """scale_space_blobs finds uniform discs of known radius, localizes them, and reads their diameter."""
    from spectrast.spatial import scale_space_blobs
    N, BIN = 240, 5.0
    yy, xx = np.mgrid[0:N, 0:N].astype(float); field = np.zeros((N, N))
    truth = [(60, 60, 12), (180, 70, 24), (120, 180, 8)]      # (cx, cy, radius_px)
    for cxp, cyp, rp in truth:
        field += (((xx - cxp) ** 2 + (yy - cyp) ** 2) <= rp ** 2).astype(float)
    m = np.ones((N, N), bool)
    r = scale_space_blobs(field, m, [20, 40, 60, 90, 130], bin_size=BIN, threshold_rel=0.3)
    assert 3 <= len(r["x_um"]) <= 5                           # ~one detection per disc (overlap-suppressed)
    for cxp, cyp, rp in truth:                                # each true disc has a nearby detection of right size
        d2 = (r["x_um"] - cxp * BIN) ** 2 + (r["y_um"] - cyp * BIN) ** 2
        j = int(np.argmin(d2))
        assert np.sqrt(d2[j]) < 30                            # localized within 30 um
        assert abs(r["diameter_um"][j] - 2 * rp * BIN) < 0.6 * rp * BIN   # diameter within ~half a radius


def test_cross_spectral_density_detects_scale_sign_flip():
    """cross_spectral_density recovers a scale-dependent co-expression SIGN FLIP that the global bivariate
    scalar averages to ~0: A=macro+fine, B=macro-fine -> positive at long wavelengths, negative at short."""
    from spectrast.spatial import cross_spectral_density
    N, BIN = 200, 10.0
    yy, xx = np.mgrid[0:N, 0:N].astype(float); xx *= BIN; yy *= BIN
    def band(lmin, lmax, seed):
        r = np.random.default_rng(seed); f = np.zeros((N, N))
        for _ in range(60):
            lam = r.uniform(lmin, lmax); th = r.uniform(0, np.pi); ph = r.uniform(0, 2 * np.pi)
            f += np.cos(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / lam + ph)
        return (f - f.mean()) / f.std()
    macro = band(180, 320, 1); fine = band(30, 60, 2)
    A, B = macro + fine, macro - fine; m = np.ones((N, N), bool)
    scalar = float(((A - A.mean()) / A.std() * (B - B.mean()) / B.std()).mean())
    assert abs(scalar) < 0.15                                   # global scalar cancels
    r = cross_spectral_density(A, B, m, bin_size=BIN); lam = r["wavelength_um"]; co = r["coexpr"]
    assert np.nanmean(co[lam > 150]) > 0.5                      # positive at macro scale
    assert np.nanmean(co[(lam < 90) & (lam > 20)]) < -0.3       # negative at fine scale


def test_flagged_cells_match_exact(data):
    """The cells fusion touches must equal the exact graph there (that's the whole point)."""
    xy, tree = data["xy"], data["tree"]
    v = data["x"]; z = (v - v.mean()) / v.std()
    exact = _exact(xy, z, combine_moran, tree)
    hybrid, meta = adaptive_local_stat(xy, z, combine_moran, RADIUS, bin_size=COARSE_BIN,
                                       tree=tree, return_meta=True)
    # recompute which cells were flagged, assert those entries equal exact
    ns, nc = _fft_neighbor_reduce(xy, z, RADIUS, COARSE_BIN)
    fft_b = combine_moran(ns, nc, z)
    ns2, nc2 = _fft_neighbor_reduce(xy, z, RADIUS, COARSE_BIN / 2)
    proxy = np.abs(fft_b - combine_moran(ns2, nc2, z)) / (np.std(fft_b) + 1e-12)
    flagged = proxy > meta["tol"]
    assert np.allclose(hybrid[flagged], exact[flagged], atol=1e-9)


def test_isolated_cells_return_nan():
    """M14: a cell with NO neighbour within radius has an UNDEFINED local statistic → NaN, not a spurious 0
    (a 0 would be counted as 'no autocorrelation' by a downstream nanmean/threshold). Clustered cells stay
    finite."""
    from spectrast.adaptive import adaptive_local_moran
    rng = np.random.default_rng(0)
    cluster = rng.normal([100, 100], 5, (200, 2))
    isolated = np.array([[400.0, 400.0], [10.0, 380.0], [390.0, 10.0]])   # far from everything (no neighbour @40)
    xy = np.vstack([cluster, isolated])
    v = np.r_[rng.standard_normal(200), np.array([3.0, -3.0, 3.0])]
    out = adaptive_local_moran(xy, v, RADIUS, bin_size=COARSE_BIN, fuse=True)
    assert np.all(np.isnan(out[200:])), f"isolated cells not NaN: {out[200:]}"
    assert np.isfinite(out[:200]).mean() > 0.9, "clustered cells should be defined"


@pytest.mark.parametrize("fuse", [False, True])
def test_isolated_cells_nan_on_fft_branch(fuse):
    """2026-10-28: the same guarantee on the FFT branch of _conv2d (disc side >= 9, i.e. r/b >= 4.5 — the
    autotuned r/b~5 operating point). The FFT leaves ~1e-15 round-off in the neighbour COUNT, which a `nc > 0`
    guard read as a neighbour: a cell with none got noise/noise (values to ~1e4 on a sparse MERFISH subsample)
    instead of NaN, and the inflated SD hid it from the b-vs-b/2 proxy. Counts are integers by construction, so
    _fft_neighbor_reduce must return them exactly."""
    from spectrast.adaptive import adaptive_local_moran
    rng = np.random.default_rng(0)
    cluster = rng.normal([100, 100], 5, (200, 2))
    isolated = np.array([[400.0, 400.0], [10.0, 380.0], [390.0, 10.0]])
    xy = np.vstack([cluster, isolated])
    v = np.r_[rng.standard_normal(200), np.array([3.0, -3.0, 3.0])]
    b = RADIUS / 8.0                                                    # disc side 17 -> FFT branch
    _, nc = _fft_neighbor_reduce(xy, v, RADIUS, b)
    assert np.array_equal(nc, np.rint(nc)), "FFT-branch neighbour counts must be exact integers"
    out = adaptive_local_moran(xy, v, RADIUS, bin_size=b, fuse=fuse)
    assert np.all(np.isnan(out[200:])), f"isolated cells not NaN on the FFT branch: {out[200:]}"
    assert np.isfinite(out[:200]).mean() > 0.9, "clustered cells should be defined"


def test_adaptive_false_is_plain_fft(data):
    xy = data["xy"]; v = data["x"]; z = (v - v.mean()) / v.std()
    plain = adaptive_local_stat(xy, z, combine_moran, RADIUS, bin_size=COARSE_BIN, fuse=False)
    ns, nc = _fft_neighbor_reduce(xy, z, RADIUS, COARSE_BIN)
    assert np.allclose(plain, combine_moran(ns, nc, z), equal_nan=True)   # NaN at isolated cells matches


def test_proxy_predicts_error(data):
    xy, tree = data["xy"], data["tree"]
    v = data["x"]; z = (v - v.mean()) / v.std()
    exact = _exact(xy, z, combine_moran, tree)
    ns, nc = _fft_neighbor_reduce(xy, z, RADIUS, COARSE_BIN)
    fft_b = combine_moran(ns, nc, z)
    ns2, nc2 = _fft_neighbor_reduce(xy, z, RADIUS, COARSE_BIN / 2)
    proxy = np.abs(fft_b - combine_moran(ns2, nc2, z)) / (np.nanstd(fft_b) + 1e-12)
    true_err = np.abs(fft_b - exact)
    m = np.isfinite(proxy) & np.isfinite(true_err)                        # isolated cells are NaN in both (M14)
    assert spearmanr(proxy[m], true_err[m]).correlation > 0.5


def test_determinism(data):
    xy = data["xy"]; v = data["x"]
    a = spatial.local_morans_i_sc(xy, v, RADIUS, bin_size=COARSE_BIN)
    b = spatial.local_morans_i_sc(xy, v, RADIUS, bin_size=COARSE_BIN)
    assert np.array_equal(a, b, equal_nan=True)          # deterministic incl. NaN at isolated cells (M14)


# --------------------------------------------------------------------- spatial.py entry points
def test_spatial_sc_entrypoints_run(data):
    xy = data["xy"]
    assert spatial.local_morans_i_sc(xy, data["x"], RADIUS).shape == (len(xy),)
    assert spatial.getis_ord_sc(xy, data["x"], RADIUS).shape == (len(xy),)
    assert spatial.bivariate_morans_sc(xy, data["x"], data["y"], RADIUS).shape == (len(xy),)
    assert spatial.local_diversity_sc(xy, data["labels"], RADIUS).shape == (len(xy),)
    assert spatial.colocalization_sc(xy, data["labels"], "B", RADIUS).shape == (len(xy),)
    _, meta = spatial.local_morans_i_sc(xy, data["x"], RADIUS, return_meta=True)
    assert "bin_um" in meta and "flagged_frac" in meta


def test_autotune_and_rasterize_auto(data):
    xy = data["xy"]
    at = autotune_grid(xy, RADIUS, probe_values=data["x"])
    assert at["bin_um"] <= at["r_min_um"]            # never coarser than the radius
    grid, mask, bin_um = spatial.rasterize_auto(xy, data["x"], RADIUS)
    assert grid.ndim == 3 and mask.shape == grid.shape[:2] and bin_um > 0


# --------------------------------------------------------------------- reduced-rank spatial stats
@pytest.fixture(scope="module")
def gene_matrix(data):
    """(N, G) matrix with a few shared spatial factors + noise, and its noise-free truth."""
    rng = np.random.default_rng(1)
    xy = data["xy"]; n = len(xy); G = 60
    f1 = np.sin(xy[:, 0] / 130); f2 = np.cos(xy[:, 1] / 110)
    load = rng.normal(0, 1, (2, G))
    Xtrue = np.outer(f1, load[0]) + np.outer(f2, load[1])
    X = Xtrue + rng.normal(0, 0.6, (n, G))
    return dict(xy=xy, X=X, Xtrue=Xtrue)


def test_rr_exact_identity(gene_matrix):
    """RR(rank-k) == direct statistic computed on the rank-k reconstructed gene (the identity)."""
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    genes = [0, 1, 2, 3, 4]
    for k in (2, 8):
        sv = rr.svd_embeddings(X, k)
        Xk = sv["Ec"] @ sv["V"].T                    # centred rank-k reconstruction
        g_rr = rr.morans_global(xy, X, RADIUS, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        g_dir = rr.morans_global(xy, Xk, RADIUS, reduced_rank=False, bin_size=COARSE_BIN)
        assert np.max(np.abs(g_rr - g_dir)) < 1e-6
        l_rr = rr.morans_local(xy, X, RADIUS, genes, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        l_dir = rr.morans_local(xy, Xk, RADIUS, genes, reduced_rank=False, bin_size=COARSE_BIN)
        assert np.max(np.abs(l_rr - l_dir)) < 1e-6


def test_rr_denoises(gene_matrix):
    """Low-rank reconstruction recovers the noise-free Moran's I better than the noisy genes."""
    from spectrast import spatial_sc as rr
    xy, X, Xtrue = gene_matrix["xy"], gene_matrix["X"], gene_matrix["Xtrue"]
    g_true = rr.morans_global(xy, Xtrue, RADIUS, reduced_rank=False, bin_size=COARSE_BIN)
    g_noisy = rr.morans_global(xy, X, RADIUS, reduced_rank=False, bin_size=COARSE_BIN)
    g_rr = rr.morans_global(xy, X, RADIUS, reduced_rank=True, rank=2, bin_size=COARSE_BIN)
    assert spearmanr(g_rr, g_true).correlation > spearmanr(g_noisy, g_true).correlation


def test_rr_spatial_entrypoints(gene_matrix):
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    assert spatial.morans_global_genes(xy, X, RADIUS, reduced_rank=True, rank=5).shape == (X.shape[1],)
    assert spatial.local_morans_i_genes(xy, X, RADIUS, [0, 1], reduced_rank=True, rank=5).shape == (len(xy), 2)
    assert spatial.getis_ord_genes(xy, X, RADIUS, [0, 1], reduced_rank=True, rank=5).shape == (len(xy), 2)


RADII = [15.0, 30.0, 60.0]


def test_rr_scalogram_geary_variogram_identity(gene_matrix):
    """RR(rank-k) == direct statistic on the rank-k reconstructed genes, for scalogram/Geary/variogram."""
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    for k in (3, 8):
        sv = rr.svd_embeddings(X, k)
        Xk = sv["Ec"] @ sv["V"].T
        sc_rr = rr.morans_scalogram(xy, X, RADII, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        sc_dir = rr.morans_scalogram(xy, Xk, RADII, reduced_rank=False, bin_size=COARSE_BIN)
        assert np.max(np.abs(sc_rr - sc_dir)) < 1e-5
        gc_rr = rr.geary_global(xy, X, RADIUS, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        gc_dir = rr.geary_global(xy, Xk, RADIUS, reduced_rank=False, bin_size=COARSE_BIN)
        assert np.max(np.abs(gc_rr - gc_dir)) < 1e-5
        vg_rr = rr.variogram(xy, X, RADII, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        vg_dir = rr.variogram(xy, Xk, RADII, reduced_rank=False, bin_size=COARSE_BIN)
        assert np.max(np.abs(vg_rr - vg_dir)) < 1e-5


def test_rr_local_geary_identity(gene_matrix):
    """Reduced-rank local Geary (via k² product-fields) == direct local Geary on the rank-k gene."""
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    genes = [0, 1, 2, 3]
    for k in (3, 8):
        sv = rr.svd_embeddings(X, k)
        Xk = sv["Ec"] @ sv["V"].T
        c_rr = rr.local_geary(xy, X, RADIUS, genes, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        c_dir = rr.local_geary(xy, Xk, RADIUS, genes, reduced_rank=False, bin_size=COARSE_BIN)
        assert c_rr.shape == (len(xy), len(genes))
        assert np.max(np.abs(c_rr - c_dir)) < 1e-5


def test_rr_bivariate_matrix_identity(gene_matrix):
    """All-pairs cross-correlation: RR matrix == direct pairwise on the rank-k genes; symmetric; diag=Moran."""
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    genes = [0, 1, 2, 3, 4, 5]
    k = 6
    sv = rr.svd_embeddings(X, k); Xk = sv["Ec"] @ sv["V"].T
    C_rr = rr.bivariate_matrix(xy, X, genes, RADIUS, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
    C_dir = rr.bivariate_matrix(xy, Xk, genes, RADIUS, reduced_rank=False, bin_size=COARSE_BIN)
    assert C_rr.shape == (len(genes), len(genes))
    assert np.allclose(C_rr, C_rr.T, atol=1e-8)                 # symmetric
    assert np.max(np.abs(C_rr - C_dir)) < 1e-5                  # identity
    mg = rr.morans_global(xy, X, RADIUS, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
    assert np.allclose(np.diag(C_rr), mg[genes], atol=1e-6)     # diagonal == Moran's I


def test_rr_getis_local_identity(gene_matrix):
    """Getis-Ord Gi*: RR(rank-k) == direct Getis on the rank-k reconstructed gene (completes the family)."""
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    genes = [0, 1, 2, 3]
    for k in (3, 8):
        sv = rr.svd_embeddings(X, k)
        Xk = sv["Ec"] @ sv["V"].T
        g_rr = rr.getis_local(xy, X, RADIUS, genes, reduced_rank=True, rank=k, bin_size=COARSE_BIN, svd=sv)
        g_dir = rr.getis_local(xy, Xk, RADIUS, genes, reduced_rank=False, bin_size=COARSE_BIN)
        assert g_rr.shape == (len(xy), len(genes))
        assert np.max(np.abs(g_rr - g_dir)) < 1e-5


def test_rr_stress_focal_field_wins(gene_matrix):
    """Reduced rank recovers SHARP focal truth better than the noisy genes — the win is gene-axis
    (low gene-rank), not spatial smoothing."""
    from spectrast.simulate import simulate_st
    from spectrast.spatial import morans_global_genes
    xy = np.random.default_rng(1).uniform(0, 800, (3000, 2))
    sim = simulate_st(xy, n_factors=6, n_genes=150, noise=2.0, dropout=0.2, seed=0, field="focal")
    t = morans_global_genes(xy, sim["X_true"], 40.0, reduced_rank=False, bin_size=10.0)
    f = morans_global_genes(xy, sim["X_noisy"], 40.0, reduced_rank=False, bin_size=10.0)
    r = morans_global_genes(xy, sim["X_noisy"], 40.0, reduced_rank=True, rank=6, bin_size=10.0)
    assert spearmanr(r, t).correlation > spearmanr(f, t).correlation


def test_gr_anndata_api(gene_matrix):
    """The scverse-style spectrast.gr wrappers read AnnData and write var/obsm/uns, matching the arrays."""
    anndata = pytest.importorskip("anndata")
    from spectrast import gr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    A = anndata.AnnData(X.astype("float32"))
    A.obsm["spatial"] = xy
    A.var_names = [f"g{i}" for i in range(X.shape[1])]
    r = float(gr.default_radius(A))
    assert r > 0
    m = gr.spatial_autocorr(A, mode="moran", radius=RADIUS, bin_size=COARSE_BIN)
    assert "spectrast_moranI" in A.var and A.var["spectrast_moranI"].shape[0] == X.shape[1]
    assert np.allclose(A.var["spectrast_moranI"].values, m)
    gr.scale_spectrum(A, radii=[20.0, 40.0, 80.0], bin_size=COARSE_BIN)
    assert "spectrast_char_scale" in A.var and "spectrast_scalogram" in A.uns
    gr.local_moran(A, genes=["g0", "g1"], radius=RADIUS, bin_size=COARSE_BIN)
    assert A.obsm["spectrast_local_moran"].shape == (len(xy), 2)
    C, names = gr.cross_correlation(A, genes=["g0", "g1", "g2"], radius=RADIUS, bin_size=COARSE_BIN)
    assert C.shape == (3, 3) and names == ["g0", "g1", "g2"]


def test_rr_sparse_matches_dense():
    """Reduced-rank Moran on a SPARSE matrix (memory-safe centred randomized SVD — no densify) agrees
    with the dense path on structured data. Enables whole-transcriptome (~18k-gene) analysis."""
    import scipy.sparse as sp
    from spectrast.spatial import morans_global_genes
    from spectrast.simulate import simulate_st
    xy = np.random.default_rng(3).uniform(0, 700, (3500, 2))
    sim = simulate_st(xy, n_factors=8, n_genes=180, noise=1.0, dropout=0.3, seed=2)
    Xd = np.clip(sim["X_noisy"], 0, None)
    gd = morans_global_genes(xy, Xd, 40.0, reduced_rank=True, rank=30, bin_size=8.0)
    gs = morans_global_genes(xy, sp.csr_matrix(Xd), 40.0, reduced_rank=True, rank=30, bin_size=8.0)
    assert spearmanr(gd, gs).correlation > 0.9      # same denoised ranking, sparse path never densifies


def test_per_radius_kernel_selection():
    """Multi-radius tools choose one convolution kernel/bin PER radius (~r/5) when bin_size=None — not a
    fixed kernel — and give the SAME scale as a fixed fine bin (both accurate); user bin_size overrides."""
    from spectrast.spatial_sc import _radius_bins, morans_scalogram, svd_embeddings
    from spectrast.simulate import simulate_st
    xy = np.random.default_rng(0).uniform(0, 700, (4000, 2))
    radii = [20.0, 40.0, 80.0, 160.0]
    bins = _radius_bins(xy, radii, None)
    assert len(bins) == len(radii)
    assert all(3.0 <= radii[i] / bins[i] <= 14.0 for i in range(len(radii)))   # each honours ~r/5 envelope
    assert _radius_bins(xy, radii, 8.0) == [8.0] * len(radii)                   # explicit bin -> fixed
    sim = simulate_st(xy, n_factors=6, n_genes=60, noise=1.0, seed=1)
    X = np.clip(sim["X_noisy"], 0, None); sv = svd_embeddings(X, 12)
    auto = morans_scalogram(xy, X, radii, reduced_rank=True, svd=sv, bin_size=None)
    fine = morans_scalogram(xy, X, radii, reduced_rank=True, svd=sv, bin_size=float(min(bins)))
    w = lambda s: (np.clip(s, 0, None) * np.array(radii)[:, None]).sum(0) / (np.clip(s, 0, None).sum(0) + 1e-9)
    assert spearmanr(w(auto), w(fine)).correlation > 0.95   # same characteristic scale, both accurate


def test_memory_auto_chunk():
    """The memory layer returns a positive budget and a chunk that shrinks as rows grow."""
    from spectrast import memory
    assert memory.available_bytes() > 0
    big, small = memory.gene_chunk(1_000_000), memory.gene_chunk(1000)
    assert 64 <= big <= small <= 8192           # more rows -> smaller (or equal, if both clamped) chunk
    assert memory.densify_gb(np.zeros((1000, 1000))) > 0


def test_sparse_local_paths_no_densify(gene_matrix):
    """The reduced-rank LOCAL family accepts SPARSE X and matches dense EXACTLY given the same SVD
    (X is unused in the reduced-rank path — proof it never densifies)."""
    import scipy.sparse as sp
    from spectrast import spatial_sc as rr
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    Xs = sp.csr_matrix(X); sv = rr.svd_embeddings(X, 6); genes = [0, 1, 2, 3]
    for fn in (lambda M: rr.morans_local(xy, M, RADIUS, genes, reduced_rank=True, svd=sv, bin_size=COARSE_BIN),
               lambda M: rr.getis_local(xy, M, RADIUS, genes, reduced_rank=True, svd=sv, bin_size=COARSE_BIN),
               lambda M: rr.local_geary(xy, M, RADIUS, genes, reduced_rank=True, svd=sv, bin_size=COARSE_BIN),
               lambda M: rr.geary_global(xy, M, RADIUS, reduced_rank=True, svd=sv, bin_size=COARSE_BIN)):
        assert np.max(np.abs(np.asarray(fn(X)) - np.asarray(fn(Xs)))) < 1e-9


def test_focality_streams_and_matches(gene_matrix):
    """focality (streamed Gini of the local map) == per-gene morans_local + Gini; runs on sparse; all genes."""
    import scipy.sparse as sp
    from spectrast import spatial_sc as rr
    from spectrast.spatial import focality_genes
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    sv = rr.svd_embeddings(X, 6)
    f = rr.focality(xy, X, RADIUS, reduced_rank=True, svd=sv, bin_size=COARSE_BIN)
    assert f.shape == (X.shape[1],) and np.all((f >= 0) & (f <= 1))
    L = rr.morans_local(xy, X, RADIUS, list(range(X.shape[1])), reduced_rank=True, svd=sv, bin_size=COARSE_BIN)
    fdir = np.array([rr._gini(L[:, j]) for j in range(X.shape[1])])
    assert np.max(np.abs(f - fdir)) < 1e-9
    # sparse input, via the spatial wrapper, gives the same shape (auto memory-safe path)
    assert focality_genes(xy, sp.csr_matrix(X), RADIUS, reduced_rank=True, rank=6, bin_size=COARSE_BIN).shape == (X.shape[1],)


def test_autotune_rank_finds_true_rank():
    """Data-driven rank selection lands near the true simulator rank; GD heuristic is sane."""
    from spectrast.simulate import simulate_st
    from spectrast.spatial import autotune_rank_genes
    xy = np.random.default_rng(0).uniform(0, 800, (2500, 2))
    true_rank = 8
    sim = simulate_st(xy, n_factors=true_rank, n_genes=150, noise=2.0, dropout=0.2, seed=0)
    res = autotune_rank_genes(xy, sim["X_noisy"], 40.0, candidate_ranks=[2, 4, 8, 12, 20, 40], bin_size=10.0)
    assert res["rank"] in (4, 8, 12), f"picked rank {res['rank']} far from true {true_rank}"
    assert 1 <= res["gd_rank"] <= 60
    assert len(res["reproducibility"]) == len(res["candidate_ranks"])


def test_power_threshold_filter():
    """Keeping 100% of power reconstructs the field; less keeps fewer coefficients and denoises."""
    from spectrast.spatial import power_threshold_filter
    from spectrast.simulate import _grf
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 600, (2000, 2))
    truth = _grf(xy, 100.0, seed=1)
    from spectrast.adaptive import grid_index
    r, c, H, W, _ = grid_index(xy, 10.0)
    g = np.zeros((H, W)); n = np.zeros((H, W)); np.add.at(g, (r, c), truth); np.add.at(n, (r, c), 1.0)
    m = n > 0; field = np.zeros((H, W)); field[m] = g[m] / n[m]
    full, frac_full = power_threshold_filter(field, keep_frac=1.0, mask=m)
    assert np.allclose(full[m], field[m], atol=1e-6) and frac_full <= 1.0
    part, frac_part = power_threshold_filter(field, keep_frac=0.9, mask=m)
    assert frac_part < frac_full                       # 90% power in fewer coefficients


def test_concordance_layer(gene_matrix):
    """Concordance layer returns per-gene direct vs reduced-rank with a valid flag category."""
    from spectrast.spatial import reduced_rank_concordance
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    df = reduced_rank_concordance(xy, X, RADIUS, bin_size=COARSE_BIN, n_split=2, rank=8)
    assert {"I_direct", "I_rr", "detection", "rank_shift", "concordant", "category"}.issubset(df.columns)
    assert len(df) == X.shape[1]
    assert df.category.isin(["concordant", "rr_promoted", "rr_demoted"]).all()
    assert df.concordant.dtype == bool


def test_power_spectrum_scale_monotone():
    """RAPSD characteristic scale increases with the true field scale (Wiener–Khinchin sanity)."""
    from spectrast.spatial import power_spectrum_scale_genes
    from spectrast.simulate import _grf
    xy = np.random.default_rng(0).uniform(0, 800, (3000, 2))
    X = np.stack([_grf(xy, s, seed=i) for i, s in enumerate([20.0, 60.0, 150.0])], 1)
    sc, an = power_spectrum_scale_genes(xy, X, bin_size=10.0)
    assert sc[0] < sc[1] < sc[2] and np.all(np.isfinite(an))


def test_simulator_denoising_recovers_truth():
    """Ground-truth: reduced-rank Moran's I recovers the noise-free truth better than full-data."""
    from spectrast.simulate import simulate_st
    from spectrast.spatial import morans_global_genes
    xy = np.random.default_rng(0).uniform(0, 800, (2500, 2))
    sim = simulate_st(xy, n_factors=8, n_genes=150, noise=3.0, dropout=0.2, seed=0)
    def mi(X, rr): return morans_global_genes(xy, X, 40.0, reduced_rank=rr, rank=8, bin_size=10.0)
    t = mi(sim["X_true"], False); f = mi(sim["X_noisy"], False); r = mi(sim["X_noisy"], True)
    c_full = spearmanr(f, t).correlation; c_rr = spearmanr(r, t).correlation
    assert c_rr > c_full, f"RR ({c_rr:.3f}) should beat full ({c_full:.3f})"
    assert c_rr > 0.85, f"RR should recover truth ({c_rr:.3f})"


def test_composition_fuse_runs_and_aggregate_robust(gene_matrix):
    """The fusion axis composes into reduced-rank; aggregate Moran is ~unchanged (bilinear form robust)."""
    from spectrast.spatial_sc import morans_global
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    a = morans_global(xy, X, RADIUS, reduced_rank=True, rank=8, bin_size=COARSE_BIN)
    b = morans_global(xy, X, RADIUS, reduced_rank=True, rank=8, bin_size=COARSE_BIN, fuse=True)
    assert b.shape == a.shape and np.all(np.isfinite(b))
    assert spearmanr(a, b).correlation > 0.95


def test_rr_new_spatial_entrypoints(gene_matrix):
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    assert spatial.morans_scalogram_genes(xy, X, RADII, reduced_rank=True, rank=5).shape == (len(RADII), X.shape[1])
    assert spatial.correlation_length_genes(xy, X, RADII, reduced_rank=True, rank=5).shape == (X.shape[1],)
    assert spatial.geary_c_genes(xy, X, RADIUS, reduced_rank=True, rank=5).shape == (X.shape[1],)
    assert spatial.variogram_genes(xy, X, RADII, reduced_rank=True, rank=5).shape == (len(RADII), X.shape[1])
    assert spatial.bivariate_spatial_matrix(xy, X, [0, 1, 2], RADIUS, reduced_rank=True, rank=5).shape == (3, 3)


def test_autotune_returns_coarsest_feasible_bin():
    """autotune_grid must return the COARSEST bin that passes BOTH the occupancy cap and the fuse-fraction
    cap — not merely 'a feasible bin' (the old test only asserted bin <= r_min, true for every candidate;
    the review showed returning the FINEST bin still passed). Non-vacuous: the fixture forces coarse bins
    to exceed the occupancy cap, so >=1 coarser candidate is genuinely rejected."""
    from spectrast.adaptive import autotune_grid
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.uniform(0, 400, (3000, 2)), rng.normal([120, 120], 15, (2000, 2))])
    occ_cap, max_fuse = 2.0, 0.25                              # the autotune_grid defaults
    out = autotune_grid(xy, r_min_um=40.0, occ_cap=occ_cap, max_fuse_frac=max_fuse)
    pick = out["bin_um"]
    assert out["feasible"] and out["median_occupancy"] <= occ_cap + 1e-9 \
        and out["flagged_frac"] <= max_fuse + 1e-9, "returned bin violates a constraint"
    coarser = [r for r in out["table"] if r["bin_um"] > pick + 1e-9]
    assert coarser, "fixture must reject >=1 coarser candidate (else the 'coarsest' claim is vacuous)"
    for r in coarser:                                          # every coarser candidate must have FAILED
        feas = (r["median_occupancy"] <= occ_cap + 1e-9) and (r["flagged_frac"] <= max_fuse + 1e-9)
        assert not feas, f"coarser bin {r['bin_um']} was feasible but not chosen (not coarsest-feasible)"


def test_gr_api_coverage_and_auto_defaults(gene_matrix):
    """Cover the spectrast.gr wrappers the review found untested (getis_ord, local_geary, focality, autotune_rank,
    default_radii) AND the 'just works' auto-default paths (radius=None -> default_radius; mode='geary')."""
    anndata = pytest.importorskip("anndata")
    from spectrast import gr
    from spectrast.spatial import getis_ord_genes, geary_c_genes
    xy, X = gene_matrix["xy"], gene_matrix["X"]
    A = anndata.AnnData(X.astype("float32")); A.obsm["spatial"] = xy
    A.var_names = [f"g{i}" for i in range(X.shape[1])]

    # default_radii: n log-spaced positive radii, increasing
    rad = gr.default_radii(A, n=5)
    assert len(rad) == 5 and np.all(rad > 0) and np.all(np.diff(rad) > 0)

    # mode='geary' auto-default path (radius=None -> default_radius); writes var, matches the array
    val = gr.spatial_autocorr(A, mode="geary", bin_size=COARSE_BIN)
    assert "spectrast_gearyC" in A.var
    r_auto = float(gr.default_radius(A))
    assert np.allclose(A.var["spectrast_gearyC"].values, geary_c_genes(xy, X, r_auto, bin_size=COARSE_BIN))

    # getis_ord wrapper: obsm (N, n_genes), matches the underlying array
    G = gr.getis_ord(A, genes=["g0", "g1"], radius=RADIUS, bin_size=COARSE_BIN)
    assert A.obsm["spectrast_getis_ord"].shape == (len(xy), 2)
    # AnnData stores X as float32, so compare to the float32 array (else tiny f32-vs-f64 drift trips allclose)
    assert np.allclose(G, getis_ord_genes(xy, X.astype("float32"), RADIUS, [0, 1], bin_size=COARSE_BIN), atol=1e-4)

    # local_geary wrapper: obsm (N, n_genes)
    gr.local_geary(A, genes=["g0", "g2"], radius=RADIUS, bin_size=COARSE_BIN)
    assert A.obsm["spectrast_local_geary"].shape == (len(xy), 2)

    # focality wrapper: per-gene var with NaN-fill for UNSELECTED genes
    gr.focality(A, genes=["g0", "g1"], radius=RADIUS, bin_size=COARSE_BIN)
    fo = A.var["spectrast_focality"].values
    assert np.isfinite(fo[:2]).all() and np.isnan(fo[2:]).all(), "focality must NaN-fill unselected genes"

    # autotune_rank wrapper: writes uns + returns a usable rank
    res = gr.autotune_rank(A, radius=RADIUS, bin_size=COARSE_BIN)
    assert "spectrast_autotune_rank" in A.uns and int(res["rank"]) >= 1

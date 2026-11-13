"""LOSH — local spatial heteroscedasticity (Ord & Getis, Annals of Regional Science 48:529-539, 2012).

The reference here is a dependency-free brute-force evaluation of the published formula on an explicit
weight matrix, so the test does not assume the FFT engine is right about anything. `esda.LOSH` is the
external cross-check where it is installed; it is not a hard dependency of this repo, so that test skips.

Run: `pytest tests/test_losh.py -q`.
"""
import numpy as np
import pytest

from spectrast.spatial import losh, neighbor_kernel


def _brute_losh(field, mask, radius, a=2.0):
    """H_i = Σ_j w_ij |e_j|^a / (h_1 Σ_j w_ij), built from an explicit (n, n) weight matrix.

    e_j = x_j - x̄_j against the row-standardized neighbour mean AT j; h_1 = mean_j |e_j|^a.
    """
    mb = np.asarray(mask, bool)
    ys, xs = np.nonzero(mb)
    x = np.asarray(field, float)[mb]
    d2 =(ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2
    Wm = (d2 <= radius ** 2).astype(float)
    np.fill_diagonal(Wm, 0.0)                      # the disc excludes the focal cell
    deg = Wm.sum(1)
    xbar = np.where(deg > 0, (Wm @ x) / np.maximum(deg, 1e-12), 0.0)
    e = np.where(deg > 0, np.abs(x - xbar) ** a, 0.0)
    h1 = e.mean()
    H = np.where(deg > 0, (Wm @ e) / np.maximum(deg, 1e-12) / h1, np.nan)
    out = np.full(mb.shape, np.nan)
    out[mb] = H
    return out


@pytest.mark.parametrize("a", [2.0, 1.0])
def test_losh_matches_bruteforce(a):
    """The exact published statistic, on a masked tissue with a hole so edge correction is exercised."""
    rng = np.random.default_rng(0); H = W = 24
    field = np.cumsum(rng.standard_normal((H, W)), 0) + rng.standard_normal((H, W)) * 0.4
    mask = np.ones((H, W), bool)
    mask[6:11, 4:15] = False                       # interior hole
    mask[:2, :] = False                            # ragged border
    code = losh(field * mask, mask, radius=4, a=a)
    bf = _brute_losh(field, mask, radius=4, a=a)
    ok = np.isfinite(bf)
    assert np.isfinite(code[ok]).all()
    assert np.allclose(code[ok], bf[ok], rtol=1e-8, atol=1e-8), (
        f"losh(a={a}) != brute force: max |Δ| = {np.nanmax(np.abs(code - bf)):.3e}")
    assert np.isnan(code[~mask]).all(), "off-mask cells must be NaN"


def test_losh_mean_is_about_one():
    """h_1 normalizes the statistic so the tissue mean sits at 1 — the property that makes H_i readable
    as 'noisier / smoother than this section' without a per-gene reference."""
    rng = np.random.default_rng(1); H = W = 40
    field = rng.standard_normal((H, W))
    mask = np.ones((H, W), bool)
    H_map = losh(field, mask, radius=5)
    assert abs(np.nanmean(H_map) - 1.0) < 0.05


def test_losh_flags_the_noisier_half():
    """A field that is smooth on the left and speckled on the right, with the SAME local mean and the
    same marginal variance overall: local Moran cannot separate these, LOSH must."""
    rng = np.random.default_rng(2); H = W = 60
    field = np.zeros((H, W))
    field[:, :30] = np.repeat(np.linspace(0, 1, H)[:, None], 30, axis=1)      # smooth ramp
    field[:, 30:] = np.repeat(np.linspace(0, 1, H)[:, None], 30, axis=1) + \
        rng.standard_normal((H, 30)) * 0.5                                    # same ramp + speckle
    mask = np.ones((H, W), bool)
    H_map = losh(field, mask, radius=4)
    left = np.nanmean(H_map[:, 5:25]); right = np.nanmean(H_map[:, 35:55])
    assert right > 5 * left, f"LOSH did not separate speckle from ramp: left={left:.3f} right={right:.3f}"


def test_losh_is_affine_invariant():
    """|e|^a and h_1 both scale as |c|^a and e is shift-invariant, so H is invariant to x -> c·x + d.
    This is why the toolkit does not standardize the field before calling it."""
    rng = np.random.default_rng(3); H = W = 30
    field = rng.standard_normal((H, W))
    mask = np.ones((H, W), bool); mask[3:8, 3:8] = False
    base = losh(field, mask, radius=4)
    scaled = losh((field * -7.5 + 100.0) * mask, mask, radius=4)
    ok = np.isfinite(base)
    assert np.allclose(base[ok], scaled[ok], rtol=1e-8, atol=1e-8)


def test_losh_batched_matches_single():
    """The (H, W, G) path is the (H, W) path gene by gene — one FFT, same numbers."""
    rng = np.random.default_rng(4); H = W = 26; G = 5
    grid = rng.standard_normal((H, W, G))
    mask = np.ones((H, W), bool); mask[10:14, :6] = False
    batched = losh(grid * mask[..., None], mask, radius=3)
    for g in range(G):
        single = losh(grid[..., g] * mask, mask, radius=3)
        ok = np.isfinite(single)
        assert np.allclose(batched[..., g][ok], single[ok], rtol=1e-9, atol=1e-9), f"gene {g}"


def test_losh_a1_is_not_a2():
    """a=1 and a=2 are different statistics, not a rescaling of one another — worth locking, because
    the cheap way to get this wrong is to normalize the difference away."""
    rng = np.random.default_rng(5); H = W = 40
    field = rng.standard_normal((H, W)) ** 3                    # heavy tails: where the exponent bites
    mask = np.ones((H, W), bool)
    h2 = losh(field, mask, radius=4); h1 = losh(field, mask, radius=4, a=1.0)
    r = np.corrcoef(h1[np.isfinite(h1)], h2[np.isfinite(h2)])[0, 1]
    assert r < 0.95, f"a=1 and a=2 are indistinguishable here (r={r:.3f}); check the exponent is applied"


def test_losh_genes_sc_matches_grid():
    """The point-cloud entry point reproduces the grid statistic when cells sit ON lattice centres —
    the same equivalence the other local statistics are held to."""
    from spectrast.spatial import losh_genes
    rng = np.random.default_rng(6); H = W = 30
    yy, xx = np.mgrid[0:H, 0:W]
    xy = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    vals = rng.standard_normal((H, W))
    X = vals.ravel()[:, None]
    mask = np.ones((H, W), bool)
    grid_H = losh(vals, mask, radius=4)
    sc_H = losh_genes(xy, X, 4.0, [0], bin_size=1.0)[:, 0].reshape(H, W)
    # the rasterized neighbourhood is the same disc; agreement is to FFT roundoff, not approximate
    assert np.allclose(sc_H, grid_H, rtol=1e-6, atol=1e-6), \
        f"sc vs grid max |Δ| = {np.nanmax(np.abs(sc_H - grid_H)):.3e}"


def test_losh_reduced_rank_is_exact_at_full_rank():
    """LOSH is not linear in the field, so the reduced-rank path reconstructs the gene rather than
    riding the embedding sums. At full rank the reconstruction is the gene, so the two must agree."""
    from spectrast.spatial import losh_genes
    rng = np.random.default_rng(8); H = W = 20; G = 6
    yy, xx = np.mgrid[0:H, 0:W]
    xy = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    X = rng.standard_normal((H * W, G))
    direct = losh_genes(xy, X, 3.0, [0, 2, 5], bin_size=1.0)
    lowrank = losh_genes(xy, X, 3.0, [0, 2, 5], bin_size=1.0, reduced_rank=True, rank=G)
    assert np.allclose(direct, lowrank, rtol=1e-6, atol=1e-6)


def test_gr_losh_writes_obsm():
    """The AnnData entry point lands where the other local maps land, and passes ``a`` through."""
    ad = pytest.importorskip("anndata")
    from spectrast import gr
    rng = np.random.default_rng(9)
    adata = ad.AnnData(np.abs(rng.standard_normal((400, 8))).astype("float32"))
    adata.obsm["spatial"] = rng.random((400, 2)) * 100
    adata.var_names = [f"g{i}" for i in range(8)]
    M = gr.losh(adata, genes=["g0", "g1"], radius=20.0)
    assert M.shape == (400, 2)
    assert adata.obsm["spectrast_losh"].shape == (400, 2)
    assert list(adata.uns["spectrast_losh_genes"]) == ["g0", "g1"]
    M1 = gr.losh(adata, genes=["g0", "g1"], radius=20.0, a=1.0)
    assert not np.allclose(M, M1)


def test_losh_matches_esda():
    """External cross-check against the reference implementation, where it is installed."""
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    rng = np.random.default_rng(7); H = W = 12
    field = rng.standard_normal((H, W))
    mask = np.ones((H, W), bool)
    w = libpysal.weights.DistanceBand.from_array(
        np.column_stack([np.mgrid[0:H, 0:W][1].ravel(), np.mgrid[0:H, 0:W][0].ravel()]).astype(float),
        threshold=3.0, binary=True, silence_warnings=True)
    ref = esda.LOSH(connectivity=w, inference=None).fit(field.ravel())
    ours = losh(field, mask, radius=3).ravel()
    assert np.allclose(ours, ref.Hi, rtol=1e-6, atol=1e-6)


def test_losh_kernel_is_center_excluded():
    """Guard the weight convention the formula assumes: the disc must not include the focal cell,
    or e_j leaks into its own neighbourhood sum and H_i is biased toward 1."""
    k = neighbor_kernel(3, "disc")
    assert k[3, 3] == 0.0


# ---------------------------------------------------------------------------------------------
# The two multiscale reductions the shared-radius curve bank needs, beside Moran and Geary.
# ---------------------------------------------------------------------------------------------
def _brute_general_g(field, mask, radius):
    """G(d) = sum_i sum_{j!=i} w_ij x_i x_j / sum_i sum_{j!=i} x_i x_j, on an explicit weight matrix."""
    mb = np.asarray(mask, bool)
    ys, xs = np.nonzero(mb)
    x = np.asarray(field, float)[mb]
    d2 = (ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2
    Wm = (d2 <= radius ** 2).astype(float)
    np.fill_diagonal(Wm, 0.0)
    num = float(x @ Wm @ x)
    den = float(x.sum() ** 2 - (x ** 2).sum())
    return num / den


def test_general_g_multiscale_matches_bruteforce():
    from spectrast.spatial import general_g_multiscale
    rng = np.random.default_rng(10); H = W = 22
    field = np.abs(rng.standard_normal((H, W))) + 0.2        # non-negative: G is defined on values
    mask = np.ones((H, W), bool); mask[4:8, 12:18] = False
    radii = [2, 3, 5]
    code = general_g_multiscale(field * mask, mask, radii)[:, 0]
    bf = np.array([_brute_general_g(field, mask, r) for r in radii])
    assert np.allclose(code, bf, rtol=1e-8, atol=1e-10), f"general G != brute:\n{code}\n{bf}"


def test_general_g_separates_hot_from_cold_clustering():
    """The property that distinguishes G from Moran: a field of clustered HIGH values and its mirror
    image (clustered LOW values) have the same Moran's I and must differ in G."""
    from spectrast.spatial import general_g_multiscale, morans_i_multiscale
    H = W = 40
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 5.0 ** 2))
    hot = 1.0 + 3.0 * blob                                    # a high-value island
    cold = 1.0 + 3.0 * (1.0 - blob)                           # the same geometry, low-value island
    mask = np.ones((H, W), bool)
    r = [4]
    gi_hot = morans_i_multiscale(hot, mask, r)[0, 0]
    gi_cold = morans_i_multiscale(cold, mask, r)[0, 0]
    g_hot = general_g_multiscale(hot, mask, r)[0, 0]
    g_cold = general_g_multiscale(cold, mask, r)[0, 0]
    assert abs(gi_hot - gi_cold) < 1e-6, "Moran should not distinguish these"
    assert g_hot > g_cold, f"General G failed to separate hot from cold clustering ({g_hot} vs {g_cold})"


def test_general_g_warns_on_negative_values():
    from spectrast.spatial import general_g_multiscale
    rng = np.random.default_rng(11)
    f = rng.standard_normal((16, 16))                          # centred: not a valid G input
    with pytest.warns(RuntimeWarning, match="non-negative"):
        general_g_multiscale(f, np.ones((16, 16), bool), [3])


def test_losh_multiscale_matches_the_single_radius_map():
    """The scalogram must be the per-radius map reduced, not a separately-derived quantity."""
    from spectrast.spatial import losh, losh_multiscale
    rng = np.random.default_rng(12); H = W = 26
    grid = rng.standard_normal((H, W, 3))
    mask = np.ones((H, W), bool); mask[9:13, 3:9] = False
    radii = [2, 4, 6]
    got = losh_multiscale(grid * mask[..., None], mask, radii)
    for si, r in enumerate(radii):
        Hm = losh(grid * mask[..., None], mask, radius=r)
        for g in range(3):
            v = Hm[..., g][mask]
            v = np.clip(np.where(np.isfinite(v), v, 0.0), 0.0, None)
            pos = np.sort(v); n = len(pos)
            gini = (2 * np.arange(1, n + 1) - n - 1).dot(pos) / (n * pos.sum())
            assert abs(got[si, g] - gini) < 1e-8, f"radius {r}, gene {g}: {got[si, g]} vs {gini}"


def test_losh_multiscale_sd_reduction_tracks_heterogeneity():
    """A field that is smooth on one half and speckled on the other must give a larger LOSH spread
    than a uniformly smooth one, at every radius."""
    from spectrast.spatial import losh_multiscale
    rng = np.random.default_rng(13); H = W = 48
    ramp = np.repeat(np.linspace(0, 1, H)[:, None], W, axis=1)
    even = ramp + 0.1 * rng.standard_normal((H, W))
    mixed = ramp.copy()
    mixed[:, W // 2:] += 0.6 * rng.standard_normal((H, W // 2))
    mask = np.ones((H, W), bool)
    radii = [3, 5, 8]
    s_even = losh_multiscale(even, mask, radii, reduce="sd")[:, 0]
    s_mixed = losh_multiscale(mixed, mask, radii, reduce="sd")[:, 0]
    assert (s_mixed > s_even).all(), f"LOSH spread did not track heterogeneity: {s_mixed} vs {s_even}"


# ---------------------------------------------------------------------------------------------
# The single-cell General G (spatial.general_g_genes), the entry point the speed benchmarks call.
# ---------------------------------------------------------------------------------------------
def _lattice_cells(H=24, W=24, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    xy = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    vals = np.abs(rng.standard_normal((H, W, 3))) + 0.2      # non-negative: G is defined on values
    return xy, vals


def test_general_g_genes_matches_bruteforce_and_grid():
    """Cells ON lattice centres: the point-cloud path, the grid path and the explicit double sum agree."""
    from spectrast.spatial import general_g_genes, general_g_multiscale
    xy, vals = _lattice_cells()
    H, W, G = vals.shape
    mask = np.ones((H, W), bool)
    sc = general_g_genes(xy, vals.reshape(-1, G), 4.0, bin_size=1.0)
    grid = general_g_multiscale(vals, mask, [4])[0]
    brute = np.array([_brute_general_g(vals[..., g], mask, 4) for g in range(G)])
    assert np.allclose(sc, brute, rtol=1e-8), f"sc {sc} vs brute {brute}"
    assert np.allclose(grid, brute, rtol=1e-8), f"grid {grid} vs brute {brute}"


def test_general_g_genes_refuses_negative_values():
    from spectrast.spatial import general_g_genes
    xy, vals = _lattice_cells()
    z = vals.reshape(-1, vals.shape[2]) - vals.mean()
    with pytest.raises(ValueError):
        general_g_genes(xy, z, 4.0, bin_size=1.0)


def test_general_g_genes_matches_esda():
    """External cross-check against esda.G on a binary distance band, where esda is installed."""
    esda = pytest.importorskip("esda")
    libpysal = pytest.importorskip("libpysal")
    from spectrast.spatial import general_g_genes
    xy, vals = _lattice_cells(H=14, W=14)
    w = libpysal.weights.DistanceBand.from_array(xy, threshold=3.0, binary=True,
                                                 silence_warnings=True)
    ours = general_g_genes(xy, vals.reshape(-1, vals.shape[2]), 3.0, bin_size=1.0)
    ref = [esda.G(vals[..., g].ravel(), w, permutations=0).G for g in range(vals.shape[2])]
    assert np.allclose(ours, ref, rtol=1e-8), f"ours {ours} vs esda {ref}"


# ---- a field with no residual variance where it is measured: H is 0/0, so NaN (spectrast 0.13.2) ----------------
def _constant_and_random(H=40, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((H, H)) < 0.6
    cols = [np.full((H, H), c) for c in (2.5, 0.1, 7.3, 0.0)] + [rng.random((H, H))]
    return np.stack(cols, -1) * mask[..., None], mask


def test_losh_is_nan_on_a_constant_field():
    """H = (local mean of |e|^a) / h1, and a constant field has e = 0 up to rounding of the local mean: 0/0. A
    nonzero constant used to read rounding noise (Gini about 0.5; 0.50 on the Mac against 0.47 on x86) and an absent
    field 0; both now read NaN, as Moran does. A column with real variation is untouched."""
    from spectrast.spatial import losh_multiscale
    grid, mask = _constant_and_random()
    ms = losh_multiscale(grid, mask, [1, 2, 4, 8])
    assert np.isnan(ms[:, :4]).all() and np.isfinite(ms[:, 4]).all()
    np.testing.assert_allclose(ms[:, 4], losh_multiscale(grid[..., 4:], mask, [1, 2, 4, 8])[:, 0], rtol=1e-12)
    Hm = losh(grid, mask, radius=3)
    assert np.isnan(Hm[..., :4]).all()
    np.testing.assert_allclose(Hm[..., 4], losh(grid[..., 4], mask, radius=3), rtol=1e-12, atol=1e-12)  # batch vs alone: rounding


def test_losh_genes_is_nan_on_a_constant_gene():
    from spectrast.spatial import losh_genes
    rng = np.random.default_rng(3); H = W = 20
    yy, xx = np.mgrid[0:H, 0:W]
    xy = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    X = np.column_stack([np.full(H * W, 0.1), np.zeros(H * W), rng.standard_normal(H * W)])
    out = losh_genes(xy, X, 3.0, [0, 1, 2], bin_size=1.0)
    assert np.isnan(out[:, :2]).all() and np.isfinite(out[:, 2]).all()


def test_losh_is_nan_where_each_fragment_is_constant_below_the_gap():
    """Two fragments 31 bins apart, constant at 1 and 3. Below the gap every bin equals its own local mean, so H is
    0/0 even though the field is not constant overall; once the disc spans the gap, H is defined. Pipelle biopsies
    are scattered fragments, so this is the realistic form of the case."""
    from spectrast.spatial import losh_multiscale
    mask = np.zeros((20, 60), bool)
    mask[5:15, 5:15] = True
    mask[5:15, 45:55] = True
    field = np.where(np.arange(60)[None, :] < 30, 1.0, 3.0) * np.ones((20, 1)) * mask
    ms = losh_multiscale(field, mask, [2, 4, 40])[:, 0]
    assert np.isnan(ms[:2]).all() and np.isfinite(ms[2])


def test_the_constant_rule_keeps_affine_invariance_on_a_large_offset():
    """The rule compares h1 with rounding of the local mean, which scales with the field's magnitude; a field that
    really varies is still tested, and still invariant, on an offset a million times its spread."""
    from spectrast.spatial import losh_multiscale
    rng = np.random.default_rng(5); H = 40
    mask = rng.random((H, H)) < 0.6
    f = rng.random((H, H))
    base = losh_multiscale(f * mask, mask, [1, 2, 4])
    shifted = losh_multiscale((f + 1e6) * mask, mask, [1, 2, 4])
    assert np.isfinite(shifted).all()
    np.testing.assert_allclose(shifted, base, rtol=1e-6)

"""Per-gene calibrated null at every radius.

The load-bearing tests are the two gates that decide whether the analytic path may be trusted at
all: G1 checks the closed-form S0/S1/S2 for the engine's row-standardised disc weights against an
explicit double sum, and G2 checks the analytic mean and variance against a permutation null on a
lattice where the two nulls coincide by construction (one cell per bin). Everything downstream —
the calibration driver, the gradient test, any figure — rests on those two passing.
"""
import numpy as np
import pytest
from scipy.stats import kstest

from spectrast import null as N
from spectrast import spatial as sp


# --------------------------------------------------------------------------- fixtures
def _toy(seed=0, n=600, side=30, hole=True):
    """Cells scattered on a side x side lattice, one radius grid, a square hole in the mask."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, size=(n, 2))
    if hole:
        keep = ~((xy[:, 0] > 10) & (xy[:, 0] < 18) & (xy[:, 1] > 10) & (xy[:, 1] < 18))
        xy = xy[keep]
    row = xy[:, 1].astype(int); col = xy[:, 0].astype(int)
    HW = (side, side)
    mask = np.zeros(HW, bool); mask[row, col] = True
    return xy, row, col, HW, mask, rng


def _planted(row, col, HW, rng, r_um=4.0):
    """A field organised at roughly `r_um`: a smooth random surface sampled at the cells."""
    from scipy.ndimage import gaussian_filter
    surf = gaussian_filter(rng.standard_normal(HW), sigma=r_um)
    return surf[row, col] + 0.3 * rng.standard_normal(len(row))


# --------------------------------------------------------------------------- Task 1: permutations as columns
def test_permute_values_columns_are_permutations():
    x = np.arange(50, dtype=float)
    V = N._permute_values(x, 7, np.random.default_rng(0))
    assert V.shape == (50, 7)
    for j in range(7):
        assert sorted(V[:, j]) == list(range(50))
    assert not np.array_equal(V[:, 0], V[:, 1])


def test_permute_values_honours_strata():
    x = np.arange(60, dtype=float)
    lab = np.repeat([0, 1, 2], 20)
    V = N._permute_values(x, 5, np.random.default_rng(1), strata=lab)
    # a value born in stratum s must land in stratum s: the value IS its index here
    for j in range(5):
        assert (lab[V[:, j].astype(int)] == lab).all()


def test_rasterize_matrix_is_the_bin_mean():
    xy, row, col, HW, mask, rng = _toy()
    x = rng.standard_normal(len(row))
    G = N._rasterize_matrix(x[:, None], row, col, HW)
    assert G.shape == (*HW, 1)
    # explicit bin mean at a bin holding several cells
    key = row * HW[1] + col
    k, cnt = np.unique(key, return_counts=True)
    kk = k[cnt.argmax()]; sel = key == kk
    assert np.isclose(G[kk // HW[1], kk % HW[1], 0], x[sel].mean())
    assert (G[~mask] == 0).all()
    const = N._rasterize_matrix(np.full((len(row), 2), 3.5), row, col, HW)
    assert np.allclose(const[mask], 3.5)


# --------------------------------------------------------------------------- Task 2: N0 / N1 for one gene
def test_planted_structure_is_significant_and_noise_is_not():
    xy, row, col, HW, mask, rng = _toy()
    radii = [1, 2, 4, 8]
    x = _planted(row, col, HW, rng, r_um=4.0)
    res = N.moran_permutation_null(x, row, col, HW, mask, radii, n_perm=199, seed=0)
    assert set(res) >= {"I", "mu", "sd", "p", "z"}
    assert res["I"].shape == (4,)
    assert res["p"][radii.index(4)] < 0.01
    noise = rng.standard_normal(len(row))
    res0 = N.moran_permutation_null(noise, row, col, HW, mask, radii, n_perm=199, seed=0)
    assert res0["p"].min() > 0.01


def test_null_p_is_uniform_for_noise():
    xy, row, col, HW, mask, rng = _toy(hole=False)
    ps = []
    for s in range(120):
        x = np.random.default_rng(1000 + s).standard_normal(len(row))
        ps.append(N.moran_permutation_null(x, row, col, HW, mask, [2], n_perm=99, seed=s)["p"][0])
    assert kstest(ps, "uniform").pvalue > 0.01


def test_single_stratum_reproduces_the_unrestricted_null():
    xy, row, col, HW, mask, rng = _toy()
    x = _planted(row, col, HW, rng)
    a = N.moran_permutation_null(x, row, col, HW, mask, [2, 4], n_perm=50, seed=3)
    b = N.moran_permutation_null(x, row, col, HW, mask, [2, 4], n_perm=50, seed=3,
                                 strata=np.zeros(len(row), int))
    assert np.array_equal(a["null"], b["null"])


def test_restricted_null_absorbs_a_gradient_the_free_null_calls():
    """A gene that IS the section-wide gradient: N0 calls it, N1 (strata along the trend) does not.

    RESOLUTION MATTERS, and this test records it. With ten bands over a thirty-bin gradient each
    band still carries a within-band trend of ~0.1 against noise of 0.15, and the restricted null
    legitimately lacks that residual, so p stays small at the finest radius (measured 0.005 at
    r = 2). Strata at the analysis resolution (one bin wide) absorb it. The same coarse-vs-fine
    behaviour is documented for the composition null; N1 inherits it and the calibration driver
    reports the stratum resolution it used.
    """
    xy, row, col, HW, mask, rng = _toy(hole=False)
    grad = xy[:, 0] / HW[1]                              # left-to-right trend, the artefact of 2026-09-17
    x = grad + 0.15 * rng.standard_normal(len(row))
    radii = [2, 4, 8]
    free = N.moran_permutation_null(x, row, col, HW, mask, radii, n_perm=199, seed=0)
    assert free["p"].max() < 0.01
    fine = np.minimum((grad * HW[1]).astype(int), HW[1] - 1)       # one band per bin column
    restricted = N.moran_permutation_null(x, row, col, HW, mask, radii, n_perm=499, seed=0,
                                          strata=fine)
    # absorbed at every radius LARGER than the stratum width (measured 0.44, 0.27 at r = 4, 8);
    # at r = 2, equal to the width, a residual remains (0.04) and only the weaker bound below holds
    assert restricted["p"][1:].min() > 0.05
    assert (restricted["p"] > free["p"]).all()
    coarse = np.minimum((grad * 10).astype(int), 9)                # ten bands: a residual leaks
    partial = N.moran_permutation_null(x, row, col, HW, mask, radii, n_perm=199, seed=0,
                                       strata=coarse)
    assert (partial["p"] >= free["p"]).all()                       # weaker, never stronger, than N0 (ties at the p floor)


# --------------------------------------------------------------------------- Task 3: A0 (gates G1, G2)
def _explicit_sums(mask, r):
    """S0, S1, S2 by brute force for the engine's weights: w_ij = 1/n_i on the disc, centre excluded."""
    H, W = mask.shape
    occ = np.argwhere(mask)
    idx = {tuple(p): i for i, p in enumerate(occ)}
    n = len(occ)
    Wm = np.zeros((n, n))
    for i, (y, x) in enumerate(occ):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                if dy * dy + dx * dx <= r * r:
                    j = idx.get((y + dy, x + dx))
                    if j is not None:
                        Wm[i, j] = 1.0
    rs = Wm.sum(1)
    Wm = np.divide(Wm, rs[:, None], out=np.zeros_like(Wm), where=rs[:, None] > 0)
    S0 = Wm.sum()
    S1 = 0.5 * ((Wm + Wm.T) ** 2).sum()
    S2 = ((Wm.sum(1) + Wm.sum(0)) ** 2).sum()
    return n, S0, S1, S2, Wm


def test_G1_analytic_S_terms_match_explicit_sums():
    xy, row, col, HW, mask, rng = _toy()
    radii = [1, 2, 4]
    M = N.moran_analytic_moments(mask, radii)
    for k, r in enumerate(radii):
        n, S0, S1, S2, _ = _explicit_sums(mask, r)
        assert M["N"] == n
        for name, mine, ref in (("S0", M["S0"][k], S0), ("S1", M["S1"][k], S1), ("S2", M["S2"][k], S2)):
            assert abs(mine - ref) <= 1e-10 * max(1.0, abs(ref)), (name, r, mine, ref)


def test_G2_analytic_moments_match_permutation_when_bins_are_cells():
    """One cell per bin, so permuting cells IS permuting bins and the two nulls must coincide."""
    xy, row, col, HW, mask, rng = _toy(hole=True)
    # collapse to one cell per occupied bin
    key = row * HW[1] + col
    _, first = np.unique(key, return_index=True)
    row, col = row[first], col[first]
    radii = [1, 2, 4]
    x = _planted(row, col, HW, rng)
    M = N.moran_analytic_moments(mask, radii)
    G = N._rasterize_matrix(x[:, None], row, col, HW)
    scal = sp.morans_i_multiscale(G, mask, radii)
    E, V, z = N.moran_analytic_z(scal, G, mask, M)
    P = 2000
    emp = N.moran_permutation_null(x, row, col, HW, mask, radii, n_perm=P, seed=0)
    for k in range(len(radii)):
        assert abs(E[k, 0] - emp["mu"][k]) < 3 * emp["sd"][k] / np.sqrt(P), radii[k]
        assert 0.9 < np.sqrt(V[k, 0]) / emp["sd"][k] < 1.1, radii[k]
    # and the closed-form mean is exact
    for k in range(len(radii)):
        assert np.isclose(E[k, 0], -(M["S0"][k] / M["N"]) / (M["N"] - 1))


# --------------------------------------------------------------------------- Task 4: strata from a covariate
def test_covariate_strata_follow_a_smoothed_gradient():
    xy, row, col, HW, mask, rng = _toy(hole=False, n=1500)
    cov = xy[:, 0] + 0.5 * rng.standard_normal(len(xy))            # a trend plus per-cell noise
    lab = N.covariate_strata(xy, cov, bin_um=1.0, radius_um=4.0, n_strata=5, HW=HW, row=row, col=col)
    assert lab.min() == 0 and lab.max() == 4
    # smoothing at 4 bins makes the strata bands along x: label rises monotonically with x-position
    from scipy.stats import spearmanr
    assert spearmanr(xy[:, 0], lab).statistic > 0.95
    # and a cell's label is decided by its neighbourhood, not its own noise: cells at the same x share it
    xb = col
    same = [len(np.unique(lab[xb == c])) for c in np.unique(xb)]
    assert np.median(same) <= 2


def test_covariate_strata_collapse_and_warn_on_a_constant():
    xy, row, col, HW, mask, rng = _toy(hole=False)
    with pytest.warns(RuntimeWarning):
        lab = N.covariate_strata(xy, np.ones(len(xy)), bin_um=1.0, radius_um=2.0, n_strata=10,
                                 HW=HW, row=row, col=col)
    assert len(np.unique(lab)) == 1


# --------------------------------------------------------------------------- Task 5: gr.moran_null
def _toy_adata(seed=0, n=900, side=40, n_genes=6):
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side * 10.0, size=(n, 2))                 # "µm", pitch ~ 13
    from scipy.ndimage import gaussian_filter
    X = np.empty((n, n_genes))
    for j in range(n_genes):
        if j < 3:                                                 # organised
            surf = gaussian_filter(rng.standard_normal((side, side)), sigma=3.0)
            X[:, j] = surf[(xy[:, 1] / 10).astype(int), (xy[:, 0] / 10).astype(int)] + 0.3 * rng.standard_normal(n)
        else:                                                     # noise
            X[:, j] = rng.standard_normal(n)
    X = X - X.min() + 0.01
    a = ad.AnnData(X=X, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    a.obsm["spatial"] = xy
    a.obs["band"] = pd.Categorical(np.where(xy[:, 0] < side * 5.0, "left", "right"))
    return a


def test_gr_moran_null_analytic_table_and_provenance():
    from spectrast import gr
    a = _toy_adata()
    df = gr.moran_null(a, method="analytic")
    assert set(df.columns) >= {"gene", "r_bin", "r_um", "I", "mu", "sd", "z", "p"}
    assert df.gene.nunique() == 6
    assert df.attrs["spectrast_null_method"] == "analytic"
    assert df.attrs["spectrast_null_kind"] == "A0"
    for k in ("spectrast_backend", "bin_um", "radii_bins", "radii_um", "n_cells", "n_bins",
              "cells_per_bin", "seed"):
        assert k in df.attrs, k
    # organised genes are called somewhere; pure noise is not called at the Bonferroni level
    zmax = df.groupby("gene").z.max()
    assert (zmax[["g0", "g1", "g2"]] > 3.9).all()
    assert (zmax[["g3", "g4", "g5"]] < 3.9).all()
    # written back to var
    assert "spectrast_null_zmax" in a.var and "spectrast_null_r_zmax_um" in a.var


def test_gr_moran_null_permutation_matches_module_call_bit_for_bit():
    from spectrast import gr
    a = _toy_adata()
    df = gr.moran_null(a, genes=["g0", "g4"], method="permutation", n_perm=30, seed=7)
    assert df.attrs["spectrast_null_kind"] == "N0"
    # rebuild the lattice exactly as the wrapper does and call the module directly
    xy = a.obsm["spatial"]; b = df.attrs["bin_um"]
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / b).astype(int); row = (xyz[:, 1] / b).astype(int)
    HW = (int(row.max()) + 1, int(col.max()) + 1)
    mask = np.zeros(HW, bool); mask[row, col] = True
    j = list(a.var_names).index("g4")
    res = N.moran_permutation_null(np.asarray(a.X[:, j]).ravel(), row, col, HW, mask,
                                   df.attrs["radii_bins"], n_perm=30, seed=7 + j)
    got = df[df.gene == "g4"].sort_values("r_bin")
    assert np.array_equal(got.z.to_numpy(), res["z"])


def test_gr_moran_null_strata_options_are_exclusive_and_analytic_refuses_them():
    from spectrast import gr
    a = _toy_adata()
    with pytest.raises(ValueError):
        gr.moran_null(a, method="permutation", n_perm=10, strata_key="band",
                      covariate=np.ones(a.n_obs))
    with pytest.raises(ValueError):
        gr.moran_null(a, method="analytic", strata_key="band")
    df = gr.moran_null(a, genes=["g0"], method="permutation", n_perm=20, strata_key="band")
    assert df.attrs["spectrast_null_kind"] == "N1"
    assert df.attrs["strata"]["source"] == "obs:band"
    cov = a.obsm["spatial"][:, 0]                                     # a continuous covariate
    # smoothing radius set explicitly: the default (largest swept radius, ~520 um) is wider than
    # this 400 um toy, flattens the gradient, and the strata honestly collapse to two
    df2 = gr.moran_null(a, genes=["g0"], method="permutation", n_perm=20, covariate=cov,
                        n_strata=4, covariate_radius_um=30.0)
    assert df2.attrs["spectrast_null_kind"] == "N1"
    assert df2.attrs["strata"]["n_strata"] == 4
    assert df2.attrs["strata"]["smoothed_at_um"] == 30.0

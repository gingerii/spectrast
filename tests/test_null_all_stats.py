"""Per-gene nulls for every organization statistic, not just Moran."""
import warnings

import numpy as np
import pytest

from spectrast import spatial as sp
from spectrast.null import (PERM_STATS, moran_analytic_moments, moran_permutation_null,
                            permutation_null, geary_analytic_z, _rasterize_matrix)


def _lattice(n=5000, H=48, W=48, seed=0):
    """48x48 rather than something smaller because TEXTURE needs Fourier modes per band: the same
    structured field separates at 2.4x on 26x26 and 6.1x here (11.7x on 72x72). The disc statistics
    are fine either way; sizing the fixture to the weakest statistic keeps the test honest."""
    rng = np.random.default_rng(seed)
    row = rng.integers(0, H, n); col = rng.integers(0, W, n)
    cnt = np.zeros((H, W)); np.add.at(cnt, (row, col), 1.0)
    return row, col, (H, W), cnt > 0, [2, 3, 5, 8, 12]


def _structured(row, col, HW, seed=0):
    """A gene whose value depends on position: structure by construction.

    The blob's WIDTH and CENTRE vary with the seed, not just the noise. An earlier version fixed
    both, so different seeds produced near-identical genes (analytic z 80.1, 80.0, 80.2) and any
    rank correlation over them was measuring permutation noise inside a tie."""
    rng = np.random.default_rng(seed)
    cx = HW[1] * (0.35 + 0.30 * rng.random())
    cy = HW[0] * (0.35 + 0.30 * rng.random())
    width = HW[0] * (0.10 + 0.18 * rng.random())
    d2 = (col - cx) ** 2 + (row - cy) ** 2
    # clipped at zero: General G is defined for a non-negative variable and warns otherwise
    return np.clip(np.exp(-d2 / (2 * width ** 2)) * 5.0 + rng.normal(0, 0.1, row.size), 0, None)


def test_dispatch_reproduces_the_pinned_moran_function():
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    a = moran_permutation_null(x, row, col, HW, mask, rb, n_perm=10, seed=7)
    b = permutation_null(x, row, col, HW, mask, rb, stat="moran", n_perm=10, seed=7)
    assert np.allclose(a["z"], b["z"], atol=0, rtol=0)
    assert np.allclose(a["mu"], b["mu"], atol=0, rtol=0)


@pytest.mark.parametrize("stat", PERM_STATS)
def test_every_statistic_has_a_usable_null(stat):
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    r = permutation_null(x, row, col, HW, mask, rb, stat=stat, n_perm=12, seed=0)
    assert r["sd"].shape == (len(rb),)
    assert np.all(r["sd"] > 0), "a degenerate null cannot support inference"
    assert np.isfinite(r["z"]).all()
    assert np.all((r["p_two"] >= 0) & (r["p_two"] <= 1))


@pytest.mark.parametrize("stat", PERM_STATS)
def test_a_structured_gene_separates_from_an_unstructured_one(stat):
    row, col, HW, mask, rb = _lattice()
    rng = np.random.default_rng(1)
    hi = _structured(row, col, HW)
    lo = rng.permutation(hi)                      # same values, arrangement destroyed
    zh = permutation_null(hi, row, col, HW, mask, rb, stat=stat, n_perm=12, seed=0)["z"]
    zl = permutation_null(lo, row, col, HW, mask, rb, stat=stat, n_perm=12, seed=0)["z"]
    assert np.nanmax(np.abs(zh)) > 3 * np.nanmax(np.abs(zl))


def test_the_default_is_two_sided_and_no_direction_is_assumed():
    """An earlier version derived the tail from whether the curve rises or falls with radius, which
    is right for a half-maximum and wrong here: General G rises because it is CUMULATIVE, and that
    version returned p = 1.0 for a gene sitting hundreds of sd above its own null."""
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    r = permutation_null(x, row, col, HW, mask, rb, stat="general_g", n_perm=12, seed=0)
    assert "p" not in r, "a one-sided p must be asked for, not assumed"
    assert r["higher"] is None
    k = int(np.nanargmax(np.abs(r["z"])))
    assert r["p_two"][k] < 0.5, "a strongly deviating gene must not read as non-significant"


def test_a_one_sided_p_is_returned_only_when_the_direction_is_given():
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    up = permutation_null(x, row, col, HW, mask, rb, stat="moran", n_perm=12, higher=True, seed=0)
    dn = permutation_null(x, row, col, HW, mask, rb, stat="moran", n_perm=12, higher=False, seed=0)
    assert "p" in up and "p" in dn
    assert np.all(up["p"] <= dn["p"]), "Moran: structure sits in the upper tail"


def test_dispatch_rejects_an_unknown_statistic():
    row, col, HW, mask, rb = _lattice()
    with pytest.raises(ValueError, match="unknown statistic"):
        permutation_null(np.zeros(row.size), row, col, HW, mask, rb, stat="nope", n_perm=4)


def test_geary_analytic_tracks_its_own_permutation_null():
    """Ranking, not absolute calibration -- the two routes answer different nulls and the Geary mean
    offset is measured at ~0.5 sd (see geary_analytic_z)."""
    from scipy.stats import spearmanr
    row, col, HW, mask, rb = _lattice()
    mom = moran_analytic_moments(mask, rb)
    rng = np.random.default_rng(3)
    za, zp = [], []
    for k in range(6):
        x = _structured(row, col, HW, seed=k) if k % 2 == 0 else np.abs(rng.normal(1, 1, row.size))
        r = permutation_null(x, row, col, HW, mask, rb, stat="geary", n_perm=80, seed=k)
        G = _rasterize_matrix(x[:, None], row, col, HW)
        _E, _V, z = geary_analytic_z(r["obs"][:, None], G, mask, mom)
        za.append(z[:, 0]); zp.append(r["z"])
    za = np.concatenate(za); zp = np.concatenate(zp)
    ok = np.isfinite(za) & np.isfinite(zp)
    assert spearmanr(za[ok], zp[ok]).statistic > 0.95


def test_texture_multiscale_has_the_shared_signature():
    row, col, HW, mask, rb = _lattice()
    grid = _rasterize_matrix(_structured(row, col, HW)[:, None], row, col, HW)
    out = sp.texture_multiscale(grid, mask, rb, kind="disc")
    assert out.shape == (len(rb), 1)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------ General G analytic

def test_general_g_binary_weight_sums_match_an_explicit_double_sum():
    """The weight convention is the trap: General G is NOT row standardised, so it needs its own
    sums. Checked against the literal double sum rather than against another FFT."""
    from spectrast.null import general_g_binary_moments
    rng = np.random.default_rng(0)
    H, W = 12, 13
    mask = rng.random((H, W)) > 0.35
    radii = [2, 3]
    mom = general_g_binary_moments(mask, radii)
    idx = np.argwhere(mask)
    for k, r in enumerate(radii):
        K = sp.neighbor_kernel(r, "disc"); off = (K.shape[0] - 1) // 2
        Wm = np.zeros((len(idx), len(idx)))
        for a, (i0, j0) in enumerate(idx):
            for b, (i1, j1) in enumerate(idx):
                di, dj = i1 - i0 + off, j1 - j0 + off
                if 0 <= di < K.shape[0] and 0 <= dj < K.shape[1] and K[di, dj] > 0:
                    Wm[a, b] = 1.0
        assert np.allclose(Wm, Wm.T) and np.all(np.diag(Wm) == 0)
        assert np.isclose(mom["W"][k], Wm.sum())
        assert np.isclose(mom["S1"][k], 0.5 * ((Wm + Wm.T) ** 2).sum())
        assert np.isclose(mom["S2"][k], ((Wm.sum(1) + Wm.sum(0)) ** 2).sum())


def test_general_g_analytic_tracks_its_own_permutation_null():
    from scipy.stats import spearmanr
    from spectrast.null import general_g_analytic_z, general_g_binary_moments
    row, col, HW, mask, rb = _lattice()
    mom = general_g_binary_moments(mask, rb)
    rng = np.random.default_rng(5)
    za, zp = [], []
    for k in range(6):
        x = (_structured(row, col, HW, seed=k) if k % 2 == 0
             else np.abs(rng.normal(1, 1, row.size)))
        r = permutation_null(x, row, col, HW, mask, rb, stat="general_g", n_perm=80, seed=k)
        G = _rasterize_matrix(x[:, None], row, col, HW)
        _E, _V, z = general_g_analytic_z(r["obs"][:, None], G, mask, mom)
        za.append(z[:, 0]); zp.append(r["z"])
    za = np.concatenate(za); zp = np.concatenate(zp)
    ok = np.isfinite(za) & np.isfinite(zp)
    assert spearmanr(za[ok], zp[ok]).statistic > 0.95


def test_general_g_moments_are_not_interchangeable_with_morans():
    """Mixing the two weight conventions is silent, so make it loud in a test instead."""
    from spectrast.null import general_g_binary_moments, moran_analytic_moments
    row, col, HW, mask, rb = _lattice()
    gg = general_g_binary_moments(mask, rb)
    mo = moran_analytic_moments(mask, rb)
    assert not np.allclose(gg["S1"], mo["S1"]), "binary and row-standardised S1 must differ"
    assert "W" in gg and "S0" in mo


# ------------------------------------------------------------------ the unified entry point

def test_every_statistic_has_a_route_and_a_default():
    from spectrast.null import ROUTES
    assert set(ROUTES) == set(PERM_STATS)
    for stat, spec in ROUTES.items():
        assert spec["default"] in ("analytic", "surrogate", "permutation")
        if spec["default"] == "analytic":
            assert spec["analytic"], f"{stat} defaults to a route it does not have"
        if spec["default"] == "surrogate":
            assert spec["surrogate"], f"{stat} defaults to a route it does not have"


@pytest.mark.parametrize("stat,expected", [("moran", "analytic"), ("geary", "analytic"),
                                           ("general_g", "analytic"), ("focality", "surrogate"),
                                           ("losh", "surrogate"), ("texture", "permutation")])
def test_auto_picks_the_measured_default(stat, expected):
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    V = np.column_stack([_structured(row, col, HW, seed=k) for k in range(30)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = gene_inference(V, row, col, HW, mask, rb, stat=stat, method="auto",
                             n_perm=8, n_sample=25)
    assert out["route"] == expected
    assert out["z"].shape == (len(rb), V.shape[1])


def test_permutation_is_available_for_every_statistic():
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    for stat in PERM_STATS:
        out = gene_inference(x, row, col, HW, mask, rb, stat=stat, method="permutation", n_perm=8)
        assert out["route"] == "permutation"
        assert "p_two" in out, "only the permutation route may offer a tail probability"


def test_asking_for_an_analytic_route_that_does_not_exist_explains_why():
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    with pytest.raises(ValueError, match="no analytic null"):
        gene_inference(x, row, col, HW, mask, rb, stat="losh", method="analytic")


def test_the_surrogate_is_refused_for_texture_with_its_measured_reason():
    """Not discouraged -- refused. Its agreement with the reference is non-monotonic in sample
    fraction, so a bigger sample is not a remedy."""
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    V = np.column_stack([_structured(row, col, HW, seed=k) for k in range(30)])
    with pytest.raises(ValueError, match="refused"):
        gene_inference(V, row, col, HW, mask, rb, stat="texture", method="surrogate")


def test_losh_warns_that_its_default_is_ranking_only():
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    V = np.column_stack([_structured(row, col, HW, seed=k) for k in range(30)])
    with pytest.warns(RuntimeWarning, match="RANKING ONLY"):
        gene_inference(V, row, col, HW, mask, rb, stat="losh", method="auto",
                       n_perm=8, n_sample=25)


def test_only_the_permutation_route_offers_a_tail_probability():
    """Analytic and surrogate rank well and neither is calibrated as a sigma count on real tissue,
    so neither returns a p that a caller could mistake for one."""
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    V = np.column_stack([_structured(row, col, HW, seed=k) for k in range(30)])
    a = gene_inference(V, row, col, HW, mask, rb, stat="moran", method="analytic")
    assert "p_two" not in a and "p" not in a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_ = gene_inference(V, row, col, HW, mask, rb, stat="focality", method="surrogate",
                            n_perm=8, n_sample=25)
    assert "p_two" not in s_ and "p" not in s_


def test_the_surrogate_refuses_a_single_gene():
    from spectrast.null import gene_inference
    row, col, HW, mask, rb = _lattice()
    x = _structured(row, col, HW)
    with pytest.raises(ValueError, match="at least 25"):
        gene_inference(x, row, col, HW, mask, rb, stat="focality", method="surrogate")


def test_texture_bands_refuse_a_single_radius():
    """A band is bounded by the midpoints to its neighbours; one radius has none, and the old code
    read an uninitialised edge array instead of saying so."""
    with pytest.raises(ValueError):
        sp.texture_bands((40, 40), [3])

"""Cell-type interaction range: density-stratified label-permutation null."""
import warnings
import numpy as np
import anndata as ad
import pytest

from spectrast import spatial as sp, null as nl, gr, composition

from _reference import assert_close_to_reference  # tests/_reference.py


def _blob_fields(seed=0, n=1500, L=300.0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * L
    a = np.exp(-((xy[:, 0] - 120) ** 2 + (xy[:, 1] - 150) ** 2) / (2 * 40 ** 2))
    b = np.exp(-((xy[:, 0] - 140) ** 2 + (xy[:, 1] - 150) ** 2) / (2 * 40 ** 2))
    bin_um = 10.0
    col = (xy[:, 0] / bin_um).astype(int); row = (xy[:, 1] / bin_um).astype(int)
    HW = (row.max() + 1, col.max() + 1)
    def field(v):
        num = np.zeros(HW); cnt = np.zeros(HW)
        np.add.at(num, (row, col), v); np.add.at(cnt, (row, col), 1.0)
        return np.where(cnt > 0, num / np.maximum(cnt, 1), 0.0), cnt
    ma, na = field(a); mb, nb = field(b)
    mask = na > 0
    return ma, mb, na, nb, mask, bin_um


_PINNED = {
    "shift": {'g': [4.398441573905466, 4.382049018069991, 4.077952166304188, 3.785500263511375, 3.2787462805080736, 2.8253086798719025], 'z': [3.757881275973104, 3.8221958168302375, 3.3943870054410348, 3.0398629575303366, 2.5958558540219383, 2.0855387185171543], 'pval': [0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616], 'null_mean': [0.9096461265962791, 0.9656859140388416, 1.02722713546819, 1.1151623580762782, 1.180068131427285, 1.267936565883423], 'null_sd': [0.928394270892915, 0.8938221032496324, 0.8987558065553034, 0.8784402266622402, 0.8084725297166104, 0.7467481184409713], 'T_max': 3.8221958168302375, 'p_max': 0.047619047619047616, 'r_at_max_um': 26.25},
    "reshuffle": {'g': [4.398441573905466, 4.382049018069991, 4.077952166304188, 3.785500263511375, 3.2787462805080736, 2.8253086798719025], 'z': [30.211018079509227, 27.954658836200416, 30.73031365078684, 30.704417983415322, 28.286434746631578, 18.650575908674607], 'pval': [0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616], 'null_mean': [1.0780045351366758, 1.1703007520806774, 1.2011588926333392, 1.265831142127865, 1.3008168268157283, 1.3555645029779304], 'null_sd': [0.1099081477502704, 0.11489134189790931, 0.09361418521015288, 0.08206210333459124, 0.06992501781893465, 0.07880422481808598], 'T_max': 30.73031365078684, 'p_max': 0.047619047619047616, 'r_at_max_um': 37.083333333333336},
}


@pytest.mark.parametrize("null", ["shift", "reshuffle"])
def test_refactor_keeps_mcc_run_output(null):
    """_mcc_run through _mcc_curves/_mcc_score reproduces the pre-refactor output.

    The reference values were computed once by the public entry point at a fixed seed and pinned
    here; they hold to floating-point rounding on any machine.
    """
    ma, mb, na, nb, mask, bin_um = _blob_fields()
    out = sp.mark_cross_correlation(ma, mb, mask, density_a=na, density_b=nb,
                                    normalize="density", bin_size=bin_um, n_bins=6,
                                    null=null, n_perm=20, seed=3)
    ref = _PINNED[null]
    for k in ("g", "z", "pval", "null_mean", "null_sd", "T_max", "p_max", "r_at_max_um"):
        assert_close_to_reference(out[k], ref[k], k)


# --------------------------------------------------------------------------- Task 2: density strata
def _lattice(xy, bin_um):
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / bin_um).astype(int); row = (xyz[:, 1] / bin_um).astype(int)
    return row, col, (int(row.max()) + 1, int(col.max()) + 1)


def test_local_density_is_a_disc_sum_per_area():
    rng = np.random.default_rng(0)
    xy = rng.random((4000, 2)) * 500
    row, col, HW = _lattice(xy, 10.0)
    d = nl.local_density(row, col, HW, radius_bins=3)
    # a disc of radius 3 bins holds 29 lattice bins; density is cells per bin, so the mean over
    # interior bins is ~ 4000 / (50*50) = 1.6
    interior = d[5:-5, 5:-5]
    assert abs(interior.mean() - 1.6) < 0.15
    assert d.shape == HW and d.min() >= 0


def test_density_strata_follow_a_gradient_and_collapse_when_constant():
    rng = np.random.default_rng(1)
    # left half twice as dense as the right half
    left = rng.random((6000, 2)) * [250, 500]; right = rng.random((3000, 2)) * [250, 500] + [250, 0]
    xy = np.vstack([left, right])
    row, col, HW = _lattice(xy, 10.0)
    lab = nl.density_strata(row, col, HW, radius_bins=5, n_strata=3)
    assert lab.shape == (9000,) and lab.max() == 2
    # high strata sit on the dense side
    assert (lab[:6000] >= 1).mean() > 0.75
    assert (lab[6000:] == 0).mean() > 0.75
    # a regular grid has constant density -> one stratum, with a warning
    gx, gy = np.meshgrid(np.arange(0, 500, 5.0), np.arange(0, 500, 5.0))
    row, col, HW = _lattice(np.c_[gx.ravel(), gy.ravel()], 10.0)
    with pytest.warns(RuntimeWarning, match="strata"):
        lab = nl.density_strata(row, col, HW, radius_bins=5, n_strata=4)
    assert len(np.unique(lab[np.isin(row, range(8, 42)) & np.isin(col, range(8, 42))])) == 1


def test_density_scale_um_recovers_the_nest_scale():
    """Matérn nests of radius 60 um give a density scale in the right decade; a uniform field
    gives NaN (no crossing) or a scale at the grid floor — either way far below the nest case."""
    # 8000 cells (the brief's literal figure) never clears DENSITY_SCALE_MIN_I at this bin_um
    # (curve.max ~0.028 vs the 0.05 gate) -- a shot-noise/power limit of the sparse count field,
    # not a bounds miss; 16000 clears it with margin (~0.076-0.094 across seeds 0-3) and lands the
    # scale at 83-94 um, inside the physically sensible decade for a 60 um nest.
    pts = nl.simulate_typed_points(n_types=1, n_cells=16000, L_um=2000.0, nest_um=60.0,
                                   gradient_um=None, seed=0)
    row, col, HW = _lattice(pts["xy"], 10.0)
    mask = np.zeros(HW, bool); mask[row, col] = True
    s = nl.density_scale_um(row, col, HW, mask, bin_um=10.0)
    assert 30 < s < 200
    rng = np.random.default_rng(0)
    row, col, HW = _lattice(rng.random((8000, 2)) * 2000, 10.0)
    mask = np.zeros(HW, bool); mask[row, col] = True
    s0 = nl.density_scale_um(row, col, HW, mask, bin_um=10.0)
    assert not np.isfinite(s0)


# --------------------------------------------------------------------------- Task 3: typed simulator
def test_simulator_types_follow_their_preferences_and_ring_is_planted():
    pts = nl.simulate_typed_points(n_types=5, n_cells=20000, L_um=3000.0, nest_um=100.0,
                                   gradient_um=1000.0, ring=(0, 4, 120.0, 800), seed=0)
    xy, lab, dens = pts["xy"], pts["labels"], pts["density"]
    assert xy.shape == (20800, 2) and lab.shape == (20800,) and lab.max() == 4
    # dense-loving type sits at higher density than the indifferent type; nanmedian because the
    # planted ring cells (also labelled 4) were never thinned and carry density = nan by design
    assert np.nanmedian(dens[lab == 0]) > np.nanmedian(dens[lab == 4]) * 1.3
    # sparse-loving type sits at LOWER density than the indifferent type, and -- since the
    # gradient is low on the low-x side -- on the low-x side. The factor is 1.1, not the 1.3 kept
    # for the dense-loving check above: log(intensity) in [-0.69, +1.79] is asymmetric (nest is a
    # 1x-4x multiplier, gradient only 0.5x-1.5x), so a unit sparse-loving slope's downside is
    # structurally smaller than a unit dense-loving slope's upside.
    assert np.nanmedian(dens[lab == 2]) < np.nanmedian(dens[lab == 4]) / 1.1
    assert np.mean(xy[lab == 2, 0]) < np.mean(xy[lab == 4, 0]) - 200
    # axis_prefs (the deliberately density-independence-breaking model) reproduces the old
    # high-x behaviour for a gradient-preferring type -- it exists to size the hidden-covariate
    # limit, not to describe the default model above
    axis = nl.simulate_typed_points(n_types=5, n_cells=20000, L_um=3000.0, nest_um=100.0,
                                    gradient_um=1000.0,
                                    axis_prefs=[(2.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 2.0),
                                               (0.0, 0.0)],
                                    ring=(0, 4, 120.0, 800), seed=0)
    xa, la = axis["xy"], axis["labels"]
    assert np.mean(xa[la == 2, 0]) > np.mean(xa[la == 4, 0]) + 200
    # the planted ring cells lie 100-140 um from SOME type-0 cell
    from scipy.spatial import cKDTree
    t0 = cKDTree(xy[lab == 0])
    ring_xy = xy[-800:]
    d, _ = t0.query(ring_xy)
    assert (d <= 140.0).mean() > 0.9
    assert (lab[-800:] == 4).all()
    # source-side contract: every ring cell sits at r_um +/- 15% from ITS OWN anchor (not merely
    # its nearest type-0 neighbour, which can be closer by chance) and strictly inside the field
    ring_src = pts["ring_src"]
    src_dist = np.linalg.norm(ring_xy - xy[ring_src], axis=1)
    assert np.all(np.abs(src_dist - 120.0) <= 0.15 * 120.0 + 1e-9)
    assert np.all((ring_xy >= 0) & (ring_xy <= 3000.0))


def test_simulator_is_seeded_and_switchable():
    a = nl.simulate_typed_points(seed=3); b = nl.simulate_typed_points(seed=3)
    np.testing.assert_array_equal(a["xy"], b["xy"]); np.testing.assert_array_equal(a["labels"], b["labels"])
    u = nl.simulate_typed_points(n_types=1, n_cells=3000, nest_um=None, gradient_um=None, seed=0)
    assert np.ptp(u["density"]) < 1e-9          # no structure -> flat density


def test_density_scale_um_is_readable_on_a_sparse_lattice():
    """Design decision: block-aggregate the count field before the scalogram so a sparse lattice
    (~0.2 cells/bin, what real MERFISH sections hit at 10 um bins) is not shot-noise-blind."""
    pts = nl.simulate_typed_points(n_types=1, n_cells=8000, L_um=2000.0, nest_um=60.0,
                                   gradient_um=None, seed=0)
    row, col, HW = _lattice(pts["xy"], 10.0)
    mask = np.zeros(HW, bool); mask[row, col] = True
    s = nl.density_scale_um(row, col, HW, mask, bin_um=10.0)
    assert 30 <= s <= 200


def test_density_scale_block_factor_tracks_sparsity():
    """Review fix: the block factor must be measured over the TISSUE (occupied COARSE blocks), not
    over fine bins that already hold >=1 cell (which are >=1 cell/bin by definition and could never
    see a true density below 1 cell/bin, capping f at 2 no matter how sparse the field really is)."""
    rng = np.random.default_rng(0)
    sparse = rng.poisson(0.2, size=(300, 300)).astype(float)
    assert nl._density_block_factor(sparse) == 5
    rng2 = np.random.default_rng(0)
    dense = rng2.poisson(2.0, size=(300, 300)).astype(float)
    assert nl._density_block_factor(dense) == 2


# --------------------------------------------------------------------------- Task 4: cross-PCF null
def _typed_lattice(pts, bin_um=10.0):
    row, col, HW = _lattice(pts["xy"], bin_um)
    mask = np.zeros(HW, bool); mask[row, col] = True
    return row, col, HW, mask


def _direct_cross_pcf(xy, labels, a, b, edges, mask_area_bins, bin_um):
    """Brute-force cross-PCF on the LATTICE (bin centres), for the exactness gate G1."""
    from scipy.spatial import cKDTree
    # bin the cells exactly as the engine does, then count bin-pairs weighted by counts
    row, col, HW = _lattice(xy, bin_um)
    key = row * HW[1] + col
    def counts(t):
        c = np.bincount(key[labels == t], minlength=HW[0] * HW[1]).reshape(HW)
        return c
    ca, cb = counts(a), counts(b)
    occ = np.zeros(HW); np.add.at(occ, (row, col), 1.0); occ = occ > 0
    ys, xs = np.nonzero(occ)
    P = np.c_[ys, xs] * bin_um
    tree = cKDTree(P)
    wa, wb = ca[ys, xs], cb[ys, xs]
    N = occ.sum()
    g = np.empty(len(edges) - 1)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        pairs_hi = tree.query_pairs(hi - 1e-9, output_type="ndarray")
        pairs_lo = tree.query_pairs(lo - 1e-9, output_type="ndarray") if lo > 0 else np.empty((0, 2), int)
        def S(pp, w1, w2):
            return (w1[pp[:, 0]] * w2[pp[:, 1]]).sum() + (w2[pp[:, 0]] * w1[pp[:, 1]]).sum()
        num = S(pairs_hi, wa, wb) - S(pairs_lo, wa, wb)
        npair = 2 * (len(pairs_hi) - len(pairs_lo))
        g[k] = (num / npair) / ((wa.sum() / N) * (wb.sum() / N))
    return g


def test_g1_observed_curve_equals_direct_pair_counting():
    pts = nl.simulate_typed_points(n_types=3, n_cells=1500, L_um=600.0, nest_um=None,
                                   gradient_um=None, seed=0)
    row, col, HW, mask = _typed_lattice(pts, 10.0)
    edges = np.array([10.0, 30.0, 60.0, 100.0])
    strata = np.zeros(len(row), int)
    out = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, edges, strata=strata,
                                   n_perm=0, bin_size=10.0)
    k = out["pairs"].index((0, 1))
    ref = _direct_cross_pcf(pts["xy"], pts["labels"], 0, 1, edges, mask.sum(), 10.0)
    np.testing.assert_allclose(out["g"][k], ref, rtol=1e-9)


def test_one_stratum_is_random_labelling_and_draws_are_shared_across_pairs():
    pts = nl.simulate_typed_points(n_types=3, n_cells=3000, L_um=800.0, seed=1)
    row, col, HW, mask = _typed_lattice(pts, 10.0)
    edges = np.geomspace(10, 200, 6)
    one = np.zeros(len(row), int)
    out = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, edges, strata=one,
                                   n_perm=5, seed=7, bin_size=10.0)
    # reproduce draw 0 by hand: a plain permutation of the labels with the same generator
    rng = np.random.default_rng(7)
    perm = composition.stratified_permutation(one, rng)
    lab0 = pts["labels"][perm]
    ref = nl.cross_pcf_strata_null(lab0, row, col, HW, mask, edges, strata=one, n_perm=0,
                                   bin_size=10.0)
    np.testing.assert_array_equal(out["draws"][0], ref["g"])
    # one draw serves every pair: the pair list's order does not change any pair's draws
    sub = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, edges, strata=one,
                                   n_perm=5, seed=7, bin_size=10.0, self_pairs=False)
    for p, pr in enumerate(sub["pairs"]):
        np.testing.assert_array_equal(sub["draws"][:, p], out["draws"][:, out["pairs"].index(pr)])


def test_untyped_cells_stay_in_tissue_and_out_of_the_permutation():
    pts = nl.simulate_typed_points(n_types=3, n_cells=3000, L_um=800.0, seed=2)
    row, col, HW, mask = _typed_lattice(pts, 10.0)
    lab = pts["labels"].copy(); lab[lab == 2] = -1
    strata = nl.density_strata(row, col, HW, radius_bins=10, n_strata=4)
    out = nl.cross_pcf_strata_null(lab, row, col, HW, mask, np.geomspace(10, 200, 6),
                                   strata=strata, n_perm=3, seed=0, bin_size=10.0)
    assert out["pairs"] == [(0, 0), (0, 1), (1, 1)]
    # the tissue mask is the whole section (untyped cells included): n_pairs matches all cells
    full = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, np.geomspace(10, 200, 6),
                                    strata=strata, n_perm=0, bin_size=10.0)
    np.testing.assert_array_equal(out["n_pairs"], full["n_pairs"])
    # the claim itself, directly on the group encoding the function permutes: an untyped cell never
    # ends up at a position that was untyped before the permutation, or vice versa, and the typed
    # cells' labels are merely REARRANGED among themselves (same value counts), never touched by an
    # untyped cell's identity
    rng = np.random.default_rng(0)
    groups = np.where(lab >= 0, strata, -1 - strata)
    perm = composition.stratified_permutation(groups, rng)
    assert np.array_equal(lab[perm] == -1, lab == -1)
    typed = lab >= 0
    np.testing.assert_array_equal(np.sort(lab[perm][typed]), np.sort(lab[typed]))


def test_planted_ring_is_detected():
    """The synthetic of the sensitivity design at its reference cell, one section.

    Only the planted pair is asserted here. Calibration under this cell's strong (nest-driven)
    density heterogeneity is a residual-bias property of the null in the sparse regime -- strata
    are built on a SMOOTHED but still REALISED density (~18 cells per 50 um disc at this cell's
    density), and the part of P(type | true density) the realised proxy misses is itself spatially
    autocorrelated, so the permutation (which repositions same-stratum labels uniformly across the
    whole section) destroys structure the proxy never captured in the first place -- not a defect
    in this implementation. It is MEASURED, not asserted, by the G2 sensitivity study;
    at this exact cell (R_s=50 um, n_strata=10, n_perm=99,
    seed=0) the unrelated pair (0,1) sits at T_max ~= 10 and the indifferent type's own self-pair
    (4,4) at T_max ~= 13, both above this n_perm's floor (the noisy-proxy mechanism).
    """
    pts = nl.simulate_typed_points(n_types=5, n_cells=20000, L_um=3000.0, nest_um=100.0,
                                   gradient_um=1000.0, ring=(0, 4, 120.0, 1500), seed=0)
    row, col, HW, mask = _typed_lattice(pts, 10.0)
    edges = np.geomspace(10, 500, 13)
    strata = nl.density_strata(row, col, HW, radius_bins=5, n_strata=10)    # R_s = 50 um
    out = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, edges, strata=strata,
                                   n_perm=99, seed=0, bin_size=10.0)
    k = out["pairs"].index((0, 4))
    assert out["p_max"][k] <= 0.02
    assert 80 <= out["r_at_max_um"][k] <= 180


def test_homogeneous_density_is_calibrated():
    """With no true density heterogeneity, the null's mechanics alone are well-calibrated.

    Unlike the deterministic grid in `density_strata`'s own collapse test, a STOCHASTIC point
    process -- even under a perfectly uniform intensity -- has real Poisson shot-noise variance in
    local density at any finite smoothing radius, so the strata here do NOT collapse (measured: 10,
    i.e. `n_strata` in full -- no collapse warning fires, so no `warnings.catch_warnings()` guard is
    needed here). At `n_perm=99` one pair, (1,3), dips to `p_max=0.03`: pure Monte Carlo /
    multiple-comparison noise at 99 draws' 1/100 resolution across 15 simultaneous pairs, not a real
    bias -- the SAME pair's p_max rises back above 0.05 as `n_perm` grows (0.04 at 199, 0.063 at
    299, stabilising 0.06-0.07 by 499-999), and WHICH pair (if any) dips is a property of the DATA
    seed, not the engine (a second data seed showed no dip anywhere across 10 permutation seeds; a
    third showed three different, unrelated pairs elevated) -- the signature of finite-sample luck
    in one fixed realisation, unlike the reproducible, N-cells-growing bias the reference-cell test
    documents. `n_perm=499` clears the noise floor with a comfortable, stable margin while keeping
    `seed=0` for reproducibility."""
    pts = nl.simulate_typed_points(n_types=5, n_cells=20000, L_um=3000.0, nest_um=None,
                                   gradient_um=None, seed=0)
    row, col, HW, mask = _typed_lattice(pts, 10.0)
    edges = np.geomspace(10, 500, 13)
    strata = nl.density_strata(row, col, HW, radius_bins=5, n_strata=10)
    out = nl.cross_pcf_strata_null(pts["labels"], row, col, HW, mask, edges, strata=strata,
                                   n_perm=499, seed=0, bin_size=10.0)
    assert np.all(out["p_max"] > 0.05)
    assert out["n_strata_used"] == 10


# --------------------------------------------------------------------------- Task 5: gr.celltype_range
def _typed_adata(seed=0, **kw):
    pts = nl.simulate_typed_points(seed=seed, **kw)
    a = ad.AnnData(np.zeros((len(pts["xy"]), 1), dtype="float32"))
    a.obsm["spatial"] = pts["xy"]
    a.obs["ct"] = np.array([f"T{t}" for t in pts["labels"]])
    a.obs["side"] = np.where(pts["xy"][:, 0] < 1500, "left", "right")
    return a


def test_celltype_range_table_matches_cross_pcf_and_carries_provenance():
    a = _typed_adata(n_types=3, n_cells=4000, L_um=1000.0, seed=0)
    with pytest.warns(RuntimeWarning, match="not calibrated"):
        tab = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, radii=np.geomspace(10, 200, 7),
                                strata_radius_um=100.0, n_perm=9, seed=0)
    assert set(tab.type_a) <= {"T0", "T1", "T2"} and len(tab) == 6
    gcols = [c for c in tab.columns if c.startswith("g_")]
    ref = gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", bin_um=10.0,
                       radii=np.geomspace(10, 200, 7), null=None, n_perm=0)
    row = tab[(tab.type_a == "T0") & (tab.type_b == "T1")].iloc[0]
    np.testing.assert_allclose(row[gcols].to_numpy(float), ref["g"], rtol=1e-12)
    for k in ("strata_radius_um", "n_strata", "n_perm", "backend", "levels", "dropped_levels",
              "L_density_um", "strata_radius_source", "null"):
        assert k in tab.attrs
    assert tab.attrs["strata_radius_source"] == "given"
    assert "T0" in a.uns["spectrast_celltype_range"]["levels"]


def test_celltype_range_requires_a_radius_until_the_rule_is_set():
    a = _typed_adata(n_types=3, n_cells=2000, L_um=800.0, seed=1)
    if gr.STRATA_RADIUS_FACTOR is None:
        with pytest.warns(RuntimeWarning, match="not calibrated"):
            with pytest.raises(ValueError, match="strata_radius_um"):
                gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, n_perm=3)
    else:
        with pytest.warns(RuntimeWarning, match="not calibrated"):
            tab = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, n_perm=3)
        assert tab.attrs["strata_radius_source"] == "estimated"
        assert np.isclose(tab.attrs["strata_radius_um"],
                          gr.STRATA_RADIUS_FACTOR * tab.attrs["L_density_um"])


def test_celltype_range_within_keeps_the_tissue_and_warns_on_a_majority_type():
    a = _typed_adata(n_types=3, n_cells=4000, L_um=1000.0, seed=2)
    a.obs["ct"] = np.where(np.arange(a.n_obs) % 10 < 7, "big", a.obs["ct"])   # 70 % one type
    # n_perm=3 also fires the "not calibrated" warning; pytest.warns re-emits whatever
    # in the block doesn't match its pattern, so both expected warnings must be in it.
    with pytest.warns(RuntimeWarning, match="majority|not calibrated"):
        tab = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, strata_radius_um=100.0,
                                n_perm=3, within=("side", "left"), radii=np.geomspace(10, 200, 5))
    assert tab.attrs["majority_warned"] == ["big"]
    # "big" is 70% globally too (not just within the left side), so this n_perm=0 call still trips
    # the majority check even though it builds no null (pre-existing behaviour, unrelated to §7).
    with pytest.warns(RuntimeWarning, match="majority"):
        full = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, strata_radius_um=100.0,
                                 n_perm=0, radii=np.geomspace(10, 200, 5))
    # the support is the whole section either way: identical n_pairs in uns
    np.testing.assert_array_equal(a.uns["spectrast_celltype_range"]["n_pairs"],
                                  full.attrs["n_pairs"])


def test_cross_pcf_refuses_the_labels_null_and_accepts_strata():
    a = _typed_adata(n_types=2, n_cells=2000, L_um=800.0, seed=3)
    with pytest.raises(ValueError, match="labels"):
        gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", null="labels", n_perm=5)
    # this 2-type synthetic splits close to 50/50 and lands one side just over MAJORITY_WARN by
    # construction, so the majority-type RuntimeWarning is expected here too; n_perm=5 also fires
    # the "not calibrated" warning, and pytest.warns re-emits whatever in the block
    # doesn't match its pattern, so both expected warnings must be in it.
    with pytest.warns(RuntimeWarning, match="majority|not calibrated"):
        res = gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", null="strata", strata_radius_um=100.0,
                           n_perm=5, bin_um=10.0)
    assert res["null"] == "strata" and "p_max" in res and res["strata_radius_um"] == 100.0


def test_cross_pcf_defaults_need_no_strata_radius_at_n_perm_zero():
    """No null is built at n_perm=0, so STRATA_RADIUS_FACTOR being unset must not block a plain
    g(r) read — only an actual permutation call (n_perm>0) needs a radius. n_types=3 (rather than
    2) keeps every type under MAJORITY_WARN for this seed, so no incidental warning fires."""
    a = _typed_adata(n_types=3, n_cells=2000, L_um=800.0, seed=4)
    res = gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", bin_um=10.0)
    assert "g" in res
    assert np.isnan(res["strata_radius_um"])
    with pytest.warns(RuntimeWarning, match="not calibrated"):
        with pytest.raises(ValueError, match="strata_radius_um"):
            gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", bin_um=10.0, n_perm=5)
    # the descriptive default is quiet and unused-source, straight from celltype_range too
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tab = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0)
    assert np.isnan(tab.attrs["strata_radius_um"])
    assert tab.attrs["strata_radius_source"] == "unused"


def test_cross_pcf_keeps_its_own_uns_key_same_type_pairs_and_refuses_copy():
    a = _typed_adata(n_types=3, n_cells=4000, L_um=1000.0, seed=0)
    full = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, strata_radius_um=100.0, n_perm=0,
                             radii=np.geomspace(10, 200, 5))
    res = gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", null="strata", strata_radius_um=100.0,
                       n_perm=0, bin_um=10.0, radii=np.geomspace(10, 200, 5))
    assert "spectrast_cross_pcf" in a.uns
    assert list(a.uns["spectrast_celltype_range"]["levels"]) == list(full.attrs["levels"])
    same = gr.cross_pcf(a, "T0", "T0", cell_type_key="ct", null="strata", strata_radius_um=100.0,
                        n_perm=0, bin_um=10.0, radii=np.geomspace(10, 200, 5))
    assert "g" in same
    with pytest.raises(ValueError, match="copy"):
        gr.cross_pcf(a, "T0", "T1", cell_type_key="ct", null="strata", strata_radius_um=100.0,
                    n_perm=0, bin_um=10.0, copy=True)


def test_celltype_range_rejects_an_unknown_explicit_type():
    a = _typed_adata(n_types=2, n_cells=2000, L_um=800.0, seed=5)
    with pytest.raises(ValueError, match="T9"):
        gr.celltype_range(a, cell_type_key="ct", types=["T0", "T9"], strata_radius_um=100.0,
                          n_perm=0, bin_um=10.0)


def test_range_readouts_and_matrix():
    """Range is read off the OBSERVED curve g, not a null z."""
    import pandas as pd
    r = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    g = np.array([[1.5, 2.0, 1.2, 0.8, 0.7],
                  [1.02, 1.03, 1.01, 0.99, 0.98],
                  [0.5, 0.4, 0.8, 1.3, 1.1],
                  [1.5, 2.0, 0.0, 1.2, 0.8],
                  [1.0, 1.1, 1.3, 1.6, 2.2],     # rising to the window edge: no peak, no length
                  [1.0, 1.1, 1.8, 0.0, 0.0]])    # peak at the last VALID annulus: the same
    peak, cross, half = gr._range_readouts(g, r)
    assert np.isnan(peak[4:]).all() and np.isnan(half[4:]).all() and np.isnan(cross[4:]).all()
    peak, cross, half = peak[:4], cross[:4], half[:4]
    assert peak[0] == 20.0 and cross[0] == 80.0
    assert peak[1] == 20.0 and np.isnan(cross[1])   # peak |g-1| below RANGE_FLOOR -> no crossing
    assert peak[2] == 20.0 and cross[2] == 80.0      # |log 0.4| is the largest deviation
    # a zero-pair annulus (g==0 exactly) is a MISSING observation, not a real dev=-1: it must be
    # skipped by the crossing scan too, not just the peak search (review fix)
    assert peak[3] == 20.0 and cross[3] == 160.0
    # r_half, hand-computed: y = |log g|, level = y_peak / 2, log-interpolated
    # between the last valid annulus above the level and the first at/below it.
    # row 0: y = [.405 .693 .182 .223 .357], level .347, k=2 is below -> f = .677 between 20 and 40
    # row 2: y = [.693 .916 .223 .262 .095], level .458 -> f = .661 between 20 and 40
    # row 3: k=2 missing, so the interpolation spans 20 -> 80 with f = .677
    f0 = (np.log(2.0) / 2) / (np.log(2.0) - np.log(1.2))          # = 0.6785 (rounded: 0.677)
    f2 = (np.log(2.5) / 2) / (np.log(2.5) - np.log(1.25))         # = 0.6610
    assert np.allclose(half, [20 * 2 ** f0, np.nan, 20 * 2 ** f2, 20 * 4 ** f0],
                       rtol=1e-3, equal_nan=True)
    assert abs(half[0] - 32.0) < 0.05                              # hand-computed 32.0 um
    assert np.isnan(half[1])                          # below the floor: no half-decay either
    tab = pd.DataFrame(dict(type_a=["A", "A", "B"], type_b=["A", "B", "B"],
                            r_cross_um=[np.nan, 80.0, 30.0], r_half_um=[15.0, 40.0, 20.0]))
    assert gr.celltype_range_matrix(tab).loc["A", "B"] == 40.0 and gr.celltype_range_matrix(tab).loc["B", "A"] == 40.0
    M = gr.celltype_range_matrix(tab, "r_cross_um")
    assert M.loc["A", "B"] == 80.0 == M.loc["B", "A"] and M.loc["B", "B"] == 30.0


def test_default_celltype_range_is_descriptive_and_quiet():
    """The re-defaulted, everyday call (design decision G2): n_perm=0, no null built,
    no warning, and every row still carries a finite descriptive g(r) and r_peak_um."""
    a = _typed_adata(n_types=3, n_cells=4000, L_um=1000.0, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tab = gr.celltype_range(a, cell_type_key="ct", bin_um=10.0, radii=np.geomspace(10, 200, 7))
    assert tab["p_max"].isna().all()
    assert tab["T_max"].isna().all()
    gcols = [c for c in tab.columns if c.startswith("g_")]
    assert np.isfinite(tab[gcols].to_numpy(float)).all()
    assert np.isfinite(tab["r_peak_um"].to_numpy(float)).all()


def test_celltype_range_refuses_colliding_annuli():
    """A mm/um coordinate mix-up rounds every annulus to the same integer column name (g_0, g_0,
    ...); celltype_range must refuse rather than silently overwrite columns."""
    a = _typed_adata(n_types=3, n_cells=4000, L_um=1000.0, seed=0)
    a.obsm["spatial"] = a.obsm["spatial"] / 1000.0            # mm-scale coordinates, mislabelled um
    with pytest.raises(ValueError, match="same integer"):
        gr.celltype_range(a, cell_type_key="ct", bin_um=0.01, radii=np.geomspace(0.01, 0.2, 7))

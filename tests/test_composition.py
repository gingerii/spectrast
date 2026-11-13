"""Composition-aware nulls (feedback item #6).

The permutation code is easy; the null calibration is the work, so the load-bearing test here is
``test_within_stratum_null_kills_the_composition_artefact`` — a synthetic tissue where pure
cell-type markers have NO gene-level arrangement at all, and the ordinary null calls every one of
them organized.
"""
import warnings

import numpy as np
import anndata as ad
import pytest

from spectrast import gr, diff, composition as C


# --------------------------------------------------------------------------- primitives
def test_stratified_permutation_stays_inside_its_stratum():
    lab = np.array(["a"] * 30 + ["b"] * 50 + ["c"] * 20)
    rng = np.random.default_rng(0)
    p = C.stratified_permutation(lab, rng)
    assert sorted(p) == list(range(len(lab)))              # a permutation
    assert (lab[p] == lab).all()                           # every cell keeps its own stratum
    assert not np.array_equal(p, np.arange(len(lab)))      # ...and it actually moved things


def test_stratified_partition_preserves_per_stratum_arm_sizes():
    lab = np.array(["a"] * 40 + ["b"] * 60)
    obs = np.zeros(100, bool); obs[:15] = True; obs[40:70] = True     # 15 of a, 30 of b
    new = C.stratified_partition(lab, obs, np.random.default_rng(0))
    for v, k in (("a", 15), ("b", 30)):
        assert int((new & (lab == v)).sum()) == k
    assert not np.array_equal(new, obs)


def test_min_cells_guard_pools_drops_or_refuses():
    lab = np.array(["big"] * 200 + ["tiny"] * 5)
    with pytest.warns(UserWarning, match="merged into"):
        out, keep, rep = C.apply_min_cells(lab, min_cells=50, policy="pool")
    assert set(out) == {"big", C.POOLED_LABEL} and keep.all() and rep["small"] == {"tiny": 5}
    with pytest.warns(UserWarning, match="dropped"):
        out, keep, rep = C.apply_min_cells(lab, min_cells=50, policy="drop")
    assert int((~keep).sum()) == 5 and rep["n_dropped"] == 5
    with pytest.raises(ValueError, match="below min_cells"):
        C.apply_min_cells(lab, min_cells=50, policy="error")


def test_residualize_removes_the_linear_composition_part():
    comp = np.column_stack([np.linspace(0, 1, 40), np.linspace(1, 0, 40)])
    pure = 2.0 + 3.0 * comp[:, 0]                          # entirely explained by composition
    assert np.allclose(C.residualize(pure, comp), 0, atol=1e-8)
    rng = np.random.default_rng(0)
    extra = rng.standard_normal(40)
    res = C.residualize(pure + extra, comp)
    assert np.corrcoef(res, extra)[0, 1] > 0.9             # the non-compositional part survives
    with pytest.raises(ValueError, match="rows"):
        C.residualize(np.zeros(10), np.zeros((9, 2)))


def test_resolution_flip_report_needs_two_resolutions_and_finds_the_flip():
    import pandas as pd
    a = pd.DataFrame({"gene": ["g0", "g1", "g2"], "significant": [True, True, False]})
    b = pd.DataFrame({"gene": ["g0", "g1", "g2"], "significant": [True, False, False]})
    rep = C.resolution_flip_report({"coarse": a, "fine": b})
    assert rep.set_index("gene").loc["g1", "flips"]
    assert not rep.set_index("gene").loc["g0", "flips"]
    assert not rep.set_index("gene").loc["g2", "flips"]
    with pytest.raises(ValueError, match=">= 2 resolutions"):
        C.resolution_flip_report({"only": a})


def test_circularity_warns_for_annotation_markers():
    with pytest.warns(UserWarning, match="annotation markers"):
        hits = C.check_circularity(["EPCAM", "VIM", "XIST"], {"epi": ["EPCAM"], "str": ["VIM"]})
    assert hits == ["EPCAM", "VIM"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert C.check_circularity(["XIST"], {"epi": ["EPCAM"]}) == []
        assert C.check_circularity(["EPCAM"], None) == []


# --------------------------------------------------------------------------- the calibration
def _typed_tissue(n=6000, seed=0, L=600.0):
    """Clustered cell types; pure markers (no within-type arrangement) + genuine within-type structure."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * L
    bumps = np.array([[.25, .25], [.75, .3], [.5, .75]]) * L
    f = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * (L * .16) ** 2)) for c in bumps)
    t = np.where(f > np.quantile(f, 0.55), "epi", "stroma")
    cols = []
    for _ in range(6):                                     # pure markers
        cols.append(np.where(t == "epi", rng.uniform(3, 6), rng.uniform(0, 1)) + rng.normal(0, .3, n))
    for _ in range(3):                                     # structured WITHIN epi
        b = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * (L * .07) ** 2)) for c in rng.random((5, 2)) * L)
        cols.append(3.0 + 4.0 * b * (t == "epi") + 6.0 * (t == "epi") + rng.normal(0, .3, n))
    for _ in range(4):
        cols.append(rng.normal(3, 1, n))
    a = ad.AnnData(np.clip(np.column_stack(cols), 0, None).astype("float32"))
    a.obsm["spatial"] = xy
    a.var_names = ([f"mark{i}" for i in range(6)] + [f"reorg{i}" for i in range(3)]
                   + [f"noise{i}" for i in range(4)])
    a.obs["ctype"] = t
    return a


def _clears_null(a, strata=None, seed=0):
    xy = np.asarray(a.obsm["spatial"], float); X = np.asarray(a.X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    bu = max(np.ptp(xy, 0).min() / 400.0, 1e-6)
    obs = gr._isotropic_fingerprint(xy, Z, bu, axes=("scale_peak_moran",))["scale_peak_moran"]
    rng = np.random.default_rng(seed)
    perm = C.stratified_permutation(strata, rng) if strata is not None else rng.permutation(len(xy))
    nul = gr._isotropic_fingerprint(xy[perm], Z, bu, axes=("scale_peak_moran",))["scale_peak_moran"]
    sig = obs > np.quantile(nul, 0.95)
    return {p: int(sum(sig[i] for i, g in enumerate(a.var_names) if g.startswith(p)))
            for p in ("mark", "reorg", "noise")}


def test_within_stratum_null_kills_the_composition_artefact():
    """THE acceptance test. Pure cell-type markers have no gene-level arrangement whatsoever — their
    apparent organization is entirely where their type sits. The ordinary null calls every one of
    them organized; the within-stratum null must not, WITHOUT killing genuine within-type structure."""
    a = _typed_tissue()
    t = np.asarray(a.obs["ctype"])
    naive = _clears_null(a)
    cond = _clears_null(a, t)
    assert naive["mark"] == 6, f"the artefact should be present under the naive null, got {naive}"
    assert cond["mark"] <= 1, f"within-stratum null still calls markers organized: {cond}"
    assert cond["reorg"] >= 2, f"within-stratum null destroyed real within-type signal: {cond}"
    assert naive["noise"] == 0 and cond["noise"] == 0


def test_fingerprint_presence_strata_key_end_to_end():
    """The same effect through the public entry point, with provenance stamped."""
    a = _typed_tissue(n=4000, seed=1)
    plain = gr.fingerprint_presence(a, axes=("scale_peak_moran",), n_perm=1, seed=0)
    cond = gr.fingerprint_presence(a, axes=("scale_peak_moran",), n_perm=1, seed=0,
                                   strata_key="ctype")
    assert cond.n_sig.iloc[0] < plain.n_sig.iloc[0]
    assert plain.attrs["spectrast_composition"] == {"conditioned": False}
    prov = cond.attrs["spectrast_composition"]
    assert prov["conditioned"] and prov["strata_key"] == "ctype" and prov["n_strata"] == 2


def test_strata_failures_are_loud():
    """A composition-aware null that silently falls back to the ordinary one would report exactly the
    false positive it exists to prevent."""
    a = _typed_tissue(n=1500, seed=2)
    with pytest.raises(KeyError, match="composition adjustment needs"):
        gr.fingerprint_presence(a, axes=("scale_peak_moran",), n_perm=1, strata_key="no_such_col")
    a.obs["one_level"] = "all"
    with pytest.raises(ValueError, match="single level"):
        gr.fingerprint_presence(a, axes=("scale_peak_moran",), n_perm=1, strata_key="one_level")
    lab = np.asarray(a.obs["ctype"], dtype=object); lab[:5] = np.nan
    a.obs["holey"] = lab
    with pytest.raises(ValueError, match="unlabelled"):
        gr.fingerprint_presence(a, axes=("scale_peak_moran",), n_perm=1, strata_key="holey")


def test_differential_refuses_strata_key_under_the_replicate_null():
    """N1 permutes SAMPLE labels, so a within-stratum variant of it is not defined. Refusing beats
    running the unconditioned null under a conditioned-sounding argument."""
    def toy(seed):
        rng = np.random.default_rng(seed)
        x = ad.AnnData(np.abs(rng.standard_normal((300, 4))).astype("float32"))
        x.obsm["spatial"] = rng.random((300, 2)) * 200
        x.var_names = [f"g{i}" for i in range(4)]
        x.obs["ctype"] = rng.choice(["a", "b"], 300)
        return x
    A = [toy(0), toy(1)]; B = [toy(2), toy(3)]
    with pytest.raises(ValueError, match="not supported with the replicate"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=10, strata_key="ctype")
    # the matched null does support it, and stamps what it conditioned on
    r = diff.differential([A[0]], [B[0]], axis="moran", features=["g0", "g1"], n_perm=20,
                          null="matched", strata_key="ctype")
    prov = r.attrs["spectrast_composition"]
    assert prov["conditioned"] and prov["null"] == "matched-within-stratum"


# --------------------------------------------------------------------------- mechanism B
def _cohort(n_per_arm=4, n=1200, seed=0, comp_lo=0.3, comp_hi=0.7, reorg=False):
    """Replicate design where composition varies BETWEEN samples within each arm, so the covariate
    has variation to regress on that is not perfectly confounded with the arm label."""
    rng = np.random.default_rng(seed)
    arms = {}
    for arm, base in (("A", comp_lo), ("B", comp_hi)):
        out = []
        for s in range(n_per_arm):
            frac = np.clip(base + rng.uniform(-.18, .18), .05, .95)   # spread within the arm
            xy = rng.random((n, 2)) * 400.0
            f = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * 70.0 ** 2)) for c in rng.random((3, 2)) * 400)
            t = np.where(f > np.quantile(f, 1 - frac), "epi", "stroma")
            cols = []
            for _ in range(5):
                v = np.where(t == "epi", 4.0, 1.0) + rng.normal(0, .3, n)
                cols.append(v)
            if reorg and arm == "B":         # gene 0 additionally rearranges within epi in arm B
                b = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * 35.0 ** 2)) for c in rng.random((4, 2)) * 400)
                cols[0] = cols[0] + 5.0 * b * (t == "epi")
            x = ad.AnnData(np.clip(np.column_stack(cols), 0, None).astype("float32"))
            x.obsm["spatial"] = xy; x.var_names = [f"g{i}" for i in range(5)]
            x.obs["ctype"] = t
            out.append(x)
        arms[arm] = out
    return arms["A"], arms["B"]


def test_regress_out_runs_and_stamps_its_design():
    A, B = _cohort(seed=3)
    r = diff.differential(A, B, axis="moran", features=["g0", "g1"], n_perm=50,
                          composition="ctype", seed=0)
    prov = r.attrs["spectrast_composition"]
    assert prov["conditioned"] and prov["mode"] == "residual"
    assert prov["n_covariates"] == 2 and prov["residual_df"] == 8 - 2 - 1
    assert "arm_r2_on_composition" in prov
    # a residual curve has no characteristic length, so delta_scale_um must not pretend otherwise
    assert r.delta_scale_um.isna().all()


def test_regress_out_refuses_a_saturated_design():
    """Too few units for the covariates: residuals are ~0 and every gene reads non-significant, which
    is not a null result but the absence of a test."""
    A, B = _cohort(n_per_arm=2, seed=4)
    comp = np.random.default_rng(0).random((4, 6))          # 6 covariates, 4 units
    with pytest.raises(ValueError, match="more units than composition columns"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=20, composition=comp)


def test_regress_out_warns_when_composition_separates_the_arms():
    """The usual case when conditions differ compositionally — and then residualizing removes the
    contrast rather than a nuisance."""
    A, B = _cohort(n_per_arm=4, seed=5)
    sep = np.zeros((8, 1)); sep[:4] = 0.0; sep[4:] = 1.0     # composition == the arm label
    with pytest.warns(UserWarning, match="predictable from composition"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=20, composition=sep)


def test_condition_on_and_regress_out_are_mutually_exclusive():
    A, B = _cohort(n_per_arm=2, seed=6)
    with pytest.raises(ValueError, match="not both"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=10,
                          strata_key="ctype", composition="ctype")


def test_composition_row_count_is_checked():
    A, B = _cohort(n_per_arm=2, seed=7)
    with pytest.raises(ValueError, match="rows but there are"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=10,
                          composition=np.zeros((3, 2)))


def test_regress_out_removes_the_artefact_and_keeps_the_signal():
    """The calibration for mechanism B, in miniature. Read on RAW p, not FDR: a k-vs-k replicate
    design has only C(2k,k)/2 distinct label partitions, so at 4-vs-4 the smallest attainable p is
    1/36 and FDR<0.05 is unreachable BY CONSTRUCTION — an empty table there means no test, not no
    effect. Six per arm gives 462 partitions."""
    fA = [0.32, 0.38, 0.44, 0.48, 0.52, 0.56]
    fB = [0.44, 0.48, 0.52, 0.56, 0.62, 0.68]

    def mk(frac, seed, reorg=0.0, n=1500, side=400.0):
        rng = np.random.default_rng(seed)
        xy = rng.random((n, 2)) * side
        f = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * 70.0 ** 2)) for c in rng.random((3, 2)) * side)
        t = np.where(f > np.quantile(f, 1 - frac), "epi", "stroma")
        cols = [np.where(t == "epi", 4.0, 1.0) + rng.normal(0, .3, n) for _ in range(8)]
        if reorg:
            b = sum(np.exp(-((xy - c) ** 2).sum(1) / (2 * 30.0 ** 2)) for c in rng.random((4, 2)) * side)
            cols[0] = cols[0] + reorg * b * (t == "epi")
        x = ad.AnnData(np.clip(np.column_stack(cols), 0, None).astype("float32"))
        x.obsm["spatial"] = xy; x.var_names = [f"g{i}" for i in range(8)]
        x.obs["ctype"] = t
        return x

    feats = [f"g{i}" for i in range(8)]           # driver's exact configuration
    run = lambda A, B, **k: diff.differential(A, B, axis="moran", features=feats, n_perm=500,
                                              seed=0, radii=(1, 2, 4, 8), bin_size=25.0, **k)
    # NEGATIVE: arms differ in composition; every gene is a pure marker
    A = [mk(f, i) for i, f in enumerate(fA)]; B = [mk(f, 100 + i) for i, f in enumerate(fB)]
    n_naive = int((run(A, B).p < 0.05).sum())
    n_reg = int((run(A, B, composition="ctype").p < 0.05).sum())
    assert n_naive > 0, "the composition artefact should be present under the naive null"
    assert n_reg < n_naive, f"regress-out did not reduce the artefact ({n_reg} vs {n_naive})"

    # POSITIVE: composition matched; g0 genuinely rearranges within epi
    A2 = [mk(f, 50 + i) for i, f in enumerate(fA)]
    B2 = [mk(f, 150 + i, reorg=6.0) for i, f in enumerate(fA)]
    assert float(run(A2, B2, composition="ctype").set_index("gene").loc["g0", "p"]) < 0.05


def test_matched_null_drop_policy_really_removes_the_small_stratum():
    """Regression (0.14.1): small_strata="drop" computed a keep-mask and discarded it, so the
    matched test still used the cells its warning said were excluded. Dropping must equal removing those cells
    by hand before the call."""
    def toy(seed):
        rng = np.random.default_rng(seed)
        x = ad.AnnData(np.abs(rng.standard_normal((400, 3))).astype("float32"))
        x.obsm["spatial"] = rng.random((400, 2)) * 200
        x.var_names = ["g0", "g1", "g2"]
        lab = rng.choice(["a", "b"], 400).astype(object); lab[:6] = "rare"        # 12 rare cells over both arms
        x.obs["ctype"] = lab
        x.X[:6] += 8.0                                                              # and they carry signal
        return x
    A, B = toy(0), toy(1)
    kw = dict(axis="moran", features=["g0", "g1", "g2"], n_perm=30, seed=0, null="matched", strata_key="ctype",
              density_match=False, min_cells_per_stratum=20, radii=[2, 4, 8], bin_size=8.0)   # explicit lattice:
    # the default comes from the cells' spacing, which removing cells by hand would move
    with pytest.warns(UserWarning, match="excluded from the test"):
        dropped = diff.differential([A], [B], small_strata="drop", **kw)
    keepA, keepB = (A.obs["ctype"] != "rare").to_numpy(), (B.obs["ctype"] != "rare").to_numpy()
    by_hand = diff.differential([A[keepA].copy()], [B[keepB].copy()], small_strata="error", **kw)
    for col in ("peak_signal", "T_max", "p"):
        assert np.allclose(dropped[col].to_numpy(), by_hand[col].to_numpy(), equal_nan=True), col

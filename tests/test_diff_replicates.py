"""Replicate differential: design floor, effect-size columns, per-arm scale spread."""
import warnings

import numpy as np
import pandas as pd
import pytest

from spectrast import diff

from _reference import assert_close_to_reference  # tests/_reference.py


def test_design_floor_unpaired_counts_the_samplers_own_support():
    d = diff.design_floor(4, 4, n_features=300)
    assert d["kind"] == "unpaired"
    assert d["n_partitions"] == 70                      # C(8,4)
    assert d["p_floor"] == pytest.approx(2 / 70)        # the library's exact-enumeration convention
    assert d["fdr_reachable"] is False                  # 0.029 > 0.05/300 for any real gene panel
    assert diff.design_floor(4, 4, n_features=1)["fdr_reachable"] is True   # one feature at alpha=0.05
    big = diff.design_floor(14, 14, n_features=15000)
    assert big["n_partitions"] == 40116600 and big["fdr_reachable"] is True


def test_design_floor_paired_uses_sign_flips():
    d = diff.design_floor(5, 5, paired=True)
    assert d["kind"] == "signflip" and d["n_partitions"] == 32
    assert d["p_floor"] == pytest.approx(2 / 32)


def _toy_arms(n_a=3, n_b=3, n_cells=600, seed=0):
    """Small paired-free replicate design: n_a + n_b AnnData sections, 8 genes, planted blob."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_a + n_b):
        xy = rng.random((n_cells, 2)) * 400.0
        X = np.abs(rng.standard_normal((n_cells, 8))).astype("float32") * 0.4
        # arm B (k >= n_a) carries a tighter blob in gene 0 — a real organisation difference
        s = 60.0 if k < n_a else 25.0
        X[:, 0] += 3 * np.exp(-((xy[:, 0] - 200) ** 2 + (xy[:, 1] - 200) ** 2) / (2 * s ** 2))
        a = ad.AnnData(X, var=pd.DataFrame(index=[f"g{i}" for i in range(8)]))
        a.obsm["spatial"] = xy
        out.append(a)
    return out[:n_a], out[n_a:]


_GENES = [f"g{i}" for i in range(8)]


def test_design_attrs_travel_with_the_result():
    A, B = _toy_arms()
    df = diff.differential(A, B, axis="moran", features=_GENES, null="replicate", n_perm=50, seed=0)
    d = df.attrs["spectrast_design"]
    assert d["kind"] == "unpaired" and d["n_units_A"] == 3 and d["n_units_B"] == 3
    assert d["n_partitions"] == 20 and d["p_floor"] == pytest.approx(0.1)


def test_auto_warns_when_it_falls_back_to_the_cell_null_with_units_given():
    A, B = _toy_arms(n_a=1, n_b=1)
    with pytest.warns(UserWarning, match="treats cells as replicates"):
        diff.differential(A, B, axis="moran", features=_GENES, null="auto", n_perm=20, seed=0,
                          units_A=["u1"], units_B=["u2"])


def test_explicit_matched_null_with_units_does_not_warn():
    """An explicit choice is not a silent fallback: only 'auto' -> 'matched' warns."""
    A, B = _toy_arms(n_a=1, n_b=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*treats cells as replicates.*")
        diff.differential(A, B, axis="moran", features=_GENES, null="matched", n_perm=20, seed=0,
                          units_A=["u1"], units_B=["u2"])


def test_new_effect_size_columns_are_defined_and_consistent():
    A, B = _toy_arms(seed=1)
    df = diff.differential(A, B, axis="moran", features=_GENES, null="replicate", n_perm=50, seed=0)
    for c in ("r_diverge_lo_um", "r_diverge_hi_um", "dI_raw_peak_um", "dI_raw_at_peak",
              "scale_A_um", "scale_B_um", "scale_A_sd_log2", "scale_B_sd_log2",
              "n_units_A", "n_units_B", "delta_scale_resolvable"):
        assert c in df.columns, c
    assert (df.n_units_A == 3).all() and (df.n_units_B == 3).all()
    # the divergence interval is the span of the envelope exceedances, and agrees with the count
    for _, r in df.iterrows():
        ex = [float(x) for x in r.radii_exceeded_um.split("|")] if r.radii_exceeded_um else []
        if ex:
            # radii_exceeded_um is formatted with %g (6 sig figs); r_diverge_lo/hi_um carry the
            # underlying float's full precision, so the string round-trip alone can cost ~1e-6
            # relative -- widen past that rather than the default rel=1e-6.
            assert r.r_diverge_lo_um == pytest.approx(min(ex), rel=1e-5)
            assert r.r_diverge_hi_um == pytest.approx(max(ex), rel=1e-5)
            assert r.n_radii_exceeded == len(ex)
        else:
            assert np.isnan(r.r_diverge_lo_um) and np.isnan(r.r_diverge_hi_um)
    # the raw peak is a radius on the grid, and dI at it is the largest |dI| of the curve
    radii_um = np.asarray(df.attrs["spectrast_grid"]["radii_um"])
    assert set(df.dI_raw_peak_um.dropna()).issubset(set(radii_um))
    # restricted to rows where both are finite: an all-NaN curve has NaN dI_raw_at_peak by
    # construction (see test_raw_peak_columns_are_nan_on_an_all_nan_curve below) and comparing NaN
    # with >= is always False, which would otherwise fail this row for the wrong reason.
    both_finite = np.isfinite(df.dI_raw_at_peak) & np.isfinite(df.dI_at_max)
    assert (np.abs(df.dI_raw_at_peak[both_finite]) >= np.abs(df.dI_at_max[both_finite]) - 1e-12).all()


def test_delta_scale_resolvable_judges_the_same_shift_delta_scale_um_reports():
    """delta_scale_resolvable must adjudicate delta_scale_um itself -- the MEAN-CURVE reduced scale
    sA/sB (delta_scale_um = sA - sB) -- not the per-unit median scale_A_um/scale_B_um. Those are a
    different estimator (see _arm_scale's docstring: a geometric median over units, vs. delta_scale's
    reduction of the arm's mean curve) and can disagree in magnitude or even sign with the column the
    docstring tells the reader to check the flag against. Recompute sA/sB exactly as the
    implementation does, from the internal mean curves, rather than from scale_A_um/scale_B_um."""
    from spectrast import scale as spscale
    A, B = _toy_arms(seed=2)
    df, internals = diff.differential(A, B, axis="moran", features=_GENES, null="replicate",
                                      n_perm=50, seed=0, _return_internals=True)
    meanA, meanB, radii_um = internals["meanA"], internals["meanB"], internals["radii_um"]
    sgn = spscale.orient_sign(np.concatenate([meanA, meanB], axis=1))
    sA = diff._char_scale(meanA, radii_um, None, sign=sgn)
    sB = diff._char_scale(meanB, radii_um, None, sign=sgn)
    # sA - sB must equal the shipped delta_scale_um exactly -- this IS how it's computed internally.
    np.testing.assert_array_equal(sA - sB, df.delta_scale_um.to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.abs(np.log2(np.where(sA > 0, sA, np.nan)) - np.log2(np.where(sB > 0, sB, np.nan)))
    spread = 0.5 * (df.scale_A_sd_log2.to_numpy() + df.scale_B_sd_log2.to_numpy())
    valid = np.isfinite(shift) & np.isfinite(spread)
    expected = shift > spread
    got = df.delta_scale_resolvable
    for g in range(len(df)):
        if valid[g]:
            assert bool(got.iloc[g]) == bool(expected[g]), g
        else:
            assert got.iloc[g] is pd.NA, g


def test_delta_scale_resolvable_is_a_nullable_boolean_column():
    """So `df[df.delta_scale_resolvable]` filters cleanly (NA rows drop out) instead of treating a
    NaN-as-float-0.0 as excludable-but-not-really and 1.0/0.0 as needing an explicit != 0 cast."""
    A, B = _toy_arms(seed=2)
    df = diff.differential(A, B, axis="moran", features=_GENES, null="replicate", n_perm=50, seed=0)
    assert df.delta_scale_resolvable.dtype == pd.BooleanDtype()
    filtered = df[df.delta_scale_resolvable]      # nullable boolean mask: NA rows drop out, not raise
    assert bool((filtered.delta_scale_resolvable == True).all())  # noqa: E712 (nullable-bool compare)


def test_arm_scale_columns_are_nan_for_pair_axis():
    """A pair axis has no per-gene characteristic length, so the per-arm scale columns must be NaN
    throughout -- same reasoning as the existing `delta_scale_um` guard for pair axes."""
    A, B = _toy_arms(seed=3)
    df = diff.differential(A, B, axis="coloc", pairs=[("g0", "g1"), ("g2", "g3")],
                           null="replicate", n_perm=30, seed=0)
    for c in ("scale_A_um", "scale_B_um", "scale_A_sd_log2", "scale_B_sd_log2"):
        assert df[c].isna().all(), c
    assert df.delta_scale_resolvable.isna().all()


def test_arm_scale_columns_are_nan_under_composition_residual_mode():
    """delta_scale_um is already NaN in residual (regress-out) mode by design (a residual curve has
    no characteristic length in um) -- a per-arm scale +/- spread would contradict that."""
    A, B = _toy_arms(seed=4)
    rng = np.random.default_rng(0)
    for ad in A + B:
        ad.obs["ctype"] = rng.choice(["a", "b", "c"], size=ad.n_obs)
    df = diff.differential(A, B, axis="moran", features=_GENES, null="replicate", n_perm=30, seed=0,
                           composition="ctype")
    assert df.delta_scale_um.isna().all()
    for c in ("scale_A_um", "scale_B_um", "scale_A_sd_log2", "scale_B_sd_log2"):
        assert df[c].isna().all(), c
    assert df.delta_scale_resolvable.isna().all()


def test_matched_null_leaves_new_arm_scale_columns_nan():
    """The matched (N2) null has no units at all (a single repartitioned coordinate frame), so the
    per-arm scale/spread columns must be NaN rather than raising -- this is the guard against the
    IA/IB hoisting introducing a NameError on the non-replicate path."""
    A, B = _toy_arms(n_a=1, n_b=1, seed=5)
    df = diff.differential(A, B, axis="moran", features=_GENES, null="matched", n_perm=30, seed=0)
    for c in ("scale_A_um", "scale_B_um", "scale_A_sd_log2", "scale_B_sd_log2"):
        assert df[c].isna().all(), c
    assert df.n_units_A.isna().all() and df.n_units_B.isna().all()


def _add_constant_gene(sections):
    """Append a 9th gene ("g8") that is literally zero in every cell of every section -- Moran's I
    is 0/0 (undefined) for a zero-variance field, so this gene's `obs` curve is NaN at EVERY radius.
    Returns fresh AnnData objects (X/var must be set together at construction)."""
    import anndata as ad
    out = []
    for sec in sections:
        X = np.hstack([np.asarray(sec.X), np.zeros((sec.n_obs, 1), dtype="float32")])
        a = ad.AnnData(X, var=pd.DataFrame(index=list(sec.var_names) + ["g8"]))
        a.obsm["spatial"] = sec.obsm["spatial"]
        out.append(a)
    return out


def test_raw_peak_columns_are_nan_on_an_all_nan_curve():
    """`np.abs(obs).argmax(0)` returns index 0 for an all-NaN column (no comparison there is ever
    True), which used to read back as `radii_um[0]` -- reporting the grid's smallest radius as a
    "raw peak" that does not exist. `dI_raw_at_peak` was already correctly NaN (indexing an all-NaN
    row is still NaN); only `dI_raw_peak_um` needed the explicit mask. `r_diverge_lo/hi_um` were
    already correct (an all-NaN curve never exceeds the envelope), checked here for the record."""
    A, B = _toy_arms(seed=6)
    A, B = _add_constant_gene(A), _add_constant_gene(B)
    feats = [f"g{i}" for i in range(9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the usual one-armed-NaN / n_perm-support UserWarnings
        df, internals = diff.differential(A, B, axis="moran", features=feats, null="replicate",
                                          n_perm=50, seed=0, _return_internals=True)
    g8 = df[df.gene == "g8"].iloc[0]
    assert np.isnan(internals["obs"][:, feats.index("g8")]).all()   # the premise: a genuinely all-NaN curve
    assert np.isnan(g8.dI_raw_peak_um)
    assert np.isnan(g8.dI_raw_at_peak)
    assert np.isnan(g8.r_diverge_lo_um)
    assert np.isnan(g8.r_diverge_hi_um)


# Pinned at HEAD before this branch's column additions: every pre-existing column of
# `differential` must keep its exact value. Regenerate ONLY with a recorded decision -- a change
# here means a shipped number moved. Generated from `_toy_arms(seed=3)`, axis="moran",
# features=[g0..g7], null="replicate", n_perm=50, seed=0.
_PINNED_TOY = {
    "delta_scale_um": [55.67342750271375, float("nan"), float("nan"), 7.065624305256943,
                        float("nan"), float("nan"), 2.379091864803332, -6.210887784182393],
    "dI_at_max": [-0.06023721868084574, -0.0012293585961428903, -0.01300659675374306,
                  -0.0003133315016491708, 0.03761844669103106, 0.00040663734080832476,
                  0.004771415801954587, -0.02608827532593632],
    "T_max": [2.5249894286875465, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "r_at_max_um": [285.27200592583523, 350.4770358517404, 81.50628740738149, 431.9833232591219,
                    32.6025149629526, 431.9833232591219, 236.36823348140632, 32.6025149629526],
    "p": [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "fdr": [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "null_used": ["replicate"] * 8,
    "structured": [True, False, False, False, False, False, False, False],
    "peak_signal": [0.887887178183493, 0.00455828047827844, 0.012381858762129098,
                    0.01879122023469135, 0.02401492123268852, 0.006941233155770978,
                    0.01989573152214016, 0.015806300634558193],
    "n_radii_exceeded": [16, 1, 1, 0, 0, 1, 2, 0],
    "radii_exceeded_um": [
        "16.3013|24.4519|32.6025|40.7531|48.9038|57.0544|65.205|81.5063|105.958|130.41|154.862|"
        "187.464|236.368|285.272|350.477|431.983",
        "350.477", "81.5063", "", "", "431.983", "236.368|285.272", "",
    ],
}

# The pre-existing columns' PHYSICAL order/position (unchanged since before this branch) -- gene/pair
# and axis first, then the 11 columns pinned above in the order they actually sit in df.columns (not
# the order _PINNED_TOY happens to list them in above). New columns must be appended after this
# prefix, never spliced into the middle, so a reorder is caught by position, not just by value.
_PRE_EXISTING_COLUMN_ORDER = [
    "gene", "axis", "delta_scale_um", "dI_at_max", "T_max", "r_at_max_um", "p", "fdr",
    "n_radii_exceeded", "radii_exceeded_um", "null_used", "structured", "peak_signal",
]


def test_existing_columns_are_unchanged_on_the_toy_design():
    """Regression guard: the existing output columns, pinned on a design small enough to recompute in
    seconds. Numeric columns hold to floating-point rounding on any machine; the rest exactly."""
    A, B = _toy_arms(seed=3)
    feats = [f"g{i}" for i in range(8)]
    df = diff.differential(A, B, axis="moran", features=feats, null="replicate", n_perm=50, seed=0)
    # position, not just value: new columns must be appended after the pre-existing prefix
    n = len(_PRE_EXISTING_COLUMN_ORDER)
    assert list(df.columns)[:n] == _PRE_EXISTING_COLUMN_ORDER, (
        f"pre-existing columns were reordered -- expected {_PRE_EXISTING_COLUMN_ORDER} as the "
        f"first {n} columns, got {list(df.columns)[:n]}")
    for col, ref in _PINNED_TOY.items():
        got = df[col].to_numpy()
        if got.dtype.kind in "fc":
            assert_close_to_reference(got, ref, col)
        else:
            assert list(got) == list(ref), col


def test_design_floor_warning_uses_the_exact_two_over_support_convention():
    """The "nothing can reach alpha" warning must fire on the SAME floor `design_floor` reports.

    `_clamp_perms` compared `1/support` against `_P_FLOOR_WARN` while `design_floor` reports
    `2/support` -- the exact floor, because the null enumerates every relabelling except the
    observed one and the observed labelling's complement is always at least as extreme. The two
    disagree exactly on the designs a user is most likely to run: at 3 v 3 (support 20) the exact
    floor is 0.1, so NOTHING in the contrast can reach 0.05, yet `1/20 = 0.05` is not `> 0.05` and
    the warning stayed silent. Seen for real on the CCI k=3 splits of the replicate-differential
    study, where every p was floored at 0.1 with no warning in the log.
    """
    assert diff.design_floor(3, 3)["p_floor"] > diff._P_FLOOR_WARN          # 2/20 = 0.1
    A, B = _toy_arms()                                                      # 3 v 3
    with pytest.warns(UserWarning, match="admits only 20 distinct relabelings"):
        diff.differential(A, B, axis="moran", features=_GENES, null="replicate", n_perm=50, seed=0)

    # ...and a design whose exact floor clears alpha stays silent: 5 v 5 -> C(10,5) = 252, 2/252.
    assert diff.design_floor(5, 5)["p_floor"] < diff._P_FLOOR_WARN
    A2, B2 = _toy_arms(n_a=5, n_b=5, seed=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        diff.differential(A2, B2, axis="moran", features=_GENES, null="replicate",
                          n_perm=50, seed=0)
    assert not any("NOTHING in this contrast" in str(w.message) for w in rec)


def test_design_floor_is_one_over_support_when_the_arms_are_unbalanced():
    """The complement argument needs the complement to BE a relabelling of the same design.

    The enumerated null holds every relabelling but the observed one, so the floor counts the
    relabellings guaranteed at least as extreme: the observed one, plus its complement *when that
    complement is in the label set*. Swapping the arms of a k-vs-k design maps a k-subset to another
    k-subset, so it is; for n_A != n_B the complement of an n_A-subset has size n_B and is not one of
    the C(n_A + n_B, n_A) labellings at all.

    MEASURED against the sampler, not argued (2026-09-28) — the smallest p an enumerated replicate
    null actually returns on a planted effect: 4v4 = 2/70, 3v3 = 2/20, but 3v6 = 1/84 and 5v6 = 1/462.
    Reporting 2/support for the unbalanced designs overstated the floor twofold, in the direction that
    declares a capable design incapable.
    """
    from math import comb

    for n_A, n_B in [(4, 4), (3, 3), (6, 6)]:                      # balanced: complement exists
        d = diff.design_floor(n_A, n_B)
        assert d["p_floor"] == pytest.approx(2 / comb(n_A + n_B, n_A)), (n_A, n_B)

    for n_A, n_B in [(3, 6), (5, 6), (28, 6)]:                     # unbalanced: it does not
        d = diff.design_floor(n_A, n_B)
        assert d["p_floor"] == pytest.approx(1 / comb(n_A + n_B, n_A)), (n_A, n_B)

    assert diff.design_floor(5, 5, paired=True)["p_floor"] == pytest.approx(2 / 2 ** 5)

    # the direction that matters: 3v6 over 84 partitions reaches 0.0119, so alpha=0.05 IS reachable
    # for a single feature. The old twofold-conservative floor said 0.0238 -- still reachable there,
    # but the same factor decides `fdr_reachable` at the panel sizes this library is used at.
    assert diff.design_floor(3, 6, n_features=1)["fdr_reachable"] is True
    assert diff._perm_floor_num("unpaired", 3, 6) == 1
    assert diff._perm_floor_num("unpaired", 4, 4) == 2
    assert diff._perm_floor_num("signflip", 4) == 2


def test_clamp_perms_warning_quotes_the_same_floor_design_floor_does():
    """The warning and `design_floor` must never state two different floors for one design.

    BOTH warnings, and an UNBALANCED design in each. The first version of this test passed
    `n_perm=10_000` for every case, which always routes to the *cap* warning — so it never reached
    the "nothing can clear alpha" floor warning, which was still hardcoding `2/{support}` while the
    value beside it was `1/support`. It printed "the smallest attainable p is 2/15 = 0.0667": a
    literal falsehood, on exactly the small unbalanced designs the warning exists to protect.
    A floor warning needs `p_min > _P_FLOOR_WARN` AND `support - 1 >= n_perm` to fire alone.
    """
    from math import comb

    # --- the CAP warning (n_perm above the support) ---
    for n_A, n_B in [(3, 3), (3, 6)]:
        support = comb(n_A + n_B, n_A)
        num = diff._perm_floor_num("unpaired", n_A, n_B)
        assert diff.design_floor(n_A, n_B)["p_floor"] == pytest.approx(num / support)
        with pytest.warns(UserWarning, match=rf"{num}/{support}"):
            diff._clamp_perms(10_000, support, "replicate label", floor_num=num)

    # --- the FLOOR warning, reached only with n_perm BELOW the support ---
    # 2v4: support 15, floor 1/15 = 0.0667 > 0.05. The balanced 3v3 (2/20 = 0.1) is the control.
    for n_A, n_B, want in [(2, 4, "1/15"), (3, 3, "2/20")]:
        support = comb(n_A + n_B, n_A)
        num = diff._perm_floor_num("unpaired", n_A, n_B)
        floor = diff.design_floor(n_A, n_B)["p_floor"]
        assert floor > diff._P_FLOOR_WARN and floor == pytest.approx(num / support)
        with pytest.warns(UserWarning, match=rf"smallest attainable p is {want} = ") as rec:
            diff._clamp_perms(3, support, "replicate label", floor_num=num)
        msg = str(rec[0].message)
        assert f"{num}/{support}" in msg
        # and it must never assert the other convention in the same breath
        wrong = f"{3 - num}/{support}"
        assert wrong not in msg, f"warning for {n_A}v{n_B} also claims {wrong}: {msg}"


def test_design_floor_refuses_a_paired_design_with_unequal_arms():
    """`paired=True` used to ignore n_B, reporting the floor of a design nobody described."""
    with pytest.raises(ValueError, match="one unit per pair"):
        diff.design_floor(3, 5, paired=True)
    assert diff.design_floor(5, 5, paired=True)["n_partitions"] == 32


def _sparse_unit_design(paired, n_units=6, seed=0):
    """Replicate arms in which unit 0 of arm A has two cells 269 um apart: no radius has a neighbouring pair,
    so that unit's Geary curve is NaN while its Moran curve, the structure gate, stays finite."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    A, B, uA, uB = [], [], [], []
    for k in range(n_units):
        for arm, out, units in (("A", A, uA), ("B", B, uB)):
            n = 2 if (k == 0 and arm == "A") else 400
            xy = np.array([[0.0, 0.0], [190.0, 190.0]]) if n == 2 else rng.random((n, 2)) * 200
            X = np.abs(rng.standard_normal((n, 3))) + (xy[:, :1] / 200 if arm == "A" else 0)
            a = ad.AnnData(X.astype("float64"), var=pd.DataFrame(index=["f0", "f1", "f2"]))
            a.obsm["spatial"] = xy
            out.append(a)
            units.append(f"d{k}" if paired else f"{arm}{k}")
    return A, B, uA, uB


@pytest.mark.parametrize("paired", [True, False])
def test_a_nan_in_the_tested_curve_is_gated_out_never_a_floor_p(paired):
    """Found 2026-10-13 on a compartment re-measurement: a unit too sparse for some radius to have
    neighbouring bins has a NaN Geary curve while its Moran gate is finite. T_max was then NaN, no null
    value compared as larger, and p sat at its floor, 1/(n_perm + 1), on a row marked structured: a false
    positive. Such a feature is now gated out (p = 1, structured False), with a warning saying why."""
    A, B, uA, uB = _sparse_unit_design(paired)
    kw = dict(axis="geary", features=["f0", "f1", "f2"], radii=[2, 4, 8], bin_size=5.0, n_perm=63, seed=0)
    with pytest.warns(UserWarning, match="NaN in the tested curve"):
        df = diff.differential(A, B, units_A=uA, units_B=uB, **kw)
    assert (df.p == 1.0).all() and not df.structured.any() and (df.T_max == 0.0).all()
    # without the sparse unit the same features are tested as usual, and nothing is gated
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*(support|characteristic length|smallest attainable).*")
        ok = diff.differential(A[1:], B[1:] if paired else B, units_A=uA[1:], units_B=uB[1:] if paired else uB, **kw)
    assert ok.structured.all() and np.isfinite(ok.T_max).all()


def test_a_finite_tested_curve_is_untouched_by_the_nan_gate():
    """The gate only acts where T_max is not finite: Moran on the same sparse design is finite in every
    unit, so it is tested as before, with no warning about the tested curve."""
    A, B, uA, uB = _sparse_unit_design(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = diff.differential(A, B, units_A=uA, units_B=uB, axis="moran", features=["f0", "f1", "f2"],
                               radii=[2, 4, 8], bin_size=5.0, n_perm=63, seed=0)
    assert not any("NaN in the tested curve" in str(w.message) for w in caught)
    assert np.isfinite(df.T_max).all() and df.structured.all()


def test_paired_gate_uses_only_the_units_that_are_tested():
    """Regression (0.14.1): the paired test compares only units present in both arms, but the
    structure gate averaged EVERY unit, so a section with no partner could move a gene across structured_min.
    Adding an unmatched unit must leave a paired result untouched."""
    A, B, uA, uB = _sparse_unit_design(paired=True, n_units=5, seed=3)
    A, B, uA, uB = A[1:], B[1:], uA[1:], uB[1:]                      # drop the deliberately sparse unit 0
    # an explicit lattice: the default is resolved from the sections' own spacing, so adding a section would move
    # every curve and the comparison would test the lattice rather than the gate
    kw = dict(axis="moran", features=["f0", "f1"], n_perm=200, seed=0, paired=True, null="replicate",
              radii=[2, 4, 8], bin_size=10.0)
    base = diff.differential(A, B, units_A=uA, units_B=uB, **kw)
    extra = A[0].copy(); extra.X = extra.X * 0 + np.linspace(0, 5, extra.n_obs)[:, None]   # strongly organised
    more = diff.differential(A + [extra], B, units_A=uA + ["unpaired"], units_B=uB, **kw)
    for col in ("peak_signal", "structured", "T_max", "p"):
        assert np.array_equal(base[col].to_numpy(), more[col].to_numpy(), equal_nan=True), col


@pytest.mark.parametrize("which", ["A", "B"])
def test_one_unit_list_without_the_other_is_refused(which):
    """Regression (0.14.1): passing only units_A (or only units_B) silently skipped all unit
    grouping, so repeated sections from one donor counted as independent replicates. It is now an error."""
    A, B = _toy_arms(n_a=4, n_b=4, n_cells=300)
    units = dict(units_A=["d0", "d0", "d1", "d1"]) if which == "A" else dict(units_B=["d0", "d0", "d1", "d1"])
    with pytest.raises(ValueError, match="units_A and units_B"):
        diff.differential(A, B, axis="moran", features=["g0"], n_perm=20, null="replicate", **units)


def test_grid_p_is_not_the_floor_where_the_difference_is_undefined():
    """A gene absent from one arm has no difference at any radius. The main table gates it out
    (p = 1); the per-radius grid must not give it the smallest possible p instead."""
    A, B = _toy_arms(seed=4)
    for a in A:
        a.X[:, 3] = 0.0                                           # g3 unexpressed in every A section
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, grid = diff.differential(A, B, axis="moran", features=_GENES, null="replicate",
                                     n_perm=19, seed=0, return_grid=True)
    g3 = grid[grid.gene == "g3"]
    assert g3.dI.isna().all()
    assert (g3.p == 1.0).all(), g3.p.tolist()
    assert (g3.fdr == 1.0).all()
    assert float(df.set_index("gene").loc["g3", "p"]) == 1.0
    rest = grid[grid.gene != "g3"]
    assert np.isfinite(rest.p).all() and (rest.p >= 1 / 20).all()

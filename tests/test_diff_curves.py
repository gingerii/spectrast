"""``differential(..., return_curves=True)``: the curves dict behind the plots.

Locks the contract the plotting layer relies on: asking for the curves never changes the result
table (bit for bit, attrs included), the returned difference is the tested one, and the envelope
in the dict is the one the table's exceedance columns were counted from.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from spectrast import diff

_GENES = [f"g{i}" for i in range(6)]
# The default lattice is derived from the sections' own spacing; pin it so every call here measures
# the same grid.
_GRID = dict(radii=[2, 3, 5, 8], bin_size=12.0)


def _arms(n_a=3, n_b=3, n_cells=500, seed=0):
    """Replicate design: arm B carries a tighter blob in g0 (a real organisation difference)."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_a + n_b):
        xy = rng.random((n_cells, 2)) * 400.0
        X = np.abs(rng.standard_normal((n_cells, len(_GENES)))).astype("float32") * 0.4
        s = 70.0 if k < n_a else 25.0
        X[:, 0] += 3 * np.exp(-((xy[:, 0] - 200) ** 2 + (xy[:, 1] - 200) ** 2) / (2 * s ** 2))
        X[:, 1] += np.sin(xy[:, 0] / 40.0) + 1.0                      # shared structure
        a = ad.AnnData(X, var=pd.DataFrame(index=_GENES))
        a.obsm["spatial"] = xy
        out.append(a)
    return out[:n_a], out[n_a:]


def _same_table(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.attrs.keys() == b.attrs.keys()
    for k in a.attrs:
        assert repr(a.attrs[k]) == repr(b.attrs[k]), k


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def test_table_is_bit_identical_with_and_without_curves_replicate_null():
    A, B = _arms()
    kw = dict(axis="moran", features=_GENES, null="replicate", n_perm=40, seed=0, **_GRID)
    plain = _quiet(diff.differential, A, B, **kw)
    df, curves = _quiet(diff.differential, A, B, return_curves=True, **kw)
    _same_table(plain, df)
    # and with the grid too: (df, grid, curves)
    pg, gg = _quiet(diff.differential, A, B, return_grid=True, **kw)
    df2, g2, c2 = _quiet(diff.differential, A, B, return_grid=True, return_curves=True, **kw)
    _same_table(pg, df2)
    pd.testing.assert_frame_equal(gg, g2, check_exact=True)
    assert isinstance(c2, dict) and np.array_equal(c2["diff"], curves["diff"])


def test_table_is_bit_identical_with_and_without_curves_paired_design():
    A, B = _arms(n_a=4, n_b=4, seed=3)
    donors = ["d0", "d1", "d2", "d3"]
    kw = dict(axis="moran", features=_GENES, units_A=donors, units_B=donors, n_perm=40, seed=0, **_GRID)
    plain = _quiet(diff.differential, A, B, **kw)
    df, curves = _quiet(diff.differential, A, B, return_curves=True, **kw)
    assert df.null_used.iloc[0] == "replicate-paired"
    _same_table(plain, df)
    assert curves["null"] == "replicate-paired"
    assert curves["units_A"] == donors and curves["unit_sections_A"] == [[0], [1], [2], [3]]
    # the paired difference is the mean of within-unit differences: equal to meanA - meanB up to rounding
    np.testing.assert_allclose(curves["diff"], curves["meanA"] - curves["meanB"], rtol=0, atol=1e-12)


def test_curves_contract_replicate_null():
    A, B = _arms()
    df, c = _quiet(diff.differential, A, B, axis="moran", features=_GENES, null="replicate",
                   n_perm=40, seed=0, return_curves=True, **_GRID)
    nR, G = len(_GRID["radii"]), len(_GENES)
    assert c["ids"] == list(df["gene"]) and c["id_col"] == "gene" and c["axis"] == "moran"
    np.testing.assert_allclose(c["radii_um"], np.array(_GRID["radii"]) * _GRID["bin_size"])
    assert c["radii_bins"] == _GRID["radii"] and c["bin_size"] == _GRID["bin_size"]
    assert c["unit_curves_A"].shape == (3, nR, G) and c["unit_curves_B"].shape == (3, nR, G)
    assert c["units_A"] == ["A0", "A1", "A2"] and c["units_B"] == ["B0", "B1", "B2"]
    for k in ("meanA", "meanB", "diff", "sd", "band", "exceed"):
        assert c[k].shape == (nR, G), k
    # the tested difference, exactly
    assert np.array_equal(c["diff"], c["meanA"] - c["meanB"])
    assert np.array_equal(c["meanA"], c["unit_curves_A"].mean(0))
    # the envelope behind the table's exceedance columns
    assert np.array_equal(c["band"], c["T_alpha"][None, :] * c["sd"])
    assert np.array_equal(c["exceed"], np.abs(c["diff"]) > c["band"])
    assert np.array_equal(c["exceed"].sum(0), df["n_radii_exceeded"].to_numpy())
    # dI_at_max is the difference read at the tested radius
    i = np.abs(c["diff"] / c["sd"]).argmax(0)
    np.testing.assert_array_equal(c["diff"][i, np.arange(G)], df["dI_at_max"].to_numpy())
    assert c["envelope_alpha"] == 0.05 and c["labels"] is None and c["n_perm"] == 19   # C(6,3)-1, enumerated


def test_curves_carry_the_arm_mean_scales_whose_difference_is_delta_scale():
    """``scale_meanA_um`` / ``scale_meanB_um`` are the two arm-mean-curve scales ``delta_scale_um``
    differences (not the per-unit medians ``scale_A_um`` / ``scale_B_um``), NaN where undefined."""
    from spectrast import scale
    A, B = _arms()
    kw = dict(axis="moran", features=_GENES, null="replicate", n_perm=40, seed=0, **_GRID)
    plain = _quiet(diff.differential, A, B, **kw)
    df, c = _quiet(diff.differential, A, B, return_curves=True, **kw)
    _same_table(plain, df)                                          # asking for them changes nothing
    sA, sB = c["scale_meanA_um"], c["scale_meanB_um"]
    assert sA.shape == sB.shape == (len(_GENES),)
    assert np.isfinite(sA - sB).any() and np.isnan(sA).any()       # both cases are exercised
    assert np.array_equal(sA - sB, df["delta_scale_um"].to_numpy(), equal_nan=True)
    # reduced from the arm MEAN curves on one shared orientation sign, exactly as the test does
    sgn = scale.orient_sign(np.concatenate([c["meanA"], c["meanB"]], axis=1))
    assert np.array_equal(sA, diff._char_scale(c["meanA"], c["radii_um"], sign=sgn), equal_nan=True)
    assert np.array_equal(sB, diff._char_scale(c["meanB"], c["radii_um"], sign=sgn), equal_nan=True)


def test_arm_mean_scales_are_nan_where_the_axis_has_no_length():
    A, B = _arms(seed=7)
    df, c = _quiet(diff.differential, A, B, axis="coloc", pairs=[("g0", "g1")], null="replicate",
                   n_perm=20, seed=0, return_curves=True, **_GRID)
    assert np.isnan(c["scale_meanA_um"]).all() and np.isnan(c["scale_meanB_um"]).all()
    assert c["scale_meanA_um"].shape == (1,)


def test_curves_under_the_matched_null_have_no_units():
    A, B = _arms(n_a=1, n_b=1)
    df, c = _quiet(diff.differential, A, B, axis="moran", features=_GENES, null="matched",
                   n_perm=20, seed=0, return_curves=True, **_GRID)
    assert c["unit_curves_A"] is None and c["unit_curves_B"] is None
    assert c["units_A"] is None and c["unit_sections_A"] is None
    assert np.array_equal(c["diff"], c["meanA"] - c["meanB"])
    assert np.array_equal(c["exceed"].sum(0), df["n_radii_exceeded"].to_numpy())


def test_units_aggregate_sections_and_record_which():
    A, B = _arms(n_a=4, n_b=4, seed=5)
    uA, uB = ["p", "p", "q", "r"], ["s", "t", "t", "u"]
    df, c = _quiet(diff.differential, A, B, axis="moran", features=_GENES, units_A=uA, units_B=uB,
                   paired=False, n_perm=30, seed=0, return_curves=True, **_GRID)
    assert c["units_A"] == ["p", "q", "r"] and c["unit_sections_A"] == [[0, 1], [2], [3]]
    assert c["units_B"] == ["s", "t", "u"] and c["unit_sections_B"] == [[0], [1, 2], [3]]
    assert c["unit_curves_A"].shape[0] == 3


def test_pair_axis_returns_curves_per_pair():
    A, B = _arms(seed=7)
    pairs = [("g0", "g1"), ("g1", "g2")]
    kw = dict(axis="coloc", pairs=pairs, null="replicate", n_perm=20, seed=0, **_GRID)
    plain = _quiet(diff.differential, A, B, **kw)
    df, c = _quiet(diff.differential, A, B, return_curves=True, **kw)
    _same_table(plain, df)
    assert c["id_col"] == "pair" and c["ids"] == ["g0~g1", "g1~g2"]
    assert c["unit_curves_A"].shape == (3, len(_GRID["radii"]), 2)
    assert np.array_equal(c["exceed"].sum(0), df["n_radii_exceeded"].to_numpy())


def _cohort(n=400, seed=0):
    import anndata as ad
    A, B = _arms(n_a=3, n_b=3, n_cells=n, seed=seed)
    parts = []
    for cond, arm in (("ctrl", A), ("case", B)):
        for i, p in enumerate(arm):
            p = p.copy()
            p.obs["cond"] = cond
            p.obs["section"] = f"{cond}_{i}"
            parts.append(p)
    return ad.concat(parts, index_unique="-")


def test_cohort_threads_return_curves_and_names_the_sections():
    a = _cohort()
    kw = dict(sample_key="section", axis="moran", features=_GENES, n_perm=30, seed=0, **_GRID)
    plain = _quiet(diff.differential_cohort, a, "cond", ["ctrl"], ["case"], **kw)
    df, c = _quiet(diff.differential_cohort, a, "cond", ["ctrl"], ["case"], return_curves=True, **kw)
    _same_table(plain, df)
    assert c["labels"] == ("ctrl", "case")
    assert c["section_labels_A"] == ["ctrl_0", "ctrl_1", "ctrl_2"]
    assert c["units_B"] == ["case_0", "case_1", "case_2"]
    df2, g2, c2 = _quiet(diff.differential_cohort, a, "cond", "ctrl", "case", return_grid=True,
                         return_curves=True, **kw)
    assert c2["labels"] == ("ctrl", "case") and isinstance(g2, pd.DataFrame)

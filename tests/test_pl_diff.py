"""spectrast.pl differential and cohort plots: each draws what it is given, honours ax=/save=, and
names the call to run when an input is missing."""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spectrast import diff, pl  # noqa: E402
from spectrast.pl import _diff  # noqa: E402

_GENES = [f"g{i}" for i in range(5)]
# the default lattice is derived from the sections' own spacing; pin it
_GRID = dict(radii=[2, 3, 5, 8], bin_size=12.0)


def _arms(n_a=3, n_b=3, n_cells=500, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_a + n_b):
        xy = rng.random((n_cells, 2)) * 400.0
        X = np.abs(rng.standard_normal((n_cells, len(_GENES)))).astype("float32") * 0.4
        s = 70.0 if k < n_a else 25.0                                   # g0 tighter in arm B
        X[:, 0] += 3 * np.exp(-((xy[:, 0] - 200) ** 2 + (xy[:, 1] - 200) ** 2) / (2 * s ** 2))
        X[:, 1] += np.sin(xy[:, 0] / 40.0) + 1.0
        a = ad.AnnData(X, var=pd.DataFrame(index=_GENES))
        a.obsm["spatial"] = xy
        out.append(a)
    return out[:n_a], out[n_a:]


def _run(A, B, axis="moran", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diff.differential(A, B, axis=axis, features=_GENES, seed=0, return_curves=True,
                                 **_GRID, **kw)


@pytest.fixture(scope="module")
def rep():
    A, B = _arms()
    df, curves = _run(A, B, null="replicate", n_perm=50)
    return A, B, df, curves


@pytest.fixture(scope="module")
def four(rep):
    """The same replicate contrast on the four axes, each as (df, curves)."""
    A, B, df, c = rep
    out = {"moran": (df, c)}
    for a in ("geary", "losh", "focality"):
        out[a] = _run(A, B, axis=a, null="replicate", n_perm=50)
    return out


@pytest.fixture(scope="module")
def matched():
    A, B = _arms(n_a=1, n_b=1, seed=4)
    return _run(A, B, null="matched", n_perm=20)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _table(n=40, seed=0):
    """A hand-made result table: 6 called features, 4 of them with no scale shift."""
    rng = np.random.default_rng(seed)
    fdr = np.r_[np.full(6, 0.01), rng.uniform(0.2, 1.0, n - 6)]
    ds = rng.normal(0, 20, n)
    ds[[0, 1, 2, 3, 10, 11]] = np.nan
    res = pd.array([True, False] * (n // 2), dtype="boolean")
    res[np.isnan(ds)] = pd.NA
    return pd.DataFrame({"gene": [f"f{i}" for i in range(n)], "dI_at_max": rng.normal(0, 0.05, n),
                         "delta_scale_um": ds, "fdr": fdr, "delta_scale_resolvable": res})


# --------------------------------------------------------------------------------------------- volcano
def test_volcano_draws_every_feature_and_highlights_the_called_ones(tmp_path):
    df = _table()
    ax = pl.volcano(df, fdr=0.05, n_label=3, save=tmp_path / "v.png")
    rest, called = ax.collections[0].get_offsets(), ax.collections[1].get_offsets()
    m = df.fdr.to_numpy() < 0.05
    np.testing.assert_allclose(called, np.c_[df.dI_at_max[m], -np.log10(df.fdr[m])])
    np.testing.assert_allclose(rest, np.c_[df.dI_at_max[~m], -np.log10(df.fdr[~m])])
    assert len(ax.texts) == 3
    assert any(np.allclose(l.get_ydata(), -np.log10(0.05)) and l.get_linestyle() == "--" for l in ax.lines)
    assert (tmp_path / "v.png").stat().st_size > 0


def test_volcano_honours_ax_and_names_the_call_when_a_column_is_missing():
    fig, ax = plt.subplots()
    assert pl.volcano(_table(), ax=ax) is ax
    with pytest.raises(KeyError, match="differential"):
        pl.volcano(_table().drop(columns="fdr"))
    with pytest.raises(TypeError, match="differential"):
        pl.volcano({"fdr": [0.1]})


# --------------------------------------------------------------------------------------------- diff_curves
def test_diff_curves_draws_the_means_the_difference_and_the_exceedances(rep, tmp_path):
    A, B, df, c = rep
    fig = pl.diff_curves(c, "g0", labels=("wide", "tight"), save=tmp_path / "d.png")
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 2
    top, bot = fig.axes
    np.testing.assert_allclose(top.lines[0].get_ydata(), c["meanA"][:, 0])
    np.testing.assert_allclose(top.lines[1].get_ydata(), c["meanB"][:, 0])
    np.testing.assert_allclose(bot.lines[-1].get_ydata(), c["diff"][:, 0])
    ex = c["exceed"][:, 0]
    pts = bot.collections[-1].get_offsets()
    np.testing.assert_allclose(np.asarray(pts)[:, 0], c["radii_um"][ex])
    assert bot.get_xscale() == "log"
    assert (tmp_path / "d.png").stat().st_size > 0


def test_diff_curves_units_need_the_replicate_null(rep, matched):
    _, _, _, c = rep
    fig = pl.diff_curves(c, "g0", show_units=True)
    assert len(fig.axes[0].lines) == 3 + 3 + 2                    # every unit, then the two means
    _, cm = matched
    pl.diff_curves(cm, "g0")                                       # the means alone are fine
    with pytest.raises(ValueError, match="replicate"):
        pl.diff_curves(cm, "g0", show_units=True)


def test_diff_curves_errors_name_the_call(rep):
    _, _, df, c = rep
    with pytest.raises(TypeError, match="return_curves=True"):
        pl.diff_curves(df, "g0")
    with pytest.raises(KeyError, match="not found"):
        pl.diff_curves(c, "nope")


# --------------------------------------------------------------------------------------------- unit_curves
def test_unit_curves_draws_each_unit_and_the_arm_means(rep):
    _, _, _, c = rep
    fig, ax = plt.subplots()
    out = pl.unit_curves(c, "g1", ax=ax)
    assert out is ax
    nA, nB = c["unit_curves_A"].shape[0], c["unit_curves_B"].shape[0]
    assert len(ax.lines) == nA + 1 + nB + 1
    for u in range(nA):
        np.testing.assert_allclose(ax.lines[u].get_ydata(), c["unit_curves_A"][u, :, 1])
    np.testing.assert_allclose(ax.lines[nA].get_ydata(), c["meanA"][:, 1])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), c["meanB"][:, 1])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [f"A (n = {nA})", f"B (n = {nB})"]


def test_unit_curves_refuses_a_result_without_units(matched):
    _, cm = matched
    with pytest.raises(ValueError, match="matched"):
        pl.unit_curves(cm, "g0")


# --------------------------------------------------------------------------------------------- diff_heatmap
def _fake_curves(n_feat, radii_um, seed=0, labels=None):
    """A hand-made curves dict and table: every feature called, signs and tested radii known."""
    rng = np.random.default_rng(seed)
    r = np.asarray(radii_um, float)
    ids = [f"f{i}" for i in range(n_feat)]
    D = rng.normal(0, 0.05, (len(r), n_feat))
    sd = np.full_like(D, 0.02)
    am = np.abs(D / sd).argmax(0)
    c = dict(radii_um=r, ids=ids, meanA=D + 0.3, meanB=np.full_like(D, 0.3), diff=D, sd=sd,
             band=2 * sd, exceed=np.abs(D) > 2 * sd, labels=labels, axis="moran")
    df = pd.DataFrame({"gene": ids, "dI_at_max": D[am, np.arange(n_feat)], "r_at_max_um": r[am],
                       "fdr": 0.001})
    return c, df


def _heatmap_order(d):
    """Expected rows: called, higher in A first, each block by tested radius then |change|."""
    called = d[d.fdr < 0.05].assign(_a=lambda t: -t.dI_at_max.abs())
    up = called[called.dI_at_max > 0].sort_values(["r_at_max_um", "_a"], kind="stable")
    dn = called[~(called.dI_at_max > 0)].sort_values(["r_at_max_um", "_a"], kind="stable")
    return list(up.gene), list(dn.gene)


def test_diff_heatmap_draws_the_raw_difference_split_by_sign_and_ordered_by_tested_radius(rep, tmp_path):
    _, _, df, c = rep
    d = df.copy()
    d["fdr"] = [0.01, 0.01, 0.5, 0.02, 0.9]                          # g0, g1, g3 called
    c = dict(c, labels=("wide", "tight"))
    ax = pl.diff_heatmap(c, df=d, save=tmp_path / "h.png")
    up, dn = _heatmap_order(d)
    assert up and dn                                                 # both blocks exercised
    want = up + dn
    assert [t.get_text() for t in ax.get_yticklabels()] == want
    idx = [c["ids"].index(g) for g in want]
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), c["diff"][:, idx].T)
    lo, hi = ax.images[0].get_clim()
    assert lo == -hi                                                 # diverging, symmetric
    # one dot per row, at the column of its tested radius
    dots = np.asarray(ax.collections[0].get_offsets())
    r_t = d.set_index("gene").loc[want, "r_at_max_um"].to_numpy()
    cols = [int(np.flatnonzero(c["radii_um"] == r)[0]) for r in r_t]
    np.testing.assert_array_equal(dots, np.c_[cols, np.arange(len(want))])
    # the two blocks are separated, and the colour bar names the arms
    assert any(np.allclose(l.get_ydata(), len(up) - 0.5) for l in ax.lines)
    assert "wide − tight" in ax.images[0].colorbar.ax.get_ylabel()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["24", "36", "60", "96"]
    assert ax.get_xlabel() == "radius (µm)"
    assert (tmp_path / "h.png").stat().st_size > 0


def test_diff_heatmap_labels_log_spaced_radii_in_um_and_thins_names_when_many():
    r = np.geomspace(31, 1226, 19)
    c, d = _fake_curves(90, r)
    ax = pl.diff_heatmap(c, df=d)
    ticks = [50, 100, 200, 500, 1000]
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(v) for v in ticks]
    np.testing.assert_allclose(ax.get_xticks(), np.interp(np.log(ticks), np.log(r), np.arange(19)))
    up, dn = _heatmap_order(d)
    rows = up + dn
    assert [t.get_text() for t in ax.get_yticklabels()] == rows[::2]       # every other row
    np.testing.assert_array_equal(ax.get_yticks(), np.arange(90)[::2])
    ax = pl.diff_heatmap(c, df=d, n_label=100)
    assert [t.get_text() for t in ax.get_yticklabels()] == rows             # every row
    assert ax.images[0].colorbar.ax.get_ylabel().endswith("(A − B)")        # no labels: A, B


def test_diff_heatmap_honours_ax_and_explains_bad_input(rep):
    _, _, df, c = rep
    d = df.copy()
    d["fdr"] = [0.01, 0.5, 0.5, 0.5, 0.5]
    fig, ax = plt.subplots()
    assert pl.diff_heatmap(c, df=d, ax=ax) is ax
    with pytest.raises(TypeError):
        pl.diff_heatmap(c)                                           # the table is required
    with pytest.raises(ValueError, match="FDR"):
        pl.diff_heatmap(c, df=d.assign(fdr=1.0))
    with pytest.raises(KeyError, match="differential"):
        pl.diff_heatmap(c, df=d.drop(columns="r_at_max_um"))
    with pytest.raises(TypeError, match="return_curves=True"):
        pl.diff_heatmap(d, df=d)
    with pytest.raises(KeyError, match="not found"):
        pl.diff_heatmap(c, df=d.assign(gene=["zz", "g1", "g2", "g3", "g4"]))


# --------------------------------------------------------------------------------------------- diff_fingerprint
_AXES4 = ("moran", "geary", "losh", "focality")


def _fp_tables(spec, n=6):
    """Hand-made tables: ``spec`` maps axis -> {gene: (p, T_max, dI_at_max)}; the rest are null."""
    out = {}
    for a in _AXES4:
        rows = []
        for i in range(n):
            g = f"f{i}"
            p, t, d = spec.get(a, {}).get(g, (0.9, 0.5, 0.01))
            rows.append(dict(gene=g, p=p, fdr=p, T_max=t, dI_at_max=d))
        out[a] = pd.DataFrame(rows)
    return out


def _fp_expected(tables, genes):
    return np.array([[float(tables[a].set_index("gene").loc[g, "T_max"]
                            * np.sign(tables[a].set_index("gene").loc[g, "dI_at_max"]))
                      for a in tables] for g in genes])


def _squares(ax):
    pts = [np.asarray(c.get_offsets()).reshape(-1, 2) for c in ax.collections[:2]]
    return {(int(x), int(y)) for p in pts for x, y in p}


def test_diff_fingerprint_draws_the_signed_studentised_change_and_marks_pooled_calls(tmp_path):
    spec = {"moran": {"f0": (1e-3, 4.0, 0.2), "f1": (1e-3, 3.0, -0.1)},
            "geary": {"f0": (1e-3, 3.5, 0.3)},
            "losh": {"f2": (1e-3, 2.5, -0.05), "f0": (0.04, 2.0, 0.1)}}     # f0's losh: 0.04 * 24/4 > 0.05
    t = _fp_tables(spec)
    ax = pl.diff_fingerprint(t, cluster=False, save=tmp_path / "f.png")
    rows = [x.get_text() for x in ax.get_yticklabels()]
    assert rows == ["f0", "f1", "f2"]                     # by axes called, then strength
    img = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(img, _fp_expected(t, rows))
    lo, hi = ax.images[0].get_clim()
    assert lo == -hi
    assert [x.get_text() for x in ax.get_xticklabels()] == ["Moran's I", "Geary (1 − C)", "LOSH",
                                                             "Focality"]
    # BH over all 6 genes x 4 axes from the p column
    q = diff.bh_fdr(np.concatenate([t[a].p.to_numpy() for a in _AXES4])).reshape(4, 6).T
    want = {(j, i) for i, g in enumerate(rows) for j in range(4) if q[int(g[1:]), j] < 0.05}
    assert _squares(ax) == want == {(0, 0), (1, 0), (0, 1), (2, 2)}
    assert "1, 2, 3, 4 axes: 2, 1, 0, 0" in ax.get_title()
    assert (tmp_path / "f.png").stat().st_size > 0


def test_diff_fingerprint_per_axis_scope_reads_each_tables_own_fdr():
    t = _fp_tables({"moran": {"f0": (1e-3, 4.0, 0.2)}, "losh": {"f3": (0.03, 2.0, 0.1)}})
    t["losh"].loc[3, "fdr"] = 0.01                                  # the table's own FDR calls it
    fig, ax = plt.subplots()
    assert pl.diff_fingerprint(t, fdr_scope="per_axis", cluster=False, ax=ax) is ax
    assert [x.get_text() for x in ax.get_yticklabels()] == ["f0", "f3"]
    assert _squares(ax) == {(0, 0), (2, 1)}
    with pytest.raises(ValueError, match="fdr_scope"):
        pl.diff_fingerprint(t, fdr_scope="global")


def test_diff_fingerprint_clusters_rows_and_contrasts_squares_on_dark_cells():
    up = {f"f{i}": (1e-4, 5.0, 0.2) for i in (0, 2, 4)}
    dn = {f"f{i}": (1e-4, 5.0, -0.2) for i in (1, 3, 5)}
    t = _fp_tables({"moran": {**up, **dn}, "geary": {**up, **dn}})
    ax = pl.diff_fingerprint(t, cluster=True)
    rows = [x.get_text() for x in ax.get_yticklabels()]
    img = np.asarray(ax.images[0].get_array())
    np.testing.assert_allclose(img, _fp_expected(t, rows))        # labels follow their rows
    signs = np.sign(img[:, 0])
    assert np.count_nonzero(np.diff(signs)) == 1                    # the two profiles are contiguous
    dark = np.asarray(ax.collections[1].get_offsets())               # white squares on dark cells
    assert len(dark) == 12 and len(ax.collections[0].get_offsets()) == 0


def test_diff_fingerprint_alternates_names_when_many_rows():
    spec = {"moran": {f"f{i}": (1e-4, 3.0 + i / 10, 0.1) for i in range(60)}}
    t = _fp_tables(spec, n=60)
    ax = pl.diff_fingerprint(t, cluster=False)
    left = [x.get_text() for x in ax.get_yticklabels()]
    right = [x.get_text() for x in ax.child_axes[0].get_yticklabels()]
    assert len(left) == 30 and len(right) == 30 and not set(left) & set(right)
    np.testing.assert_array_equal(ax.get_yticks(), np.arange(0, 60, 2))


def test_diff_fingerprint_on_real_tables_keeps_every_column_rising_with_clustering(four):
    """diff's geary curve is already 1 - C, so its column carries the change as tested: g0 (wider
    blob in A) rises on Moran and on Geary (1 - C) alike."""
    res = {a: four[a][0].assign(p=np.where(four[a][0].gene == "g0", 1e-4, four[a][0].p)) for a in _AXES4}
    ax = pl.diff_fingerprint(res, cluster=False)
    rows = [x.get_text() for x in ax.get_yticklabels()]
    img = np.asarray(ax.images[0].get_array())
    g0 = img[rows.index("g0")]
    assert np.sign(g0[0]) == np.sign(g0[1]) == 1
    np.testing.assert_allclose(img, _fp_expected(res, rows))


def test_diff_fingerprint_explains_bad_input():
    t = _fp_tables({})
    with pytest.raises(ValueError, match="no feature"):
        pl.diff_fingerprint(t)
    with pytest.raises(TypeError, match="axis"):
        pl.diff_fingerprint([t["moran"]])
    with pytest.raises(KeyError, match="differential"):
        pl.diff_fingerprint({"moran": t["moran"].drop(columns="T_max")})


# --------------------------------------------------------------------------------------------- scale_shift
def _shift_inputs(four):
    """The four axes with g0 called (rises in A) and g1 called (falls in A); g1's shift unresolvable."""
    out = {}
    for a in _AXES4:
        df, c = four[a]
        d = df.copy()
        d["fdr"] = np.where(d.gene.isin(["g0", "g1"]), 0.01, 0.6)
        d["dI_at_max"] = np.where(d.gene == "g0", 0.2, np.where(d.gene == "g1", -0.2, d.dI_at_max))
        res = d["delta_scale_resolvable"].copy()
        res[d.gene == "g0"] = True
        res[d.gene == "g1"] = False
        d["delta_scale_resolvable"] = res
        out[a] = (d, dict(c, labels=("wide", "tight")))
    return out


def test_scale_shift_puts_arm_b_on_x_and_arm_a_on_y_and_colours_the_calls(four, tmp_path):
    res = _shift_inputs(four)
    fig = pl.scale_shift(res, save=tmp_path / "s.png")
    assert isinstance(fig, plt.Figure)
    panels = [a for a in fig.axes if a.get_xscale() == "log"]
    assert len(panels) == 4
    for ax, a in zip(panels, _AXES4):
        d, c = res[a]
        x, y = c["scale_meanB_um"], c["scale_meanA_um"]
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        called = d.fdr.to_numpy() < 0.05
        rise = d.dI_at_max.to_numpy() > 0
        groups = (ok & ~called, ok & called & rise, ok & called & ~rise)
        for coll, sel in zip(ax.collections[:3], groups):
            np.testing.assert_allclose(np.asarray(coll.get_offsets()).reshape(-1, 2), np.c_[x[sel], y[sel]])
        # filled when resolvable, open otherwise
        for coll, sel in zip(ax.collections[1:3], groups[1:]):
            want = np.asarray(d.delta_scale_resolvable[sel].eq(True).fillna(False), bool)
            fc = coll.get_facecolors()
            if len(want):
                assert np.array_equal(fc[:, 3] > 0, want)
        assert ax.get_yscale() == "log"
        assert ax.get_xlabel() == "scale in tight (µm)" and ax.get_ylabel() == "scale in wide (µm)"
        assert any(l.get_linestyle() == "--" for l in ax.lines)
        title = ax.get_title(loc="left")
        assert title.startswith(_diff._AXIS_TITLE[a])
        assert f"no scale in one or both arms: {int((~ok).sum())}" in title
    assert (tmp_path / "s.png").stat().st_size > 0


def test_scale_shift_explains_bad_input(four):
    df, c = four["moran"]
    with pytest.raises(TypeError, match="return_curves=True"):
        pl.scale_shift({"moran": df})
    with pytest.raises(TypeError, match="axis"):
        pl.scale_shift([(df, c)])
    old = {k: v for k, v in c.items() if k not in ("scale_meanA_um", "scale_meanB_um")}
    with pytest.raises(KeyError, match="scale_meanA_um"):
        pl.scale_shift({"moran": (df, old)})
    with pytest.raises(ValueError, match="same call"):
        pl.scale_shift({"moran": (df.iloc[::-1], c)})
    with pytest.raises(KeyError, match="differential"):
        pl.scale_shift({"moran": (df.drop(columns="delta_scale_resolvable"), c)})


# --------------------------------------------------------------------------------------------- effect_plane
def test_effect_plane_keeps_features_without_a_scale_shift(tmp_path):
    df = _table()
    fig, ax = plt.subplots()
    assert pl.effect_plane(df, ax=ax, n_label=2, save=tmp_path / "e.png") is ax
    fin = np.isfinite(df.delta_scale_um.to_numpy())
    called = df.fdr.to_numpy() < 0.05
    off = [np.asarray(c.get_offsets()) for c in ax.collections[:4]]
    np.testing.assert_allclose(off[0], np.c_[df.dI_at_max[fin & ~called], df.delta_scale_um[fin & ~called]])
    np.testing.assert_allclose(off[1], np.c_[df.dI_at_max[fin & called], df.delta_scale_um[fin & called]])
    assert len(off[2]) == int((~fin & ~called).sum()) and len(off[3]) == int((~fin & called).sum())
    # nothing dropped, and the band sits below every finite shift
    assert sum(len(o) for o in off) == len(df)
    assert np.all(np.r_[off[2][:, 1], off[3][:, 1]] < df.delta_scale_um.min())
    assert any("no scale shift defined (n = 6; 4 called)" in t.get_text() for t in ax.texts)
    # an explicitly unresolvable shift is hollow
    fc = ax.collections[0].get_facecolors()
    res = df.delta_scale_resolvable[fin & ~called].to_numpy()
    assert np.all(fc[np.asarray(res == False, bool)][:, 3] == 0)                       # noqa: E712
    with pytest.raises(KeyError, match="differential"):
        pl.effect_plane(df.drop(columns="delta_scale_um"))


def test_effect_plane_on_a_real_result(rep):
    _, _, df, _ = rep
    ax = pl.effect_plane(df)
    assert sum(len(c.get_offsets()) for c in ax.collections[:4]) == len(df)


# --------------------------------------------------------------------------------------------- neighborhood_maps
def test_neighborhood_field_reproduces_the_differentials_moran_curve(rep):
    """The mapped field is the one the test correlates: averaged as z*lag over occupied bins it
    returns the unit's Moran's I at that radius."""
    A, B, _, c = rep
    for arm, lst in (("A", A), ("B", B)):
        for u in range(len(lst)):
            _, vals, ex = _diff._neighborhood_field(lst[u], "g0", c["radii_bins"], c["bin_size"])
            m, z = ex["mask"], ex["z"]
            for ri, lag in enumerate(ex["lag"]):
                I = np.nansum((z * lag)[m]) / np.sum(z[m] ** 2)
                assert I == pytest.approx(c[f"unit_curves_{arm}"][u, ri, 0], rel=1e-11, abs=1e-13)
            np.testing.assert_array_equal(vals[0], ex["lag"][0].ravel()[ex["binidx"]])


def test_neighborhood_maps_picks_the_most_typical_section_and_shares_one_scale_per_radius(rep, tmp_path):
    A, B, _, c = rep
    fig = pl.neighborhood_maps(A, B, "g0", [36, 96], curves=c, save=tmp_path / "n.png")
    axes = np.array(fig.axes[:4]).reshape(2, 2)
    want = []
    for arm in ("A", "B"):
        d = np.linalg.norm(c[f"unit_curves_{arm}"][:, :, 0] - c[f"mean{arm}"][:, 0], axis=1)
        want.append(int(np.argmin(d)))
    assert axes[0, 0].get_ylabel().endswith(f"A{want[0]}")
    assert axes[1, 0].get_ylabel().endswith(f"B{want[1]}")
    for j, rb in enumerate((3, 8)):                                 # 36 and 96 um on a 12 um lattice
        assert axes[0, j].get_title() == f"r = {rb * 12} µm"
        assert axes[0, j].collections[0].get_clim() == axes[1, j].collections[0].get_clim()
        for row, (lst, i) in enumerate(((A, want[0]), (B, want[1]))):
            _, vals, _ = _diff._neighborhood_field(lst[i], "g0", [rb], 12.0)
            np.testing.assert_allclose(np.asarray(axes[row, j].collections[0].get_array()), vals[0])
    assert (tmp_path / "n.png").stat().st_size > 0


def test_neighborhood_maps_explicit_sections_and_errors(rep):
    A, B, _, c = rep
    fig = pl.neighborhood_maps(A, B, "g1", 48, sections=(2, "B1"), curves=c, labels=("x", "y"))
    assert fig.axes[0].get_ylabel() == "x\nA2" and fig.axes[1].get_ylabel() == "y\nB1"
    fig = pl.neighborhood_maps(A[0], B[0], "g1", 48, bin_um=12.0)  # no curves: first sections
    assert fig.axes[0].get_title() == "r = 48 µm"
    with pytest.raises(ValueError, match="one to three"):
        pl.neighborhood_maps(A, B, "g1", [10, 20, 40, 80], curves=c)
    with pytest.raises(ValueError, match="same section lists"):
        pl.neighborhood_maps(A[:2], B, "g1", 48, curves=c)
    with pytest.raises(KeyError):
        pl.neighborhood_maps(A, B, "g1", 48, sections=("nope", 0), curves=c)


def test_neighborhood_maps_leaves_the_inputs_untouched(rep):
    A, B, _, c = rep
    before = [(a.X.copy(), dict(a.obsm), a.obs.copy(), dict(a.uns)) for a in (A[0], B[0])]
    pl.neighborhood_maps(A, B, "g0", 36, sections=(0, 0), curves=c)
    for a, (X, obsm, obs, uns) in zip((A[0], B[0]), before):
        assert np.array_equal(a.X, X) and set(a.obsm) == set(obsm) and set(a.uns) == set(uns)
        pd.testing.assert_frame_equal(a.obs, obs)


# --------------------------------------------------------------------------------------------- section_scales
def _tidy():
    rng = np.random.default_rng(1)
    secs = ["s0", "s1", "s2", "s3"]
    rows = []
    for g in range(8):
        for s in secs:
            rows.append(dict(gene=f"f{g}", section=s, n_obs=100, scale_um=float(20 * 2 ** (g / 2) * rng.uniform(0.8, 1.25)),
                             focality=rng.random()))
    t = pd.DataFrame(rows)
    t.loc[(t.gene == "f7") & (t.section == "s3"), "scale_um"] = np.nan   # one section without a length
    return t


def test_section_scales_strip_draws_every_section_value(tmp_path):
    t = _tidy()
    groups = {"s0": "ctrl", "s1": "ctrl", "s2": "case", "s3": "case"}
    fig, ax = plt.subplots()
    assert pl.section_scales(t, groups=groups, ax=ax, save=tmp_path / "s.png") is ax
    ctrl, case, med = (np.asarray(c.get_offsets()) for c in ax.collections[:3])
    assert len(ctrl) == 16 and len(case) == 15                       # f7 has no value in s3
    np.testing.assert_allclose(np.sort(ctrl[:, 0]), np.sort(t.scale_um[t.section.isin(["s0", "s1"])]))
    shown = [x.get_text() for x in ax.get_yticklabels()]
    medians = t.groupby("gene").scale_um.median()
    assert shown == list(medians.loc[shown].sort_values().index)     # rows ordered by median
    np.testing.assert_allclose(med[:, 0], medians.loc[shown].to_numpy())
    assert ax.get_xscale() == "log"
    assert [x.get_text() for x in ax.get_legend().get_texts()][:2] == ["ctrl (n = 2 sections)",
                                                                       "case (n = 2 sections)"]


def test_section_scales_strip_colours_groups_in_the_order_given():
    """Groups take colours in the order of the ``groups`` mapping, so a caller can keep one arm's
    colour across plots whatever order the sections come in."""
    t = _tidy()
    groups = {"s2": "case", "s3": "case", "s0": "ctrl", "s1": "ctrl"}
    ax = pl.section_scales(t, groups=groups)
    first = np.asarray(ax.collections[0].get_offsets())
    np.testing.assert_allclose(np.sort(first[:, 0]),
                               np.sort(t.scale_um[t.section.isin(["s2", "s3"])].dropna()))
    assert [x.get_text() for x in ax.get_legend().get_texts()][:2] == ["case (n = 2 sections)",
                                                                       "ctrl (n = 2 sections)"]
    assert np.allclose(ax.collections[0].get_facecolors()[0][:3], matplotlib.colors.to_rgb(_diff.U.ARM_COLORS[0]))


def test_section_scales_default_genes_prefer_complete_then_amplitude():
    t = _tidy()
    t["scale_peak_moran"] = t.gene.str[1:].astype(int) / 10.0       # f7 highest amplitude, but incomplete
    ax = pl.section_scales(t, axis="focality")
    shown = {x.get_text() for x in ax.get_yticklabels()}
    assert shown == {f"f{i}" for i in range(8)}                     # fewer than 30: every gene
    assert ax.get_xscale() == "linear"


def test_section_scales_pairs_reports_spearman():
    from scipy.stats import spearmanr
    t = _tidy()
    ax = pl.section_scales(t, kind="pairs", sections=("s0", "s1"))
    a = t[t.section == "s0"].set_index("gene").scale_um
    b = t[t.section == "s1"].set_index("gene").scale_um.loc[a.index]
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_offsets()), np.c_[a, b])
    assert ax.get_title().startswith(f"Spearman ρ = {spearmanr(a, b)[0]:.2f}")
    with pytest.raises(ValueError, match="sections"):
        pl.section_scales(t, kind="pairs")
    with pytest.raises(KeyError, match="available"):
        pl.section_scales(t, kind="pairs", sections=("s0", "zz"))
    with pytest.raises(KeyError, match="available"):
        pl.section_scales(t, axis="texture")
    with pytest.raises(TypeError, match="fingerprint_by"):
        pl.section_scales(t.drop(columns="gene"))


def test_section_scales_on_a_real_fingerprint_by_table():
    import anndata as ad
    from spectrast import gr
    rng = np.random.default_rng(0)
    a = ad.AnnData(np.abs(rng.standard_normal((600, 4))).astype("float32"))
    a.obsm["spatial"] = rng.random((600, 2)) * 100
    a.obs["section"] = ["s0"] * 300 + ["s1"] * 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tidy = gr.fingerprint_by(a, "section", min_cells=10)
    ax = pl.section_scales(tidy, groups=pd.Series({"s0": "x", "s1": "y"}))
    assert len(ax.get_yticklabels()) == 4

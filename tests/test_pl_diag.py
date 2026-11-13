"""spectrast.pl diagnostics (celltype_range, presence, qc), plus the unchanged-numbers locks on
the two library functions they read (fingerprint_presence's attrs, section_report's by_radius)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import anndata as ad  # noqa: E402
import pytest  # noqa: E402

from spectrast import gr, qc, orient  # noqa: E402
from spectrast import pl  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------------------------- synthetic data
def _presence_section(n=1200, n_org=12, n_noise=28, extent=600.0, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, extent, size=(n, 2))
    cols = []
    for k in range(n_org):
        cx, cy = rng.uniform(0.2 * extent, 0.8 * extent, 2)
        s = 40.0 + 15.0 * k
        d = np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2 * s ** 2))
        cols.append(rng.poisson(0.3 + 5.0 * d))
    for k in range(n_noise):
        cols.append(rng.poisson(0.8, n))
    a = ad.AnnData(np.column_stack(cols).astype("float32"),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(n_org + n_noise)]))
    a.obsm["spatial"] = xy
    return a


def _qc_section(n=1500, n_org=4, n_noise=6, extent=900.0, sigma=120.0, seed=0):
    r = np.random.default_rng(seed)
    xy = r.uniform(0, extent, size=(n, 2))
    d = np.exp(-((xy[:, 0] - extent / 2) ** 2 + (xy[:, 1] - extent / 2) ** 2) / (2 * sigma ** 2))
    org = r.poisson(np.tile((6 * d + 0.4)[:, None], (1, n_org)))
    noise = r.poisson(0.7, size=(n, n_noise))
    X = np.c_[org, noise].astype("float32")
    a = ad.AnnData(X, var=pd.DataFrame(index=[f"g{i}" for i in range(n_org + n_noise)]))
    a.obsm["spatial"] = xy
    return a


def _typed_section(n=3000, extent=800.0, seed=1):
    """Three cell types: A clustered in one corner, B ringing A, C everywhere."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, extent, size=(n, 2))
    d = np.hypot(xy[:, 0] - 250, xy[:, 1] - 250)
    lab = np.where(d < 120, "A", np.where((d > 150) & (d < 260), "B", "C"))
    lab[rng.random(n) < 0.15] = "C"
    a = ad.AnnData(np.ones((n, 1), "float32"))
    a.obsm["spatial"] = xy
    a.obs["ct"] = pd.Categorical(lab)
    return a


# --------------------------------------------------------------- fingerprint_presence: numbers unchanged
# Pinned from fingerprint_presence BEFORE attrs["values"] was added (same machine, same seed): the
# table must not move by a bit. Order: n_tested, n_sig, n_expected, enrichment, presence_p, score,
# effect_size.
_PRESENCE_PIN = {
    "scale_peak_moran": [40.0, 13.0, 2.0, 6.5, 0.015752717499885237, 0.325, 0.5785847491681227],
    "focality": [40.0, 8.0, 2.0, 4.0, 0.029324045900299866, 0.2, 0.7921584219580615],
    "texture": [40.0, 9.0, 2.0, 4.5, 0.13319826359822434, 0.225, 0.022685419290243748],
}
_PRESENCE_PIN2 = {   # axes=(scale_peak_moran, focality), n_perm=2, seed=7
    "scale_peak_moran": [40.0, 17.0, 2.0, 8.5, 0.0007758090331155967, 0.425, 0.5785847491681227],
    "focality": [40.0, 8.0, 2.0, 4.0, 0.017313873148006694, 0.2, 0.7921584219580615],
}
_NUM = ["n_tested", "n_sig", "n_expected", "enrichment", "presence_p", "score", "effect_size"]


@pytest.fixture(scope="module")
def presence_df():
    return gr.fingerprint_presence(_presence_section())


# Focality is a Gini over a sorted local map, which amplifies rounding: on Linux its effect size differs
# from these values (computed on macOS) by 4e-9 relative. The other columns agree to 1e-15.
_PIN_RTOL = 1e-6


def test_presence_table_is_unchanged(presence_df):
    df = presence_df
    for _, r in df.iterrows():
        assert [float(r[c]) for c in _NUM] == pytest.approx(_PRESENCE_PIN[r.axis], rel=_PIN_RTOL), r.axis
    assert list(df.underpowered) == [False, False, False]
    assert list(df.present) == [True, True, False]
    df2 = gr.fingerprint_presence(_presence_section(), axes=("scale_peak_moran", "focality"),
                                  n_perm=2, seed=7)
    for _, r in df2.iterrows():
        assert [float(r[c]) for c in _NUM] == pytest.approx(_PRESENCE_PIN2[r.axis], rel=_PIN_RTOL), r.axis


def test_presence_values_are_what_the_test_ranked(presence_df):
    """Re-running the rank test on the stored vectors reproduces every row exactly."""
    df = presence_df.set_index("axis")
    vals = presence_df.attrs["values"]
    assert set(vals) == set(df.index)
    for ax_, v in vals.items():
        assert v["genes"] == [f"g{i}" for i in range(40)]
        assert len(v["observed"]) == 40 and len(v["null"]) == 40 * v["n_perm"]
        pr = orient.axis_presence(v["observed"], v["null"], alpha=v["alpha"], higher=v["higher"],
                                  n_tested=len(v["genes"]))
        for k in ("n_sig", "enrichment", "presence_p", "score", "present"):
            assert pr[k] == df.loc[ax_, k], (ax_, k)
    df2 = gr.fingerprint_presence(_presence_section(), axes=("focality",), n_perm=3)
    assert len(df2.attrs["values"]["focality"]["null"]) == 3 * 40


@pytest.mark.filterwarnings("ignore:window sweep spans only")   # a small synthetic section
def test_presence_orientation_row_carries_its_values_too():
    df = gr.fingerprint_presence(_presence_section(), axes=("focality",), orientation=True,
                                 orient_n_perm=5)
    v = df.attrs["values"]["orientation"]
    row = df.set_index("axis").loc["orientation"]
    pr = orient.axis_presence(v["observed"], v["null"], alpha=v["alpha"], higher=True)
    for k in ("n_sig", "enrichment", "presence_p", "present"):
        assert pr[k] == row[k], k
    fig = pl.presence(df)
    assert len([a for a in fig.axes if a.get_xlabel()]) == 2


def test_presence_table_prints_when_truncated(presence_df):
    """pandas compares attrs with == when it truncates a wide table for display; an ndarray there makes
    repr raise. The values are plain Python lists and scalars, so the table prints."""
    def plain(x):
        if isinstance(x, dict):
            return all(plain(v) for v in x.values())
        if isinstance(x, list):
            return all(isinstance(v, (str, float, int, bool)) for v in x)
        return isinstance(x, (str, float, int, bool))
    assert plain(presence_df.attrs["values"])
    with pd.option_context("display.width", 60, "display.max_columns", 4):
        assert "axis" in repr(presence_df) or "..." in repr(presence_df)
        presence_df._repr_html_()


def test_presence_tables_still_concatenate(presence_df):
    """attrs hold plain lists: numpy arrays there make pd.concat of two tables raise."""
    out = pd.concat([presence_df, presence_df])
    assert len(out) == 6


# ---------------------------------------------------------------- section_report: numbers unchanged
_QC_PIN = {
    "lattice.n_cells": 1500, "lattice.n_bins_occupied": 1324, "lattice.bin_um": 11.672068742900116,
    "lattice.cells_per_bin": 1.1329305135951662, "lattice.n_radii": 18,
    "lattice.r_min_um": 23.344137485800232, "lattice.r_max_um": 933.7654994320093,
    "lattice.pitch_um": 23.344137485800232, "lattice.r_min_below_pitch": False,
    "lattice.cells_per_bin_high": False, "saturation.alpha": 0.05, "saturation.n_genes": 10,
    "saturation.n_cells_tested": 180, "saturation.frac_gene_radius_significant": 0.3277777777777778,
    "saturation.frac_genes_significant_any_radius": 0.5,
    "saturation.median_abs_I_significant": 0.5355422360629493,
    "saturation.median_abs_I_all": 0.007430493462339238, "saturation.saturated": False,
}


@pytest.fixture(scope="module")
def report():
    return qc.section_report(_qc_section(), n_units_A=3, n_units_B=3)


def test_section_report_numbers_unchanged_and_by_radius_consistent(report):
    for k, v in _QC_PIN.items():
        blk, key = k.split(".")
        assert report[blk][key] == (pytest.approx(v, rel=1e-12) if isinstance(v, float) else v), k
    br = report["saturation"]["by_radius"]
    assert br["radius_um"] == report["provenance"]["radii_um"]
    f = np.asarray(br["frac_significant"])
    assert ((0 <= f) & (f <= 1)).all()
    # every gene is tested at every radius, so the per-radius fractions average to the overall one
    assert np.mean(f) == pytest.approx(report["saturation"]["frac_gene_radius_significant"], abs=1e-12)
    qc.format_report(report)                                      # rendering still works


# ---------------------------------------------------------------------------------- celltype_range
@pytest.fixture(scope="module")
def ct_table():
    with pytest.warns(RuntimeWarning, match="majority"):      # C is over half of the cells, by design
        return gr.celltype_range(_typed_section(), cell_type_key="ct", min_cells=50)


def test_celltype_range_heatmap_colours_by_sign_and_labels_blanks(ct_table):
    tab = ct_table
    ax = pl.celltype_range(tab)
    assert isinstance(ax, plt.Axes)
    img = ax.get_images()[0].get_array()
    lv = list(dict.fromkeys(list(tab.type_a) + list(tab.type_b)))
    assert img.shape[:2] == (len(lv), len(lv))
    assert [t.get_text() for t in ax.get_yticklabels()] == lv
    sign, reason = pl._diag._peak_sign_and_reason(tab, "r_half_um")
    for k, row in tab.reset_index(drop=True).iterrows():
        i, j = lv.index(row.type_a), lv.index(row.type_b)
        px = img[i, j]
        if np.isfinite(row.r_half_um):
            # enriched pairs are red-dominant, excluded ones blue-dominant
            assert (px[0] > px[2]) if sign[k] > 0 else (px[2] > px[0]), (row.type_a, row.type_b)
        else:
            assert np.allclose(px[:3], 1.0)                       # blank
            assert reason[k] in ("F", "W", "E")
    # values and blank codes are written once per matrix cell (both triangles)
    n_cells = sum(1 if r.type_a == r.type_b else 2 for _, r in tab.iterrows())
    assert len(ax.texts) == n_cells
    n_blank = int((~np.isfinite(tab.r_half_um)).sum())
    assert ("Blank pairs" in ax.get_xlabel()) == (n_blank > 0)


def test_celltype_range_reasons_cover_floor_window_and_edge():
    """A hand-built table: one enriched, one excluded, one below the floor, one never decaying, one
    whose peak sits at the window edge."""
    r = [10.0, 20.0, 40.0, 80.0]
    rows = [
        dict(type_a="A", type_b="A", r_peak_um=10.0, r_half_um=30.0, r_cross_um=np.nan, g=[3.0, 2.0, 1.2, 1.0]),
        dict(type_a="A", type_b="B", r_peak_um=10.0, r_half_um=25.0, r_cross_um=np.nan, g=[0.3, 0.6, 0.9, 1.0]),
        dict(type_a="B", type_b="B", r_peak_um=10.0, r_half_um=np.nan, r_cross_um=np.nan, g=[1.03, 1.02, 1.0, 1.0]),
        dict(type_a="A", type_b="C", r_peak_um=10.0, r_half_um=np.nan, r_cross_um=np.nan, g=[2.0, 1.9, 1.8, 1.7]),
        dict(type_a="B", type_b="C", r_peak_um=np.nan, r_half_um=np.nan, r_cross_um=np.nan, g=[1.0, 1.2, 1.5, 2.0]),
    ]
    tab = pd.DataFrame([{**{k: v for k, v in d.items() if k != "g"},
                         **{f"g_{int(x)}": gv for x, gv in zip(r, d["g"])}} for d in rows])
    tab.attrs["radii_um"] = r
    sign, reason = pl._diag._peak_sign_and_reason(tab, "r_half_um")
    assert list(sign[:2]) == [1.0, -1.0]
    assert list(reason) == ["", "", "F", "W", "E"]
    ax = pl.celltype_range(tab)
    lab = ax.get_xlabel()
    assert "F =" in lab and "W =" in lab and "E =" in lab


def test_celltype_range_honours_ax_save_and_errors(ct_table, tmp_path):
    fig, ax0 = plt.subplots()
    out = tmp_path / "ct.png"
    ax = pl.celltype_range(ct_table, value="r_peak_um", ax=ax0, save=str(out))
    assert ax is ax0 and out.exists()
    with pytest.raises(KeyError, match=r"gr\.celltype_range"):
        pl.celltype_range(pd.DataFrame({"a": [1]}))


# ---------------------------------------------------------------------------------------- presence
def test_presence_one_panel_per_axis(presence_df, tmp_path):
    out = tmp_path / "presence.png"
    fig = pl.presence(presence_df, save=str(out))
    assert isinstance(fig, plt.Figure) and out.exists()
    axs = [a for a in fig.axes if a.patches]
    assert len(axs) == 3
    for a, name in zip(axs, presence_df.axis):
        assert name in a.get_title()
        txt = " ".join(t.get_text() for t in a.texts)
        row = presence_df.set_index("axis").loc[name]
        assert f"{float(row.enrichment):.2f}x" in txt and f"{float(row.presence_p):.2g}" in txt
    fig2 = pl.presence(presence_df, axes=["focality"])
    assert len([a for a in fig2.axes if a.patches]) == 1


def test_presence_errors_without_values(presence_df):
    bare = presence_df.copy()
    bare.attrs = {}
    with pytest.raises(ValueError, match=r"fingerprint_presence"):
        pl.presence(bare)
    with pytest.raises(KeyError, match="not in this presence table"):
        pl.presence(presence_df, axes=["orientation"])


# --------------------------------------------------------------------------------------------- qc
def test_qc_page_with_and_without_adata(report, tmp_path):
    a = _qc_section()
    out = tmp_path / "qc.png"
    fig = pl.qc(report, adata=a, save=str(out))
    assert isinstance(fig, plt.Figure) and out.exists()
    titles = [x.get_title(loc="left") or x.get_title() for x in fig.axes]
    assert any("Saturation" in t for t in titles) and any("occupancy" in t for t in titles)
    assert any("Design floor" in t for t in titles)
    # the occupancy image counts every cell once
    img = [x for x in fig.axes if x.get_images()][0].get_images()[0].get_array()
    assert int(np.ma.filled(img, 0).sum()) == a.n_obs
    # the saturation curve is the report's by_radius
    sat = [x for x in fig.axes if "Saturation" in x.get_title(loc="left")][0]
    line = sat.get_lines()[0]
    np.testing.assert_array_equal(line.get_ydata(), report["saturation"]["by_radius"]["frac_significant"])
    fig2 = pl.qc(qc.section_report(_qc_section()))
    assert not any(x.get_images() for x in fig2.axes)


def test_qc_errors(report):
    with pytest.raises(KeyError, match=r"qc\.section_report"):
        pl.qc({"lattice": {}})
    old = {k: (dict(v) if isinstance(v, dict) else v) for k, v in report.items()}
    del old["saturation"]["by_radius"]
    with pytest.raises(KeyError, match=r"qc\.section_report"):
        pl.qc(old)
    with pytest.raises(ValueError, match="cells"):
        pl.qc(report, adata=_qc_section()[:100].copy())

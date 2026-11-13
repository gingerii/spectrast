"""spectrast.pl single-section plots: each returns the right object, draws the data it was given,
honours ax= and save=, and names the call to run when its input is missing."""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spectrast import gr  # noqa: E402
from spectrast import organization as org  # noqa: E402
from spectrast import pl  # noqa: E402
from spectrast.pl import _section as S  # noqa: E402


def _make(n_cells=1500, seed=0):
    """Uniform cells; eight blob genes of increasing width (40-200 µm) and four shuffled copies;
    a categorical with a patchy and a scattered type; a numeric obs column."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 800, size=(n_cells, 2))
    cols, names = [], []
    for k, sig in enumerate(np.geomspace(40, 200, 8)):
        cx, cy = rng.uniform(250, 550, 2)
        d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        cols.append(np.clip(np.exp(-d2 / (2 * sig ** 2)) * 4 + rng.normal(0, 0.3, n_cells), 0, None))
        names.append(f"blob{k}")
    for k in range(4):
        cols.append(rng.permutation(cols[k])); names.append(f"noise{k}")
    a = ad.AnnData(np.column_stack(cols))
    a.var_names = names
    a.obsm["spatial"] = xy
    lab = np.full(n_cells, "other", dtype=object)
    lab[rng.choice(n_cells, 300, replace=False)] = "scattered"
    lab[(np.hypot(xy[:, 0] - 300, xy[:, 1] - 300) < 110) & (lab == "other")] = "patchy"
    a.obs["ctype"] = pd.Categorical(lab)
    a.obs["depth"] = xy[:, 0] / 800.0
    return a


@pytest.fixture(scope="module")
def data():
    a = _make()
    gr.scale_spectrum(a)
    gr.focality(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # the analytic null's cells-per-bin note
        null = gr.moran_null(a, genes=["blob0", "blob3", "noise0"])
    tab, cur = org.organization(a, return_curves=True)
    return dict(adata=a, null=null, table=tab, curves=cur)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _by_label(ax, label):
    hits = [c for c in ax.collections if c.get_label() == label]
    assert hits, f"no collection labelled {label!r}"
    return hits[0]


def _rows(table, stat="moran"):
    return table[table.statistic == stat].reset_index(drop=True)


# --------------------------------------------------------------------------- scale_plane
def test_scale_plane_draws_the_read_genes_at_their_scale_and_amplitude(data):
    t = _rows(data["table"])
    ax = pl.scale_plane(data["table"], gate=0.5)
    assert isinstance(ax, matplotlib.axes.Axes)
    thr = np.quantile(t.amplitude, 0.5)
    want = t[(t.amplitude >= thr) & np.isfinite(t.scale_um)]
    got = _by_label(ax, "read").get_offsets()
    np.testing.assert_allclose(np.sort(np.asarray(got)[:, 0]), np.sort(want.scale_um.to_numpy()))
    np.testing.assert_allclose(np.sort(np.asarray(got)[:, 1]), np.sort(want.amplitude.to_numpy()))
    gate_lines = [ln for ln in ax.lines if np.allclose(ln.get_ydata(), thr)]
    assert gate_lines and gate_lines[0].get_linestyle() == "--"


def test_scale_plane_absolute_gate_and_colour_from_var(data):
    t = _rows(data["table"])
    thr = float(np.median(t.amplitude))
    ax = pl.scale_plane(data["table"], gate=("abs", thr), color="spectrast_focality",
                        adata=data["adata"])
    sc = _by_label(ax, "read")
    read = t[(t.amplitude >= thr) & np.isfinite(t.scale_um)]
    foc = data["adata"].var["spectrast_focality"].reindex(read.gene).to_numpy()
    np.testing.assert_allclose(np.sort(sc.get_array()), np.sort(foc))


def test_scale_plane_colour_array_must_align(data):
    n = len(_rows(data["table"]))
    pl.scale_plane(data["table"], color=np.arange(n, dtype=float))
    with pytest.raises(ValueError, match="rows"):
        pl.scale_plane(data["table"], color=np.arange(n + 1, dtype=float))


def test_scale_plane_counts_unplaced_genes_and_draws_the_scramble(data):
    t = data["table"].copy()
    i = t.index[(t.statistic == "moran")][0]
    t.loc[i, "scale_um"] = np.nan
    fig, ax = plt.subplots()
    out = pl.scale_plane(t, scramble=data["table"], ax=ax)
    assert out is ax
    n_nan = int(_rows(t).scale_um.isna().sum())
    assert n_nan >= 1
    assert f"{n_nan} with no scale not placed" in ax.get_title()
    from matplotlib.contour import ContourSet
    assert any(isinstance(c, ContourSet) for c in ax.collections) or len(ax.collections) > 2


def test_scale_plane_errors_name_the_call(data):
    with pytest.raises(KeyError, match="organization"):
        pl.scale_plane(pd.DataFrame({"gene": ["a"]}))
    with pytest.raises(ValueError, match="no rows for statistic"):
        pl.scale_plane(data["table"], stat="not_a_stat")
    with pytest.raises(ValueError, match="adata="):
        pl.scale_plane(data["table"], color="spectrast_focality")
    with pytest.raises(KeyError, match="fingerprint"):
        pl.scale_plane(data["table"], color="spectrast_focality_x", adata=data["adata"])
    with pytest.raises(ValueError, match="abs"):
        pl.scale_plane(data["table"], gate=1.5)


def test_scale_plane_saves(data, tmp_path):
    p = tmp_path / "plane.png"
    pl.scale_plane(data["table"], save=str(p))
    assert p.exists() and p.stat().st_size > 1000


# --------------------------------------------------------------------------- curves
def test_curves_draw_the_stored_scalogram_and_mark_r_star(data):
    a = data["adata"]
    st = a.uns["spectrast_scalogram"]
    ax = pl.curves(a, ["blob0", "blob5"])
    j = list(st["genes"]).index("blob5")
    line = [ln for ln in ax.lines if ln.get_label() == "blob5"][0]
    np.testing.assert_array_equal(line.get_ydata(), st["scalogram"][:, j])
    np.testing.assert_array_equal(line.get_xdata(), st["radii"])
    rs = a.var.loc["blob5", "spectrast_char_scale"]
    assert any(ln.get_marker() == "D" and np.isclose(ln.get_xdata()[0], rs) for ln in ax.lines)


def test_curves_with_null_fill_markers_where_significant(data):
    a, null = data["adata"], data["null"]
    fig, ax = plt.subplots()
    out = pl.curves(a, ["blob0", "noise0"], null=null, alpha=0.05, ax=ax)
    assert out is ax
    n_sig = int((null[null.gene.isin(["blob0", "noise0"])].p < 0.05).sum())
    n_all = int(null.gene.isin(["blob0", "noise0"]).sum())
    pts = [c for c in ax.collections if hasattr(c, "get_offsets") and len(c.get_offsets())]
    filled = sum(len(c.get_offsets()) for c in pts
                 if c.get_facecolor().size and c.get_facecolor()[0][:3].tolist() != [1.0, 1.0, 1.0]
                 and type(c).__name__ == "PathCollection")
    total = sum(len(c.get_offsets()) for c in pts if type(c).__name__ == "PathCollection")
    assert total == n_all
    assert filled == n_sig


def test_curves_errors(data):
    import anndata as ad
    with pytest.raises(KeyError, match="scale_spectrum"):
        pl.curves(ad.AnnData(np.zeros((3, 2))), ["a"])
    with pytest.raises(KeyError, match="not found"):
        pl.curves(data["adata"], ["not_a_gene"])
    with pytest.raises(KeyError, match="moran_null"):
        pl.curves(data["adata"], ["blob1"], null=data["null"])


# --------------------------------------------------------------------------- significance_heatmap
def test_significance_heatmap_image_is_the_z_table_in_r_star_order(data):
    a, null = data["adata"], data["null"]
    ax = pl.significance_heatmap(null, adata=a)
    img = ax.images[0].get_array()
    rs = a.var["spectrast_char_scale"].reindex(["blob0", "blob3", "noise0"])
    order = rs.sort_values(na_position="last").index
    piv = null.pivot(index="gene", columns="r_um", values="z").loc[order]
    np.testing.assert_allclose(np.asarray(img), piv.to_numpy())
    assert [t.get_text() for t in ax.get_yticklabels()] == list(order)


def test_significance_heatmap_log_p_and_zmax_order(data):
    null = data["null"]
    fig, ax = plt.subplots()
    out = pl.significance_heatmap(null, value="-log10p", order="zmax", ax=ax)
    assert out is ax
    img = np.asarray(ax.images[0].get_array())
    assert img.shape == (3, null.r_um.nunique())
    assert np.isfinite(img).all() and (img >= 0).all()


def test_significance_heatmap_errors(data):
    with pytest.raises(ValueError, match="adata="):
        pl.significance_heatmap(data["null"])
    with pytest.raises(KeyError, match="moran_null"):
        pl.significance_heatmap(pd.DataFrame({"gene": ["a"]}), order=None)
    with pytest.raises(ValueError, match="value"):
        pl.significance_heatmap(data["null"], value="q", order=None)


# --------------------------------------------------------------------------- gene_card
def test_gene_card_draws_the_given_curves(data, tmp_path):
    cur = data["curves"]
    p = tmp_path / "card.png"
    fig = pl.gene_card(data["adata"], "blob4", curves=cur, save=str(p))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert p.exists()
    j = list(cur["genes"]).index("blob4")
    titles = {ax.get_title().split(":")[0].split("\n")[0]: ax for ax in fig.axes}
    for s in org.ORG_STATS:
        ax = titles[s]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), cur[s][:, j])


def test_gene_card_computes_one_gene_when_no_curves_given(data):
    a = data["adata"]
    keys = (set(a.obsm), set(a.uns), set(a.var.columns))
    fig = pl.gene_card(a, "blob2", stats=("moran", "geary"))
    assert len([ax for ax in fig.axes if ax.get_title().startswith(("moran", "geary"))]) == 2
    assert (set(a.obsm), set(a.uns), set(a.var.columns)) == keys


def test_gene_card_errors(data):
    with pytest.raises(KeyError, match="not in adata.var_names"):
        pl.gene_card(data["adata"], "nope")
    with pytest.raises(KeyError, match="return_curves"):
        pl.gene_card(data["adata"], "blob1", curves={"radii_um": [1, 2], "genes": ["blob1"]})
    with pytest.raises(ValueError, match="unknown statistics"):
        pl.gene_card(data["adata"], "blob1", stats=("moran", "bogus"))


# --------------------------------------------------------------------------- local_map
def test_local_map_mean_is_the_stored_curve_at_that_radius(data):
    """The map is the global statistic resolved over cells: at a stored radius its mean over cells
    equals the stored scalogram's value there (same lattice, same neighbourhood)."""
    a = data["adata"]
    st = a.uns["spectrast_scalogram"]
    j = list(st["genes"]).index("blob3")
    for k in (2, 9):
        ax = pl.local_map(a, "blob3", radius=float(st["radii"][k]), show_curve=False)
        vals = np.asarray(ax.collections[0].get_array(), float)
        assert vals.size == a.n_obs
        np.testing.assert_allclose(vals.mean(), st["scalogram"][k, j], rtol=1e-9, atol=1e-12)


def test_local_map_has_no_side_effects_and_defaults_to_r_star(data):
    a = data["adata"]
    before = (set(a.obsm), set(a.uns), set(a.var.columns), set(a.obs.columns))
    ax = pl.local_map(a, "blob6")
    assert (set(a.obsm), set(a.uns), set(a.var.columns), set(a.obs.columns)) == before
    rs = a.var.loc["blob6", "spectrast_char_scale"]
    assert f"{rs:.0f} µm (r*)" in ax.get_title()
    curve_ax = [x for x in ax.figure.axes if x is not ax and x.get_xscale() == "log"][0]
    np.testing.assert_array_equal(curve_ax.lines[0].get_ydata(),
                                  a.uns["spectrast_scalogram"]["scalogram"][:, 6])
    from matplotlib.patches import Circle
    circ = [p for p in ax.patches if isinstance(p, Circle)]
    assert circ and np.isclose(circ[0].get_radius(), rs)
    lo, hi = ax.collections[0].get_clim()
    assert np.isclose(lo, -hi)


def test_local_map_indicator_and_obs_column_and_ax(data):
    a = data["adata"]
    fig, ax = plt.subplots()
    out = pl.local_map(a, "ctype==patchy", radius=60.0, ax=ax)
    assert out is ax
    assert len(ax.child_axes) == 1                       # the curve inset beneath
    vals = np.asarray(ax.collections[0].get_array(), float)
    z = (a.obs.ctype == "patchy").to_numpy(float)
    assert vals.size == z.size
    assert vals[np.argsort(-np.abs(vals))][:50].mean() > 0
    pl.local_map(a, "depth", radius=80.0, show_curve=False, circle=False)


def test_local_map_errors(data):
    with pytest.raises(KeyError, match="neither a gene"):
        pl.local_map(data["adata"], "nope", radius=50.0)
    with pytest.raises(KeyError):
        pl.local_map(data["adata"], "missing_key==x", radius=50.0)


# --------------------------------------------------------------------------- local_filmstrip
def test_local_filmstrip_one_panel_per_radius(data, tmp_path):
    a = data["adata"]
    p = tmp_path / "film.png"
    fig = pl.local_filmstrip(a, "blob5", radii=[30, 60, 120], shared_scale=True, save=str(p))
    maps = [ax for ax in fig.axes if ax.get_title().startswith("r = ")]
    assert len(maps) == 3 and p.exists()
    clims = {ax.collections[0].get_clim() for ax in maps}
    assert len(clims) == 1
    fig = pl.local_filmstrip(a, "blob5")
    assert len([ax for ax in fig.axes if ax.get_title().startswith("r = ")]) == 5


# --------------------------------------------------------------------------- curve_embedding
def test_curve_embedding_places_every_read_gene(data):
    a = data["adata"]
    ax = pl.curve_embedding(a, gate=None)
    n = sum(len(c.get_offsets()) for c in ax.collections)
    assert n == a.n_vars
    fig, ax2 = plt.subplots()
    assert pl.curve_embedding(data["curves"], color="amplitude", gate=0.5, ax=ax2) is ax2


def test_curve_embedding_pca_matches_numpy_svd(data):
    a = data["adata"]
    ax = pl.curve_embedding(a, gate=None, color="amplitude")
    C, *_ = S._read_curves(a, None)
    M = C - C.mean(0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    E = U[:, :2] * s[:2]
    got = np.asarray(ax.collections[-1].get_offsets())
    np.testing.assert_allclose(np.sort(np.abs(got[:, 0])), np.sort(np.abs(E[:, 0])), rtol=1e-9)


def test_curve_embedding_takes_array_colours(data):
    """Values aligned to the source's genes, as an ndarray or a Series, colour the points; only a
    string is compared with the named colourings."""
    for src in (data["adata"], data["curves"]):
        *_, idx = S._read_curves(src, None)
        vals = np.arange(len(S._curve_source(src)[2]), dtype=float) + 1.0
        for colour in (vals, pd.Series(vals)):
            ax = pl.curve_embedding(src, gate=None, color=colour)
            np.testing.assert_array_equal(np.asarray(ax.collections[-1].get_array()), vals[idx])
            plt.close("all")


def test_curve_embedding_family_and_errors(data, monkeypatch):
    a = data["adata"]
    fig, fam = pl.curve_families(a, k=3, gate=None, return_labels=True)
    ax = pl.curve_embedding(a, gate=None, color="family", families=fam)
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"1", "2", "3"}
    with pytest.raises(ValueError, match="families="):
        pl.curve_embedding(a, color="family")
    with pytest.raises(KeyError, match="nope"):
        pl.curve_embedding(a, color="nope")
    import sys
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match="umap-learn"):
        pl.curve_embedding(a, method="umap", gate=None)


# --------------------------------------------------------------------------- curve_families
def test_curve_families_numbered_fine_to_coarse(data):
    a = data["adata"]
    fig, fam = pl.curve_families(a, k=3, gate=None, return_labels=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert set(fam.unique()) == {1, 2, 3}
    rs = a.var["spectrast_char_scale"].reindex(fam.index)
    med = [np.nanmedian(rs[fam == f]) for f in (1, 2, 3)]
    assert med == sorted(med)
    img = np.asarray(fig.axes[0].images[0].get_array())
    assert img.shape == (len(fam), len(a.uns["spectrast_scalogram"]["radii"]))
    assert np.nanmin(img) >= 0 and np.nanmax(img) <= 1
    fam2 = pl.curve_families(a, k=3, gate=None, return_labels=True)[1]
    pd.testing.assert_series_equal(fam, fam2)


def test_curve_families_terms_column(data, tmp_path):
    a = data["adata"]
    terms = pd.DataFrame({"family": [1, 1, 2, 3], "term": ["t1", "t2", "t3", "t4"],
                          "q": [1e-3, 1e-2, 1e-4, 0.03], "fold": [3.0, 2.0, 5.0, 1.5]})
    p = tmp_path / "fam.png"
    fig = pl.curve_families(a, k=3, gate=None, terms=terms, save=str(p))
    bars = fig.axes[-1].patches
    assert sorted(b.get_width() for b in bars) == [1.5, 2.0, 3.0, 5.0]
    assert p.exists()


def test_curve_families_errors(data):
    with pytest.raises(ValueError, match="lower k"):
        pl.curve_families(data["adata"], k=50, gate=None)
    with pytest.raises(KeyError, match="'q' or a 'fold'"):
        pl.curve_families(data["adata"], k=2, gate=None,
                          terms=pd.DataFrame({"family": [1], "term": ["x"]}))
    with pytest.raises(KeyError, match="return_curves"):
        pl.curve_families({"radii_um": [1.0, 2.0]}, k=2)


# --------------------------------------------------------------------------- celltype_ladder
def test_celltype_ladder_places_types_fine_to_coarse_and_notes_the_unplaced(data):
    a = data["adata"]
    tab = org.celltype_scale(a, "ctype", min_cells=100)
    tab = pd.concat([tab, pd.DataFrame({"cell_type": ["ghost"], "n_cells": [600],
                                        "scale_um": [np.nan], "amplitude": [0.001]})],
                    ignore_index=True)
    ax = pl.celltype_ladder(tab)
    off = np.asarray(ax.collections[0].get_offsets())
    placed = tab[np.isfinite(tab.scale_um)].sort_values("scale_um")
    np.testing.assert_allclose(off[:, 0], placed.scale_um.to_numpy())
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_array()), placed.amplitude.to_numpy())
    assert [t.get_text() for t in ax.get_yticklabels()] == list(placed.cell_type)
    assert any("ghost" in t.get_text() for t in ax.texts)
    fig, ax2 = plt.subplots()
    assert pl.celltype_ladder(tab, ax=ax2) is ax2


def test_celltype_ladder_error_names_the_call():
    with pytest.raises(KeyError, match="celltype_scale"):
        pl.celltype_ladder(pd.DataFrame({"gene": ["a"]}))


def test_every_section_plot_is_exported():
    for name in ("scale_plane", "curves", "significance_heatmap", "gene_card", "local_map",
                 "local_filmstrip", "curve_embedding", "curve_families", "celltype_ladder"):
        assert callable(getattr(pl, name))

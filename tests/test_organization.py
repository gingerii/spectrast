"""The unified organization reduction: the two criteria, and the flags that make a scale auditable."""
import numpy as np
import pandas as pd
import pytest

from spectrast import organization as org


def _curve(n_r=19, n_g=40, kind="decay", seed=0):
    rng = np.random.default_rng(seed)
    r = np.geomspace(30.0, 600.0, n_r)
    if kind == "decay":                       # flattens well inside the window
        xi = rng.uniform(60, 150, n_g)
        C = np.exp(-r[:, None] / xi[None, :])
    elif kind == "cumulative":                # still climbing at the edge, like General G
        C = (r[:, None] / r[-1]) ** 1.6 * rng.uniform(0.5, 1.5, n_g)[None, :]
    return C, r


def test_reduction_returns_a_scale_for_a_curve_that_flattens():
    C, r = _curve(kind="decay")
    R = org._reduce_curve(C, r)
    assert R["scale_supported"].mean() > 0.9
    assert not R["window_limited"].any()
    assert np.all(R["scale_um"][R["scale_supported"]] > 0)


def test_cumulative_curve_is_flagged_window_limited_even_though_it_yields_a_scale():
    """The General G case, and the reason grid_suspect alone is not enough.

    A cumulative curve flipped to the decaying convention produces a confident half-maximum for a
    quantity that has no characteristic scale. The scale is PLACED BY THE WINDOW, not by the tissue,
    and it is nowhere near the grid's centre of mass -- so only window_limited catches it."""
    C, r = _curve(kind="cumulative")
    R = org._reduce_curve(C, r)
    assert R["scale_supported"].mean() > 0.9, "the artefact looks supported -- that is the point"
    assert R["window_limited"].all(), "a curve still climbing at r_max must be flagged"


def test_the_two_flags_are_not_redundant():
    decay, r = _curve(kind="decay")
    cum, _ = _curve(kind="cumulative")
    assert not org._reduce_curve(decay, r)["window_limited"].any()
    assert org._reduce_curve(cum, r)["window_limited"].all()


def test_scale_is_nan_not_extrapolated_when_the_half_maximum_is_never_crossed():
    """A flat curve has no crossing to find. Note that ANY monotone decline does cross -- the level
    is half of the curve's OWN range, not an absolute -- so the non-crossing cases are a flat curve
    and one whose maximum is interior rather than at the short-radius end."""
    r = np.geomspace(30.0, 600.0, 19)
    flat = np.ones((19, 5))
    R = org._reduce_curve(flat, r)
    assert np.isnan(R["scale_um"]).all()
    assert not R["scale_supported"].any()


def test_a_curve_peaking_mid_window_reports_no_scale_rather_than_a_wrong_one():
    r = np.geomspace(30.0, 600.0, 19)
    bump = np.exp(-((np.arange(19) - 9.0) ** 2) / 8.0)        # peak in the middle, ends equal
    R = org._reduce_curve(np.repeat(bump[:, None], 4, axis=1), r)
    assert np.isnan(R["scale_um"]).all()


def test_an_undefined_radius_before_the_crossing_is_skipped_not_reported_as_the_scale():
    """A texture band with no Fourier mode is NaN. When that radius sits just before the half-max
    crossing, the crossing is interpolated from the last defined radius, never read off the NaN."""
    from spectrast import scale as sc
    r = np.geomspace(30.0, 600.0, 19)
    C = np.exp(-r / 100.0)[:, None]
    full = org._reduce_curve(C, r)["scale_um"][0]
    j = int(np.flatnonzero(C[:, 0] <= C.min() + 0.5 * (C.max() - C.min()))[0])
    gap = C.copy()
    gap[j - 1] = np.nan
    got = org._reduce_curve(gap, r)["scale_um"][0]
    ok = np.isfinite(gap[:, 0])
    want = sc.half_max(gap[ok], r[ok], sign=1.0)[0]
    assert got != r[j - 1]
    assert got == pytest.approx(want, rel=1e-12)
    assert r[j - 2] < got < r[j] and got != full
    # a curve peaking mid-window has no scale; an undefined first radius must not give it one
    bump = np.repeat(np.exp(-((np.arange(19) - 9.0) ** 2) / 8.0)[:, None], 2, axis=1)
    bump[0, 0] = np.nan
    assert np.isnan(org._reduce_curve(bump, r)["scale_um"]).all()


def test_reduction_rejects_a_radius_mismatch():
    C, r = _curve()
    with pytest.raises(ValueError, match="radii"):
        org._reduce_curve(C, r[:-1])


# ------------------------------------------------------------------ the second criterion

def _tidy(sections, stats, values, underpowered=()):
    rows = [dict(section=s, statistic=t, effect_size=v, underpowered=s in underpowered)
            for s, vs in zip(sections, values) for t, v in zip(stats, vs)]
    return pd.DataFrame(rows)


def test_presence_variation_refuses_a_single_section():
    """Cross-tissue variation is not a property one section can exhibit, and a column of NaN would
    be an answer this function does not have."""
    d = _tidy(["a"], ["moran"], [[0.9]])
    with pytest.raises(ValueError, match="at least two sections"):
        org.presence_variation(d)


def test_presence_variation_separates_a_stable_statistic_from_a_variable_one():
    d = _tidy(["a", "b", "c", "d"], ["stable", "variable"],
              [[0.90, 0.10], [0.91, 0.60], [0.89, 0.35], [0.90, 0.85]])
    out = org.presence_variation(d).set_index("statistic")
    assert not out.loc["stable", "varies_across_tissues"]
    assert out.loc["variable", "varies_across_tissues"]
    assert out.loc["stable", "cv_pct"] < out.loc["variable", "cv_pct"]


def test_a_section_underpowered_on_every_statistic_is_dropped_not_averaged_in():
    """One section at the noise floor otherwise manufactures apparent variation in EVERY statistic
    at once -- a statement about that section, not about any statistic."""
    good = [[0.90, 0.92], [0.91, 0.93], [0.89, 0.91]]
    d = _tidy(["a", "b", "c", "dead"], ["s1", "s2"], good + [[0.05, 0.04]], underpowered=["dead"])
    out = org.presence_variation(d)
    assert (out.n_sections == 3).all(), "the dead section must not be counted"
    assert out.n_underpowered.iloc[0] == 1
    assert not out.varies_across_tissues.any()


def test_dropping_underpowered_sections_never_leaves_fewer_than_two():
    d = _tidy(["a", "dead"], ["s1"], [[0.9], [0.05]], underpowered=["dead"])
    with pytest.warns(RuntimeWarning, match="fewer than two"):
        out = org.presence_variation(d)
    assert out.n_sections.iloc[0] == 2


def test_presence_variation_requires_its_columns():
    d = pd.DataFrame({"statistic": ["a"], "section": ["s"]})
    with pytest.raises(ValueError, match="effect_size"):
        org.presence_variation(d)


# ------------------------------------------------------------------ the adata entry points

def _section(n_cells=6000, n_each=10, seed=0):
    """A synthetic section with genes of KNOWN structure: four smooth (a broad spatial gradient or
    blob) and four unstructured (expression shuffled across positions, so the marginal is identical
    and only the arrangement differs)."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, size=(n_cells, 2))
    cols, names = [], []
    for k in range(n_each):                                # structured: broad blobs
        cx, cy = rng.uniform(200, 800, 2)
        d2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        v = np.exp(-d2 / (2 * 250.0 ** 2)) * 5.0 + rng.normal(0, 0.2, n_cells)
        cols.append(np.clip(v, 0, None)); names.append(f"struct{k}")
    for k in range(n_each):                                # unstructured: same values, shuffled
        cols.append(rng.permutation(cols[k].copy())); names.append(f"shuf{k}")
    X = np.column_stack(cols).astype(float)
    a = ad.AnnData(X)
    a.var_names = names
    a.obsm["spatial"] = xy
    return a


@pytest.fixture(scope="module")
def section():
    return _section()


def test_organization_returns_one_row_per_gene_and_statistic(section):
    stats = ("moran", "general_g")
    out = org.organization(section, stats=stats)
    assert len(out) == section.n_vars * len(stats)
    assert set(out.statistic) == set(stats)
    for c in ("scale_um", "amplitude", "scale_supported", "grid_suspect",
              "window_limited", "n_valid_radii"):
        assert c in out.columns, c


def test_organization_rejects_an_unknown_statistic(section):
    with pytest.raises(ValueError, match="unknown statistics"):
        org.organization(section, stats=("moran", "not_a_statistic"))


def test_general_g_is_flagged_window_limited_on_real_geometry(section):
    """The flag's whole purpose, exercised through the public entry point rather than the helper:
    General G is cumulative, so it yields a confident scale that must not be trusted.

    Only the CONTRAST is asserted, not the absolute rate. This synthetic section is uniform random
    cells with broad Gaussian blobs, which does not flatten inside the default radius window, so
    Moran reads ~0.55 here against 0.0-0.1% on real tissue. The flag's calibration against real
    banked curves lives in the WINDOW_LIMITED_TOL docstring; this test guards the ordering."""
    out = org.organization(section, stats=("moran", "general_g"))
    g = out[out.statistic == "general_g"]
    m = out[out.statistic == "moran"]
    assert g.window_limited.mean() > m.window_limited.mean()


def test_every_scale_carries_its_flags(section):
    """A scale without its distrust flags is the failure mode this module exists to prevent, so a
    NaN scale must never be reported as supported."""
    out = org.organization(section, stats=("moran", "losh"))
    assert not out.loc[out.scale_um.isna(), "scale_supported"].any()


def test_presence_separates_structured_from_shuffled(section):
    """The paired criterion, on data whose answer is known by construction, tested BOTH ways.

    Pooling structured and shuffled genes caps frac_outside by construction, so each set is tested
    on its own: the structured genes must clear the criterion and the shuffled ones must not."""
    struct = [g for g in section.var_names if g.startswith("struct")]
    shuf = [g for g in section.var_names if g.startswith("shuf")]
    hi = org.presence(section[:, struct], stats=("moran",), n_perm=8, seed=0).iloc[0]
    lo = org.presence(section[:, shuf], stats=("moran",), n_perm=8, seed=0).iloc[0]
    assert hi.separates_genes, f"structured genes must separate (frac_outside={hi.frac_outside})"
    assert not lo.separates_genes, f"shuffled genes must not (frac_outside={lo.frac_outside})"
    assert hi.frac_outside > lo.frac_outside
    assert hi.median_abs_z > lo.median_abs_z


def test_presence_refuses_too_few_permutations(section):
    """With a small n_perm the threshold exceeds 1 and separates_genes could never be True --
    a flag that silently reads False for every input however strong the signal."""
    with pytest.raises(ValueError, match="too small for the separation criterion"):
        org.presence(section, stats=("moran",), n_perm=4)


def test_presence_names_the_unpaired_column_as_a_guard(section):
    """Reaching for the unpaired readout as a selection criterion should require typing the word
    'guard'. This project selected on it once already and got focality and texture backwards."""
    s = org.presence(section, stats=("moran",), n_perm=8)
    assert "auc_sep_unpaired_guard" in s.columns
    assert "auc_sep" not in s.columns


def test_separates_genes_follows_the_paired_readout_not_the_guard(section):
    s = org.presence(section, stats=("moran",), n_perm=8).iloc[0]
    assert bool(s.separates_genes) == bool(
        s.frac_outside > org.SEPARATES_MULTIPLE * s.chance_frac_outside)


def test_presence_does_not_return_a_cross_tissue_column(section):
    """One section cannot exhibit cross-tissue variation; the column belongs to presence_variation."""
    s = org.presence(section, stats=("moran",), n_perm=8)
    assert "varies_across_tissues" not in s.columns


def test_presence_rejects_an_unknown_statistic(section):
    with pytest.raises(ValueError, match="unknown statistics"):
        org.presence(section, stats=("nope",), n_perm=8)


# ------------------------------------------------------------------ return_curves and celltype_scale

def test_return_curves_leaves_the_table_bit_identical(section):
    """Asking for the curves must not change a single number in the table."""
    stats = ("moran", "geary", "losh")
    plain = org.organization(section, stats=stats)
    tab, cur = org.organization(section, stats=stats, return_curves=True)
    pd.testing.assert_frame_equal(plain, tab, check_exact=True)
    assert set(plain.attrs) == set(tab.attrs)
    for k in plain.attrs:
        np.testing.assert_array_equal(np.asarray(plain.attrs[k]), np.asarray(tab.attrs[k]))
    assert set(cur) == {"radii_um", "genes", *stats}
    np.testing.assert_array_equal(cur["radii_um"], tab.attrs["radii_um"])
    np.testing.assert_array_equal(cur["genes"], np.asarray(section.var_names, str))
    for s in stats:
        assert cur[s].shape == (len(cur["radii_um"]), section.n_vars)


def test_returned_curves_reduce_to_the_table_through_scale_half_max(section):
    """The curves are the ones the table was reduced from: the package's one half-max reduction,
    applied to them with the population orientation, gives the table's scale, and their oriented
    peak gives its amplitude. ``half_max`` needs a curve defined at every radius, and the table
    skips undefined radii (texture bands holding no Fourier mode), so each curve is reduced over the
    radii where it is defined."""
    from spectrast import scale as sc
    stats = ("moran", "geary", "general_g", "losh", "focality", "texture")
    tab, cur = org.organization(section, stats=stats, return_curves=True)
    r = cur["radii_um"]
    for s in stats:
        C = cur[s]
        sign = sc.orient_sign(C)
        t = tab[tab.statistic == s].set_index("gene").loc[cur["genes"]]
        want = np.full(C.shape[1], np.nan)
        for i in range(C.shape[1]):
            ok = np.isfinite(C[:, i])
            if ok.any():
                want[i] = sc.half_max(C[ok, i:i + 1], r[ok], sign=sign)[0]
        np.testing.assert_allclose(want, t.scale_um.to_numpy(), rtol=1e-12, equal_nan=True, err_msg=s)
        np.testing.assert_allclose(np.nanmax(sign * C, axis=0), t.amplitude.to_numpy(), rtol=1e-12,
                                   err_msg=s)


def _typed_section(n_cells=6000, seed=1):
    """Uniform cells with four labels: 'scattered' is a random 15% of ALL cells (drawn first, so it
    has no hole where the patches are), 'patchy' fills the rest of three 120 µm disks, 'rare' has
    40 cells, the remainder are 'other'."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, size=(n_cells, 2))
    lab = np.full(n_cells, "other", dtype=object)
    lab[rng.choice(n_cells, int(0.15 * n_cells), replace=False)] = "scattered"
    for cx, cy in [(250, 250), (700, 300), (450, 750)]:
        lab[(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < 120) & (lab == "other")] = "patchy"
    free = np.flatnonzero(lab == "other")
    lab[rng.choice(free, 40, replace=False)] = "rare"
    a = ad.AnnData(rng.random((n_cells, 1)))
    a.obs["ctype"] = pd.Categorical(lab)
    a.obsm["spatial"] = xy
    return a


@pytest.fixture(scope="module")
def typed():
    return _typed_section()


def test_celltype_scale_places_a_patchy_type_coarser_than_a_scattered_one(typed):
    tab = org.celltype_scale(typed, "ctype", min_cells=100).set_index("cell_type")
    assert set(tab.index) == {"patchy", "scattered", "other"}
    assert np.isfinite(tab.loc["patchy", "scale_um"])
    sc_scat = tab.loc["scattered", "scale_um"]
    assert np.isnan(sc_scat) or tab.loc["patchy", "scale_um"] > 2 * sc_scat
    assert tab.loc["patchy", "amplitude"] > 5 * abs(tab.loc["scattered", "amplitude"])
    assert tab.loc["patchy", "n_cells"] == int((typed.obs.ctype == "patchy").sum())


def test_celltype_scale_lists_the_types_it_skipped(typed):
    tab = org.celltype_scale(typed, "ctype", min_cells=100)
    assert "rare" not in set(tab.cell_type)
    assert ("rare", 40) in tab.attrs["skipped"]
    for k in ("radii_um", "bin_um", "grid_com_um", "cell_type_key"):
        assert k in tab.attrs


def test_celltype_scale_is_organization_on_the_indicator_matrix(typed):
    """The documented route, checked to the bit: a type's row is organization()'s Moran row for
    its 0/1 indicator on the same coordinates, and the curves are the ones it reduced."""
    import anndata as ad
    tab, cur = org.celltype_scale(typed, "ctype", min_cells=100, return_curves=True)
    types = list(tab.cell_type)
    ind = np.column_stack([(typed.obs.ctype.astype(str).to_numpy() == t).astype(float) for t in types])
    b = ad.AnnData(ind)
    b.var_names = types
    b.obsm["spatial"] = typed.obsm["spatial"]
    ref, rc = org.organization(b, stats=("moran",), return_curves=True)
    for c in ("scale_um", "amplitude", "scale_supported", "grid_suspect", "window_limited"):
        np.testing.assert_array_equal(tab[c].to_numpy(), ref[c].to_numpy(), err_msg=c)
    np.testing.assert_array_equal(cur["moran"], rc["moran"])
    np.testing.assert_array_equal(cur["types"], np.asarray(types, str))
    assert cur["moran"].shape == (len(cur["radii_um"]), len(types))


def test_celltype_scale_uses_the_lattice_a_gene_call_would(typed):
    """Commensurate with genes: default radii and bin equal those organization() resolves."""
    tab = org.celltype_scale(typed, "ctype", min_cells=100)
    genes = org.organization(typed, stats=("moran",))
    np.testing.assert_array_equal(tab.attrs["radii_um"], genes.attrs["radii_um"])
    assert tab.attrs["bin_um"] == genes.attrs["bin_um"]


def test_celltype_scale_errors_name_the_problem(typed):
    with pytest.raises(KeyError, match="not_a_column"):
        org.celltype_scale(typed, "not_a_column")
    with pytest.raises(ValueError, match="min_cells"):
        org.celltype_scale(typed, "ctype", min_cells=10 ** 6)
    with pytest.raises(KeyError, match="nope"):
        org.celltype_scale(typed, "ctype", types=["patchy", "nope"])


# ------------------------------------------------------------------ texture on the radius axis

def _textured_section(n_cells=4000, extent=400.0, seed=0):
    """A stripe of period 60 µm and three smoothed-noise fields (Gaussian smoothing of 5, 10 and
    20 µm on a 1 µm grid, sampled at the cells), so each gene's texture has a known length."""
    import anndata as ad
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, extent, size=(n_cells, 2))
    cols = [2 + np.sin(2 * np.pi * xy[:, 0] / 60.0) + 0.3 * rng.normal(size=n_cells)]
    names = ["stripe60"]
    m = int(extent) + 1
    ij = np.clip(xy.astype(int), 0, m - 1)
    for s in (5, 10, 20):
        f = ndimage.gaussian_filter(rng.normal(size=(m, m)), s, mode="wrap")
        cols.append(f[ij[:, 1], ij[:, 0]]); names.append(f"smooth{s}")
    a = ad.AnnData(np.column_stack(cols).astype(float))
    a.var_names = names
    a.obsm["spatial"] = xy
    return a


@pytest.fixture(scope="module")
def textured():
    return _textured_section()


def test_texture_curve_is_spatial_texture_multiscale_at_the_measured_radii(textured):
    """organization's texture curve is the band power of spatial.texture_multiscale on the same
    lattice and the same radii in bins, so each value belongs to the radius it is plotted at."""
    from spectrast import spatial as spa

    b = 5.0
    _, cur = org.organization(textured, stats=("texture",), radii=[10, 20, 40, 80, 160],
                              bin_um=b, return_curves=True)
    # the lattice, built independently: per-bin mean expression, empty bins zero and off the mask
    xy = np.asarray(textured.obsm["spatial"], float)
    X = np.asarray(textured.X, float)
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / b).astype(int)
    row = (xyz[:, 1] / b).astype(int)
    H, W = row.max() + 1, col.max() + 1
    tot = np.zeros((H, W, X.shape[1]))
    cnt = np.zeros((H, W))
    np.add.at(tot, (row, col), X)
    np.add.at(cnt, (row, col), 1.0)
    mask = cnt > 0
    grid = np.where(mask[..., None], tot / np.maximum(cnt, 1.0)[..., None], 0.0)
    r_bin = np.round(cur["radii_um"] / b).astype(int)
    want = spa.texture_multiscale(grid, mask, r_bin)
    np.testing.assert_allclose(cur["texture"], want, rtol=1e-12, atol=0, equal_nan=True)


def test_texture_follows_the_radii_not_their_positions(textured):
    """Doubling every radius must not simply double the texture scale: the stripe's power stays in
    the band of its 60 µm wavelength, and a smoothed field's scale is a length of the field."""
    R = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    runs = {k: org.organization(textured, stats=("texture",), radii=rr, bin_um=5.0,
                                return_curves=True) for k, rr in (("R", R), ("2R", 2 * R))}
    for k, (_, cur) in runs.items():
        stripe = cur["texture"][:, list(cur["genes"]).index("stripe60")]
        # wavelength 60 µm is radius 30 µm, inside the band centred on 40 µm in both sweeps
        assert cur["radii_um"][np.nanargmax(stripe)] == 40.0, k
    s1 = runs["R"][0].set_index("gene").scale_um
    s2 = runs["2R"][0].set_index("gene").scale_um
    for g in ("smooth10", "smooth20"):
        assert np.isfinite(s1[g]) and np.isfinite(s2[g]), g
        assert s2[g] / s1[g] < 1.5, (g, s1[g], s2[g])
    assert s1["smooth5"] < s1["smooth10"] < s1["smooth20"]


# ------------------------------------------------------------------ sparse expression stays sparse

def test_sparse_and_dense_expression_give_identical_results(section):
    """A sparse matrix is densified one gene chunk at a time; the numbers are the dense input's."""
    import scipy.sparse as sps
    sp_ = section.copy()
    sp_.X = sps.csr_matrix(section.X)
    stats = ("moran", "geary", "texture")
    td, cd = org.organization(section, stats=stats, chunk=7, return_curves=True)
    ts, cs = org.organization(sp_, stats=stats, chunk=7, return_curves=True)
    pd.testing.assert_frame_equal(td, ts, check_exact=True)
    for s in stats:
        np.testing.assert_array_equal(cd[s], cs[s], err_msg=s)
    sub = ["struct1", "shuf2", "struct4"]
    pd.testing.assert_frame_equal(org.organization(section, stats=("moran",), genes=sub),
                                  org.organization(sp_, stats=("moran",), genes=sub),
                                  check_exact=True)
    pd.testing.assert_frame_equal(org.presence(section, stats=("moran",), n_perm=6, chunk=7),
                                  org.presence(sp_, stats=("moran",), n_perm=6, chunk=7),
                                  check_exact=True)


def test_a_sparse_matrix_is_never_densified_whole():
    """Peak memory stays far below the dense matrix: only one chunk of genes is dense at a time."""
    import tracemalloc

    import anndata as ad
    import scipy.sparse as sps

    n, G = 40_000, 1_000
    rng = np.random.default_rng(3)
    a = ad.AnnData(sps.random(n, G, density=0.01, format="csr", random_state=4, dtype=np.float64))
    a.obsm["spatial"] = rng.uniform(0, 2000, size=(n, 2))
    dense = n * G * 8                                            # 320 MB
    for name, kw in (("organization", {}), ("presence", dict(n_perm=6))):
        tracemalloc.start()
        try:
            getattr(org, name)(a, stats=("moran",), radii=[40.0, 80.0, 160.0], bin_um=20.0,
                               chunk=16, **kw)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < dense / 4, f"{name}: peak {peak / 1e6:.0f} MB against {dense / 1e6:.0f} MB dense"


def test_organization_and_presence_accept_a_gene_subset(section):
    """genes= selects columns by name (it used to index X with the (positions, names) tuple)."""
    sub = ["struct1", "shuf2"]
    full = org.organization(section, stats=("moran",))
    one = org.organization(section, stats=("moran",), genes=sub)
    assert list(one.gene) == sub
    pd.testing.assert_frame_equal(
        one.reset_index(drop=True),
        full.set_index("gene").loc[sub].reset_index()[one.columns].reset_index(drop=True))
    p = org.presence(section, stats=("moran",), genes=sub, n_perm=8)
    assert p.n_tested.iloc[0] == 2

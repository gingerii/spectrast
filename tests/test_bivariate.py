"""Bivariate mark cross-correlation: batched path, outputs layer (O1-O4), differential pair-axis.

The contract these protect is that the batched path is not a second implementation. It memoizes
transforms and shares null draws across pairs, so the only thing standing between it and silent
divergence from the looped path is that both route through one engine — asserted here bit for bit,
nulls included.
"""
import numpy as np
import anndata as ad
import pytest

import spectrast
from spectrast import bivariate as bv, diff, gr, spatial


def _fields(seed=0, H=56, W=56):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    def blob(cy, cx, s):
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s ** 2))

    mask = np.ones((H, W), bool)
    mask[:5, :] = False                                     # a non-rectangular support
    marks = [blob(20, 20, 6) + 0.1 * rng.random((H, W)),
             blob(24, 26, 6) + 0.1 * rng.random((H, W)),
             blob(42, 40, 8) + 0.1 * rng.random((H, W))]
    dens = [1.0 + rng.random((H, W)) for _ in marks]
    return marks, dens, mask


def _adata(n=1800, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 400
    X = np.abs(rng.standard_normal((n, 6))).astype("float32") * 0.4

    def bump(cx, cy, s):
        return np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2.0 * s ** 2))

    X[:, 0] += 3 * bump(180, 200, 40)                       # g0 and g1 share a territory
    X[:, 1] += 3 * bump(210, 200, 40)
    X[:, 2] += 3 * bump(340, 90, 40)                        # g2 sits elsewhere
    a = ad.AnnData(X)
    a.obsm["spatial"] = xy
    a.var_names = [f"g{i}" for i in range(6)]
    a.obs["ct"] = np.array(["A", "B"])[rng.integers(0, 2, n)]
    return a


@pytest.mark.parametrize("null,n_perm", [(None, 0), ("shift", 11), ("reshuffle", 11)])
def test_batch_equals_looped_singles(null, n_perm):
    """S5: the batched core must reproduce looped single calls EXACTLY, nulls included.

    Not merely close: both go through one engine, so any difference means the batching changed the
    statistic rather than only the bookkeeping. The null draws are generated from the seed and the
    mask alone — never from a mark — which is what lets one draw serve every pair.
    """
    marks, dens, mask = _fields()
    pairs = [(0, 1), (0, 2), (1, 2)]
    kw = dict(n_bins=8, r_max=20.0, null=null, n_perm=n_perm, seed=7)
    b = spatial.mark_cross_correlation_batch(marks, mask, pairs, densities=dens, **kw)
    assert b["g"].shape == (3, 8)
    for p, (i, j) in enumerate(pairs):
        s = spatial.mark_cross_correlation(marks[i], marks[j], mask, density_a=dens[i],
                                           density_b=dens[j], **kw)
        assert np.array_equal(b["g"][p], s["g"], equal_nan=True)
        if null:
            for k in ("z", "pval", "null_mean", "null_sd"):
                assert np.array_equal(b[k][p], s[k], equal_nan=True), k
            assert b["p_max"][p] == s["p_max"] and b["T_max"][p] == s["T_max"]


def test_gr_matrix_equals_looped_singles_and_shares_fields():
    """S5 at the AnnData layer, which has its own rasterization and normalizer-selection code."""
    a = _adata()
    pairs = [("g0", "g1"), ("g0", "g2"), ("g1", "g3")]
    R = np.linspace(20, 200, 9)
    m = gr.mark_cross_correlation_matrix(a, pairs, radii=R, null="shift", n_perm=11, seed=3)
    assert m["n_fields"] == 4, "g0 appears in two pairs and must be rasterized once"
    for p, (x, y) in enumerate(pairs):
        s = gr.mark_cross_correlation(a, x, y, radii=R, null="shift", n_perm=11, seed=3)
        for k in ("g", "z", "pval"):
            assert np.array_equal(m[k][p], s[k], equal_nan=True), k


def test_within_restricts_the_mark_not_the_tissue():
    """Confining one side to a population must not shrink the support the pair count normalizes by.

    If it did, a stratified comparison would silently renormalize each stratum against its own
    footprint and the strata would no longer be on a common scale.
    """
    a = _adata()
    R = np.linspace(20, 200, 9)
    full = gr.mark_cross_correlation(a, "g0", "g1", radii=R)
    part = gr.mark_cross_correlation(a, "g0", "g1", radii=R, b_within=("ct", "A"))
    assert np.array_equal(full["n_pairs"], part["n_pairs"])
    assert not np.allclose(full["g"], part["g"]), "b_within had no effect on the statistic"


def test_o1_recovers_the_planted_scale():
    """O1: on a pair whose fields are offset by a known distance, the peak lands at that offset."""
    H = W = 96
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.ones((H, W), bool)
    ring = np.exp(-((np.sqrt((yy - 48.0) ** 2 + (xx - 48.0) ** 2) - 24.0) ** 2) / (2 * 3.0 ** 2))
    core = np.exp(-((yy - 48.0) ** 2 + (xx - 48.0) ** 2) / (2 * 4.0 ** 2))
    res = spatial.mark_cross_correlation(core, ring, mask, n_bins=16, r_max=40.0)
    o1 = bv.interaction_distance(res)
    assert o1["sign"] == "attraction"
    assert abs(o1["peak_r_um"] - 24.0) <= 4.0, o1


def test_o2_o3_table_and_heatmap():
    """O2 ranks the planted pair first and carries an FDR; O3 renders at the table's row order."""
    import matplotlib
    matplotlib.use("Agg")
    a = _adata()
    pairs = [("g3", "g4"), ("g0", "g1"), ("g0", "g2")]
    res = gr.mark_cross_correlation_matrix(a, pairs, radii=np.linspace(20, 260, 13),
                                           null="shift", n_perm=39, seed=0)
    t = bv.pair_table(res)
    assert list(t.columns[:3]) == ["pair", "a", "b"]
    assert t.loc[0, "pair"] == "g0~g1" and t.loc[0, "sign"] == "attraction"
    assert {"T_max", "p_max", "fdr"} <= set(t.columns)
    assert t["fdr"].between(0, 1).all()
    names, r, M = bv.annulus_matrix(res)
    assert M.shape == (3, 12) and names == list(t["pair"])
    ax = bv.heatmap(res, top=2)
    assert len(ax.get_yticklabels()) == 2


def test_o4_stratified_table_covers_every_level():
    """O4: one row per (stratum, pair), with the cell count that produced it."""
    a = _adata()
    pairs = [("g0", "g1"), ("g0", "g2")]
    st = bv.stratified_pair_table(a, pairs, "ct", radii=np.linspace(20, 200, 9))
    assert set(st["stratum"]) == {"A", "B"}
    assert len(st) == 4 and (st["n_cells"] > 0).all()


def test_single_pair_result_reduces_to_one_row():
    """The outputs layer must accept a single-pair result, not only a batched one."""
    a = _adata()
    res = gr.lric(a, "g0", "g1", radii=np.linspace(20, 200, 9))
    t = bv.pair_table(res)
    assert len(t) == 1 and t.loc[0, "pair"] == "g0~g1"


def test_p_max_prices_in_the_peak_selection():
    """The reported p must be the max-over-annuli statistic, never the per-annulus p at the peak.

    Reading a p at the annulus where the curve happens to peak is optimistic — the annulus was
    chosen by looking at the data.
    """
    marks, dens, mask = _fields()
    res = spatial.mark_cross_correlation(marks[0], marks[1], mask, density_a=dens[0],
                                         density_b=dens[1], n_bins=8, r_max=20.0,
                                         null="shift", n_perm=49, seed=1)
    j = int(np.nanargmax(np.abs(res["z"])))
    assert res["p_max"] >= res["pval"][j] - 1e-12
    assert np.isclose(res["T_max"], np.abs(res["z"]).max())


def test_cross_axis_negative_control_and_detection():
    """S6: same process both arms -> no hits; a pair whose coupling RANGE moves -> detected.

    The control pair in the planted run is generated identically in both arms, so it also has to
    stay null while its neighbour fires — a single-pair test could not tell a working axis from one
    that flags everything.
    """
    def sec(seed, sep):
        rng = np.random.default_rng(seed)
        n = 1600
        xy = rng.random((n, 2)) * 400
        X = np.abs(rng.standard_normal((n, 4))).astype("float32") * 0.4

        def bump(cx, cy, s):
            return np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2.0 * s ** 2))

        X[:, 0] += 3 * bump(200 - sep / 2, 200, 40)
        X[:, 1] += 3 * bump(200 + sep / 2, 200, 40)
        X[:, 2] += 3 * bump(120, 320, 40)
        X[:, 3] += 3 * bump(300, 110, 40)
        a = ad.AnnData(X)
        a.obsm["spatial"] = xy
        a.var_names = [f"g{i}" for i in range(4)]
        return a

    P = [("g0", "g1"), ("g2", "g3")]
    kw = dict(axis="cross", pairs=P, radii=(1, 2, 4, 8), bin_size=30.0, n_perm=200, seed=0)
    A = [sec(s, 20) for s in range(1, 6)]
    neg = diff.differential(A, [sec(s, 20) for s in range(11, 16)], **kw)
    assert (neg["fdr"] >= 0.1).all(), neg[["pair", "fdr"]]
    pos = diff.differential(A, [sec(s, 160) for s in range(11, 16)], **kw)
    hit = pos.set_index("pair")
    assert hit.loc["g0~g1", "fdr"] < 0.1, pos[["pair", "T_max", "fdr"]]
    assert hit.loc["g2~g3", "fdr"] >= 0.1, "the untouched pair fired too"


def test_cross_axis_refuses_signed_expression():
    """A centred/scaled matrix has no interpretation here; refuse rather than return a number."""
    a = _adata(n=600)
    a.X = np.asarray(a.X) - 1.0
    with pytest.raises(ValueError, match="non-negative"):
        diff.differential([a, a], [a, a], axis="cross", pairs=[("g0", "g1")],
                          radii=(1, 2, 4), bin_size=40.0, n_perm=10)


def test_cross_edges_are_fixed_across_grids():
    """The annuli come from the caller's radii, so curves align across samples of different size."""
    e1 = diff._cross_edges(np.array([30.0, 60.0, 120.0]), 30.0)
    e2 = diff._cross_edges(np.array([30.0, 60.0, 120.0]), 30.0)
    assert np.array_equal(e1, e2) and len(e1) == 4 and np.all(np.diff(e1) > 0)
    with pytest.raises(ValueError, match="increasing"):
        diff._cross_edges(np.array([60.0, 30.0]), 30.0)


def test_array_mark_is_a_first_class_field():
    """A derived per-cell field must be usable as a mark without being written to obs first."""
    a = _adata()
    R = np.linspace(20, 200, 9)
    v = np.asarray(a[:, "g1"].X).ravel()
    direct = gr.mark_cross_correlation(a, "g0", "g1", radii=R)
    viaarr = gr.mark_cross_correlation(a, "g0", v, b_type="array", radii=R)
    assert np.allclose(direct["g"], viaarr["g"], equal_nan=True)
    with pytest.raises(ValueError, match="one value per cell"):
        gr.mark_cross_correlation(a, "g0", v[:10], b_type="array", radii=R)


def test_array_marks_sharing_a_repr_stay_separate_fields():
    """Two distinct per-cell fields must never be collapsed into one registry entry.

    ``repr`` of a numpy array abbreviates the middle, so fields agreeing on their first and last
    three values stringify identically. Keying the field registry on that substituted one field for
    the other, which silently turned a cross-pair into a self-pair — the collapsed pair reported the
    autocorrelation and topped its mark. Observed on 4 of 5 tissues in the mark matrix.
    """
    a = _adata()
    n = a.n_obs
    u, v = np.zeros(n), np.zeros(n)                          # identical head and tail, different middle
    u[3:-3] = np.asarray(a[:, "g0"].X).ravel()[3:-3]
    v[3:-3] = np.asarray(a[:, "g2"].X).ravel()[3:-3]
    assert repr(u) == repr(v), "fixture must reproduce the repr collision"
    assert not np.array_equal(u, v)

    R = np.linspace(20, 200, 9)
    res = gr.mark_cross_correlation_matrix(a, [(u, v), (u, u)], a_type="array", b_type="array",
                                           radii=R)
    assert res["n_fields"] == 2, "u and v collapsed into one field"
    cross, self_ = np.asarray(res["g"])[0], np.asarray(res["g"])[1]
    assert not np.allclose(cross, self_, equal_nan=True), "cross-pair became a self-pair"
    # and the label must not be an abbreviated array dump
    assert res["pair_names"][0] == f"array[{n}]~array[{n}]"


def test_fingerprint_cross_runs_on_organization_not_expression():
    """#8: the marks are local-Moran fields, so the result must differ from the expression cross."""
    a = _adata()
    R = np.linspace(20, 200, 9)
    fc = gr.fingerprint_cross(a, "g0", "g1", axis="local_moran", radii=R, null="shift",
                              n_perm=19, seed=0)
    assert fc["fingerprint_axis"] == "local_moran" and fc["a"] == "g0"
    assert not np.allclose(fc["g"], gr.lric(a, "g0", "g1", radii=R)["g"], equal_nan=True)
    with pytest.raises(ValueError, match="axis must be"):
        gr.fingerprint_cross(a, "g0", "g1", axis="focality")


def test_regulon_cross_uses_the_target_mean():
    a = _adata()
    R = np.linspace(20, 200, 9)
    res = gr.regulon_cross(a, "g0", ["g1", "g2"], radii=R)
    assert res["targets"] == ["g1", "g2"] and res["b"] == "regulon[2]"
    v = np.asarray(a[:, ["g1", "g2"]].X).mean(axis=1)
    assert np.allclose(res["g"], gr.mark_cross_correlation(a, "g0", v, b_type="array",
                                                           radii=R)["g"], equal_nan=True)


@pytest.mark.parametrize("normalize", ["density", "csr", "coherence"])
def test_first_order_map_sums_to_the_curve_numerator(normalize):
    """S5c: the map must be a DECOMPOSITION of the curve, not a lookalike computed another way.

    If its sum did not reproduce the numerator, the map would be a second heuristic that merely
    resembles the statistic, and pairing them would be decorative. Holding the identity to
    floating-point is what lets a reader trust that the highlighted regions are the ones that
    produced the reported number.
    """
    marks, dens, mask = _fields()
    edges = np.linspace(1.0, 21.0, 11)
    fa, na = spatial._mcc_fields(marks[0], mask, dens[0], name="a")
    fb, nb = spatial._mcc_fields(marks[1], mask, dens[1], name="b")
    from scipy.fft import next_fast_len
    H, W = mask.shape
    lag = int(np.ceil(edges[-1])) + 1
    bn = spatial._mcc_binner(edges, 1.0, next_fast_len(H + lag), next_fast_len(W + lag), False)
    XA, XB = {"density": (na * fa, nb * fb), "csr": (na, nb),
              "coherence": (fa, fb)}[normalize]
    num = spatial._mcc_radial(XA, XB, bn)
    for i in (0, 3, 7):
        out = spatial.mark_cross_map(marks[0], marks[1], mask, r_lo=edges[i], r_hi=edges[i + 1],
                                     density_a=dens[0], density_b=dens[1], normalize=normalize)
        assert out["numerator"] == pytest.approx(num[i], rel=1e-10), (normalize, i)


def test_first_order_map_localizes_the_coupling():
    """S5c acceptance: the map must light up where BOTH marks are high, not where either alone is.

    A map that merely tracked one mark's abundance would pass a sum check and still be useless.
    """
    rng = np.random.default_rng(0)
    n = 6000
    xy = rng.random((n, 2)) * 600
    X = np.abs(rng.standard_normal((n, 4))).astype("float32") * 0.3

    def bump(cx, cy, s):
        return np.exp(-((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) / (2.0 * s ** 2))

    X[:, 0] += 4 * bump(120, 120, 55) + 4 * bump(480, 120, 55)      # coupled region + g0 alone
    X[:, 1] += 4 * bump(120, 120, 55) + 4 * bump(120, 480, 55)      # coupled region + g1 alone
    a = ad.AnnData(X)
    a.obsm["spatial"] = xy
    a.var_names = [f"g{i}" for i in range(4)]

    res = gr.mark_cross_correlation_matrix(a, [("g0", "g1")], radii=np.geomspace(20, 300, 10),
                                           bin_um=15.0)
    edges = np.asarray(res["edges"])
    m = gr.mark_cross_map(a, ("g0", "g1"), r_lo=edges[0], r_hi=edges[2], bin_um=15.0)
    pc = m["per_cell"][:, 0]

    def region(cx, cy):
        return (np.abs(xy[:, 0] - cx) < 110) & (np.abs(xy[:, 1] - cy) < 110)

    both, only_a, only_b = region(120, 120), region(480, 120), region(120, 480)
    share = lambda s: pc[s].sum() / pc.sum()                                     # noqa: E731
    assert share(both) > 0.45, share(both)
    assert share(both) > 3 * max(share(only_a), share(only_b))
    assert a.obsm["spectrast_mark_cross_map"].shape == (n, 1)
    assert a.uns["spectrast_mark_cross_map_pairs"] == ["g0~g1"]


def test_first_order_map_helper_uses_the_pairs_own_peak():
    """bivariate.first_order_map must explain the annulus the table reports, not another one."""
    a = _adata()
    R = np.linspace(20, 200, 9)
    res = gr.mark_cross_correlation_matrix(a, [("g0", "g1"), ("g0", "g2")], radii=R, bin_um=20.0)
    tab = bv.pair_table(res)
    m = bv.first_order_map(a, res, tab.loc[0, "pair"], bin_um=20.0)
    assert m["peak_r_um"] == pytest.approx(tab.loc[0, "peak_r_um"])
    assert m["r_lo"] < m["peak_r_um"] <= m["r_hi"]
    with pytest.raises(ValueError, match="r_lo and r_hi are required"):
        gr.mark_cross_map(a, ("g0", "g1"), bin_um=20.0)


def _hotspot_setup():
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.ones((H, W), bool)
    edges = np.linspace(2.0, 26.0, 9)
    rng = np.random.default_rng(0)
    terr = np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 45.0 ** 2))
    mA = 0.3 + 2.0 * terr + 0.05 * rng.random((H, W))
    mB = 0.3 + 2.0 * terr + 0.05 * rng.random((H, W))
    nest = 1.0 + 40.0 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 14.0 ** 2))
    return mask, edges, mA, mB, np.ones((H, W)), nest, yy, xx


def test_density_robust_is_exactly_a_noop_on_homogeneous_density():
    """S5d: with uniform density the correction divides by a constant and must cancel exactly.

    Anything else would mean the knob quietly changes results on tissue it has no business
    touching, which is the one behaviour a robustness check cannot have.
    """
    mask, edges, mA, mB, flat, _, _, _ = _hotspot_setup()
    kw = dict(density_a=flat, density_b=flat, annuli=edges)
    d = spatial.mark_cross_correlation(mA, mB, mask, normalize="density", **kw)["g"]
    r = spatial.mark_cross_correlation(mA, mB, mask, normalize="density_robust", **kw)["g"]
    assert np.array_equal(d, r, equal_nan=True)


def test_density_robust_discounts_a_density_hotspot():
    """The artefact it exists to catch: a dense nest shifting a curve whose marks did not change."""
    mask, edges, mA, mB, flat, nest, _, _ = _hotspot_setup()

    def run(n, norm):
        return spatial.mark_cross_correlation(mA, mB, mask, density_a=n, density_b=n,
                                              normalize=norm, annuli=edges)["g"]

    truth = run(flat, "density")
    assert np.nanmean(np.abs(run(nest, "density_robust") - truth)) < \
        0.5 * np.nanmean(np.abs(run(nest, "density") - truth))


def test_density_robust_attenuates_real_co_located_coupling():
    """The PRICE, asserted so it cannot drift silently.

    Coupling that genuinely lives in a dense region is attenuated too — the method cannot tell it
    from an artefact, because geometrically they are the same signal. This is why a result that
    FAILS the check is ambiguous rather than refuted, and the docstring says so.
    """
    mask, edges, _, _, _, nest, yy, xx = _hotspot_setup()
    rng = np.random.default_rng(1)
    inside = np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 14.0 ** 2))
    cA = 0.3 + 2.0 * inside + 0.05 * rng.random(mask.shape)
    cB = 0.3 + 2.0 * inside + 0.05 * rng.random(mask.shape)

    def run(norm):
        return spatial.mark_cross_correlation(cA, cB, mask, density_a=nest, density_b=nest,
                                              normalize=norm, annuli=edges)["g"]

    base, rob = run("density"), run("density_robust")
    kept = (rob[0] - 1.0) / (base[0] - 1.0)
    assert 0.4 < kept < 0.95, f"retention {kept:.2%} moved; the documented tradeoff has changed"


def test_directed_pair_table_recovers_a_planted_direction():
    """O4 directed: confining BOTH sides must recover sender -> receiver, not just co-location.

    This is the one place LIANA was genuinely stronger (its ``groupby=`` is directional while our
    ``a_within``/``b_within`` was used one-sided). The planted signal runs stroma -> immune only, so
    the reverse direction has to stay flat; a symmetric statistic would report both.
    """
    rng = np.random.default_rng(0)
    n = 6000
    xy = rng.random((n, 2)) * 500
    X = np.abs(rng.standard_normal((n, 4))).astype("float32") * 0.3
    ct = np.array(["tumour", "stroma", "immune"])[rng.integers(0, 3, n)]
    hub = np.exp(-((xy[:, 0] - 150) ** 2 + (xy[:, 1] - 150) ** 2) / (2 * 60.0 ** 2))
    X[:, 0] += 4 * hub * (ct == "stroma")          # ligand on stroma
    X[:, 1] += 4 * hub * (ct == "immune")          # receptor on immune
    a = ad.AnnData(X)
    a.obsm["spatial"] = xy
    a.var_names = [f"g{i}" for i in range(4)]
    a.obs["ct"] = ct

    t = bv.directed_pair_table(a, [("g0", "g1")], "ct", radii=np.linspace(20, 200, 9),
                               bin_um=20.0, min_cells=100)
    assert len(t) == 9 and set(t["sender"]) == {"tumour", "stroma", "immune"}
    k = t.set_index(["sender", "receiver"])["peak_k"]
    assert k[("stroma", "immune")] > 2.0, k
    assert k[("immune", "stroma")] < 1.2, "the reverse direction fired; the statistic is not directional"
    senders, receivers, M = bv.directed_matrix(t, "g0~g1")
    assert M.shape == (3, 3)
    assert M[senders.index("stroma"), receivers.index("immune")] == pytest.approx(
        k[("stroma", "immune")])


def test_batch_rejects_bad_input():
    marks, dens, mask = _fields()
    with pytest.raises(IndexError):
        spatial.mark_cross_correlation_batch(marks, mask, [(0, 9)], n_bins=4, r_max=12.0)
    with pytest.raises(ValueError, match="empty"):
        spatial.mark_cross_correlation_batch(marks, mask, [], n_bins=4, r_max=12.0)
    with pytest.raises(ValueError, match="negative"):
        spatial.mark_cross_correlation_batch([marks[0], -marks[1], marks[2]], mask, [(0, 1)],
                                             n_bins=4, r_max=12.0)


def test_interaction_distance_flags_a_peak_on_the_innermost_annulus():
    """A range is only a range when the peak is interior; otherwise it is where the sweep stopped.

    Measured motivation (2026-09-29): on six autotuned sections,
    38-67 % of the 24 ligand-receptor pairs peak ON the innermost annulus and the distance column
    takes 2-8 distinct values over 24 pairs. Without this flag a censored pair is indistinguishable
    from a measured one and returns a plausible number rather than a NaN — the same silent-failure
    shape the differential's one-armed Δscale had, but worse for being plausible.
    """
    import numpy as np
    from spectrast import bivariate

    r = np.array([10.0, 20.0, 40.0, 80.0, 160.0])

    def res(k):
        return dict(a="A", b="B", r=r, g=np.asarray(k, float))

    # monotone decreasing from the first annulus: still rising where resolution ran out
    at_floor = bivariate.interaction_distance(res([2.0, 1.5, 1.2, 1.05, 1.0]))
    assert at_floor["peak_at_floor"] is True
    assert at_floor["range_resolved"] is False
    assert at_floor["peak_index"] == 0
    assert at_floor["peak_r_um"] == 10.0            # still reported — but now labelled

    # a genuine interior peak
    interior = bivariate.interaction_distance(res([1.0, 1.2, 2.0, 1.2, 1.0]))
    assert interior["peak_at_floor"] is False
    assert interior["peak_at_ceiling"] is False
    assert interior["range_resolved"] is True
    assert interior["peak_r_um"] == 40.0

    # peak on the outermost annulus: the sweep stopped before the curve turned over
    at_ceiling = bivariate.interaction_distance(res([1.0, 1.05, 1.2, 1.5, 2.0]))
    assert at_ceiling["peak_at_ceiling"] is True
    assert at_ceiling["range_resolved"] is False


def test_pair_table_warns_when_the_distance_column_is_mostly_censored():
    """The warning has to name the confound, not just the count: a BETWEEN-GROUP distance
    comparison is invalid when the groups are censored at different rates."""
    import warnings as _w

    import numpy as np
    import pytest

    from spectrast import bivariate

    r = np.array([10.0, 20.0, 40.0, 80.0])
    censored = [3.0, 1.4, 1.1, 1.0]          # peaks at the floor
    interior = [1.0, 1.1, 2.0, 1.1]          # peaks inside
    g = np.array([censored, censored, censored, interior], float)
    res = dict(pair_names=["A~B", "C~D", "E~F", "G~H"],
               pairs=[("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")], r=r, g=g)

    with pytest.warns(UserWarning, match="INNERMOST"):
        df = bivariate.pair_table(res)
    assert df["peak_at_floor"].sum() == 3
    assert int(df["range_resolved"].sum()) == 1

    # and it stays quiet when censoring is rare
    g2 = np.array([interior, interior, interior, interior], float)
    res2 = dict(pair_names=["A~B", "C~D", "E~F", "G~H"],
                pairs=[("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")], r=r, g=g2)
    with _w.catch_warnings():
        _w.simplefilter("error")
        bivariate.pair_table(res2)


def test_mcc_score_never_gives_a_floor_p_to_an_undefined_observed_curve():
    """The same false positive as the differential's (2026-10-13): a NaN observed value compares as less
    extreme than every draw, so its p fell to 1/(n_valid + 1). An observed curve with no finite annulus now
    gets p_max = 1, and a NaN annulus gets p = 1; finite values keep their p exactly."""
    rng = np.random.default_rng(0)
    draws = rng.random((20, 2, 4))                        # 20 valid draws, 2 pairs, 4 annuli
    g = np.array([[np.nan] * 4, [0.5, np.nan, 0.2, 0.9]])
    out = spatial._mcc_score(g, draws, [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(out["T_max"][0]) and out["p_max"][0] == 1.0
    assert (out["pval"][0] == 1.0).all() and out["pval"][1, 1] == 1.0
    mu = draws.mean(0)
    ge = (np.abs(draws - mu) >= np.abs(g - mu)).sum(0)
    fin = np.isfinite(g)
    np.testing.assert_array_equal(out["pval"][fin], ((ge + 1.0) / 21.0)[fin])
    assert np.isfinite(out["T_max"][1]) and out["p_max"][1] < 1.0


def _accepted_shifts(mask, n_shift, seed):
    """Replay spatial_association's shift draws and count those it accepts (overlap above 10 bins)."""
    H, W = mask.shape
    rng = np.random.default_rng(seed)
    n_ok = 0
    for _ in range(n_shift):
        dy, dx = int(rng.integers(1, H)), int(rng.integers(1, W))
        n_ok += int((np.roll(np.roll(mask, dy, 0), dx, 1) & mask).sum() > 10)
    return n_ok


def test_spatial_association_p_counts_only_accepted_shifts():
    """A rejected shift (overlap of 10 bins or fewer) is not a draw from the null, so it must not enter
    the denominator. With y = x every accepted shift is less extreme than the observed correlation, so
    p = 1 / (n_accepted + 1) exactly; counting the rejected shifts gave 1 / (n_shift + 1)."""
    rng = np.random.default_rng(0)
    mask = np.zeros((40, 40), bool); mask[2:20, 3:21] = True
    x = rng.standard_normal((40, 40))
    n_ok = _accepted_shifts(mask, 39, seed=0)
    assert 20 <= n_ok < 39                         # some shifts rejected, enough kept for a test
    r = spatial.spatial_association(x, x, mask, n_modes=4, n_shift=39, seed=0)
    assert r["p_shift"] == 1.0 / (n_ok + 1)


def test_spatial_association_p_is_nan_when_too_few_shifts_are_accepted():
    """When fewer than half of the shifts are accepted, the shift null has too few draws to be a test:
    p is NaN, with a warning, instead of a p whose denominator counted the rejected shifts."""
    rng = np.random.default_rng(7)
    y, x = rng.standard_normal((2, 40, 40))
    island = np.zeros((40, 40), bool); island[10:13, 10:13] = True    # 9 bins: every shift rejected
    small = np.zeros((40, 40), bool); small[2:16, 3:17] = True
    n_ok = _accepted_shifts(small, 39, seed=4)
    assert 0 < n_ok < 39 / 2
    for mask, seed in ((island, 0), (small, 4)):
        with pytest.warns(RuntimeWarning, match="spatial_association"):
            r = spatial.spatial_association(y, x, mask, n_modes=2, n_shift=39, seed=seed)
        assert np.isnan(r["p_shift"]) and np.isfinite(r["r_resid"])

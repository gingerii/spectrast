"""The differential over MARK SPECS — bivariate co-organization and first-order level per mark.

The brief's gate is that extending `axis="cross"` beyond gene-gene must not perturb gene-gene, so
that is the first test here and it demands BIT-identity, not closeness. The rest establish that a
derived mark behaves like the expression it was derived from, that the first-order axis is a real
test rather than a relabelled one, and that both stay calibrated on a same-condition split.
"""
import numpy as np
import anndata as ad
import pytest

from spectrast import diff


def _sec(seed, sep=20, n=1600, extent=400.0, shift=0.0):
    """One synthetic section: two bumps whose SEPARATION is the planted signal, plus a control pair."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * extent
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
    a.obs["cov"] = np.asarray(X[:, 0], float) + shift            # a covariate that tracks g0
    a.obs["ct"] = np.array(["A", "B"])[(xy[:, 0] > extent / 2).astype(int)]
    return a


# --------------------------------------------------------------------- the gate
@pytest.mark.parametrize("axis,kw", [
    ("cross", dict(pairs=[("g0", "g1"), ("g2", "g3")])),
    ("coloc", dict(pairs=[("g0", "g1"), ("g2", "g3")], radii=(2, 4, 8))),
    ("moran", dict(features=["g0", "g1", "g2"])),
])
def test_gene_gene_is_bit_identical_after_the_mark_extension(axis, kw):
    """Gene-name inputs must route through the untouched path — 0.0e+00, not 'close'.

    Guaranteed by construction (`_prep_marks` delegates to `_prep` when every spec is a string), and
    asserted here so a later refactor cannot quietly move gene-gene onto the generic resolver.
    """
    A = [_sec(s) for s in range(1, 4)]
    B = [_sec(s, sep=120) for s in range(11, 14)]
    base = dict(radii=(1, 2, 4, 8), bin_size=30.0, n_perm=50, seed=0)
    base.update(kw)
    r1 = diff.differential(A, B, axis=axis, **base)
    r2 = diff.differential(A, B, axis=axis, **base)
    for col in ("T_max", "p", "fdr", "dI_at_max"):
        assert np.array_equal(np.asarray(r1[col]), np.asarray(r2[col])), f"{axis}/{col} not identical"
    # and the resolver reproduces _prep exactly for string specs
    xy0, X0 = diff._prep(A[0], [0, 1], None, 0)
    xy1, X1 = diff._prep_marks(A[0], ["g0", "g1"], None, 0, None)
    assert np.array_equal(xy0, xy1)
    assert np.abs((X0 - X1)).max() == 0.0


def test_a_mark_spec_that_names_a_gene_equals_the_gene_itself():
    """A callable returning g0's expression must give the SAME curve as passing "g0".

    This is what makes derived marks trustworthy: the only difference between a gene and a mark is
    where the per-cell vector came from, so an identity mark has to be a no-op.
    """
    def g0_field(a):
        return np.asarray(a.X[:, 0]).ravel()
    g0_field.mark_name = "g0_as_field"

    A = [_sec(s) for s in range(1, 4)]
    B = [_sec(s, sep=120) for s in range(11, 14)]
    kw = dict(axis="cross", radii=(1, 2, 4, 8), bin_size=30.0, n_perm=50, seed=0)
    byname = diff.differential(A, B, pairs=[("g0", "g1")], **kw)
    byspec = diff.differential(A, B, pairs=[(g0_field, "g1")], **kw)
    assert byspec["pair"].iloc[0] == "g0_as_field~g1"
    assert np.allclose(byname["T_max"], byspec["T_max"], rtol=0, atol=1e-12)
    assert np.allclose(byname["dI_at_max"], byspec["dI_at_max"], rtol=0, atol=1e-12)


def test_mark_forms_resolve_to_the_right_vectors():
    """Each supported spec form returns one value per cell, matching a hand-computed reference."""
    a = _sec(1)
    X = np.asarray(a.X, float)
    assert np.allclose(diff._resolve_mark(a, "g0"), X[:, 0])
    assert np.allclose(diff._resolve_mark(a, ("obs", "cov")), np.asarray(a.obs["cov"], float))
    assert np.allclose(diff._resolve_mark(a, ("mean", ["g0", "g1"])), X[:, :2].mean(1))
    fp = diff._resolve_mark(a, ("fingerprint", "g0", "getis"))
    assert fp.shape == (a.n_obs,) and fp.min() >= 0.0 and fp.max() > 0.0   # clipped organization field
    assert diff._mark_label(("mean", ["g0", "g1"], "prog")) == "prog"
    with pytest.raises(ValueError, match="unknown mark spec"):
        diff._resolve_mark(a, ("nope", "x"))
    with pytest.raises(ValueError, match="one value per cell|returned"):
        diff._resolve_mark(a, lambda _a: np.zeros(5))


def test_derived_marks_run_on_the_cross_axis_and_stay_calibrated():
    """A same-condition split must give no hits for a pair built from derived marks (the NEG gate)."""
    prog = ("mean", ["g0", "g1"], "prog01")
    cov = ("obs", "cov")
    pairs = [(prog, cov), (prog, "g2")]
    kw = dict(axis="cross", pairs=pairs, radii=(1, 2, 4, 8), bin_size=30.0, n_perm=200, seed=0)
    A = [_sec(s) for s in range(1, 6)]
    B = [_sec(s) for s in range(11, 16)]                 # SAME process — nothing to find
    neg = diff.differential(A, B, **kw)
    assert list(neg["pair"]) == ["prog01~cov", "prog01~g2"]
    assert (neg["fdr"] >= 0.1).all(), neg[["pair", "T_max", "fdr"]]


def test_first_order_mean_axis_detects_a_level_shift_and_ignores_a_pure_range_change():
    """The first-order axis is a real, separate test — it fires on level and not on geometry.

    Planting the two separately is the point: a first-order statistic that moved whenever the
    second-order one did would be redundant, and one that never moved would be inert.
    """
    kw = dict(axis="mean", features=[("obs", "cov")], radii=(1,), bin_size=30.0,
              n_perm=200, seed=0, structured_min=0.0)
    A = [_sec(s) for s in range(1, 6)]
    lvl = diff.differential(A, [_sec(s, shift=2.0) for s in range(11, 16)], **kw)
    assert lvl["fdr"].iloc[0] < 0.1, lvl                      # level moved -> detected
    rng_only = diff.differential(A, [_sec(s, sep=160) for s in range(11, 16)], **kw)
    assert rng_only["fdr"].iloc[0] >= 0.1, rng_only           # geometry moved, level did not
    same = diff.differential(A, [_sec(s) for s in range(11, 16)], **kw)
    assert same["fdr"].iloc[0] >= 0.1, same                   # NEG control


def test_mean_axis_carries_no_scale_information():
    """`mean` is constant across radii by construction, so its 'scale of change' is meaningless."""
    a = _sec(1)
    xy, Xg = diff._prep_marks(a, ["g0", "g1"], None, 0, None)
    m = diff._mean_stat(xy, Xg, (1, 2, 4, 8), 30.0)
    assert m.shape == (4, 2)
    assert np.array_equal(m[0], m[-1])


def test_layer_is_never_silently_ignored():
    """`_prep` reads ad.X unconditionally, so delegating to it with a layer set would return the
    WRONG matrix — a silent wrong answer. With a layer, every spec must go through the resolver."""
    a = _sec(1)
    a.layers["alt"] = np.asarray(a.X) * 3.0
    _, X_default = diff._prep_marks(a, ["g0", "g1"], None, 0, None)
    _, X_layer = diff._prep_marks(a, ["g0", "g1"], None, 0, "alt")
    assert np.allclose(np.asarray(X_layer.todense()), 3.0 * np.asarray(X_default.todense()))


def test_depth_match_is_refused_for_derived_marks():
    """Thinning counts has no meaning for a program score; refuse rather than silently skip it."""
    a = _sec(1)
    with pytest.raises(ValueError, match="no meaning for derived marks"):
        diff._prep_marks(a, [("obs", "cov")], depth_to=100.0, seed=0, layer=None)

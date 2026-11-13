"""The reductions and the schedule behind spatial.multiscale_statistics must not move a single bit.

Three things are held to exact (uint64-view) equality here:

* the numpy summation-order fact that `_scatter_sum`'s fast path rests on -- for a C-ordered (n, G) float64
  array with G >= 2, `t.sum(0)` adds the rows one after another, so it equals the sum of the full (H, W, G) grid
  the rows were gathered from (zeros elsewhere) to the bit. It is numpy behaviour, not ours, so it is checked
  on whatever numpy is installed;
* `_gini_occupied` against the 2026-10-12 reference code it replaced (copied below verbatim);
* the grouped schedule of `multiscale_statistics` against the frozen pre-refactor outputs in
  tests/data/multiscale_reference.npz (the same fixture test_multiscale_equivalence.py uses).
"""
import os

import numpy as np
import pytest

from spectrast import _fft
from spectrast import spatial as sp

REF = os.path.join(os.path.dirname(__file__), "data", "multiscale_reference.npz")
SINGLE = {"moran": sp.morans_i_multiscale, "geary": sp.gearys_c_multiscale,
          "general_g": sp.general_g_multiscale, "losh": sp.losh_multiscale,
          "focality": sp.focality_scalogram, "texture": sp.texture_multiscale}


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _same_bits(a, b):
    return np.shape(a) == np.shape(b) and np.array_equal(_bits(a), _bits(b))


@pytest.fixture(autouse=True)
def cpu_backend():
    prev, prev_w = _fft.get_backend(), _fft.get_workers()
    _fft.set_backend("cpu")
    yield
    _fft.set_backend(prev)
    _fft.set_workers(prev_w)


def _masks(rng, H, W):
    yy, xx = np.mgrid[:H, :W]
    yield rng.random((H, W)) < 0.05
    yield rng.random((H, W)) < 0.6
    yield (yy - H / 2) ** 2 + (xx - W / 3) ** 2 < (min(H, W) / 3) ** 2
    yield (xx % 5) < 2
    yield np.ones((H, W), bool)
    yield np.zeros((H, W), bool)
    m = np.zeros((H, W), bool)
    m[H // 2, W // 2] = True
    yield m


def _contents(rng, n, G):
    yield rng.standard_normal((n, G))
    yield rng.lognormal(0, 3, (n, G))                                         # heavy tail, positive
    yield rng.standard_normal((n, G)) * 10.0 ** rng.integers(-8, 9, (n, G))   # 16 decades
    x = rng.standard_normal((n, G))
    x[1::2] = -x[0::2][: x[1::2].shape[0]]                                     # exact cancellation
    yield x
    x = rng.standard_normal((n, G))
    x[rng.random((n, G)) < 0.3] = -0.0                                         # signed zeros
    yield x
    x = rng.standard_normal((n, G))
    x[rng.random((n, G)) < 0.05] = np.nan
    x[rng.random((n, G)) < 0.05] = np.inf
    yield x


# ---------------------------------------------------------------------------------------------------------
# 1. the summation-order fact behind _scatter_sum's row route
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("G", [2, 3, 4, 5, 7, 8, 13, 16, 31, 32, 33, 64])
def test_occupied_rows_sum_like_the_full_grid_for_two_or_more_genes(G):
    """numpy adds a multi-column array one row at a time, so the occupied rows alone reproduce the full-grid
    sum bit for bit (the zero rows in between change nothing). If a numpy release ever reorders this
    reduction, this test is the one that says so -- and _scatter_sum must then drop its row route."""
    rng = np.random.default_rng(1000 + G)
    with np.errstate(all="ignore"):
        for H, W in [(1, 1), (3, 2), (7, 9), (60, 80), (257, 129)]:
            for mb in _masks(rng, H, W):
                n = int(mb.sum())
                for term in _contents(rng, n, G):
                    full = np.zeros((H, W, G))
                    full[mb] = term
                    ref = full.sum((0, 1))
                    assert _same_bits(term.sum(0), ref), (G, H, W, n)
                    assert _same_bits(sp._scatter_sum(term, mb, (H, W, G)), ref), (G, H, W, n)


def test_scatter_sum_on_a_real_size_grid():
    """A grid the size of the largest real section's bounding box at 5 % occupancy."""
    rng = np.random.default_rng(3)
    H, W, G = 1378, 1488, 32
    mb = rng.random((H, W)) < 0.05
    term = rng.standard_normal((int(mb.sum()), G))
    full = np.zeros((H, W, G))
    full[mb] = term
    assert _same_bits(sp._scatter_sum(term, mb, (H, W, G)), full.sum((0, 1)))


def test_scatter_sum_single_gene_and_odd_terms_take_the_scatter_and_reuse_one_zero_grid():
    """G == 1 (numpy sums that pairwise), non-contiguous and non-float64 terms go through the zero grid; one
    reused grid gives what a fresh np.zeros gives, call after call, because every call rewrites every
    occupied bin and touches nothing else."""
    rng = np.random.default_rng(5)
    H, W = 90, 70
    mb = rng.random((H, W)) < 0.3
    n = int(mb.sum())
    for G, make in [(1, lambda: rng.standard_normal((n, 1))),
                    (6, lambda: np.asfortranarray(rng.standard_normal((n, 6)))),
                    (6, lambda: rng.standard_normal((n, 6)).astype(np.float32))]:
        work = sp._MsWork(n)
        for _ in range(5):
            term = make()
            full = np.zeros((H, W, G))
            full[mb] = term
            assert _same_bits(sp._scatter_sum(term, mb, (H, W, G), work), full.sum((0, 1)))
        assert work._zeros is not None                      # the scatter path really ran, on the shared grid
        work.close()


# ---------------------------------------------------------------------------------------------------------
# 2. the Gini reduction against the code it replaced
# ---------------------------------------------------------------------------------------------------------
def _gini_reference(v, wt, nocc):
    """spatial._gini_occupied as of 195829d (with its threaded sort, which is np.sort per column)."""
    pos = np.clip(np.where(np.isfinite(v), v, 0.0), 0.0, None)
    pos = np.sort(pos, axis=0)
    s = pos.sum(0)
    return np.nan_to_num((wt * pos).sum(0) / np.where(s > 0, nocc * s, np.nan), nan=0.0)


def _gini_inputs(rng, n, G):
    yield rng.standard_normal((n, G)) * rng.random((n, G))
    x = rng.standard_normal((n, G))
    x[rng.random((n, G)) < 0.05] = np.nan
    yield x
    x = rng.standard_normal((n, G))
    x[rng.random((n, G)) < 0.03] = np.inf
    x[rng.random((n, G)) < 0.03] = -np.inf
    yield x
    x = rng.standard_normal((n, G))
    x[rng.random((n, G)) < 0.3] = -0.0
    yield x
    yield np.abs(rng.lognormal(0, 3, (n, G)))
    yield np.zeros((n, G))
    yield np.full((n, G), np.nan)
    x = np.round(rng.standard_normal((n, G)) * 3) / 4                        # many exact ties
    x[:, 0] = np.nan
    yield x


@pytest.mark.parametrize("G", [1, 2, 3, 7, 8, 9, 12, 17, 32])
def test_gini_matches_the_replaced_code_for_any_thread_count(G):
    rng = np.random.default_rng(G)
    with np.errstate(all="ignore"):
        for n in (1, 2, 5, 300, 4097):
            idx = np.arange(1, n + 1)
            wt = (2 * idx - n - 1)[:, None]
            for v in _gini_inputs(rng, n, G):
                ref = _gini_reference(v.copy(), wt, n)
                for w in (1, 3, 4, 8):
                    _fft.set_workers(w)
                    for work in (None, sp._MsWork(n)):
                        v_in = v.copy()
                        got = sp._gini_occupied(v_in, wt, n, work)
                        assert _same_bits(got, ref), (G, n, w)
                        assert np.array_equal(v_in, v, equal_nan=True)     # the input is left alone
                        if work is not None:
                            got2 = sp._gini_occupied(v.copy(), wt, n, work)  # reused buffers and pool
                            assert _same_bits(got2, ref)
                            work.close()


def test_sort_blocks_never_hold_a_single_column():
    """A one-column block would be summed pairwise; the block plan must keep every block >= 2 wide."""
    prev = _fft.get_workers()
    try:
        for w in (2, 3, 4, 8, 16, 64):
            _fft.set_workers(w)
            for G in range(sp._SORT_MIN_COLS, 70):
                nt = sp._sort_threads(G)
                edges = [k * G // nt for k in range(nt + 1)]
                assert min(np.diff(edges)) >= 2, (w, G)
    finally:
        _fft.set_workers(prev)


# ---------------------------------------------------------------------------------------------------------
# 3. the schedule: one pass or the sharing groups in turn, same bits either way
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    with np.load(REF) as z:
        d = {k: z[k] for k in z.files}
    names = sorted({k.split("/")[0] for k in d if "/" in k})
    return d, names


def _case(d, name):
    return (d[f"{name}/grid"], d[f"{name}/mask"], d[f"{name}/radii"].tolist(), str(d[f"{name}/kind"]))


def test_sharing_groups():
    full = sp.MULTISCALE_STATS
    assert sp._ms_passes(full, True) == [("moran", "focality"), ("general_g", "losh"), ("geary",), ("texture",)]
    assert sp._ms_passes(full, False) == [("moran", "focality"), ("general_g", "losh", "geary"), ("texture",)]
    assert sp._ms_passes(("geary", "moran"), False) == [("moran",), ("geary",)]
    assert sp._ms_passes(("losh",), True) == [("losh",)]
    assert sp._ms_passes((), True) == []


def test_schedule_follows_the_pass_size():
    """The two real sections the threshold was set between: 0.4 M bins stays one pass, 2.05 M bins splits."""
    r = [2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19, 23, 29, 35, 43, 53, 65, 80]
    five = ("moran", "geary", "general_g", "losh", "focality")
    assert sp._ms_pass_bytes((347, 1168, 32), r, five, True) <= sp._FUSE_MAX_BYTES
    assert sp._ms_pass_bytes((1378, 1488, 32), r, five, True) > sp._FUSE_MAX_BYTES
    # a single group's own pass is what each split pass costs -- never more than the fused one
    for grp in sp._ms_passes(five, True):
        assert sp._ms_pass_bytes((1378, 1488, 32), r, grp, True) < sp._ms_pass_bytes((1378, 1488, 32), r, five, True)


@pytest.mark.parametrize("limit", [0, float("inf")])
def test_both_schedules_reproduce_the_frozen_outputs(ref, monkeypatch, limit):
    """limit 0 forces the sharing groups in turn, inf the single pass; both must give the old code's bits."""
    monkeypatch.setattr(sp, "_FUSE_MAX_BYTES", limit)
    d, names = ref
    for name in names:
        g, m, radii, kind = _case(d, name)
        got = sp.multiscale_statistics(g, m, radii, kind=kind)
        assert list(got) == list(sp.MULTISCALE_STATS)
        for s in sp.MULTISCALE_STATS:
            assert _same_bits(got[s], d[f"{name}/{s}"]), (limit, name, s)
        sub = sp.multiscale_statistics(g, m, iter(radii), stats=("losh", "geary", "moran"), kind=kind)
        assert list(sub) == ["losh", "geary", "moran"]
        for s in sub:
            assert _same_bits(sub[s], d[f"{name}/{s}"]), (limit, name, s)


@pytest.mark.parametrize("limit", [0, float("inf")])
def test_both_schedules_with_a_separate_centred_field(ref, monkeypatch, limit):
    monkeypatch.setattr(sp, "_FUSE_MAX_BYTES", limit)
    d, _ = ref
    for name in ("A", "D"):
        g, m, radii, kind = _case(d, name)
        zs = (g - g[m].mean(0)) * m[..., None]
        zs = zs / (zs.std((0, 1)) + 1e-9)
        got = sp.multiscale_statistics(g, m, radii, kind=kind, centred=zs)
        assert list(got) == list(sp.MULTISCALE_STATS)
        for s in sp.MULTISCALE_STATS:
            src = zs if s in ("moran", "geary", "focality") else g
            assert _same_bits(got[s], SINGLE[s](src, m, radii, kind)), (limit, name, s)
        sd = sp.multiscale_statistics(g, m, radii, stats=("focality", "losh"), kind=kind, centred=zs,
                                      reduce="sd", a=1.5)
        assert _same_bits(sd["losh"], sp.losh_multiscale(g, m, radii, kind, a=1.5, reduce="sd"))


def test_grouped_schedule_warns_once_about_negative_general_g_input(ref, monkeypatch):
    monkeypatch.setattr(sp, "_FUSE_MAX_BYTES", 0)
    d, _ = ref
    g, m, radii, kind = _case(d, "C")
    with pytest.warns(RuntimeWarning, match="general_g_multiscale received negative values") as rec:
        sp.multiscale_statistics(g - g.mean(), m, radii, kind=kind)
    hits = [w for w in rec if "negative values" in str(w.message)]
    assert len(hits) == 1
    assert hits[0].filename == __file__            # stacklevel still points at the caller


def test_geary_with_no_neighbouring_bins_is_nan_without_a_divide_by_zero():
    """Regression (0.14.1): a radius at which no occupied bin has an occupied neighbour gives
    S0 = 0. The value was already NaN, but reached through 1/0 and inf * 0 with two RuntimeWarnings; it is now
    set to NaN explicitly."""
    import warnings
    from spectrast import spatial as sp
    H = W = 40
    G = np.zeros((H, W, 2)); mask = np.zeros((H, W), bool)
    mask[2, 2] = mask[35, 35] = True; G[2, 2] = [1.0, 2.0]; G[35, 35] = [3.0, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = sp.multiscale_statistics(G, mask, [1, 2], stats=("geary",))
    assert np.all(np.isnan(np.asarray(out["geary"])))


@pytest.mark.parametrize("reduced_rank", [False, True])
def test_single_radius_and_single_cell_geary_with_no_neighbours_are_nan(reduced_rank):
    """The same S0 = 0 case on the single-radius lattice route (numpy 1/0 -> inf * 0) and the single-cell route,
    where (N - 1) / (2.0 * 0.0) is Python float division and RAISED ZeroDivisionError (found alongside the
    case above, 0.14.1). Both now return NaN, silently."""
    import warnings
    from spectrast import spatial as sp, spatial_sc as sc
    H = W = 40
    G = np.zeros((H, W, 2)); mask = np.zeros((H, W), bool)
    mask[2, 2] = mask[35, 35] = True; G[2, 2] = [1.0, 2.0]; G[35, 35] = [3.0, 0.5]
    xy = np.array([[0.0, 0.0], [500.0, 500.0], [0.0, 500.0]])
    X = np.array([[1.0, 2.0, 0.5], [3.0, 0.5, 1.0], [0.2, 1.0, 2.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if not reduced_rank:
            assert np.all(np.isnan(np.asarray(sp.gearys_c(G, mask, sp.neighbor_kernel(1)))))
        g = sc.geary_global(xy, X, 20.0, reduced_rank=reduced_rank, rank=2, bin_size=5.0)
    assert np.all(np.isnan(np.asarray(g, float)))

"""The transform plumbing behind the multiscale scalograms must reproduce scipy's 2-D transforms to the bit.

`spatial._rfft2_pad`, `_irfft2_crop` and `_irfft2_occ` run scipy's padded rfft2 / irfft2 as their two 1-D passes,
skipping rows that are zero (forward) or never read (inverse); `_MsGeom` caches the mask-only neighbour arrays
across calls. Every comparison here is on the bit patterns (signed zeros included), against the plain scipy call
each helper replaces, with the pocketfft binding and with the public-call fallback, at several thread counts and
block sizes. The cache tests check that a different mask, geometry or kernel never reads a stale entry and that
the cache stays bounded.
"""
import numpy as np
import pytest
import scipy.fft as sfft
from scipy.fft import next_fast_len

from spectrast import _fft
from spectrast import spatial as sp

from _reference import assert_close_to_reference  # tests/_reference.py


def bits(a):
    a = np.ascontiguousarray(a)
    return a.shape, a.dtype.str, a.tobytes()


def assert_bits_equal(got, ref, what=""):
    assert bits(got) == bits(ref), f"{what}: max|diff| = {np.nanmax(np.abs(got - ref)):.3e}"


@pytest.fixture(autouse=True)
def cpu_backend_clean_cache():
    prev_b, prev_w = _fft.get_backend(), _fft.get_workers()
    _fft.set_backend("cpu")
    sp._ms_cache_clear()
    yield
    sp._ms_cache_clear()
    _fft.set_backend(prev_b)
    _fft.set_workers(prev_w)


@pytest.fixture(params=["binding", "public"])
def route(request, monkeypatch):
    """Run a test through the verified pocketfft binding and through the public-call fallback."""
    if request.param == "public":
        monkeypatch.setattr(sp, "_PFFT", False)
    elif not sp._pocketfft():
        pytest.skip("scipy's pocketfft binding is unavailable or unverified here; the public route is tested")
    return request.param


def _field(rng, shape, holes=True):
    x = rng.standard_normal(shape)
    if holes:                                                  # exact zeros, zero rows and a zero column
        x[rng.random(shape[:2]) < 0.3] = 0.0
        x[1] = 0.0
        x[:, -1] = 0.0
    return x


# (h, w, extra padding) -> fh, fw = next_fast_len(h + pad), next_fast_len(w + pad): even, odd, radix 3/5/7/11
GEOMS = [(13, 17, 6), (20, 9, 15), (33, 40, 11), (7, 121, 10), (61, 45, 26), (30, 30, 0)]


@pytest.mark.parametrize("h,w,padn", GEOMS)
@pytest.mark.parametrize("G", [None, 1, 5])
def test_rfft2_pad_is_scipys_rfft2(route, h, w, padn, G):
    rng = np.random.default_rng(h * 1000 + w)
    fh, fw = next_fast_len(h + padn), next_fast_len(w + padn)
    shape = (h, w) if G is None else (h, w, G)
    x = _field(rng, shape)
    ref = sfft.rfft2(x, s=(fh, fw), axes=(0, 1))
    assert_bits_equal(sp._rfft2_pad(x, fh, fw), ref, "fresh")
    # a reused pad (columns >= w stay zero) and a reused output buffer, across two different inputs
    pad = np.zeros((3, fw) + shape[2:])
    out = np.full((fh, fw // 2 + 1) + shape[2:], np.nan + 1j)
    y = _field(rng, shape)
    assert_bits_equal(sp._rfft2_pad(x, fh, fw, pad, out), ref, "pad+out")
    assert_bits_equal(sp._rfft2_pad(y, fh, fw, pad, out), sfft.rfft2(y, s=(fh, fw), axes=(0, 1)), "reused")


def test_rfft2_pad_kernel_rows_and_many_blocks(route, monkeypatch):
    """A neighbour kernel occupies 2r + 1 rows of the padded grid; tiny blocks force a ragged last block."""
    monkeypatch.setattr(sp, "_XF_BLOCK_BYTES", 1)                 # one row per block
    fh, fw = next_fast_len(50 + 2 * 13), next_fast_len(70 + 2 * 13)
    for kind in ("disc", "gaussian", "queen"):
        for r in (1, 2, 5, 13):
            k = sp.neighbor_kernel(r, kind)
            assert_bits_equal(sp._rfft2_pad(k, fh, fw), sfft.rfft2(k, s=(fh, fw)), f"{kind} r={r}")


@pytest.mark.parametrize("h,w,padn", GEOMS)
@pytest.mark.parametrize("G", [1, 4])
def test_irfft2_occ_is_scipys_cropped_irfft2(route, h, w, padn, G):
    rng = np.random.default_rng(7 * h + w)
    fh, fw = next_fast_len(h + padn), next_fast_len(w + padn)
    mb = rng.random((h, w)) < 0.35
    mb[0] = False; mb[h // 2] = False; mb[:, 0] = False             # empty rows at the edge and inside
    mb[-1, -1] = True
    F = sfft.rfft2(_field(rng, (h, w, G)), s=(fh, fw), axes=(0, 1))
    for off in sorted({0, padn // 2}):                              # the crop must fit: 2 * off <= padn
        k = sp.neighbor_kernel(max(off, 1), "disc")
        Fk = sfft.rfft2(k, s=(fh, fw))
        crop = (slice(off, off + h), slice(off, off + w))
        X = F * Fk[:, :, None]
        ref = sfft.irfft2(X, s=(fh, fw), axes=(0, 1))[crop][mb]
        for step in (1, 2, 5, h):
            gp = sp._OccGather(mb, fw, step)
            got = sp._irfft2_occ(X.copy(), fh, fw, off, crop, mb, gp)
            assert_bits_equal(got, ref, f"off={off} step={step}")
            buf = np.empty((step, fw, G))
            assert_bits_equal(sp._irfft2_occ(X.copy(), fh, fw, off, crop, mb, gp, buf), ref, f"buf step={step}")
        # a 2-D spectrum through the same path as an (fh, n1, 1) view -- how the degree map W1 is taken --
        # against scipy's 2-D irfft2 of the 2-D array
        X2 = F[..., 0] * Fk
        for step in (1, 3, h):
            got = sp._irfft2_occ(X2.copy()[:, :, None], fh, fw, off, crop, mb, sp._OccGather(mb, fw, step))[:, 0]
            assert_bits_equal(got, sfft.irfft2(X2, s=(fh, fw))[crop][mb], f"2-D off={off} step={step}")


def _ns_case(seed, h, w, G, padn):
    rng = np.random.default_rng(seed)
    fh, fw = next_fast_len(h + padn), next_fast_len(w + padn)
    mb = rng.random((h, w)) < 0.3
    mb[:2] = False; mb[h // 2: h // 2 + 3] = False; mb[-1] = False      # empty rows: edges and a band inside
    mb[h // 2 - 1, 1] = True
    return rng, mb, fh, fw


@pytest.mark.parametrize("block_bytes", [1, 600, 8 << 20])             # one row per block .. one block
@pytest.mark.parametrize("G", [1, 3])
def test_residual_spectrum_is_rfft2_of_the_zero_filled_residual(route, monkeypatch, block_bytes, G):
    """LOSH's residual is nonzero only on occupied bins; its spectrum, built a row block at a time in a reused
    zeroed buffer, must be rfft2 of the zero-filled grid -- over repeated calls with different residuals."""
    monkeypatch.setattr(sp, "_XF_BLOCK_BYTES", block_bytes)
    h, w = 23, 31
    rng, mb, fh, fw = _ns_case(G + block_bytes, h, w, G, 14)
    nb = sp._NeighbourSums(mb, fh, fw, G)
    try:
        for rep in range(3):
            eo = np.abs(rng.standard_normal((int(mb.sum()), G))) ** 2.0
            eo[rng.random(eo.shape) < 0.2] = 0.0
            e = np.zeros((h, w, G)); e[mb] = eo
            ref = sfft.rfft2(e, s=(fh, fw), axes=(0, 1))
            got = nb.residual_spectrum(eo)
            assert_bits_equal(got, ref, f"rep {rep}")
            if route == "binding":
                assert got is nb.prod                                   # written into the product buffer
                assert not nb._rpad.any()                               # the block buffer is all zero again
            # and its neighbour sums, the product taken in place in the buffer
            k = sp.neighbor_kernel(3); Fk = sfft.rfft2(k, s=(fh, fw)); crop = (slice(3, 3 + h), slice(3, 3 + w))
            want = sfft.irfft2(ref * Fk[:, :, None], s=(fh, fw), axes=(0, 1))[crop][mb]
            assert_bits_equal(nb(got, Fk, 3), want, f"sums rep {rep}")
    finally:
        nb.close()


@pytest.mark.parametrize("kind", ["disc", "gaussian", "queen"])
def test_neighbours_are_the_degree_weights_825b1a8_formed(route, kind):
    """(has_nb, inv05, invW) on occupied bins: W1 = irfft2(Fm * Fk, s=(fh, fw))[crop][mb] (the 2-D transform)
    and the two gates, exactly as the merged loop formed them -- from the cache on the second call."""
    h, w = 29, 26
    rng, mb, fh, fw = _ns_case(len(kind), h, w, 2, 16)
    m = mb.astype(float)
    Fm = sfft.rfft2(m, s=(fh, fw))
    nb = sp._NeighbourSums(mb, fh, fw, 2, kind, m)
    try:
        for r in (1, 2, 5, 8):
            k = sp.neighbor_kernel(r, kind); off = (k.shape[0] - 1) // 2
            Fk = sfft.rfft2(k, s=(fh, fw))
            W1 = sfft.irfft2(Fm * Fk, s=(fh, fw))[off:off + h, off:off + w][mb]
            has_nb = W1 > 0.5
            want = (has_nb, np.where(has_nb, 1.0 / np.maximum(W1, 1e-12), 0.0)[:, None],
                    np.where(W1 > 1e-12, 1.0 / np.maximum(W1, 1e-12), 0.0)[:, None])
            got = nb.neighbours(r, sp._rfft2_pad(k, fh, fw), off)
            for a, b_, nm in zip(got, want, ("has_nb", "inv05", "invW")):
                assert bits(a) == bits(b_), f"{nm} r={r} {kind}"
            again = nb.neighbours(r, sp._rfft2_pad(k, fh, fw), off)
            assert all(x is y for x, y in zip(again, got))               # served from the cache
    finally:
        nb.close()


def test_neighbour_sums_forward_and_crop_guard(route):
    h, w, G = 19, 27, 4
    rng, mb, fh, fw = _ns_case(2, h, w, G, 12)
    nb = sp._NeighbourSums(mb, fh, fw, G)
    try:
        for _ in range(2):                                              # the padded block buffer is reused
            x = _field(rng, (h, w, G))
            assert_bits_equal(nb.forward(x), sfft.rfft2(x, s=(fh, fw), axes=(0, 1)), "forward")
        F = sfft.rfft2(_field(rng, (h, w, G)), s=(fh, fw), axes=(0, 1))
        Fk = sfft.rfft2(sp.neighbor_kernel(2), s=(fh, fw))
        with pytest.raises(ValueError, match="does not fit"):
            nb(F, Fk, fh)                                               # a crop past the padded grid
    finally:
        nb.close()


# ------------------------------------------------------------------ the setup's grid-sized temporaries

def _special(rng, shape):
    a = rng.standard_normal(shape) * 10.0 ** rng.integers(-300, 300, shape)
    flat = a.reshape(-1)
    flat[::7] = -0.0; flat[1::11] = 0.0; flat[2::13] = np.nan; flat[3::17] = np.inf; flat[4::19] = -np.inf
    flat[5::23] = 5e-324; flat[6::29] = -1e-310
    return a


def test_square_is_multiply_bit_for_bit():
    """_ms_cpu forms (z * z).sum once and uses it for (z ** 2).sum too: numpy's ** 2 is np.square, i.e. z * z."""
    rng = np.random.default_rng(0)
    for shape in [(7,), (13, 5), (9, 8, 3)]:
        a = _special(rng, shape)
        with np.errstate(all="ignore"):
            assert bits(a ** 2) == bits(a * a) == bits(np.square(a))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.bool_])
@pytest.mark.parametrize("G", [1, 2, 5])
def test_numpy_temporaries_of_c_grids_are_c_ordered_float64(dtype, G):
    """_c_grid's premise: for a C-ordered grid the temporaries the setup used to allocate are C-ordered float64,
    so a C-ordered float64 scratch holding them is summed in the same order."""
    rng = np.random.default_rng(G)
    Zc = (rng.standard_normal((11, 9, G)) * 3).astype(dtype)
    m = (rng.random((11, 9)) > 0.4).astype(float)
    assert sp._c_grid(Zc, Zc.shape)
    zbar = (Zc * m[..., None]).sum((0, 1)) / m.sum()
    for t in (Zc * m[..., None], Zc - zbar, (Zc - zbar) * m[..., None]):
        assert t.dtype == np.float64 and t.flags.c_contiguous
    assert not sp._c_grid(np.asfortranarray(np.ones((4, 5, 3))), (4, 5, 3))
    assert not sp._c_grid(np.ones((4, 10, 3))[:, ::2], (4, 5, 3))
    assert not sp._c_grid(np.ones((4, 5, 3), complex), (4, 5, 3))


@pytest.mark.parametrize("G", [1, 4])
def test_scratch_setup_equals_the_original_expressions(monkeypatch, G):
    """The scratch route (C-ordered inputs) against the original expressions (forced): every statistic, fused
    with a separate centred field and alone, with -0.0, NaN and inf outside the mask."""
    rng = np.random.default_rng(10 + G)
    H, W = 33, 41
    yy, xx = np.mgrid[:H, :W]
    mb = ((yy - 15) ** 2 + (xx - 20) ** 2 < 14 ** 2) & (rng.random((H, W)) > 0.1)
    g = np.abs(rng.standard_normal((H, W, G))) + 0.1
    zs = (g - g[mb].mean(0)) * mb[..., None]
    radii = [1, 2, 5]
    variants = [(g, None), (g, zs)]
    gg = g.copy(); gg[~mb] = -0.0; variants.append((gg, None))
    gg = g.copy(); gg[~mb] = np.nan; gg[0, 0] = np.inf; variants.append((gg, None))
    for grid, centred in variants:
        with np.errstate(all="ignore"):
            sp._ms_cache_clear()
            got = sp.multiscale_statistics(grid, mb, radii, stats=STATS, centred=centred)
            with monkeypatch.context() as mp:
                mp.setattr(sp, "_c_grid", lambda a, shape: False)
                want = sp.multiscale_statistics(grid, mb, radii, stats=STATS, centred=centred)
            for s in STATS:
                assert_bits_equal(got[s], want[s], f"{s} centred={centred is not None}")


@pytest.mark.parametrize("layout", ["C", "F", "float32"])
def test_geary_fields_are_never_formed_but_transform_the_same(route, layout):
    """forward_masked: rfft2 of xg = Zc * m[..., None] and of xg ** 2, formed one row block at a time."""
    rng = np.random.default_rng(3)
    h, w, G = 21, 34, 3
    fh, fw = next_fast_len(h + 12), next_fast_len(w + 12)
    mb = rng.random((h, w)) > 0.3
    m = mb.astype(float)
    Zc = _field(rng, (h, w, G))
    Zc = {"C": Zc, "F": np.asfortranarray(Zc), "float32": Zc.astype(np.float32)}[layout]
    nb = sp._NeighbourSums(mb, fh, fw, G)
    try:
        xg = Zc * m[..., None]
        assert_bits_equal(nb.forward_masked(Zc, m), sfft.rfft2(xg, s=(fh, fw), axes=(0, 1)), "xg")
        assert_bits_equal(nb.forward_masked(Zc, m, square=True), sfft.rfft2(xg ** 2, s=(fh, fw), axes=(0, 1)),
                          "xg ** 2")
    finally:
        nb.close()


def test_irfft2_scale_is_the_factor_pocketfft_applies():
    for fh, fw in ((8, 8), (21, 22), (512, 1331), (1540, 1650), (77, 125)):
        delta = np.zeros((fh, fw // 2 + 1), complex); delta[0, 0] = 1.0
        probe = sfft.irfft2(delta, s=(fh, fw))
        assert np.all(probe == probe[0, 0])
        assert bits(probe[0, 0]) == bits(sp._irfft2_scale(fh, fw))


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_split_transforms_do_not_depend_on_the_thread_count(route, workers):
    rng = np.random.default_rng(3)
    h, w, G = 37, 51, 6
    fh, fw = next_fast_len(h + 20), next_fast_len(w + 20)
    x = _field(rng, (h, w, G))
    mb = rng.random((h, w)) < 0.4
    _fft.set_workers(workers)
    F = sp._rfft2_pad(x, fh, fw)
    assert_bits_equal(F, sfft.rfft2(x, s=(fh, fw), axes=(0, 1), workers=1), "forward")
    Fk = sfft.rfft2(sp.neighbor_kernel(4), s=(fh, fw))
    crop = (slice(4, 4 + h), slice(4, 4 + w))
    X = F * Fk[:, :, None]
    ref = sfft.irfft2(X, s=(fh, fw), axes=(0, 1), workers=1)[crop][mb]
    assert_bits_equal(sp._irfft2_occ(X, fh, fw, 4, crop, mb, sp._OccGather(mb, fw, 4)), ref, "inverse")


def test_invalid_worker_settings_fail_as_scipy_fails(route):
    x = np.ones((6, 7, 2))
    for w in (0, -10_000):
        _fft.set_workers(w)
        with pytest.raises(ValueError):
            sfft.rfft2(x, s=(8, 8), axes=(0, 1), workers=w)
        with pytest.raises(ValueError):
            sp.multiscale_statistics(x, np.ones((6, 7), bool), [1, 2], stats=("moran",))


def test_an_unverified_binding_is_not_used(monkeypatch):
    """If the private binding stops matching the public calls, the probe rejects it and the public route runs."""
    class Broken:
        def __getattr__(self, name):
            raise AssertionError("an unverified binding was used")
    monkeypatch.setattr(sp, "_PFFT", None)
    import scipy.fft._pocketfft as pk
    monkeypatch.setattr(pk, "pypocketfft", Broken())
    assert sp._pocketfft() is False
    rng = np.random.default_rng(5)
    x = rng.standard_normal((9, 11, 2))
    assert_bits_equal(sp._rfft2_pad(x, 16, 16), sfft.rfft2(x, s=(16, 16), axes=(0, 1)), "fallback forward")


# ------------------------------------------------------------------ the cross-call cache of mask-only arrays

def _case(seed=0, H=40, W=52, G=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    m = ((yy - 18) ** 2 + (xx - 25) ** 2 < 15 ** 2) & (rng.random((H, W)) > 0.1)
    m[1, 1] = m[H - 2, 3] = True                                  # isolated, neighbourless bins
    g = np.abs(rng.standard_normal((H, W, G))) + 0.1
    return g, m


STATS = ("moran", "geary", "general_g", "losh", "focality")


def _uncached(monkeypatch, g, m, radii, kind="disc"):
    with monkeypatch.context() as mp:
        mp.setattr(sp, "_MS_CACHE_MAX_ENTRIES", 0)
        return sp.multiscale_statistics(g, m, radii, stats=STATS, kind=kind)


def _assert_same(a, b, what):
    for s in STATS:
        assert_bits_equal(a[s], b[s], f"{what}: {s}")


def test_cache_hits_reproduce_the_uncached_result(monkeypatch):
    g, m = _case()
    radii = [1, 2, 4, 7]
    ref = _uncached(monkeypatch, g, m, radii)
    first = sp.multiscale_statistics(g, m, radii, stats=STATS)
    assert len(sp._ms_cache) == 1
    geom = next(iter(sp._ms_cache.values()))
    assert set(geom.nb) == set(radii)
    second = sp.multiscale_statistics(g, m, radii, stats=STATS)       # every neighbour array from the cache
    _assert_same(first, ref, "cold")
    _assert_same(second, ref, "warm")
    for has_nb, inv05, invW in geom.nb.values():
        assert not (has_nb.flags.writeable or inv05.flags.writeable or invW.flags.writeable)


def test_a_different_mask_never_reads_a_stale_entry(monkeypatch):
    g, m = _case()
    radii = [1, 3, 6]
    sp.multiscale_statistics(g, m, radii, stats=STATS)
    m2 = m.copy(); m2[10:14, 20:30] = ~m2[10:14, 20:30]               # same shape, different tissue
    _assert_same(sp.multiscale_statistics(g, m2, radii, stats=STATS), _uncached(monkeypatch, g, m2, radii), "m2")
    # the SAME array object edited in place between calls
    m3 = m.copy()
    sp.multiscale_statistics(g, m3, radii, stats=STATS)
    m3[5:9, 5:40] = True
    _assert_same(sp.multiscale_statistics(g, m3, radii, stats=STATS), _uncached(monkeypatch, g, m3, radii),
                 "in-place edit")


def test_a_different_geometry_or_kernel_never_reads_a_stale_entry(monkeypatch):
    g, m = _case()
    sp.multiscale_statistics(g, m, [1, 2, 4], stats=STATS)
    # a larger rmax pads to a different (fh, fw): radius 2 must be recomputed at that size, not reused
    radii = [2, 9]
    _assert_same(sp.multiscale_statistics(g, m, radii, stats=STATS), _uncached(monkeypatch, g, m, radii), "rmax")
    for kind in ("gaussian", "queen"):
        _assert_same(sp.multiscale_statistics(g, m, [1, 2, 4], stats=STATS, kind=kind),
                     _uncached(monkeypatch, g, m, [1, 2, 4], kind), kind)
    keys = list(sp._ms_cache)
    assert len({k[1:4] for k in keys}) == len(keys)                    # one entry per (fh, fw, kind) here


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(sp, "_MS_CACHE_MAX_ENTRIES", 2)
    for seed in range(5):
        g, m = _case(seed)
        sp.multiscale_statistics(g, m, [1, 2], stats=("moran",))
        assert len(sp._ms_cache) <= 2
    # a byte budget smaller than one geometry's arrays: still correct, and nothing beyond the budget is kept
    sp._ms_cache_clear()
    g, m = _case(9)
    monkeypatch.setattr(sp, "_MS_CACHE_MAX_BYTES", m.nbytes + 10)
    got = sp.multiscale_statistics(g, m, [1, 2, 3], stats=STATS)
    assert sum(x.nbytes for x in sp._ms_cache.values()) <= sp._MS_CACHE_MAX_BYTES
    assert not next(iter(sp._ms_cache.values())).nb
    _assert_same(got, _uncached(monkeypatch, g, m, [1, 2, 3]), "byte bound")


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs os.fork")
def test_a_forked_child_starts_with_a_fresh_lock_and_an_empty_cache():
    """Even if the parent's cache lock is held at the fork (another thread mid-update), the child must not
    deadlock: it gets a new lock and an empty cache, and computes the same result."""
    import os
    import time
    g, m = _case(4)
    want = sp.multiscale_statistics(g, m, [1, 2], stats=("moran", "losh"))
    assert len(sp._ms_cache) == 1
    with sp._ms_cache_lock:                                         # held across the fork
        pid = os.fork()
        if pid == 0:                                                # child
            code = 1
            try:
                ok = len(sp._ms_cache) == 0
                got = sp.multiscale_statistics(g, m, [1, 2], stats=("moran", "losh"))
                ok &= all(bits(got[s]) == bits(want[s]) for s in want)
                code = 0 if ok else 1
            finally:
                os._exit(code)
    deadline = time.time() + 60
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.time() > deadline:
            os.kill(pid, 9)
            pytest.fail("forked child deadlocked on the cache lock")
        time.sleep(0.05)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def test_no_cache_off_the_cpu_backend(monkeypatch):
    monkeypatch.setattr(sp._fft, "get_backend", lambda: "mps")
    assert sp._ms_geom(np.ones((4, 4), bool), 8, 8, "disc") is None


def test_frozen_fixture_cold_and_warm_cache():
    """Every statistic on every fixture case: the run with every neighbour array served from the cache has the
    same bit patterns (signed zeros, NaN positions) as the run with an empty cache, and the cold run matches the
    frozen reference to floating-point rounding."""
    import os
    ref = os.path.join(os.path.dirname(__file__), "data", "multiscale_reference.npz")
    with np.load(ref) as z:
        d = {k: z[k] for k in z.files}
    names = sorted({k.split("/")[0] for k in d if "/" in k})
    sp._ms_cache_clear()
    cold = {}
    for warm in (False, True):
        for name in names:
            g, m = d[f"{name}/grid"], d[f"{name}/mask"]
            radii, kind = d[f"{name}/radii"].tolist(), str(d[f"{name}/kind"])
            got = sp.multiscale_statistics(g, m, radii, kind=kind)
            for s in sp.MULTISCALE_STATS:
                if warm:
                    assert_bits_equal(got[s], cold[name][s], f"{name}/{s} warm vs cold")
                else:
                    assert_close_to_reference(got[s], d[f"{name}/{s}"], f"{name}/{s}")
            if not warm:
                cold[name] = got


# ------------------------------------------------------------------ texture: threaded per-gene loop

def test_texture_does_not_depend_on_the_thread_count():
    rng = np.random.default_rng(11)
    H, W, G = 45, 64, 7
    g = rng.standard_normal((H, W, G))
    m = rng.random((H, W)) > 0.2
    out = {}
    for w in (1, 3, 8):
        _fft.set_workers(w)
        out[w] = sp.texture_multiscale(g, m, [2, 3, 5, 8, 12])
    for w in (3, 8):
        assert_bits_equal(out[w], out[1], f"texture workers={w}")

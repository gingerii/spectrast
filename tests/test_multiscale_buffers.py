"""The per-radius buffers, threads and transform plumbing of spatial._ms_cpu must not move a single bit.

`_NeighbourSums` reuses one product buffer per call, splits the spectrum product over threads by row blocks,
runs the inverse as pocketfft's two passes (the first in place, the second on the cropped rows only, in reused
row blocks) through scipy's private pocketfft binding once `_pocketfft` has verified it, and gathers occupied
bins with `np.take`. The fixture cases are small, so by default the product runs unthreaded; these tests force
each path and hold it to the frozen reference outputs in tests/data/multiscale_reference.npz with exact
equality. (tests/test_multiscale_transforms.py checks the split transforms themselves against scipy.)
"""
import os

import numpy as np
import pytest

from spectrast import _fft
from spectrast import spatial as sp

REF = os.path.join(os.path.dirname(__file__), "data", "multiscale_reference.npz")
DISC = ("moran", "geary", "general_g", "losh", "focality")


@pytest.fixture(scope="module")
def ref():
    with np.load(REF) as z:
        d = {k: z[k] for k in z.files}
    names = sorted({k.split("/")[0] for k in d if "/" in k})
    return d, names


@pytest.fixture(autouse=True)
def cpu_backend():
    prev, prev_w = _fft.get_backend(), _fft.get_workers()
    _fft.set_backend("cpu")
    yield
    _fft.set_backend(prev)
    _fft.set_workers(prev_w)


def _case(d, name):
    return d[f"{name}/grid"], d[f"{name}/mask"], d[f"{name}/radii"].tolist(), str(d[f"{name}/kind"])


@pytest.mark.parametrize("workers", [2, 3, 7])
def test_threaded_product_reproduces_the_reference(ref, monkeypatch, workers):
    d, names = ref
    monkeypatch.setattr(sp, "_PRODUCT_MIN_THREADED", 0)          # thread every product, however small
    _fft.set_workers(workers)
    for name in names:
        g, m, radii, kind = _case(d, name)
        got = sp.multiscale_statistics(g, m, radii, stats=DISC, kind=kind)
        for s in DISC:
            np.testing.assert_array_equal(got[s], d[f"{name}/{s}"], err_msg=f"{s} on {name}, {workers} workers")


def test_public_transform_fallback_reproduces_the_reference(ref, monkeypatch):
    """Without scipy's private pocketfft binding (absent, or failing _pocketfft's probe) the public transforms
    are used -- the same values."""
    d, names = ref
    monkeypatch.setattr(sp, "_PFFT", False)
    sp._ms_cache_clear()                                          # nothing computed on the binding route reused
    for name in names:
        g, m, radii, kind = _case(d, name)
        got = sp.multiscale_statistics(g, m, radii, stats=DISC, kind=kind)
        for s in DISC:
            np.testing.assert_array_equal(got[s], d[f"{name}/{s}"], err_msg=f"{s} on {name}, public fallback")


def test_neighbour_sums_equal_the_plain_expression():
    """One radius, by hand: irfft2(F * Fk)[crop][mb] against the buffered, threaded, two-pass path."""
    from scipy.fft import next_fast_len
    rng = np.random.default_rng(7)
    H, W, G, r = 37, 53, 6, 4
    mb = rng.random((H, W)) > 0.6
    x = rng.standard_normal((H, W, G)) * mb[..., None]
    fh, fw = next_fast_len(H + 2 * r), next_fast_len(W + 2 * r)
    F = _fft.rfft2(x, s=(fh, fw), axes=(0, 1))
    k = sp.neighbor_kernel(r, "disc"); off = (k.shape[0] - 1) // 2
    Fk = _fft.rfft2(k, s=(fh, fw))
    want = _fft.irfft2(F * Fk[:, :, None], s=(fh, fw), axes=(0, 1))[off:off + H, off:off + W][mb]
    F0 = F.copy()
    nb = sp._NeighbourSums(mb, fh, fw, G)
    try:
        for _ in range(2):                                         # second call reuses the buffers
            got = nb(F, Fk, off)
            assert got.flags.c_contiguous and got.shape == want.shape
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(F, F0)                       # the field's spectrum is never overwritten
    finally:
        nb.close()

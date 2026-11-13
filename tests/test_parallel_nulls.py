"""Phase-2 determinism guardrail: the FFT-bound permutation nulls are now
fanned across a thread pool. That MUST NOT change the result — parallel output has to equal the
serial output bit-for-bit and be independent of the worker count (a correctness requirement, since
these nulls drive FDR). Achieved by pre-drawing all permutations from the master RNG (identical
draw sequence) and reducing in submission order.

Run: ``pytest tests/test_parallel_nulls.py -q``.
"""
import numpy as np
import pytest

from spectrast import _parallel
from spectrast.spatial import (celltype_colocalization_multiscale, neighborhood_enrichment,
                         morans_i, morans_i_multiscale, permutation_pvalues, permutation_test,
                         bivariate_multiscale_significance)


@pytest.fixture
def toy_labels():
    rng = np.random.default_rng(0)
    N = 900
    xy = rng.random((N, 2)) * 200.0
    labels = rng.choice(["A", "B", "C"], size=N, p=[0.5, 0.3, 0.2])
    return xy, labels


def _run(fn):
    """Run fn() serially (n_jobs=1) then threaded (n_jobs=4), restoring the setting after."""
    prev = _parallel.get_n_jobs()
    try:
        _parallel.set_n_jobs(1)
        serial = fn()
        _parallel.set_n_jobs(4)
        threaded = fn()
    finally:
        _parallel.set_n_jobs(prev if prev else None)
    return serial, threaded


def test_coloc_null_parallel_equals_serial(toy_labels):
    xy, labels = toy_labels
    radii = [3, 6, 10]
    s, t = _run(lambda: celltype_colocalization_multiscale(
        xy, labels, radii, bin_size=10.0, n_perm=32, seed=7))
    # (types, obs, z) — obs is deterministic; z rides on the threaded null and must match exactly
    assert s[0] == t[0]
    assert np.array_equal(s[1], t[1])                 # observed
    assert np.array_equal(s[2], t[2], equal_nan=True) # permutation z — bit-identical serial vs threaded


def test_coloc_counts_include_same_bin_partners():
    """m17: same-bin partners must count in the celltype coloc matrix (home bin now INCLUDED), and a cell must
    NOT count itself (self removed on the A–A diagonal). All cells sit in ONE bin (huge bin_size) so the only
    neighbours are co-binned — a home-EXCLUDED kernel would report 0 here."""
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05]])
    labels = np.array(["A", "A", "B", "B", "B"])                # 2 A, 3 B, all co-binned
    types, obs, _ = celltype_colocalization_multiscale(xy, labels, [1], bin_size=100.0, n_perm=5, seed=0)
    d = {t: i for i, t in enumerate(types)}
    ab = obs[0, d["A"], d["B"]]; aa = obs[0, d["A"], d["A"]]
    assert np.isclose(ab, 3.0), f"A–B coloc should count all 3 co-binned B per A cell: {ab}"
    assert np.isclose(aa, 1.0), f"A–A coloc should be n_A−1=1 (self excluded): {aa}"


def test_neighborhood_enrichment_null_parallel_equals_serial():
    rng = np.random.default_rng(1)
    H, W = 40, 44
    mask = rng.random((H, W)) > 0.2
    label_grid = np.where(mask, rng.integers(0, 3, size=(H, W)), -1)
    s, t = _run(lambda: neighborhood_enrichment(label_grid, mask, radius=4, n_perm=32, seed=3))
    for a, b in zip(s, t):                             # (enrichment, z, labels) — all must match exactly
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.fixture
def toy_grid():
    rng = np.random.default_rng(2)
    H, W, G = 40, 44, 8
    grid = rng.standard_normal((H, W, G))
    mask = rng.random((H, W)) > 0.2
    grid *= mask[..., None]
    return grid, mask


def test_permutation_pvalues_parallel_equals_serial(toy_grid):
    grid, mask = toy_grid
    obs = morans_i(grid, mask, radius=3)
    s, t = _run(lambda: permutation_pvalues(grid, mask, obs, radius=3, n_perm=48, seed=1))
    assert np.array_equal(s, t)


def test_permutation_test_parallel_equals_serial(toy_grid):
    grid, mask = toy_grid
    s, t = _run(lambda: permutation_test(lambda g, m: morans_i(g, m, radius=3), grid, mask,
                                         n_perm=48, seed=1))
    for k in s:
        assert np.array_equal(np.asarray(s[k]), np.asarray(t[k]), equal_nan=True)


def test_bivariate_multiscale_significance_parallel_equals_serial():
    rng = np.random.default_rng(4)
    H, W = 44, 40
    mask = rng.random((H, W)) > 0.2
    lig = rng.standard_normal((H, W)) * mask
    rec = rng.standard_normal((H, W)) * mask
    s, t = _run(lambda: bivariate_multiscale_significance(lig, rec, mask, [3, 6, 9], n_perms=40, seed=5))
    for a, b in zip(s, t):
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)

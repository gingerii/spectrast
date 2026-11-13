"""Analytic tests for the hypercomplex transforms/algebra (src/spectrast/quaternion.py, octonion.py).

These underpin the paper's quaternion multi-scale + octonion rank-7 "validated wins" but had ZERO
coverage — and they have trivial exact checks (transform round-trips = identity, algebra laws), which
are cheap to break in a refactor. Dependency-free. Run: `pytest tests/test_quaternion.py -q`.
"""
import numpy as np
import pytest

from spectrast._archive import quaternion as Q
from spectrast._archive import octonion as O

I1 = np.array([1.0, 0.0, 0.0, 0.0])          # quaternion identity


def _rand_q(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape + (4,))


def test_qdft2_roundtrip():
    q = _rand_q((16, 24), 0)
    assert np.allclose(Q.iqdft2(Q.qdft2(q)), q, atol=1e-10)


def test_octonion_dft_roundtrip():
    o = np.random.default_rng(1).standard_normal((16, 20, 8))
    assert np.allclose(O.iodft2(O.odft2(o)), o, atol=1e-10)


def test_cayley_dickson_roundtrip():
    q = _rand_q((5, 7), 2)
    assert np.allclose(Q.from_cayley_dickson(*Q.to_cayley_dickson(q)), q, atol=1e-12)


def test_qmul_identity():
    q = _rand_q((30,), 3)
    assert np.allclose(Q.qmul(q, np.broadcast_to(I1, q.shape)), q, atol=1e-12)
    assert np.allclose(Q.qmul(np.broadcast_to(I1, q.shape), q), q, atol=1e-12)


def test_qmul_associative():
    p, r, s = _rand_q((40,), 4), _rand_q((40,), 5), _rand_q((40,), 6)
    assert np.allclose(Q.qmul(Q.qmul(p, r), s), Q.qmul(p, Q.qmul(r, s)), atol=1e-12)


def test_qmul_norm_multiplicative():
    """|p r| = |p| |r| — the quaternion norm is multiplicative (a defining property)."""
    p, r = _rand_q((50,), 7), _rand_q((50,), 8)
    np_ = np.linalg.norm(p, axis=-1) * np.linalg.norm(r, axis=-1)
    assert np.allclose(np.linalg.norm(Q.qmul(p, r), axis=-1), np_, atol=1e-12)


def test_qmul_noncommutative():
    """Sanity: the Hamilton product is NOT commutative (guards against an accidental symmetric impl)."""
    p, r = _rand_q((10,), 9), _rand_q((10,), 10)
    assert not np.allclose(Q.qmul(p, r), Q.qmul(r, p))


def test_rotate_preserves_vector_norm():
    """r q r^{-1} with unit r is an isometry of the vector part (a rotation)."""
    rng = np.random.default_rng(11)
    v = np.zeros((25, 4)); v[:, 1:] = rng.standard_normal((25, 3))     # pure-vector quaternions
    r = Q.rotation_quaternion(rng.standard_normal(3), 0.7)
    out = Q.rotate(v, np.broadcast_to(r, v.shape))
    assert np.allclose(np.linalg.norm(out[:, 1:], axis=-1), np.linalg.norm(v[:, 1:], axis=-1), atol=1e-10)
    assert np.allclose(out[:, 0], 0.0, atol=1e-10)                     # scalar part stays zero


def test_qdft2_matches_componentwise_fft():
    """The quaternion DFT is exactly two complex FFTs on the Cayley–Dickson components (paper def)."""
    q = _rand_q((8, 8), 12)
    a, b = Q.to_cayley_dickson(q)
    F = Q.qdft2(q); Fa, Fb = Q.to_cayley_dickson(F)
    assert np.allclose(Fa, np.fft.fft2(a), atol=1e-12) and np.allclose(Fb, np.fft.fft2(b), atol=1e-12)

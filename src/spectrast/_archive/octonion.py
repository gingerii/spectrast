"""Octonion (8D hypercomplex) extension of the quaternion ST model.

An octonion is a pair of quaternions (Cayley-Dickson): o = a + b*e, stored as the
last axis (size 8) = [a (4), b (4)]. The vector part is 7 imaginary components, so a
location maps **7 features** (top-7 SVD comps or 7 gene sets) instead of 3 -- directly
addressing the paper's rank-3 limitation (Limitation #1).

Because each Cayley-Dickson level's DFT is built from the level below, the octonion 2D
DFT is just two quaternion DFTs (which are two complex FFTs each). The transform is
linear, so spectral filtering / large-neighborhood ops / compression all extend.

CAVEAT: octonion multiplication is non-associative AND non-commutative, so the clean
SO(3) rotation interpretation of quaternions does NOT carry over. Octonions buy
*capacity* (more features), not new geometry. Use for spectral/visualization/capacity.
"""
from __future__ import annotations

import numpy as np

from .quaternion import qdft2, iqdft2


def odft2(o: np.ndarray) -> np.ndarray:
    """2D octonion DFT (H, W, 8) -> (H, W, 8) via two quaternion DFTs (Cayley-Dickson)."""
    return np.concatenate([qdft2(o[..., :4]), qdft2(o[..., 4:])], axis=-1)


def iodft2(O: np.ndarray) -> np.ndarray:
    """Inverse 2D octonion DFT."""
    return np.concatenate([iqdft2(O[..., :4]), iqdft2(O[..., 4:])], axis=-1)


def map_svd_octonion(counts) -> np.ndarray:
    """Octonion map from the top-8 singular vectors of UNNORMALIZED counts (rank-7 vector).

    Mirrors `quaternion.map_svd` but keeps 8 components: scalar = |projection on PC1|
    (depth proxy); the 7 imaginary parts = projections on PC2..PC8, normalized to unit
    length. `counts` is (N, genes). Returns (N, 8). Uses randomized SVD (sparse-safe).
    """
    from scipy.sparse import issparse  # lazy
    sparse = issparse(counts)
    small = (min(counts.shape) <= 64 or max(counts.shape) <= 2000)
    if small:
        X = counts.toarray().astype(float) if sparse else np.asarray(counts, dtype=float)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:8].T
        proj = X @ V
    else:
        from sklearn.utils.extmath import randomized_svd  # lazy
        X = counts if sparse else np.asarray(counts, dtype=float)
        _, _, Vt = randomized_svd(X, n_components=8, n_iter=5, random_state=0)
        proj = np.asarray(X @ Vt.T)

    flip = np.where(proj[0] < 0, -1.0, 1.0)        # constant sign convention
    proj = proj * flip
    a = np.abs(proj[:, 0])
    vec = proj[:, 1:8]
    length = np.linalg.norm(vec, axis=1, keepdims=True)
    safe = np.where(length == 0, 1.0, length)
    unit = vec / safe
    unit[length[:, 0] == 0] = 0.0
    return np.concatenate([a.reshape(-1, 1), unit], axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    O = rng.standard_normal((16, 24, 8))
    err = np.abs(iodft2(odft2(O)) - O).max()
    print(f"octonion DFT round-trip max abs error: {err:.2e}")
    assert err < 1e-10
    print("OK")

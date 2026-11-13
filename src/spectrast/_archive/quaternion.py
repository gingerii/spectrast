"""Core quaternion model for spatial transcriptomics (Frost 2025, Bioinf. Adv.).

A quaternion ``q = a + b i + c j + d k`` is stored as the last axis (size 4) of a
numpy array, ordered ``[a, b, c, d]`` (scalar, then i, j, k).

Per the paper, each ST location maps to a quaternion whose
  * vector part (b, c, d) is a UNIT direction in R^3 = the relative expression
    profile (from the top-3 SVD components, 3 genes, or 3 gene-set scores), and
  * scalar part (a) encodes sequencing depth (library size) = uncertainty.

The hypercomplex 2D DFT is realized via the Cayley–Dickson decomposition
(q = alpha + beta j, alpha = a + b i, beta = c + d i) and two standard complex
FFTs, exactly as in Ell & Sangwine (2007) / Section 3.3 of the paper.

Only numpy/scipy are required; heavier deps (scanpy/anndata) are lazy-imported.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cayley–Dickson <-> quaternion packing
# ---------------------------------------------------------------------------

def to_cayley_dickson(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quaternion array (..., 4) -> (alpha, beta) complex arrays.

    alpha = a + b i, beta = c + d i, so that q = alpha + beta j.
    """
    q = np.asarray(q)
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    alpha = a + 1j * b
    beta = c + 1j * d
    return alpha, beta


def from_cayley_dickson(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """(alpha, beta) complex arrays -> quaternion array (..., 4)."""
    return np.stack(
        [alpha.real, alpha.imag, beta.real, beta.imag], axis=-1
    )


# ---------------------------------------------------------------------------
# Quaternion-domain 2D discrete Fourier transform
# ---------------------------------------------------------------------------

def qdft2(q: np.ndarray) -> np.ndarray:
    """2D quaternion DFT of a quaternion-valued matrix (H, W, 4) -> (H, W, 4).

    Realized as two complex 2D FFTs on the Cayley–Dickson components and
    repacked as F = F_alpha + F_beta j (paper Section 3.3, steps 1-4).
    """
    alpha, beta = to_cayley_dickson(q)
    Fa = np.fft.fft2(alpha)
    Fb = np.fft.fft2(beta)
    return from_cayley_dickson(Fa, Fb)


def iqdft2(F: np.ndarray) -> np.ndarray:
    """Inverse 2D quaternion DFT (H, W, 4) -> (H, W, 4)."""
    Fa, Fb = to_cayley_dickson(F)
    alpha = np.fft.ifft2(Fa)
    beta = np.fft.ifft2(Fb)
    return from_cayley_dickson(alpha, beta)


# ---------------------------------------------------------------------------
# Spectral filtering and real-mask convolution
# ---------------------------------------------------------------------------

def make_quaternion_kernel(kind: str) -> np.ndarray:
    """3x3 scalar-quaternion kernel (3, 3, 4), matching QSC ``createKernel``.

    Real-valued kernels are embedded as scalar quaternions (value, 0, 0, 0).
    """
    K = np.zeros((3, 3, 4))
    if kind == "identity":
        K[1, 1, 0] = 1.0
    elif kind == "sharpening":
        K[1, 1, 0] = 5.0
        K[0, 1, 0] = K[2, 1, 0] = K[1, 0, 0] = K[1, 2, 0] = -1.0
    elif kind == "edge":
        K[:, :, 0] = -1.0
        K[1, 1, 0] = 8.0
    elif kind == "boxblur":
        K[:, :, 0] = 1.0 / 9.0
    else:
        raise ValueError(f"unknown kernel kind: {kind}")
    return K


def make_mask(n: int, p: int, K: np.ndarray) -> np.ndarray:
    """Embed a 3x3 quaternion kernel into an (n, p, 4) mask at the origin corners.

    Mirrors QSC ``createMask``: kernel rows map to matrix rows [n-1, 0, 1] and
    kernel cols to [p-1, 0, 1], i.e. the kernel is centered at the (0,0) origin
    with wraparound for circular convolution.
    """
    M = np.zeros((n, p, 4))
    rows = [n - 1, 0, 1]
    cols = [p - 1, 0, 1]
    for i in range(3):
        for j in range(3):
            M[rows[i], cols[j]] = K[i, j]
    return M


def convolve_q(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Quaternion-domain circular convolution of (H, W, 4) matrices, matching QSC.

    QSC ``convolveMat(quat=TRUE)``: ``iqft(qft(X) * qft(M))`` where ``*`` is the
    elementwise Hamilton product in the frequency domain. (QSC also divides by
    ``length(X1)``, but ``length()`` on an onionmat returns 1, so that is a no-op;
    we reproduce the effective behavior.) Use with :func:`make_mask` /
    :func:`make_quaternion_kernel` to reproduce the package's sharpening /
    edge-detection convolutions.
    """
    F = qmul(qdft2(X), qdft2(M))
    return iqdft2(F)


def convolve_real_mask(q: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear convolution of the VECTOR part with a real kernel (scipy, mode='same').

    Convenience helper that ignores the scalar channel. NOTE: this is *linear*
    convolution and does not match QSC's circular :func:`convolve_q`; use
    :func:`convolve_q` for QSC-faithful results.
    """
    from scipy.signal import fftconvolve  # lazy

    out = q.copy().astype(float)
    for ch in (1, 2, 3):  # i, j, k
        out[..., ch] = fftconvolve(q[..., ch], kernel, mode="same")
    return out


# ---------------------------------------------------------------------------
# Quaternion algebra: multiplication and rotation (state transitions)
# ---------------------------------------------------------------------------

def qmul(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternion arrays (..., 4)."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    ], axis=-1)


def rotation_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit rotation quaternion for ``angle`` (radians) about ``axis`` (3-vector)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def rotate(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Apply rotation quaternion ``r`` to vector quaternions: r q r^{-1}.

    A transcriptomic state transition (paper Section 3.2). ``r`` is broadcast.
    """
    rconj = r.copy()
    rconj[..., 1:] *= -1  # unit quaternion -> inverse is conjugate
    return qmul(qmul(r, q), rconj)


# ---------------------------------------------------------------------------
# ST -> quaternion mapping
# ---------------------------------------------------------------------------

def map_features_to_quaternion(features: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Map a (N, 3) feature matrix + (N,) depth to quaternions (N, 4).

    Vector part = L2-normalized features (unit direction = relative profile);
    scalar part = depth scaled to [0, 1] (uncertainty / library size).
    Zero-norm rows map to (0,0,0,0).
    """
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    unit = features / safe
    unit[norms[:, 0] == 0] = 0.0

    depth = np.asarray(depth, dtype=float)
    dmax = depth.max() if depth.size and depth.max() > 0 else 1.0
    a = (depth / dmax).reshape(-1, 1)
    return np.concatenate([a, unit], axis=1)


def map_svd(counts, constant_sign: bool = True) -> np.ndarray:
    """Quaternion map from the top-4 singular vectors of UNNORMALIZED counts.

    Mirrors QSC ``mapToQuaternions``: project onto the top-4 right singular
    vectors of the *uncentered* count matrix; scalar = |projection on PC1|
    (a depth proxy), vector (i, j, k) = projections on PC2,3,4 normalized to
    unit length. `counts` is (N_locations, N_genes). Returns (N, 4).

    Uses randomized SVD (top-4, fixed seed) like QSC's rsvd, so it scales to
    Visium-HD-sized matrices; for small inputs it falls back to exact SVD.
    """
    from scipy.sparse import issparse  # lazy
    sparse = issparse(counts)
    small = (min(counts.shape) <= 64 or max(counts.shape) <= 2000)

    # Top-4 right singular vectors of the uncentered data; proj = X @ V.
    if small:
        X = counts.toarray().astype(float) if sparse else np.asarray(counts, dtype=float)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:4].T
        proj = X @ V
    else:
        # randomized SVD works on sparse directly -- never densify HD-scale matrices.
        from sklearn.utils.extmath import randomized_svd  # lazy
        X = counts if sparse else np.asarray(counts, dtype=float)
        _, _, Vt = randomized_svd(X, n_components=4, n_iter=5, random_state=0)
        V = Vt.T
        proj = np.asarray(X @ V)  # (sparse @ dense) -> dense (N, 4)

    if constant_sign:  # make first projected coordinate positive per component
        flip = np.where(proj[0] < 0, -1.0, 1.0)
        proj = proj * flip

    a = np.abs(proj[:, 0])
    vec = proj[:, 1:4]
    length = np.linalg.norm(vec, axis=1, keepdims=True)
    safe = np.where(length == 0, 1.0, length)
    unit = vec / safe
    unit[length[:, 0] == 0] = 0.0
    return np.concatenate([a.reshape(-1, 1), unit], axis=1)


def map_discriminant(counts, labels, n_components: int = 3, scalar: str = "depth"):
    """SUPERVISED quaternion map: project locations onto linear-discriminant axes.

    Unlike :func:`map_svd` (unsupervised top-variance PCs), this uses the top-3 LDA
    axes that maximally SEPARATE the provided cell-type labels, so the vector direction
    encodes cell-type identity by construction -- boundaries/transitions in the
    resulting field become cell-type boundaries. Mirrors map_svd's output contract:
    scalar = depth proxy in [0,1], vector (i,j,k) = unit LDA projection. Returns (N,4).

    `counts` is (N, G) expression (log-normalized recommended); `labels` is (N,)
    cell-type labels. To avoid leakage when the field is then used to *recover* those
    labels, fit on a subset and project all (see the driver). Needs >= n_components+1
    classes.
    """
    from scipy.sparse import issparse  # lazy
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # lazy
    X = counts.toarray() if issparse(counts) else np.asarray(counts, dtype=float)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    proj = lda.fit(X, np.asarray(labels)).transform(X)          # (N, n_components)
    if proj.shape[1] < 3:                                         # pad if <3 axes available
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
    vec = proj[:, :3]
    length = np.linalg.norm(vec, axis=1, keepdims=True)
    unit = vec / np.where(length == 0, 1.0, length)
    unit[length[:, 0] == 0] = 0.0
    if scalar == "depth":
        d = X.sum(1)
        a = np.clip(d / (np.percentile(d, 99) + 1e-9), 0, 1).reshape(-1, 1)
    else:
        a = np.clip(length / (np.percentile(length, 99) + 1e-9), 0, 1)
    return np.concatenate([a, unit], axis=1)


def false_color(q: np.ndarray) -> np.ndarray:
    """Map a quaternion matrix (..., 4) to RGBA in [0,1] (paper Section 3.1).

    (b, c, d) -> RGB via sigmoid(2x); a (depth) -> alpha (already in [0,1]).
    """
    b, c, d = q[..., 1], q[..., 2], q[..., 3]
    rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(np.stack([b, c, d], axis=-1), -50, 50)))
    alpha = np.clip(q[..., 0:1], 0.0, 1.0)
    return np.concatenate([rgb, alpha], axis=-1)


if __name__ == "__main__":
    # Round-trip self-test of the quaternion DFT.
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((16, 24, 4))
    back = iqdft2(qdft2(Q))
    err = np.abs(back - Q).max()
    print(f"QDFT round-trip max abs error: {err:.2e}")
    assert err < 1e-10, "round-trip failed"
    print("OK")

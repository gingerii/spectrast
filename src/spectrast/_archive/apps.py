"""Prototype applications of the quaternion ST model.

All operate on a quaternion grid (H, W, 4) and/or per-location quaternions (N, 4).
Grid operations are resolution-agnostic (Visium == Visium HD == binned Xenium,
just different grid sizes). Spectral ops reuse the validated qdft2/convolve_q core.
"""
from __future__ import annotations

import numpy as np

from .quaternion import (qdft2, iqdft2, rotate, rotation_quaternion,
                          convolve_q, make_mask, make_quaternion_kernel)

# ---------------------------------------------------------------------------
# #1 Rotations: transition / perturbation modeling (pointwise, no grid needed)
# ---------------------------------------------------------------------------

def mean_direction(quats: np.ndarray) -> np.ndarray:
    """Mean unit expression direction (3-vector) of a set of quaternions (N, 4)."""
    v = quats[:, 1:4].mean(axis=0)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation quaternion taking unit 3-vector u to unit 3-vector v."""
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    d = float(np.clip(u @ v, -1.0, 1.0))
    if d > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1 + 1e-9:  # opposite: rotate pi about any orthogonal axis
        axis = np.cross(u, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0, 1, 0])
        return rotation_quaternion(axis, np.pi)
    axis = np.cross(u, v)
    return rotation_quaternion(axis, np.arccos(d))


def learn_transition(q_control: np.ndarray, q_perturbed: np.ndarray) -> np.ndarray:
    """Rotation quaternion modeling the control->perturbed mean state transition.

    Uses only the mean expression direction of each group (no cell correspondence
    needed) -- the rotation between average states.
    """
    return rotation_between(mean_direction(q_control), mean_direction(q_perturbed))


def fit_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Optimal single rotation aligning paired direction vectors (Kabsch/Wahba).

    Requires correspondence (row i of v_from <-> row i of v_to). Returns a
    quaternion [w, x, y, z]. Solves the global rotation, unlike the per-pair
    minimal rotation, so it correctly recovers a true global transition.
    """
    from scipy.spatial.transform import Rotation  # lazy
    R, _ = Rotation.align_vectors(v_to[:, 1:4] if v_to.ndim == 2 and v_to.shape[1] == 4 else v_to,
                                  v_from[:, 1:4] if v_from.ndim == 2 and v_from.shape[1] == 4 else v_from)
    x, y, z, w = R.as_quat()
    return np.array([w, x, y, z])


def transition_consistency(q_control: np.ndarray, q_perturbed: np.ndarray) -> float:
    """How well a single rotation explains a perturbation (1=perfect), paired data.

    Fits the optimal global rotation (Kabsch) on matched control/perturbed direction
    vectors, then returns the mean cosine between rotated-control and perturbed
    directions. Requires equal-length, index-matched inputs.
    """
    if len(q_control) != len(q_perturbed):
        raise ValueError("transition_consistency needs index-matched pairs")
    r = fit_rotation(q_control, q_perturbed)
    rotated = rotate(q_control, r)[:, 1:4]
    a = rotated / (np.linalg.norm(rotated, axis=1, keepdims=True) + 1e-12)
    b = q_perturbed[:, 1:4]
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.clip((a * b).sum(1), -1, 1).mean())


def invariance_map(grid: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-location length of (rotated - original) vector part (paper's diff-length).

    ~0 where the local profile is aligned with the rotation axis (invariant),
    ~1 where orthogonal (maximally changed by the transition).
    """
    diff = rotate(grid, r) - grid
    return np.linalg.norm(diff[..., 1:4], axis=-1)


# ---------------------------------------------------------------------------
# #2 Registration via quaternion phase correlation (grid)
# ---------------------------------------------------------------------------

def phase_correlate(ref: np.ndarray, mov: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Estimate the integer (drow, dcol) shift aligning `mov` to `ref`.

    Quaternion phase correlation: combine the per-channel cross-power spectra of the
    two quaternion grids, inverse-FFT, and take the peak. Returns (shift, corr map).
    The returned (drow, dcol) is the shift that aligns `mov` back onto `ref`.
    """
    # cross-power across all 4 quaternion channels (complex per channel)
    cross = np.zeros(ref.shape[:2], dtype=complex)
    for ch in range(4):
        A = np.fft.fft2(ref[..., ch]); B = np.fft.fft2(mov[..., ch])
        R = A * np.conj(B)
        cross += R
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = ref.shape[:2]
    drow = peak[0] - h if peak[0] > h // 2 else peak[0]
    dcol = peak[1] - w if peak[1] > w // 2 else peak[1]
    return (int(drow), int(dcol)), corr


# ---------------------------------------------------------------------------
# #3 FFT spatial autocorrelation (Moran's I) (grid)
# ---------------------------------------------------------------------------

def _neighbor_kernel(radius: int = 1) -> np.ndarray:
    k = np.ones((2 * radius + 1, 2 * radius + 1))
    k[radius, radius] = 0.0
    return k


def morans_i_fft(field: np.ndarray, mask: np.ndarray | None = None,
                 radius: int = 1) -> float:
    """Moran's I of a scalar field on a grid, via FFT convolution.

    Numerator z'^T W z' is computed as sum(z' * conv(z', kernel)) using FFT, so the
    cost is O(N log N) and independent of neighborhood radius. `mask` selects valid
    (non-empty) cells.
    """
    from scipy.signal import fftconvolve  # lazy
    if mask is None:
        mask = np.ones(field.shape, dtype=bool)
    m = mask.astype(float)
    zbar = field[mask].mean()
    z = (field - zbar) * m
    k = _neighbor_kernel(radius)
    Wz = fftconvolve(z, k, mode="same")
    S0 = (fftconvolve(m, k, mode="same") * m).sum()  # sum of weights over valid pairs
    N = m.sum()
    num = (z * Wz).sum()
    den = (z * z).sum()
    return float((N / S0) * (num / den)) if den > 0 and S0 > 0 else np.nan


def morans_i_direct(field: np.ndarray, mask: np.ndarray | None = None,
                    radius: int = 1) -> float:
    """Reference Moran's I via explicit neighbor shifts (no FFT).

    Vectorized O(N*k) baseline (k = #offsets ~ radius^2): the realistic competitor
    to the FFT version. Cost grows with neighborhood size; the FFT version does not.
    """
    if mask is None:
        mask = np.ones(field.shape, dtype=bool)
    m = mask.astype(float)
    zbar = field[mask].mean()
    z = (field - zbar) * m
    r = radius
    num = 0.0; S0 = 0.0
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            zs = np.roll(np.roll(z, di, 0), dj, 1)
            ms = np.roll(np.roll(m, di, 0), dj, 1)
            # zero out wraparound rows/cols
            valid = np.ones_like(m)
            if di > 0: valid[:di, :] = 0
            elif di < 0: valid[di:, :] = 0
            if dj > 0: valid[:, :dj] = 0
            elif dj < 0: valid[:, dj:] = 0
            num += (z * zs * valid).sum()
            S0 += (m * ms * valid).sum()
    den = (z * z).sum(); N = m.sum()
    return float((N / S0) * (num / den)) if den > 0 and S0 > 0 else np.nan


# ---------------------------------------------------------------------------
# #4 Template / motif matching via FFT cross-correlation (grid)
# ---------------------------------------------------------------------------

def template_match(grid: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Locate `template` (h, w, 4) within `grid` via normalized cross-correlation.

    Sums FFT-based NCC (skimage.match_template) over the vector channels, which is
    invariant to local energy (unlike raw correlation). Returns the (row, col) of
    the best-match CENTER and the score map (same size as grid).
    """
    from skimage.feature import match_template  # lazy
    score = np.zeros((grid.shape[0], grid.shape[1]))
    for ch in (1, 2, 3):  # vector channels = relative expression profile
        score += match_template(grid[..., ch], template[..., ch], pad_input=True)
    peak = np.unravel_index(np.argmax(score), score.shape)
    return (int(peak[0]), int(peak[1])), score


# ---------------------------------------------------------------------------
# #5 Spectral compression / denoising (grid)
# ---------------------------------------------------------------------------

def spectral_compress(grid: np.ndarray, keep_frac: float = 0.1) -> tuple[np.ndarray, float]:
    """Keep a centered low-frequency fraction of the QDFT, inverse-transform.

    Returns (reconstruction, fraction of coefficients kept). A lossy compression /
    denoiser for the joint signal. The scalar (depth) channel is normalized to the
    vector scale first, otherwise its large magnitude leaks into the vector channels
    through the shared Cayley-Dickson complex during truncation.
    """
    grid = grid.copy()
    amax = max(np.abs(grid[..., 0]).max(), 1e-12)
    grid[..., 0] /= amax  # scale depth into [~0,1], comparable to unit vector parts
    F = qdft2(grid)
    Fs = np.fft.fftshift(F, axes=(0, 1))
    H, W = grid.shape[:2]
    kh = max(1, int(H * np.sqrt(keep_frac))); kw = max(1, int(W * np.sqrt(keep_frac)))
    r0, c0 = (H - kh) // 2, (W - kw) // 2
    mask = np.zeros((H, W, 1)); mask[r0:r0 + kh, c0:c0 + kw] = 1.0
    Fs *= mask
    recon = iqdft2(np.fft.ifftshift(Fs, axes=(0, 1)))
    return recon, float(mask.sum() / (H * W))


# ---------------------------------------------------------------------------
# #6 Boundary / interface detection (grid)  +  #7 uncertainty (pointwise)
# ---------------------------------------------------------------------------

def boundary_map(grid: np.ndarray) -> np.ndarray:
    """Edge magnitude per location via quaternion edge-detection convolution."""
    M = make_mask(grid.shape[0], grid.shape[1], make_quaternion_kernel("edge"))
    edges = convolve_q(grid, M)
    return np.linalg.norm(edges[..., 1:4], axis=-1)


def uncertainty(grid_or_quats: np.ndarray) -> np.ndarray:
    """Expression-state uncertainty = inverse of the scalar (depth) channel.

    High where library size / confidence is low. Works on a grid (..., 4) or (N, 4).
    """
    a = grid_or_quats[..., 0]
    amax = a.max() if a.max() > 0 else 1.0
    return 1.0 - a / amax

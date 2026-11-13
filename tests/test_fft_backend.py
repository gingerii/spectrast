"""Backend-shim guardrail: the pluggable FFT backend must reproduce
the previous ``numpy.fft`` path on the CPU backend. Two distinct guarantees, tested separately:

  * WORKER-COUNT INVARIANCE is EXACT — ``workers=-1`` (all cores) and ``workers=1`` give
    bit-identical output. Threading (the whole point of Phase 1) never changes a bit; scipy's
    pocketfft splits the gene batch, not the arithmetic.
  * The numpy->scipy library swap is EXACT for the forward real transforms (``rfft2``/``irfft2``,
    Δ=0) and identical to machine epsilon for the complex transforms (``fft2``/``ifft2`` differ
    by ~1e-14 absolute — two independent pocketfft builds disagreeing in the last bit, ~12 orders
    below the float32-permutation bug that mattered). We assert a tight double-precision tolerance
    for those, not bitwise equality, because bitwise numpy==scipy is not achievable and not the
    property we need — numerical identity is.

Also locks the backend-selection / graceful-degradation contract so a missing GPU never hard-fails.
Run: ``pytest tests/test_fft_backend.py -q``.
"""
import numpy as np
import pytest

from spectrast import _fft
from spectrast.spatial import morans_i_multiscale, gearys_c_multiscale, focality_scalogram


def _toy(seed=0, H=48, W=52, G=7):
    rng = np.random.default_rng(seed)
    grid = rng.standard_normal((H, W, G))
    mask = rng.random((H, W)) > 0.15
    grid *= mask[..., None]
    return grid, mask


# ---------------------------------------------------------------------------
# The four transforms: shim (scipy, threaded) == numpy.fft, bit-for-bit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_transforms_match_numpy(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((37, 41, 5))
    s = (64, 64)
    assert _fft.get_backend() == "cpu"
    # forward real transforms: EXACTLY bit-identical to numpy
    fr = _fft.rfft2(x, s=s, axes=(0, 1))
    assert np.array_equal(fr, np.fft.rfft2(x, s=s, axes=(0, 1)))
    back = _fft.irfft2(fr, s=s, axes=(0, 1))
    assert np.array_equal(back, np.fft.irfft2(np.fft.rfft2(x, s=s, axes=(0, 1)), s=s, axes=(0, 1)))
    # complex transforms: identical to machine epsilon (~1e-14), not bitwise (independent pocketfft builds)
    fc = _fft.fft2(x, s=s, axes=(0, 1))
    assert np.allclose(fc, np.fft.fft2(x, s=s, axes=(0, 1)), rtol=1e-11, atol=1e-11)
    ic = _fft.ifft2(fc, s=s, axes=(0, 1))
    assert np.allclose(ic, np.fft.ifft2(np.fft.fft2(x, s=s, axes=(0, 1)), s=s, axes=(0, 1)),
                       rtol=1e-11, atol=1e-11)


def test_worker_count_does_not_change_bits():
    """workers=-1 (all cores) vs workers=1 must agree to the bit — pocketfft splits the batch,
    not the arithmetic."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((60, 60, 12))
    prev = _fft.get_workers()
    try:
        _fft.set_workers(-1)
        a = _fft.rfft2(x, s=(72, 72), axes=(0, 1))
        _fft.set_workers(1)
        b = _fft.rfft2(x, s=(72, 72), axes=(0, 1))
    finally:
        _fft.set_workers(prev)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Real scalograms are unchanged by the routing (guards the actual hot paths).
# ---------------------------------------------------------------------------
def test_scalograms_unchanged():
    grid, mask = _toy()
    radii = [2, 4, 8, 12]
    mi = morans_i_multiscale(grid, mask, radii)
    gc = gearys_c_multiscale(grid, mask, radii)
    fs = focality_scalogram(grid, mask, radii)
    assert mi.shape == (len(radii), grid.shape[2])
    # reference: recompute Moran scalogram with the raw numpy.fft (pre-shim path). The scalogram
    # rides on rfft2/irfft2 (bit-exact vs numpy) so the residual is pure double round-off (~1e-14).
    ref = _ref_moran_multiscale(grid, mask, radii)
    assert np.allclose(mi, ref, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert np.isfinite(gc).any() and np.isfinite(fs).any()


def _ref_moran_multiscale(grid, mask, radii):
    """Byte-for-byte copy of morans_i_multiscale using raw numpy.fft (the pre-shim path)."""
    from scipy.fft import next_fast_len
    from spectrast.spatial import neighbor_kernel
    if grid.ndim == 2:
        grid = grid[..., None]
    radii = list(radii); rmax = max(radii)
    m = mask.astype(float); msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum
    z = (grid - zbar) * m[..., None]
    den = (z * z).sum((0, 1))
    H, W, G = z.shape
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fz = np.fft.rfft2(z, s=(fh, fw), axes=(0, 1)); Fm = np.fft.rfft2(m, s=(fh, fw))
    out = np.empty((len(radii), G))
    for si, r in enumerate(radii):
        k = neighbor_kernel(r, "disc"); Fk = np.fft.rfft2(k, s=(fh, fw))
        Wz = np.fft.irfft2(Fz * Fk[:, :, None], s=(fh, fw), axes=(0, 1))
        Wm = np.fft.irfft2(Fm * Fk, s=(fh, fw))
        off = (k.shape[0] - 1) // 2
        Wz = Wz[off:off + H, off:off + W]; Wm = Wm[off:off + H, off:off + W]
        invW = np.where(Wm > 1e-12, 1.0 / np.maximum(Wm, 1e-12), 0.0)[..., None]   # ROW-STANDARDIZED (B1 fix)
        num = (z * (Wz * invW)).sum((0, 1))
        out[si] = num / np.where(den == 0, np.nan, den)
    return out


# ---------------------------------------------------------------------------
# Backend selection / graceful degradation.
# ---------------------------------------------------------------------------
def test_set_backend_roundtrip_and_degradation():
    assert _fft.set_backend("cpu") == "cpu"
    # a GPU backend that isn't present must WARN and fall back to cpu, never raise.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolved = _fft.set_backend("cuda")
    assert resolved in ("cpu", "cuda")            # cpu on a machine w/o CUDA (this CI host)
    # auto never fails and always resolves to a valid backend
    assert _fft.set_backend("auto") in ("cpu", "cuda", "mps")
    _fft.set_backend("cpu")                        # restore default for the rest of the suite


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        _fft.set_backend("opencl")
    _fft.set_backend("cpu")


# ---------------------------------------------------------------------------
# Phase 4: the device (torch) scalogram matches the float64 CPU path (float32 tol).
# Runs on the torch CPU device so it's hardware-independent; skipped if torch is absent.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["moran", "geary", "focality"])
def test_torch_scalogram_matches_cpu_float32(name):
    torch = pytest.importorskip("torch")
    from spectrast.spatial import (gearys_c_multiscale, _morans_i_multiscale_torch,
                             _gearys_c_multiscale_torch, _focality_scalogram_torch)
    cpu_fn, torch_fn = {
        "moran": (morans_i_multiscale, _morans_i_multiscale_torch),
        "geary": (gearys_c_multiscale, _gearys_c_multiscale_torch),
        "focality": (focality_scalogram, _focality_scalogram_torch),
    }[name]
    grid, mask = _toy(seed=5, H=64, W=60, G=11)
    radii = [3, 6, 10, 14]
    ref = cpu_fn(grid, mask, radii)                                   # float64 CPU
    got = torch_fn(grid, mask, radii, dev=torch.device("cpu"))        # float32 torch
    m = np.isfinite(ref) & np.isfinite(got)
    assert got.shape == ref.shape
    # float32 tolerance — well below the ~1e-3 level that would perturb a statistic
    assert np.max(np.abs(got[m] - ref[m])) < 1e-4
    assert np.corrcoef(got[m], ref[m])[0, 1] > 0.9999


# ---------------------------------------------------------------------------
# GPU-device transforms survive a prior narrower transform of the same shape (found 2026-11-12).
# torch 2.10's MPS FFT caches a plan per transformed shape and reuses it across trailing-batch widths:
# after one (H, W, 1) transform over dim=(0, 1), a later (H, W, K) one at the same shape came back
# correct in column 0 only. In the package, `autotune_grid` probes each radius's lattice with one
# field before the 256-gene chunk is convolved there, so `gr.scale_spectrum` on MPS returned noise for
# genes 1-255 at every radius whose lattice the bug hits. The torch-CPU device above never shows it,
# so these run on the real GPU device and skip without one. The bug needs a transformed size with a
# prime factor above 11: the direct device scalograms pad to scipy's next_fast_len and were measured
# immune (0 of 432 shapes), so their test below guards that padding as much as the shim fix.
# ---------------------------------------------------------------------------
def _gpu_backend():
    torch = pytest.importorskip("torch")
    for name, ok in (("cuda", torch.cuda.is_available()),
                     ("mps", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())):
        if ok:
            prev = _fft.get_backend()
            got = _fft.set_backend(name)
            _fft.set_backend(prev)
            if got == name:
                return name
    pytest.skip("no usable GPU FFT backend")


@pytest.mark.parametrize("fn", ["rfft2", "irfft2"])
def test_gpu_transform_after_a_single_column_one_of_the_same_shape(fn):
    gpu = _gpu_backend()
    from scipy import fft as sfft
    H, W, s = 93, 84, (113, 104)                    # a shape the MPS plan cache was measured to corrupt
    rng = np.random.default_rng(0)
    x1, xk = rng.standard_normal((H, W, 1)), rng.standard_normal((H, W, 16))
    if fn == "irfft2":
        x1, xk = sfft.rfft2(x1, s=s, axes=(0, 1)), sfft.rfft2(xk, s=s, axes=(0, 1))
    ref = getattr(sfft, fn)(xk, s=s, axes=(0, 1))
    prev = _fft.get_backend()
    try:
        _fft.set_backend(gpu)
        getattr(_fft, fn)(x1, s=s, axes=(0, 1))     # the narrow transform that used to fix the plan
        got = getattr(_fft, fn)(xk, s=s, axes=(0, 1))
    finally:
        _fft.set_backend(prev)
    rel = np.abs(got - ref).max((0, 1)) / np.abs(ref).max((0, 1))
    assert np.all(rel < 1e-4), f"columns wrong on {gpu}: {np.flatnonzero(rel >= 1e-4)}"


@pytest.mark.parametrize("name", ["moran", "geary", "focality"])
def test_gpu_scalogram_after_a_single_gene_call_of_the_same_shape(name):
    gpu = _gpu_backend()
    import torch
    from spectrast.spatial import _morans_i_multiscale_torch, _gearys_c_multiscale_torch, _focality_scalogram_torch
    cpu_fn, torch_fn = {
        "moran": (morans_i_multiscale, _morans_i_multiscale_torch),
        "geary": (gearys_c_multiscale, _gearys_c_multiscale_torch),
        "focality": (focality_scalogram, _focality_scalogram_torch),
    }[name]
    grid, mask = _toy(seed=5, H=93, W=84, G=16)
    radii = [3, 6, 10]
    ref = cpu_fn(grid, mask, radii)
    dev = torch.device(gpu)
    torch_fn(grid[:, :, :1], mask, radii, dev=dev)  # one gene first, at the same lattice
    got = torch_fn(grid, mask, radii, dev=dev)
    m = np.isfinite(ref) & np.isfinite(got)
    assert np.max(np.abs(got[m] - ref[m])) < 1e-3


def test_gpu_scale_spectrum_matches_cpu():
    """End to end, the way it was found: autotuned bins, one 16-gene chunk, GPU vs CPU."""
    gpu = _gpu_backend()
    from spectrast.spatial_sc import morans_scalogram
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 2000, (6000, 2))
    X = np.stack([np.sin(xy[:, 0] / (40 + 15 * g)) + np.cos(xy[:, 1] / (55 + 10 * g)) + 0.3 * rng.standard_normal(6000)
                  for g in range(16)], 1)
    radii = [30.0, 60.0, 120.0, 240.0]
    prev = _fft.get_backend()
    try:
        _fft.set_backend("cpu"); ref = morans_scalogram(xy, X, radii)
        _fft.set_backend(gpu); got = morans_scalogram(xy, X, radii)
    finally:
        _fft.set_backend(prev)
    assert np.max(np.abs(got - ref)) < 1e-3


@pytest.mark.parametrize("kind", ["disc", "gaussian"])
def test_gpu_moran_treats_round_off_as_no_neighbour(kind):
    """In float32 a bin with no occupied neighbour gets a degree of ~1e-7 from FFT round-off instead of 0.
    The device Moran path gated 1/W at 1e-12, so those bins were divided by ~1e-7 and their noise entered
    the statistic (found 2026-11-12 on a 3.8 um lattice of a Xenium section: Moran's I off by up to 0.40).
    A sparse lattice puts many bins out of reach of any neighbour at small radii."""
    gpu = _gpu_backend()
    import torch
    from spectrast.spatial import _morans_i_multiscale_torch
    rng = np.random.default_rng(7)
    H, W, G = 140, 130, 12
    mask = rng.random((H, W)) < 0.04                     # ~4% occupancy: many isolated bins at r = 1-2
    grid = rng.standard_normal((H, W, G)) * mask[..., None] + 5.0 * mask[..., None]
    radii = [1, 2, 3, 6]
    ref = morans_i_multiscale(grid, mask, radii, kind=kind)
    got = _morans_i_multiscale_torch(grid, mask, radii, kind=kind, dev=torch.device(gpu))
    m = np.isfinite(ref) & np.isfinite(got)
    assert m.all(), "a curve went non-finite"
    assert np.max(np.abs(got - ref)) < 1e-3, f"max |GPU - CPU| = {np.max(np.abs(got - ref)):.3g}"

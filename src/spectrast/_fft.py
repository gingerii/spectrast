"""Pluggable FFT backend for the batched 2D transforms in the spatial engine.

Every hot path (multiscale scalograms, spectral densities, permutation nulls) drives its
compute through a small handful of batched 2D transforms — ``rfft2 / irfft2 / fft2 / ifft2``
over an ``(H, W, G)`` field stack (axes ``(0, 1)``). Routing those four calls through this
shim lets the whole toolkit swap compute device in one place:

  * ``"cpu"``  — ``scipy.fft`` with ``workers`` threading (default, universal). Bit-for-bit
                 identical to the previous ``numpy.fft`` path (both use pocketfft); worker count
                 does not change the result.
  * ``"cuda"`` — ``torch.fft`` on an NVIDIA GPU (optional; requires ``spectrast[cuda]``).
  * ``"mps"``  — ``torch.fft`` on Apple Silicon (optional; requires ``spectrast[mps]`` AND a native
                 MPS-FFT).

Design rules:
  * torch is NEVER imported at module load — the base package works with scipy only.
  * ``fftfreq`` / ``fftshift`` are cheap host-side index ops and are intentionally NOT routed
    here; call sites keep using ``numpy.fft`` for those.
  * GPU backends convert array<->tensor and move to device at the call boundary, so the rest
    of the code stays pure-numpy. The scalogram is responsible for pushing its field stack to
    the device once (via ``to_device`` / ``from_device``) and looping radii there — this shim
    only guarantees a correct single transform.
  * Graceful degradation: an unavailable GPU backend (or a runtime failure) falls back to the
    threaded CPU path with an explicit warning — never a silent CPU-masquerading-as-GPU, never
    a hard crash.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------
_VALID = ("cpu", "cuda", "mps")
_BACKEND = "cpu"          # active backend (resolved, never "auto")
_WORKERS = -1             # scipy.fft worker count for the cpu path (-1 = all cores)
_TORCH_DEVICE = None      # cached torch.device for the active GPU backend
_MPS_FFT_OK: bool | None = None   # tri-state probe cache for native MPS FFT support
_MPS_FFT_ACC: bool | None = None  # tri-state cache for MPS FFT accuracy at scale
_MPS_ACC_THRESH = 1e-3    # max relative-L2 error of MPS float32 rfft2 vs float64 truth (1024^2)
_MPS_ACC_THRESH_USED: float | None = None   # threshold the cached _MPS_FFT_ACC was computed at (m27: re-probe on change)


def set_workers(n: int) -> None:
    """Set the scipy.fft worker count for the CPU backend (-1 = all cores)."""
    global _WORKERS
    _WORKERS = int(n)


def get_workers() -> int:
    return _WORKERS


def get_backend() -> str:
    """Return the active, resolved backend ('cpu' | 'cuda' | 'mps')."""
    return _BACKEND


# ---------------------------------------------------------------------------
# Capability probes (lazy; torch imported only when a GPU backend is requested)
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _mps_fft_native() -> bool:
    """True iff torch.fft.rfft2 runs natively on MPS (no CPU fallback).

    Cached. Probes with PYTORCH_ENABLE_MPS_FALLBACK disabled inside the call so a fallback
    RAISES here instead of silently masquerading as GPU.
    """
    global _MPS_FFT_OK
    if _MPS_FFT_OK is not None:
        return _MPS_FFT_OK
    if not _mps_available():
        _MPS_FFT_OK = False
        return False
    prev = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        with warnings.catch_warnings():                        # scope the filter — don't mutate global state
            warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
            x = torch.randn(16, 16, 2, device="mps")
            y = torch.fft.rfft2(x, dim=(0, 1))
            torch.fft.irfft2(y, s=(16, 16), dim=(0, 1))
        _MPS_FFT_OK = True
    except Exception:
        _MPS_FFT_OK = False
    finally:
        if prev is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prev
    return _MPS_FFT_OK


def _mps_fft_accurate() -> bool:
    """True iff MPS float32 rfft2 is accurate enough at scale to trust for spatial statistics.

    torch's MPS FFT is known to LOSE PRECISION as the transform grows (a reported Metal-backend
    limitation), and MPS has no float64 to fall back on. So beyond the native-FFT probe we gate on
    accuracy: relative-L2 error of a 1024x1024 MPS rfft2 vs the float64 truth must stay below
    ``_MPS_ACC_THRESH``. Generic float32 sits ~2-3 orders below that; MPS-specific degradation that
    would corrupt a statistic trips the gate. Cached. Errs to False on any failure.
    """
    global _MPS_FFT_ACC, _MPS_ACC_THRESH_USED
    if _MPS_FFT_ACC is not None and _MPS_ACC_THRESH_USED == _MPS_ACC_THRESH:
        return _MPS_FFT_ACC                          # m27: re-probe if the caller changed _MPS_ACC_THRESH
    _MPS_ACC_THRESH_USED = _MPS_ACC_THRESH
    if not _mps_fft_native():
        _MPS_FFT_ACC = False
        return False
    try:
        import torch
        from scipy import fft as sfft
        with warnings.catch_warnings():                                # scope the filter — no global mutation
            warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
            rng = np.random.default_rng(0)
            x = rng.standard_normal((1024, 1024)).astype(np.float32)
            ref = sfft.rfft2(x.astype(np.float64))                     # float64 truth
            got = torch.fft.rfft2(torch.from_numpy(x).to("mps")).detach().to("cpu").numpy()
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        _MPS_FFT_ACC = rel < _MPS_ACC_THRESH
        if not _MPS_FFT_ACC:
            warnings.warn(f"spectrast: MPS FFT accuracy check FAILED (relative-L2 {rel:.2e} >= "
                          f"{_MPS_ACC_THRESH:.0e} at 1024^2); MPS float32 FFT is too imprecise on this "
                          f"build — routing to CPU. Set spectrast._fft._MPS_ACC_THRESH to override.")
    except Exception:
        _MPS_FFT_ACC = False
    return _MPS_FFT_ACC


def _resolve(backend: str) -> str:
    """Resolve a requested backend to an available one, warning + degrading if needed."""
    if backend == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available() and _mps_fft_native() and _mps_fft_accurate():
            return "mps"
        return "cpu"
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        if _cuda_available():
            return "cuda"
        warnings.warn("spectrast: CUDA backend requested but no CUDA device available; using threaded CPU.")
        return "cpu"
    if backend == "mps":
        if not _mps_available():
            warnings.warn("spectrast: MPS backend requested but no MPS device available; using threaded CPU.")
            return "cpu"
        if not _mps_fft_native():
            warnings.warn("spectrast: MPS backend requested but torch.fft falls back to CPU on this build; "
                          "using threaded CPU directly (avoids device round-trips).")
            return "cpu"
        if not _mps_fft_accurate():
            return "cpu"                      # _mps_fft_accurate already warned with the measured error
        return "mps"
    raise ValueError(f"unknown backend {backend!r}; expected one of {('auto',) + _VALID}")


def set_backend(backend: str = "auto") -> str:
    """Select the compute backend: 'auto' | 'cpu' | 'cuda' | 'mps'.

    'auto' picks cuda > mps(native FFT) > cpu. A requested-but-unavailable GPU backend warns
    and degrades to threaded CPU. Returns the resolved backend name.
    """
    global _BACKEND, _TORCH_DEVICE
    resolved = _resolve(backend)
    _BACKEND = resolved
    _TORCH_DEVICE = None
    if resolved in ("cuda", "mps"):
        import torch
        _TORCH_DEVICE = torch.device(resolved)
    return resolved


# ---------------------------------------------------------------------------
# Device transfer (GPU backends only; identity on CPU)
# ---------------------------------------------------------------------------
def to_device(a: np.ndarray):
    """Move a host array onto the active device. Identity (returns ndarray) on the CPU backend.

    On GPU backends the returned object is a torch tensor (float32/complex64) — callers that use
    ``to_device`` must run the whole transform chain through this shim and finish with
    ``from_device``. Real inputs go to float32, complex to complex64 (GPU FFT precision).
    """
    if _BACKEND == "cpu":
        return a
    import torch
    arr = np.asarray(a)
    if np.iscomplexobj(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.complex64))
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return t.to(_TORCH_DEVICE)


def from_device(t) -> np.ndarray:
    """Bring a device tensor back to a host numpy array. Identity on the CPU backend."""
    if _BACKEND == "cpu":
        return np.asarray(t)
    return t.detach().to("cpu").numpy()


def _is_tensor(x) -> bool:
    if _BACKEND == "cpu":
        return False
    import torch
    return isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# The four batched transforms (np.fft-compatible signatures: s=, axes=)
# ---------------------------------------------------------------------------
# On CPU: thin wrappers over scipy.fft with worker threading (bit-identical to np.fft).
# On GPU: torch.fft, translating axes->dim. Inputs may be host arrays (moved on the fly) or
#         tensors already on-device (kept there — the scalogram pushes its stack once).

def _torch_call(fn_name, x, s, axes):
    """Run one torch.fft transform, always over the LAST axes of a contiguous tensor.

    torch 2.10's MPS FFT reuses a cached plan across trailing-batch widths when the transformed axes
    are not the last ones and a transformed size has a prime factor above 11: after one ``(H, W, 1)``
    transform over ``dim=(0, 1)``, a later ``(H, W, K)`` one at the same shape comes back correct in
    column 0 only, silently. Measured 2026-11-12 on all four transforms: 48 of 52 such shapes wrong in
    that layout, 0 of 52 once the transformed axes are moved last (and 0 of 432 scipy
    ``next_fast_len`` shapes in either layout). It reached users through ``adaptive._conv2d``, whose
    padded size ``H + k - 1`` is arbitrary, after ``autotune_grid`` probed each lattice with one field.
    Moving the axes costs one device copy; the result is identical on every backend.
    """
    import torch
    fn = getattr(torch.fft, fn_name)
    with warnings.catch_warnings():                    # torch's MPS FFT emits a benign internal out-resize
        warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
        t = x if _is_tensor(x) else to_device(x)
        nd = t.ndim
        dims = tuple(a % nd for a in axes)
        last = tuple(range(nd - len(dims), nd))
        if dims == last:
            out = fn(t, s=s, dim=dims)
        else:                                          # s stays paired with dims: dims[i] -> last[i]
            out = fn(torch.movedim(t, dims, last).contiguous(), s=s, dim=last)
            out = torch.movedim(out, last, dims)
        return out if _is_tensor(x) else from_device(out)   # device caller keeps it; host caller gets numpy


def rfft2(x, s=None, axes=(-2, -1)):
    if _BACKEND == "cpu":
        from scipy import fft as sfft
        return sfft.rfft2(x, s=s, axes=axes, workers=_WORKERS)
    return _torch_call("rfft2", x, s, axes)


def irfft2(x, s=None, axes=(-2, -1)):
    if _BACKEND == "cpu":
        from scipy import fft as sfft
        return sfft.irfft2(x, s=s, axes=axes, workers=_WORKERS)
    return _torch_call("irfft2", x, s, axes)


def fft2(x, s=None, axes=(-2, -1)):
    if _BACKEND == "cpu":
        from scipy import fft as sfft
        return sfft.fft2(x, s=s, axes=axes, workers=_WORKERS)
    return _torch_call("fft2", x, s, axes)


def ifft2(x, s=None, axes=(-2, -1)):
    if _BACKEND == "cpu":
        from scipy import fft as sfft
        return sfft.ifft2(x, s=s, axes=axes, workers=_WORKERS)
    return _torch_call("ifft2", x, s, axes)


# ---------------------------------------------------------------------------
# High-precision (host float64 scipy) transforms — used REGARDLESS of the active backend.
# ---------------------------------------------------------------------------
# For cancellation-heavy SINGLE-SHOT statistics — variogram / cross-spectral / periodogram / lag maps and
# the single-radius neighbour convolutions — GPU float32 (complex64) can lose precision catastrophically:
# these form differences of large O(N·var) terms where the true value is tiny (short-lag semivariance, near-
# zero cross-spectral bins), and MPS has no float64 fallback and no accuracy gate on the full-complex path.
# These functions are NOT the hot batched multiscale loops (which stay on the active backend, validated to a
# float32 tolerance), so we run them on host float64 for correctness. Threaded via the CPU worker count.

def rfft2_hp(x, s=None, axes=(-2, -1)):
    from scipy import fft as sfft
    return sfft.rfft2(x, s=s, axes=axes, workers=_WORKERS)


def irfft2_hp(x, s=None, axes=(-2, -1)):
    from scipy import fft as sfft
    return sfft.irfft2(x, s=s, axes=axes, workers=_WORKERS)


def fft2_hp(x, s=None, axes=(-2, -1)):
    from scipy import fft as sfft
    return sfft.fft2(x, s=s, axes=axes, workers=_WORKERS)


def ifft2_hp(x, s=None, axes=(-2, -1)):
    from scipy import fft as sfft
    return sfft.ifft2(x, s=s, axes=axes, workers=_WORKERS)


# ---------------------------------------------------------------------------
# Worker-threading context for the CPU backend
# ---------------------------------------------------------------------------
class threaded:
    """Context manager that threads scipy.fft (and scipy.signal.fftconvolve, which honors it)
    across all cores for the CPU backend. No-op for GPU backends (torch threads internally).

    Usage::

        with _fft.threaded():
            out = morans_i_multiscale(...)   # every FFT in the loop runs multi-core
    """
    def __enter__(self):
        self._ctx = None
        if _BACKEND == "cpu":
            from scipy import fft as sfft
            self._ctx = sfft.set_workers(_WORKERS)
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            return self._ctx.__exit__(*exc)
        return False

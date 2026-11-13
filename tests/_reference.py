"""Comparing a result with reference values that were computed once and frozen in the tests."""
import numpy as np

RTOL = 1e-12


def assert_close_to_reference(got, ref, err_msg="", rtol=RTOL):
    """``got`` equals the frozen ``ref`` up to floating-point rounding.

    Reference values were written on one machine. FFT and reduction kernels round differently on
    other CPUs and operating systems, so bit equality holds only where the reference was made; across
    machines the differences are of order 1e-15 relative. NaN and infinity positions must match
    exactly, and finite values must agree to ``rtol`` relative to the largest magnitude in ``ref``.

    Bit-for-bit claims belong in comparisons made within one run (two code paths, thread counts,
    cold and warm caches), which hold on any machine.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape, f"{err_msg}: shape {got.shape} != reference {ref.shape}"
    finite = np.isfinite(ref)
    scale = float(np.max(np.abs(ref[finite]))) if finite.any() else 0.0
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol * scale, equal_nan=True, err_msg=err_msg)

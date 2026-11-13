"""How an organization curve is reduced to ONE characteristic length in micrometres.

There is exactly one such reduction in the toolkit and this module holds it, because the fingerprint
and the differential must mean the same thing by "characteristic scale" or a Delta-scale is a
difference of two different quantities.

THE DEFAULT IS THE HALF-MAXIMUM, AS OF 2026-08-18. It replaces the positive-scalogram centroid, which
the toolkit reported until now. The centroid was not wrong so much as insensitive, and three
independent measurements say so:

  1. AGREEMENT WITH AN INDEPENDENT FIT. Rank correlation against a fitted decay, by amplitude quartile,
     on four sections across two tissue types: half-max beats the centroid on every statistic, every
     section and every quartile. Moran on merfish_sagittal +0.845 against +0.700 overall and +0.915
     against +0.838 among gated genes; xenium_brain_5k +0.754/+0.893 against +0.598/+0.827;
     atera_breast +0.839/+0.851 against +0.590/+0.595. Geary and LOSH separate by more, not less.
  2. SPREAD. Among organized genes the centroid compresses the population into about a fifth of an
     octave where a fitted decay spreads it over three quarters.
  3. SEPARABILITY FROM NOISE, which is the serious one. Run the centroid on a coordinate scramble and
     the null piles up at 359 um against a radius-grid centre of mass of 370 um -- and 99% of GATED
     genes fall inside that null band. On that section the centroid was substantially reporting where
     we chose to put the radii. The half-max null instead hugs the window floor, because a flat noisy
     curve crosses half of its own range almost at once, so the organized genes sit clear of it.

WHY NOT THE FITTED SCALE, WHICH SPREADS GENES FURTHEST. Because the family is wrong: fitting the
stretched exponential to strongly organized genes gives beta < 1 for 69-99% of them, so a single decay
length misdescribes the curve it came from (see :mod:`spectrast.curvefit`, which ships that fit as a
CHECK on this reduction). The half-max is the estimator that is model-free and still tracks the fit
where the signal is strong.

THIS IS NOT THE OLD ``correlation_length`` AND THE DISTINCTION IS THE WHOLE REASON IT IS SAFE HERE.
``spatial.correlation_length`` crosses half of ``I`` AT THE SMALLEST RADIUS and assigns the largest
radius to any gene that never gets there. On a row-standardized Moran curve with a high floor that
threshold is never reached, so the estimator saturates at the grid ceiling and a differential built on
it collapses -- which is exactly why the differential moved to the centroid in the first place, and
the objection is recorded in this repo. The reduction here crosses half of the curve's OWN RANGE,
``min + (max - min)/2``, which a high floor cannot defeat, and it returns NaN rather than the ceiling
where no crossing exists. Different estimator, different failure mode. It was re-tested against the
scale-recovery contrast before the default was flipped rather than argued away.

NaN IS A REAL ANSWER HERE. A curve that never falls to its own mid-range within the sweep, or that is
already below it at the first radius, has no half-max to report and gets NaN instead of a plausible
micrometre value. The centroid never does this -- it returns a number for every curve, including the
unstructured ones, which is precisely how it ends up reporting the radius grid. Downstream, a NaN
scale propagates to a NaN Delta-scale, and that is the intended behaviour: no length, no difference.

ORIENTATION IS DECIDED ON THE POPULATION, NEVER PER CURVE. Moran scalograms decay with radius and
Geary curves rise; the reduction needs the curve to decay. Flipping each curve to its own preferred
direction would make the sign of the statistic a per-gene free parameter and destroy the comparison
between genes, so the sign is taken once from the population median. In a differential BOTH ARMS MUST
SHARE ONE SIGN -- pass the pooled sign explicitly (see :func:`orient_sign`), or the two arms are
reduced on different conventions and the difference is meaningless.
"""
from __future__ import annotations

import numpy as np

#: Reductions :func:`reduce_scale` accepts. ``"half_max"`` is the default and what the toolkit reports.
REDUCTIONS = ("half_max", "centroid")

#: What the toolkit reports. Changing this changes every published micrometre value.
DEFAULT_REDUCTION = "half_max"


def orient_sign(scalogram):
    """``+1`` if this population of curves already decays with radius, ``-1`` if it rises.

    ``scalogram`` is ``(n_radii, n_curves)``. Taken from the population median rather than per curve on
    purpose: a per-curve flip would make the sign of the statistic a free parameter. In a differential,
    compute this ONCE on the two arms pooled and pass it to both.

    Parameters
    ----------
    scalogram : array-like, shape (n_radii, n_curves)
        One curve per column. A 1-D array is read as ONE radius across many curves, so it always
        gives ``+1``; pass a single curve as shape ``(n_radii, 1)``.

    Returns
    -------
    float
        ``+1.0`` when the median over curves at the first radius is at least the median at the
        last radius, else ``-1.0``. The medians are not NaN-aware: a NaN anywhere in the first or
        last row gives ``-1.0``.
    """
    S = np.atleast_2d(np.asarray(scalogram, float))
    return 1.0 if np.median(S[0]) >= np.median(S[-1]) else -1.0


def centroid(scalogram, radii):
    """The positive-scalogram radius centroid, ``sum_r max(I(r),0)*r / sum_r max(I(r),0)``.

    What the toolkit reported before 2026-08-18, kept reachable so published numbers can be reproduced
    exactly. Retained for reproduction, not recommended for new work — see the module docstring for
    the three measurements that demoted it.

    Parameters
    ----------
    scalogram : array-like, shape (n_radii, n_curves)
        One curve per column. Must be 2-D: the shape is not checked, and a 1-D curve is broadcast
        against ``radii`` into a meaningless result. Pass a single curve as ``(n_radii, 1)``.
    radii : array-like, shape (n_radii,)
        The radius of each row, in any length unit.

    Returns
    -------
    numpy.ndarray
        ``(n_curves,)`` in the units of ``radii``. Negative values carry no weight. A curve with no
        positive value returns ``0.0``, not NaN; a NaN anywhere in a curve returns NaN.
    """
    w = np.clip(np.asarray(scalogram, float), 0, None)
    r = np.asarray(radii, float)
    return (w * r[:, None]).sum(0) / (w.sum(0) + 1e-12)


def half_max(scalogram, radii, *, sign=None):
    """Radius at which each curve falls to half of ITS OWN RANGE, log-interpolated between radii.

    ``min + (max - min)/2``, not half of the value at the smallest radius — a high floor cannot defeat
    this threshold, which is what separates it from ``spatial.correlation_length`` and its ceiling
    saturation. The curve is not normalised first, so amplitude survives as an independent number
    instead of being divided away.

    Parameters
    ----------
    scalogram : array-like, shape (n_radii, n_curves)
    radii : array-like, shape (n_radii,)
    sign : float, optional
        ``+1`` or ``-1``, applied to every curve. ``None`` derives it from this population via
        :func:`orient_sign`. **Pass it explicitly when reducing two arms that must be compared.**

    Returns
    -------
    numpy.ndarray
        ``(n_curves,)`` in the units of ``radii``. NaN where the curve never crosses its own
        mid-range inside the sweep, or is already below it at the first radius — such a curve has no
        half-max to report, and NaN says so rather than inventing a plausible value.
    """
    S = np.atleast_2d(np.asarray(scalogram, float))
    r = np.asarray(radii, float)
    if S.shape[0] != r.size:
        raise ValueError(f"scalogram has {S.shape[0]} radii but radii has {r.size}")
    if np.any(r <= 0):
        raise ValueError("radii must be positive: the crossing is interpolated in log radius")
    s = float(orient_sign(S)) if sign is None else float(sign)
    A = (s * S).T                                              # (n_curves, n_radii), decaying
    lr = np.log2(r)
    lo, hi = A.min(1), A.max(1)
    lvl = lo + 0.5 * (hi - lo)
    out = np.full(len(A), np.nan)
    for i, y in enumerate(A):
        if not np.isfinite(y).all():
            continue
        b = np.flatnonzero(y <= lvl[i])
        if b.size and b[0] > 0:
            j = b[0]
            f = (y[j - 1] - lvl[i]) / max(y[j - 1] - y[j], 1e-12)
            out[i] = 2.0 ** (lr[j - 1] + f * (lr[j] - lr[j - 1]))
    return out


def reduce_scale(scalogram, radii, how=DEFAULT_REDUCTION, *, sign=None):
    """Reduce a ``(n_radii, n_curves)`` scalogram to one characteristic length per curve.

    ``how`` is ``"half_max"`` (default, what the toolkit reports) or ``"centroid"`` (the pre-2026-08-18
    readout, for reproducing published numbers). ``sign`` is forwarded to :func:`half_max` and ignored
    by the centroid, which is orientation-free because it weights by positive mass.

    Parameters
    ----------
    scalogram : array-like, shape (n_radii, n_curves)
        One curve per column.
    radii : array-like, shape (n_radii,)
        The radius of each row, in any length unit. Must be positive for ``"half_max"``, which
        interpolates in log radius.
    how : {"half_max", "centroid"}, optional
        The reduction. Default :data:`DEFAULT_REDUCTION` (``"half_max"``). Any other value raises
        ``ValueError``.
    sign : float, optional
        ``+1`` or ``-1``, the orientation that makes the curves decay; used by ``"half_max"``
        only. ``None`` (default) derives it from this population with :func:`orient_sign`. Pass it
        explicitly when two sets of curves must be compared.

    Returns
    -------
    numpy.ndarray
        ``(n_curves,)`` characteristic lengths in the units of ``radii``. See :func:`half_max` and
        :func:`centroid` for when each returns NaN or 0.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrast import scale
    >>> r = np.array([10.0, 20.0, 40.0, 80.0])
    >>> curve = np.array([[0.8], [0.6], [0.2], [0.0]])          # one decaying curve
    >>> scale.reduce_scale(curve, r).round(1)
    array([28.3])
    """
    if how == "half_max":
        return half_max(scalogram, radii, sign=sign)
    if how == "centroid":
        return centroid(scalogram, radii)
    raise ValueError(f"reduction must be one of {REDUCTIONS}, got {how!r}")

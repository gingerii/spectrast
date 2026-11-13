"""Shape classification and landmark extraction for curves no single model family fits.

PUBLIC AS OF 2026-08-18, WITH A STATED SCOPE. This is a per-section DESCRIPTIVE coordinate. It is
independent of what the toolkit already reports -- near-orthogonal to the Moran half-max on seven of
seven supported sections (|rho| <= 0.13 on five), prominence independent of Moran amplitude, and
independent of both the amplitude and scale SHIFTS in a differential (rho -0.05 and -0.10). But it is
NOT a differential axis: measured on nineteen replicate sections its between-condition shift is 0.223
octaves against a within-condition scatter of 0.663, a ratio of 0.38, so a between-arm test cannot see
it at that design. Report it per section, with its class
label and its coverage; do not difference it between conditions.

WHY A LANDMARK AND NOT A FIT. The focality curve turns over inside the measured window for most genes,
runs monotonically for some, and dips for others. A decay family has no parameter for a peak and none
for a trough, so fitting one estimates a quantity that does not exist for most of the population --
which is why four successive attempts to reduce focality with one formula failed. Landmark
registration is the classical answer: find a feature the curve actually possesses, report its position
as one number and its prominence as another, and report the CLASS as a property in its own right
(Ramsay & Silverman 2005; Srivastava & Klassen 2016).

WHAT MADE THIS DEFENSIBLE WAS THE NOISE CONTROL, NOT THE THEORY. Counting turning points on the raw
curves suggested a zoo, with a majority multi-modal on several sections. Running the identical
classifier over coordinate-scrambled curves showed the opposite of what that implied: scrambled curves
are MORE multi-modal (62% against 32% on merfish_sagittal, 92% against 60% on trem2_merfish) and
rougher (mean |second difference| over range 0.122 against 0.061), and essentially no real gene is
wigglier than destroyed tissue. Multi-modality is the noise signature. It also falls as organization
rises -- 37% to 19% across amplitude fifths on merfish_sagittal, 30% to 1% on dcis_idc -- so the genes
a landmark is wanted for are exactly the genes whose curves are cleanly unimodal.

CLASS FRACTIONS ARE A RESULT, NOT BOOKKEEPING. They vary by tissue and platform far more than any
per-gene number does, and no axis currently reports them.

Two sections are known not to support this at all: visiumHD_brain and xenium_breast_cancer have
observed shape composition indistinguishable from their own scramble. Anything computed here for them
should be treated as undefined rather than small.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# a turning point must clear this fraction of the curve's own range to be counted. Scaled by the
# range on purpose: an absolute threshold would call every low-amplitude gene featureless and would
# be amplitude under another name.
PROM = 0.12
MONO = 0.90                      # |Spearman(curve, radius)| above this is monotone
CLASSES = ("monotone up", "monotone down", "one max", "one min", "two or more", "flat")
KIND = ("none", "max", "min")


def smooth(y):
    """Light three-point smoothing, endpoints preserved. Removes single-sample spikes without
    moving a real turning point by more than a fraction of a grid step.

    Parameters
    ----------
    y : array-like of shape (n,)
        One curve, at least three points.

    Returns
    -------
    numpy.ndarray of shape (n,)
        ``y`` convolved with the weights ``(0.25, 0.5, 0.25)``, with the first and last values
        copied from ``y``.
    """
    k = np.array([0.25, 0.5, 0.25])
    ys = np.convolve(np.asarray(y, float), k, mode="same")
    ys[0], ys[-1] = y[0], y[-1]
    return ys


def turning_points(y, prom=PROM):
    """Indices of interior maxima and minima clearing ``prom`` of the curve's own range.

    Prominence is TOPOGRAPHIC: an extremum is measured against the HIGHER of the two flanking minima
    (the key col), not the lower. Using the lower collapses to the curve's global minimum, which makes
    every global maximum score exactly 1.0 -- a saturated variable that correlates with nothing and
    lets any small bump qualify as a turning point. That error inflated the multi-modal class in a
    first pass of this analysis; it is the single most consequential line in this file.

    Parameters
    ----------
    y : array-like of shape (n_radii,)
        One curve, in radius order. It is smoothed with :func:`smooth` first, and the extrema and
        the range are those of the smoothed curve.
    prom : float, optional
        Minimum prominence, as a fraction of the smoothed curve's range; an extremum must exceed it.
        Default ``PROM = 0.12``.

    Returns
    -------
    mx : list of int
        Indices of the interior maxima that clear ``prom``.
    mn : list of int
        Indices of the interior minima that clear ``prom``.
    """
    ys = smooth(y)
    rng = max(ys.max() - ys.min(), 1e-12)
    n = len(ys)
    mx, mn = [], []
    for i in range(1, n - 1):
        if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]:
            if (ys[i] - max(ys[:i].min(), ys[i:].min())) / rng > prom:
                mx.append(i)
        if ys[i] < ys[i - 1] and ys[i] <= ys[i + 1]:
            if (min(ys[:i].max(), ys[i:].max()) - ys[i]) / rng > prom:
                mn.append(i)
    return mx, mn


def classify(A, r, prom=PROM):
    """Integer code into CLASSES, per curve, and each curve's Spearman correlation with radius.

    Parameters
    ----------
    A : array-like of shape (n_curves, n_radii)
        Curves, one per row, in radius order.
    r : array-like of shape (n_radii,)
        Radii, one per column of ``A``. Used only for the Spearman correlation, so any units.
    prom : float, optional
        Minimum prominence, as in :func:`turning_points`. Default ``PROM = 0.12``.

    Returns
    -------
    cls : numpy.ndarray of int, shape (n_curves,)
        Index into :data:`CLASSES`: ``0`` monotone up, ``1`` monotone down, ``2`` one max, ``3``
        one min, ``4`` two or more turning points, ``5`` flat (range below ``1e-9``). A curve with
        no turning point that clears ``prom`` is monotone up when its Spearman correlation with
        ``r`` is positive, and monotone down otherwise.
    rho : numpy.ndarray of float, shape (n_curves,)
        Spearman correlation of each curve with ``r``; NaN for a constant curve.
    """
    A = np.asarray(A, float)
    rho = np.array([spearmanr(r, y).statistic for y in A])
    out = np.zeros(len(A), int)
    for i, y in enumerate(A):
        if y.max() - y.min() < 1e-9:
            out[i] = 5
            continue
        mx, mn = turning_points(y, prom)
        if not mx and not mn:
            out[i] = 0 if rho[i] > 0 else 1
        elif len(mx) == 1 and not mn:
            out[i] = 2
        elif len(mn) == 1 and not mx:
            out[i] = 3
        else:
            out[i] = 4
    return out, rho


def _interp(y, i, lr):
    """Sub-grid position of an extremum at index i, by 3-point parabola in log radius.

    Not a refinement: the radius grid carries about 4.2 points per octave, so an un-interpolated
    index is quantised into fifth-of-an-octave steps and most of its between-gene variance is that
    quantisation rather than biology.
    """
    if i <= 0 or i >= len(y) - 1:
        return lr[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2.0 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return lr[i] + np.clip(d, -1.0, 1.0) * (lr[i + 1] - lr[i - 1]) / 2.0


def landmark(A, r, prom=PROM):
    """Per curve: the class, and the reduction that fits that class.

    For a multi-modal curve the DOMINANT extremum is reported and n_extrema records the ambiguity.
    Given that scrambled curves are more multi-modal than real ones, this is the right default: the
    extra extrema are usually noise, and discarding the gene entirely would discard signal with them.

    Parameters
    ----------
    A : array-like of shape (n_curves, n_radii)
        Curves, one per row, in radius order.
    r : array-like of shape (n_radii,)
        Radii, one per column of ``A``; positive, normally in µm. The extremum's position is
        interpolated in ``log2(r)`` and returned in the units of ``r``.
    prom : float, optional
        Minimum prominence, as in :func:`turning_points`. Default ``PROM = 0.12``.

    Returns
    -------
    dict of numpy.ndarray
        A dict of arrays, each of shape ``(n_curves,)``:

        - ``cls`` : class code into CLASSES (int), as in :func:`classify`.
        - ``kind`` : 0 none / 1 max / 2 min -- which feature the landmark is (int).
        - ``radius_um`` : position of the dominant extremum, from a three-point parabola in
          ``log2(r)`` on the smoothed curve; NaN for monotone and flat curves, as are
          ``prominence`` and ``value``.
        - ``prominence`` : its height above (or depth below) the best opposing excursion, as a
          fraction of the curve's own range -- so it is comparable between genes of different
          amplitude.
        - ``value`` : the curve's value at the landmark, in the statistic's own units (the
          unsmoothed curve at the extremum's grid radius, not interpolated).
        - ``rho_radius`` : Spearman of curve against radius: the summary that DOES fit a monotone
          curve.
        - ``n_extrema`` : how many turning points were found (int), so a multi-modal curve is
          never silently presented as if its landmark were unambiguous.
    """
    A = np.asarray(A, float)
    lr = np.log2(np.asarray(r, float))
    cls, rho = classify(A, r, prom)
    n = len(A)
    kind = np.zeros(n, int)
    rad = np.full(n, np.nan)
    prm = np.full(n, np.nan)
    val = np.full(n, np.nan)
    nex = np.zeros(n, int)
    for i, y in enumerate(A):
        mx, mn = turning_points(y, prom)
        nex[i] = len(mx) + len(mn)
        if not mx and not mn:
            continue
        ys = smooth(y)
        rng = max(ys.max() - ys.min(), 1e-12)
        best, bkind, bprom = None, 0, -np.inf
        for j in mx:
            p = (ys[j] - max(ys[:j].min(), ys[j:].min())) / rng
            if p > bprom:
                best, bkind, bprom = j, 1, p
        for j in mn:
            p = (min(ys[:j].max(), ys[j:].max()) - ys[j]) / rng
            if p > bprom:
                best, bkind, bprom = j, 2, p
        kind[i] = bkind
        rad[i] = 2.0 ** _interp(ys, best, lr)
        prm[i] = bprom
        val[i] = y[best]
    return dict(cls=cls, kind=kind, radius_um=rad, prominence=prm, value=val,
                rho_radius=rho, n_extrema=nex)


def roughness(A):
    """Mean ``|second difference|`` over the curve's own range: how much of the curve is jitter.

    Scaled by range deliberately -- unscaled, this would rank low-amplitude genes as rough and simply
    restate amplitude.

    Parameters
    ----------
    A : array-like of shape (n_curves, n_radii)
        Curves, one per row, in radius order; at least three radii.

    Returns
    -------
    numpy.ndarray of shape (n_curves,)
        Mean absolute second difference of each curve divided by its range (the range floored at
        ``1e-12``); ``0`` for a flat curve.
    """
    A = np.asarray(A, float)
    return np.abs(np.diff(A, n=2, axis=1)).mean(1) / np.maximum(A.max(1) - A.min(1), 1e-12)

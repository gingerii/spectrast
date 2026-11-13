"""Decay-family fits for organization curves — a CHECK on the reported scale, never the scale itself.

WHAT THIS IS FOR. The toolkit reports a model-free scale (the radius at which a gene's oriented Moran
curve falls to half its own range) and an amplitude. Those two numbers describe the curves well, and
nothing here replaces either. What they cannot do is say whether a particular gene's number is a
length the tissue has or an artefact of where the radius sweep happened to stop. That is this module's
whole job: fit the curve, and report whether the fit CORROBORATES the reported scale.

WHY THE REPORTED SCALE IS MODEL-FREE IN THE FIRST PLACE, WHICH IS A MEASURED RESULT AND NOT A
PREFERENCE. Real Moran curves are systematically not single exponentials. Fitting the stretched family
a*exp(-(r/xi)^beta)+c to the strongly organized fifth of the genes gives beta < 1 for 66-99% of them
across ten sections spanning four tissue types and a 3.6x range of cell pitch — a heavier tail than
any single decay has. The separation is a property of tissue and not of the criterion: run this same
two-family contest on coordinate-scrambled curves and the stretched family wins by more than
``D_BIC`` in 4-5% of cases, against 39-99% on the real strongly organized genes. A fitted xi therefore
MISREPORTS these curves, which is exactly why the half-maximum is what gets reported. Keep
that argument attached to the code: without it, the model-free choice reads as laziness rather than as
the answer to "why not just fit an exponential?".

TWO FLAGS ARE THE POINT, NOT THE PARAMETERS.

  ``scale_supported`` — the single decay converged and its xi landed INSIDE the swept window. Where it
  does not, the gene's half-max is being set by the ends of the measurement rather than by the tissue,
  and the number should not be quoted. This is the flag that caught a retraction: a set of genes read
  as carrying two organizational scales until the fitted components turned out to sit at fixed
  fractions of the window (xi2 / r_max = 0.58 +/- 0.09 across ten sections, rho(r_max, xi2) = +0.92)
  and to evaporate when the window was shortened.

  ``beta_pinned`` — the stretched exponent converged onto one of its bounds, meaning the data do not
  identify it. On flat curves a third to three quarters of fits pin (67-76% on the scrambles of three
  sections measured, against 0% on their strongly organized genes), and the population median swings
  2.36 -> 0.77 when the LOWER bound alone is moved. A pinned beta is not a shape measurement and must
  never be reported as one; the bound fractions are returned so a degenerate population cannot be read
  as a finding.

WHAT IS DELIBERATELY ABSENT. The two-component family a1*exp(-r/xi1)+a2*exp(-r/xi2)+c is NOT fitted
here. It wins the BIC contest on every section tested, and that win was retracted as biology: a sum of
two exponentials is simply a more flexible shape, it fits anything decaying more slowly than an
exponential, and the components it finds here are the window's ends. It stays out of the library,
because shipping it would hand users a fit that looks like a second length scale and is not one. The stretched
family is the honest version of the same flexibility: one extra parameter instead of two, nested
against the single decay at beta = 1, and it buys the slow tail WITHOUT positing a second scale.

RESOLUTION IS A HARD LIMIT HERE. Four parameters cannot be identified from five radii. The stretched
fit is refused below ``MIN_RADII_STRETCHED`` points and warned about below ``WEAK_RADII_STRETCHED``;
the single decay needs at least four. The library's own default sweep is nineteen radii (six until
0.14.0), which clears both: xi and beta are identified on a default call. Six was marginal for beta —
the same coarse-grid limit that suppresses focality landmark coverage — and it is worth knowing before
reading a beta off any call that passes fewer than eight radii.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Bounds on the stretched exponent. Loose on purpose: at 0.2 the lower bound clipped 4% of one
#: section's genes, and widening it to 0.05 moved no family share and no qualifying fraction on either
#: section tested — which is what says the fit is decided by the data and not by the box it runs in.
BETA_LO, BETA_HI = 0.05, 3.0

#: Relative tolerance for calling a fitted exponent "on the bound".
BOUND_TOL = 0.02

#: BIC difference read as strong evidence for one family over another.
D_BIC = 6.0

#: Fewer radii than this and the four-parameter stretched fit is not attempted at all.
MIN_RADII_STRETCHED = 6

#: Below this the stretched fit runs but beta is weakly identified and says so.
WEAK_RADII_STRETCHED = 8

#: Fewer radii than this and even the three-parameter single decay is refused.
MIN_RADII_ONE = 4


def one(r, a, xi, c):
    """Single exponential decay to a floor. Three parameters.

    Parameters
    ----------
    r : numpy.ndarray or float
        Radius, in the units of ``xi``.
    a : float
        Amplitude: the height above the floor at ``r = 0``.
    xi : float
        Decay length, in the units of ``r``.
    c : float
        The floor the curve decays to.

    Returns
    -------
    numpy.ndarray or float
        ``a * exp(-r / xi) + c``, with the shape of ``r``.
    """
    return a * np.exp(-r / xi) + c


def stretched(r, a, xi, beta, c):
    """One scale, one shape parameter. Four parameters.

    ``beta < 1`` is a heavier-than-exponential tail — the shape a shoulder actually has — and
    ``beta = 1`` recovers :func:`one` exactly, so the comparison against it is nested and the extra
    parameter is doing a single identifiable job.

    Parameters
    ----------
    r : numpy.ndarray or float
        Radius, in the units of ``xi``; positive.
    a, xi, c : float
        Amplitude, decay length and floor, as in :func:`one`.
    beta : float
        Stretch exponent. ``beta < 1`` gives a heavier tail; ``beta = 1`` is :func:`one`.

    Returns
    -------
    numpy.ndarray or float
        ``a * exp(-(r / xi) ** beta) + c``, with the shape of ``r``.
    """
    return a * np.exp(-((r / xi) ** beta)) + c


def bic(y, yhat, k):
    """Gaussian BIC for a fit with ``k`` free parameters.

    Parameters
    ----------
    y : array-like of shape (n,)
        The observed curve.
    yhat : array-like of shape (n,)
        The fitted values at the same radii.
    k : int
        Number of free parameters in the fit.

    Returns
    -------
    float
        ``n * log(RSS / n) + k * log(n)``, with the residual sum of squares RSS floored at
        ``1e-300``. Terms shared by every fit to the same curve are dropped, so only differences
        between fits mean anything; lower is better.
    """
    n = len(y)
    rss = float(np.sum((np.asarray(y) - np.asarray(yhat)) ** 2))
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def orient(A):
    """Flip a POPULATION of curves so it decays, or return it unchanged if it already does.

    Decided on the population median rather than per curve on purpose: flipping each curve to its own
    preferred direction would make the sign of the statistic a per-gene free parameter and destroy the
    comparison between genes. Moran scalograms decay in practice, so this is usually the identity.

    Parameters
    ----------
    A : array-like of shape (n_curves, n_radii)
        The population of curves, one per row, columns in ascending radius.

    Returns
    -------
    numpy.ndarray of shape (n_curves, n_radii)
        ``A`` as float, unchanged when the median of its first column exceeds the median of its last
        column, and ``-A`` otherwise.
    """
    A = np.asarray(A, float)
    return (1.0 if np.median(A[:, 0]) > np.median(A[:, -1]) else -1.0) * A


def pinned(b, lo=BETA_LO, hi=BETA_HI, tol=BOUND_TOL):
    """Which fitted exponents sit on a bound — the signature of a parameter the data do not identify.

    Public because every reported beta must be accompanied by this: a degenerate fit reads exactly
    like a shape finding otherwise.

    Parameters
    ----------
    b : array-like
        Fitted stretch exponents, any shape; NaN where there is no fit.
    lo, hi : float, optional
        The bounds the fit ran with. Defaults ``BETA_LO = 0.05`` and ``BETA_HI = 3.0``.
    tol : float, optional
        Relative tolerance: an exponent at or below ``lo * (1 + tol)``, or at or above
        ``hi * (1 - tol)``, is on a bound. Default ``BOUND_TOL = 0.02``.

    Returns
    -------
    numpy.ndarray of bool
        Same shape as ``b``. ``True`` where the exponent is finite and on a bound; ``False`` for
        NaN.
    """
    b = np.asarray(b, float)
    return np.isfinite(b) & ((b <= lo * (1 + tol)) | (b >= hi * (1 - tol)))


def fit_decay(A, r, *, beta_bounds=(BETA_LO, BETA_HI), maxfev_one=8000, maxfev_stretched=20000,
              warn=True):
    """Fit the single and stretched decays to every curve, and flag what the fits do not identify.

    Parameters
    ----------
    A : array-like, shape (n_curves, n_radii)
        Organization curves, ALREADY oriented to decay (see :func:`orient`). A 1-D array is one
        curve.
    r : array-like, shape (n_radii,)
        Radii in coordinate units (normally µm), positive and ascending; ``r[0]`` and ``r[-1]`` are
        the swept window. At least ``MIN_RADII_ONE = 4`` radii, or ``ValueError``.
    beta_bounds : tuple, optional
        Bounds ``(lo, hi)`` on the stretched exponent. Default ``(BETA_LO, BETA_HI) = (0.05, 3.0)``.
        Widening them should not move the population; if it does, the exponent is not identified and
        ``beta_pinned`` will say so.
    maxfev_one : int, optional
        Maximum function evaluations for the single-decay fit (``maxfev`` of
        ``scipy.optimize.curve_fit``). Default ``8000``. A fit that does not converge within it
        sets ``fit_failed``.
    maxfev_stretched : int, optional
        The same for the stretched fit. Default ``20000``. A stretched fit that does not converge
        leaves its outputs NaN.
    warn : bool, optional
        Emit the resolution warnings. Set ``False`` when calling in a loop that has already warned.

    Returns
    -------
    dict of numpy.ndarray
        Per curve, each of shape ``(n_curves,)``:

        - ``xi_um``, ``xi_se_um``, ``a_one``, ``c_one``, ``bic_one`` : the single decay's length (in
          the units of ``r``), its standard error, amplitude, floor and BIC.
        - ``beta``, ``beta_se``, ``xi_stretched_um``, ``bic_stretched`` : the stretched fit's
          exponent, its standard error, length and BIC. NaN where the fit was not run (fewer than
          ``MIN_RADII_STRETCHED = 6`` radii) or did not converge.
        - ``d_bic`` : ``bic_one - bic_stretched``, positive where the stretched family wins.
        - the booleans ``fit_failed`` (the single decay was not fitted: a non-finite or constant
          curve, or no convergence), ``xi_outside_window`` (``xi_um`` outside ``[r[0], r[-1]]``),
          ``scale_supported`` (fitted, with ``xi_um`` inside the window), ``beta_pinned`` (see
          :func:`pinned`) and ``heavier_than_exponential`` (an unpinned ``beta < 1``).
    """
    from scipy.optimize import curve_fit  # lazy: scipy.optimize is not needed to import spectrast

    A = np.atleast_2d(np.asarray(A, float))
    r = np.asarray(r, float)
    if A.shape[1] != r.size:
        raise ValueError(f"A has {A.shape[1]} radii but r has {r.size}")
    nR = r.size
    if nR < MIN_RADII_ONE:
        raise ValueError(
            f"a three-parameter decay cannot be fitted to {nR} radii; at least {MIN_RADII_ONE} are "
            f"needed for a single residual degree of freedom. Sweep a finer grid.")
    lo_b, hi_b = float(beta_bounds[0]), float(beta_bounds[1])
    do_stretched = nR >= MIN_RADII_STRETCHED
    if warn:
        if not do_stretched:
            warnings.warn(
                f"{nR} radii is too few to identify the four-parameter stretched family (needs at "
                f"least {MIN_RADII_STRETCHED}); beta is returned as NaN. The single decay still runs, "
                f"so the scale_supported flag is unaffected — only the tail-shape check is lost.",
                stacklevel=2)
        elif nR < WEAK_RADII_STRETCHED:
            warnings.warn(
                f"{nR} radii leaves the stretched fit {nR - 4} residual degrees of freedom, so beta is "
                f"weakly identified here. Read beta_pinned before reading beta, and prefer a finer "
                f"radius grid for any statement about tail shape.", stacklevel=2)

    n = len(A)
    nan = lambda: np.full(n, np.nan)                                        # noqa: E731
    out = {k: nan() for k in ("xi_um", "xi_se_um", "a_one", "c_one", "bic_one",
                              "beta", "beta_se", "xi_stretched_um", "bic_stretched", "d_bic")}
    out["fit_failed"] = np.zeros(n, bool)
    lo, hi = float(r[0]), float(r[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, y in enumerate(A):
            if not np.isfinite(y).all() or np.std(y) < 1e-12:
                out["fit_failed"][i] = True
                continue
            amp = float(y.max() - y.min())
            try:
                p1, cov1 = curve_fit(one, r, y, p0=(max(amp, 1e-6), r[nR // 3], float(y[-1])),
                                     maxfev=maxfev_one)
                out["a_one"][i], out["xi_um"][i], out["c_one"][i] = p1
                if np.all(np.isfinite(cov1)):
                    out["xi_se_um"][i] = float(np.sqrt(np.diag(cov1))[1])
                out["bic_one"][i] = bic(y, one(r, *p1), 3)
            except Exception:
                out["fit_failed"][i] = True
                continue
            if not do_stretched:
                continue
            try:
                p2, cov2 = curve_fit(
                    stretched, r, y, p0=(max(amp, 1e-6), r[nR // 3], 1.0, float(y[-1])),
                    bounds=([1e-12, lo * 0.1, lo_b, -np.inf], [np.inf, hi * 10.0, hi_b, np.inf]),
                    maxfev=maxfev_stretched)
                out["xi_stretched_um"][i], out["beta"][i] = p2[1], p2[2]
                if np.all(np.isfinite(cov2)):
                    out["beta_se"][i] = float(np.sqrt(np.diag(cov2))[2])
                out["bic_stretched"][i] = bic(y, stretched(r, *p2), 4)
            except Exception:
                pass

    out["d_bic"] = out["bic_one"] - out["bic_stretched"]
    # xi outside the swept window means the half-max is being set by where the sweep stopped, not by
    # the tissue. A failed fit corroborates nothing either, so it does not support the scale.
    out["xi_outside_window"] = np.isfinite(out["xi_um"]) & ((out["xi_um"] < lo) | (out["xi_um"] > hi))
    out["scale_supported"] = (~out["fit_failed"]) & np.isfinite(out["xi_um"]) & (~out["xi_outside_window"])
    out["beta_pinned"] = pinned(out["beta"], lo_b, hi_b)
    out["heavier_than_exponential"] = (np.isfinite(out["beta"]) & (~out["beta_pinned"])
                                       & (out["beta"] < 1.0))
    return out


def fit_summary(fit):
    """Population-level read-out of a :func:`fit_decay` result — the numbers that decide whether the
    per-gene ones may be quoted at all.

    Parameters
    ----------
    fit : dict
        The output of :func:`fit_decay`.

    Returns
    -------
    dict
        ``n_curves`` (int), then floats: ``frac_supported``, ``frac_outside_window``,
        ``frac_failed``, ``frac_beta_pinned`` (of curves with a beta), ``frac_heavier_tail`` (of
        identified betas, i.e. finite and not pinned), ``median_beta`` (identified only) and
        ``frac_stretched_wins`` (by BIC, margin ``D_BIC``: the fraction of curves with a finite
        ``d_bic`` whose ``d_bic`` exceeds ``D_BIC = 6``). Each is NaN when the set it is taken over
        is empty.
    """
    f = fit
    n = len(f["xi_um"])
    has_b = np.isfinite(f["beta"])
    ident = has_b & (~f["beta_pinned"])
    cmp_ok = np.isfinite(f["d_bic"])
    return dict(
        n_curves=int(n),
        frac_supported=float(f["scale_supported"].mean()) if n else np.nan,
        frac_outside_window=float(f["xi_outside_window"].mean()) if n else np.nan,
        frac_failed=float(f["fit_failed"].mean()) if n else np.nan,
        frac_beta_pinned=float(f["beta_pinned"][has_b].mean()) if has_b.any() else np.nan,
        frac_heavier_tail=float((f["beta"][ident] < 1.0).mean()) if ident.any() else np.nan,
        median_beta=float(np.median(f["beta"][ident])) if ident.any() else np.nan,
        frac_stretched_wins=float((f["d_bic"][cmp_ok] > D_BIC).mean()) if cmp_ok.any() else np.nan,
    )

"""Lightweight known-truth ST simulator (Aim 2c seed) for validating denoising / fidelity claims.

Builds a synthetic expression matrix whose spatial structure is KNOWN: a small number of smooth
spatial factors (Gaussian random fields at controlled scales) mixed by sparse gene loadings give a
low-rank NOISE-FREE signal ``X_true`` with an exactly-computable spatial autocorrelation; optional
per-gene ``private`` high-rank structure, additive ``noise``, and ``dropout`` produce the observed
``X_noisy``. Because ``X_true`` is available, any statistic's TRUE value is known, so we can test
whether reduced-rank / fusion recover the truth (not merely reproduce a noisy estimate).

Heavy deps imported lazily.
"""
from __future__ import annotations
import numpy as np


def _grf(coords, scale_um, seed):
    """Gaussian random field of correlation length ``scale_um`` sampled at ``coords`` (N, 2)."""
    from scipy import ndimage  # lazy
    from .adaptive import grid_index  # lazy
    fine = max(scale_um / 8.0, 1.0)
    row, col, H, W, _ = grid_index(coords, fine)
    white = np.random.default_rng(seed).standard_normal((H, W))
    f = ndimage.gaussian_filter(white, (scale_um / fine) / 2.0, mode="wrap")
    f = (f - f.mean()) / (f.std() + 1e-9)
    return f[row, col]


def _focal_field(coords, scale_um, seed, n_hot=None, sharp=6.0):
    """SHARP focal field: a few compact hotspots (super-Gaussian bumps) at correlation length
    ``scale_um``. Unlike a GRF this is spatially high-frequency (sharp edges), used to prove the
    reduced-rank win is GENE-axis denoising, not spatial smoothing."""
    coords = np.asarray(coords, float); rng = np.random.default_rng(seed)
    lo = coords.min(0); hi = coords.max(0)
    span = float(np.linalg.norm(hi - lo))
    if n_hot is None:
        n_hot = int(np.clip((span / scale_um) ** 2 * 0.15, 3, 40))
    centers = lo + rng.random((n_hot, 2)) * (hi - lo)
    signs = rng.choice([-1.0, 1.0], n_hot)
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (N, n_hot)
    # super-Gaussian (flat-topped, sharp-edged) bumps
    bumps = np.exp(-(d2 / (scale_um ** 2)) ** (sharp / 2.0))
    f = bumps @ signs
    return (f - f.mean()) / (f.std() + 1e-9)


def _domain_field(coords, scale_um, seed):
    """Piecewise-constant (Voronoi) field: sharp domain boundaries, correlation length ~``scale_um``.
    Maximally non-smooth truth — a hard case for anything that secretly relies on spatial smoothness."""
    coords = np.asarray(coords, float); rng = np.random.default_rng(seed)
    lo = coords.min(0); hi = coords.max(0)
    span = float(np.linalg.norm(hi - lo))
    n_dom = int(np.clip((span / scale_um) ** 2 * 0.5, 3, 80))
    seeds = lo + rng.random((n_dom, 2)) * (hi - lo)
    vals = rng.standard_normal(n_dom)
    # nearest-seed assignment (chunked to bound memory)
    lab = np.empty(len(coords), int)
    for s in range(0, len(coords), 20000):
        c = coords[s:s + 20000]
        d2 = ((c[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)
        lab[s:s + 20000] = d2.argmin(1)
    f = vals[lab]
    return (f - f.mean()) / (f.std() + 1e-9)


_FIELDS = {"grf": _grf, "focal": _focal_field, "domains": _domain_field}


def simulate_st(coords, n_factors=10, n_genes=400, scale_range=(15.0, 120.0),
                noise=1.0, dropout=0.0, private=0.0, seed=0, field="grf"):
    """Synthetic ST matrix with known spatial structure.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Cell coordinates (microns) — real coordinates keep realistic density/geometry.
    n_factors : int, optional
        Rank of the true low-rank spatial signal (the true GENE-axis rank). Default ``10``.
    n_genes : int, optional
        Number of genes. Default ``400``.
    scale_range : tuple of float, optional
        ``(min, max)`` correlation length (µm) of the factor fields; each factor's scale is drawn
        log-uniformly between them. Default ``(15.0, 120.0)``.
    noise : float, optional
        Additive white-noise SD as a multiple of the signal SD (0 = noise-free). The signal SD is
        one number for the whole of ``X_true``, not one per gene. Default ``1.0``.
    dropout : float, optional
        Fraction of entries zeroed (sparsity), mimicking imaging-based ST. Default ``0.0``.
    private : float, optional
        Weight of per-gene high-rank private spatial structure (0 = purely rank-``n_factors``).
        Each gene then gets its own field, at a scale drawn uniformly from ``scale_range``.
        Default ``0.0``.
    seed : int, optional
        Seed for the factor scales, the loadings, the private-structure scales and weights, the
        noise and the dropout. The spatial fields themselves come from fixed seeds (``100 + i``
        for factor ``i``, ``5000 + g`` for gene ``g``'s private field), so a different ``seed``
        changes the fields only through their scales: at equal scales, two seeds give identical
        fields. Default ``0``.
    field : {"grf", "focal", "domains"}, optional
        Spatial character of the true factors — ``"grf"`` (smooth Gaussian random field, default),
        ``"focal"`` (sharp compact hotspots) or ``"domains"`` (piecewise-constant, sharp boundaries).
        Non-smooth fields let us prove reduced-rank denoising is GENE-axis (low gene-rank), not spatial
        smoothing: sharp truth is recovered exactly as long as the gene-rank is ``<= k``. Any other
        value raises ``KeyError``.

    Returns
    -------
    dict
        * ``X_true``: ``(N, n_genes)`` noise-free signal, ``F @ L`` (plus any private structure),
          centred so that each gene has mean 0;
        * ``X_noisy``: ``(N, n_genes)`` observed matrix, ``X_true`` plus noise and then dropout;
        * ``scales``: ``(n_factors,)`` correlation length of each factor field, in the units of
          ``coords``;
        * ``F``: ``(N, n_factors)`` factor fields at the cells, each standardised to mean 0 and
          SD 1 (for ``"grf"`` over the underlying grid, so the per-cell values are close to that
          but not exact);
        * ``L``: ``(n_factors, n_genes)`` loadings, standard normal with about half set to 0.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrast.simulate import simulate_st
    >>> xy = np.random.default_rng(0).random((500, 2)) * 300
    >>> sim = simulate_st(xy, n_factors=3, n_genes=20, noise=0.5)
    >>> sim["X_noisy"].shape, sim["F"].shape, sim["L"].shape
    ((500, 20), (500, 3), (3, 20))
    """
    coords = np.asarray(coords, float); rng = np.random.default_rng(seed)
    gen = _FIELDS[field]
    scales = np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]), n_factors))
    F = np.stack([gen(coords, s, seed=100 + i) for i, s in enumerate(scales)], 1)   # (N, n_factors)
    L = rng.normal(0, 1, (n_factors, n_genes)) * (rng.random((n_factors, n_genes)) < 0.5)  # sparse
    X_true = F @ L
    if private > 0:
        Pg = np.stack([gen(coords, float(rng.uniform(*scale_range)), seed=5000 + g)
                       for g in range(n_genes)], 1)
        X_true = X_true + private * Pg * rng.normal(0, 1, n_genes)
    X_true = X_true - X_true.mean(0)
    sig = X_true.std() + 1e-9
    X_noisy = X_true + noise * rng.normal(0, sig, X_true.shape)
    if dropout > 0:
        X_noisy = X_noisy * (rng.random(X_noisy.shape) > dropout)
    return dict(X_true=X_true, X_noisy=X_noisy, scales=scales, F=F, L=L)

"""Per-gene calibrated nulls for the multiscale Moran curve.

Three nulls, one statistic — ``spatial.morans_i_multiscale`` on one lattice, exactly as ``diff`` and
the curve bank compute it.

**N0, per-gene cell scramble.** Permutations are genes to the engine: *P* permutations of one
gene's values across cells are *P* columns of one rasterisation and one batched multiscale Moran,
so the observed curve and its null share every numerical path (the observed curve is column 0 of
the same call). Bin re-averaging is part of the null because the values are permuted over CELLS and
rasterised afresh — a bin holding five cells gets the mean of five random ones.

**N1, per-gene covariate-restricted scramble.** The same, permuting only within strata
(``composition.stratified_permutation``), with strata drawn from a continuous section covariate —
total counts smoothed at tissue scale, or cell density — quantile-binned. The covariate's spatial
pattern survives in the null, so a gene that merely tracks a section-wide gradient is not called.
That gradient is what made the entire diffuse tail "significant" at large radii against a pooled
scramble.

**A0, analytic randomisation moments.** Cliff & Ord's exact permutation mean and variance of *I*
for the engine's own row-standardised, centre-excluded disc weights, from two extra FFTs per radius
shared across genes plus one kurtosis per gene. A0 permutes BINS; N0 permutes CELLS. They are
different nulls and their agreement is measured by the calibration driver, not assumed here.

Nothing in this module edits the engine.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import spatial as sp
from .composition import stratified_permutation

__all__ = ["moran_permutation_null", "permutation_null", "PERM_STATS",
           "gene_features", "fit_surrogate_null",
           "moran_analytic_moments", "moran_analytic_z", "geary_analytic_z",
           "general_g_binary_moments", "general_g_analytic_z",
           "gene_inference", "ROUTES",
           "covariate_strata", "moran_null_batch"]


# --------------------------------------------------------------------------- primitives
def _permute_values(x, n_perm, rng, strata=None):
    """``(n, P)`` — column *p* is ``x`` under the *p*-th permutation of cells.

    Permuting the value vector is identical to permuting coordinates, which is how every other
    scramble in this package is phrased; doing it on the values lets one fixed scatter matrix
    rasterise all *P* columns at once.
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty((n, n_perm), float)
    if strata is None:
        for p in range(n_perm):
            out[:, p] = x[rng.permutation(n)]
    else:
        strata = np.asarray(strata)
        for p in range(n_perm):
            out[:, p] = x[stratified_permutation(strata, rng)]
    return out


def _rasterize_matrix(V, row, col, HW):
    """``(H, W, P)`` bin means of the ``(n, P)`` value matrix; zero on empty bins.

    One sparse scatter per call, the same construction ``gr._isotropic_fingerprint`` uses (19x
    faster than ``np.add.at`` and bit-identical). Kept private and tiny so the null never grows its
    own rasterisation semantics.
    """
    import scipy.sparse as sps
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[:, None]
    H, W = HW
    n = len(row)
    key = np.asarray(row, np.int64) * W + np.asarray(col, np.int64)
    S = sps.csr_matrix((np.ones(n), (key, np.arange(n))), shape=(H * W, n))
    cnt = np.asarray(S.sum(1)).ravel()
    acc = np.asarray(S @ V)
    occ = cnt > 0
    acc[occ] /= cnt[occ, None]
    return acc.reshape(H, W, V.shape[1])


# --------------------------------------------------------------------------- N0 / N1
#: The statistics a permutation null can be taken for. All six share the
#: ``(grid, mask, radii, kind)`` signature, so the dispatch is a lookup rather than a special case.
#: ``texture`` is accepted and ignores ``kind``: it reads a frequency band, not a disc.
PERM_STATS = ("moran", "geary", "general_g", "losh", "focality", "texture")


def _stat_fn(stat):
    from . import spatial as _sp
    try:
        return {"moran": _sp.morans_i_multiscale, "geary": _sp.gearys_c_multiscale,
                "general_g": _sp.general_g_multiscale, "losh": _sp.losh_multiscale,
                "focality": _sp.focality_scalogram, "texture": _sp.texture_multiscale}[stat]
    except KeyError:
        raise ValueError(f"unknown statistic {stat!r}; choose from {list(PERM_STATS)}") from None


def permutation_null(values, row, col, HW, mask, radii, *, stat="moran", n_perm=100, higher=None,
                     strata=None, seed=0, kind="disc", keep_null=True):
    """Per-gene, per-radius permutation null for ANY of the six organization statistics.

    The generalisation of :func:`moran_permutation_null`, which is this function pinned to one
    statistic. The permutation machinery was already statistic-agnostic -- permutations are extra
    COLUMNS of one rasterisation, so the observed curve and its null share every numerical path --
    and only the scalogram call was hardcoded.

    **The default is TWO-SIDED, because the structured tail is not the same for every statistic and
    is not always known.** Moran high means structured; Geary's C is approximately ``1 - I`` here, so
    LOW does; General G high means high values cluster; focality is MEASURED to invert on carcinoma,
    where organised genes are less focal than their scramble; and LOSH and texture have no direction
    fixed a priori. ``p_two`` therefore needs no direction and is what you get by default.

    Pass ``higher=True`` or ``higher=False`` for a one-sided ``p`` when the direction IS known for
    the statistic and tissue in hand. There is no default table, deliberately.

    **Do not take the direction from the curve's shape.** An earlier version of this function derived
    it from whether the curve rises or falls with radius -- the rule the characteristic-scale
    reduction uses -- which is correct for choosing a half-maximum and wrong here. General G rises with radius because it is cumulative, not because low values mean
    structure; that version returned ``p = 1.000`` for a gene sitting 212 sd above its own null.
    ``z`` carries the signed deviation; read its sign if you want what the data says.

    ``n_perm`` defaults to 100 rather than 200. Ranking genes by ``z`` stabilises well before the tail
    probability does, and the cost is linear in ``n_perm`` above about 30.

    Parameters
    ----------
    values : array-like of shape (n_cells,)
        One gene's per-cell values, e.g. log-normalised expression. The values are permuted over
        cells and then rasterised to bin means, so bin re-averaging is part of the null.
    row, col : array-like of int, shape (n_cells,)
        Each cell's lattice bin: ``row`` indexes the y axis in ``[0, H)`` and ``col`` the x axis in
        ``[0, W)``. For square bins of side ``bin_um``, ``col = floor((x - min(x)) / bin_um)`` and
        ``row = floor((y - min(y)) / bin_um)``, which is how :func:`spectrast.gr.moran_null` builds
        the lattice.
    HW : tuple of int
        The lattice shape ``(H, W)``.
    mask : array-like of bool, shape (H, W)
        Bins the statistic is computed over, normally the bins that hold at least one cell
        (``mask[row, col] = True``). Cells in bins outside the mask are ignored.
    radii : sequence of int
        Disc radii in BINS, one curve point each (``R = len(radii)``). For ``texture`` each radius is
        a band centre at wavelength ``2 * r`` bins, and at least two radii are needed.
    stat : str, optional
        One of :data:`PERM_STATS`: ``"moran"`` (default), ``"geary"``, ``"general_g"``, ``"losh"``,
        ``"focality"`` or ``"texture"``. Any other value raises ``ValueError``.
    n_perm : int, optional
        Number of permutations ``P``. Default ``100``.
    higher : bool, optional
        ``None`` (default) returns only the two-sided ``p_two``. ``True`` also returns a one-sided
        ``p`` counting null values at or above the observed one; ``False``, at or below it.
    strata : array-like of shape (n_cells,), optional
        Stratum label per cell; each permutation moves values only within a label (N1). ``None``
        (default) permutes over all cells (N0).
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Default ``0``. The same seed and the same strata give
        the same draws.
    kind : str, optional
        Neighbourhood kernel at each radius ``r``, centre bin always excluded: ``"disc"`` (default,
        bins within distance ``r``), ``"queen"`` (the full ``(2r + 1)`` square) or ``"gaussian"`` (a
        Gaussian with sigma ``r / 2``, cut off at that square). Ignored for ``texture``.
    keep_null : bool, optional
        Return the permuted curves as ``null``. Default ``True``.

    Returns
    -------
    dict
        - ``stat`` : the statistic, as given.
        - ``obs`` : the observed curve, ``(R,)``.
        - ``mu``, ``sd`` : mean and standard deviation (``ddof=1``) of the null at each radius,
          ``(R,)``.
        - ``z`` : ``(obs - mu) / sd``, signed, ``(R,)``.
        - ``p_two`` : two-sided permutation p-value ``min(1, 2 * min(p_up, p_down))``, ``(R,)``,
          where ``p_up = (1 + #(null >= obs)) / (P + 1)`` and ``p_down`` counts ``null <= obs``.
        - ``higher`` : as given.
        - ``p`` : the one-sided p-value in the ``higher`` direction (``p_up`` or ``p_down``),
          ``(R,)``; only when ``higher`` is not ``None``.
        - ``null`` : the permuted curves, ``(P, R)``; omitted when ``keep_null=False``.
    """
    fn = _stat_fn(stat)
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    V = np.concatenate([x[:, None], _permute_values(x, n_perm, rng, strata)], axis=1)
    G = _rasterize_matrix(V, row, col, HW)
    A = fn(G, np.asarray(mask, bool), list(radii), kind=kind)                   # (R, P+1)
    obs = A[:, 0]
    nul = A[:, 1:].T                                                            # (P, R)
    mu = nul.mean(0)
    sd = nul.std(0, ddof=1) + 1e-300
    z = (obs - mu) / sd
    p_up = (1.0 + (nul >= obs[None, :]).sum(0)) / (n_perm + 1.0)
    p_dn = (1.0 + (nul <= obs[None, :]).sum(0)) / (n_perm + 1.0)
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_up, p_dn))
    out = dict(stat=stat, obs=obs, mu=mu, sd=sd, z=z, p_two=p_two, higher=higher)
    if higher is not None:
        out["p"] = p_up if higher else p_dn
    if keep_null:
        out["null"] = nul
    return out


def moran_permutation_null(values, row, col, HW, mask, radii, *, n_perm=200, strata=None, seed=0,
                           kind="disc", keep_null=True):
    """Per-gene, per-radius null for one gene's multiscale Moran curve.

    The same draws and values as ``permutation_null(..., stat="moran", higher=True)``, with the
    observed curve named ``I``.

    Parameters
    ----------
    values : array-like of shape (n_cells,)
        The gene's per-cell values, e.g. log-normalised expression.
    row, col, HW, mask
        The lattice, as in :func:`permutation_null`.
    radii : sequence of int
        Disc radii in BINS; ``R = len(radii)``.
    n_perm : int, optional
        Number of permutations ``P``. Default ``200``.
    strata : array-like of shape (n_cells,), optional
        Stratum label per cell; permutation stays within a label (N1). ``None`` (default) is N0.
    seed : int, optional
        Generator seed (``numpy.random.default_rng``). Default ``0``. Identical seed + identical
        strata gives identical draws.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``.
    keep_null : bool, optional
        Return the permuted curves as ``null``. Default ``True``.

    Returns
    -------
    dict
        - ``I`` : the observed Moran curve, ``(R,)``.
        - ``mu``, ``sd`` : mean and standard deviation (``ddof=1``) of the null, ``(R,)``.
        - ``p`` : one-sided upper permutation p-value ``(1 + #(null >= I)) / (P + 1)``, ``(R,)``.
        - ``z`` : ``(I - mu) / sd``, ``(R,)``.
        - ``null`` : the permuted curves, ``(P, R)``; omitted when ``keep_null=False``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    V = np.concatenate([x[:, None], _permute_values(x, n_perm, rng, strata)], axis=1)
    G = _rasterize_matrix(V, row, col, HW)
    A = sp.morans_i_multiscale(G, np.asarray(mask, bool), list(radii), kind=kind)   # (R, P+1)
    I = A[:, 0]
    nul = A[:, 1:].T                                                                # (P, R)
    mu = nul.mean(0)
    sd = nul.std(0, ddof=1) + 1e-300
    p = (1.0 + (nul >= I[None, :]).sum(0)) / (n_perm + 1.0)
    z = (I - mu) / sd
    out = dict(I=I, mu=mu, sd=sd, p=p, z=z)
    if keep_null:
        out["null"] = nul
    return out


def gene_features(grids, mask):
    """Cheap per-gene summaries of the BINNED field, the predictors a surrogate null is fitted on.

    Binned rather than cell-level because the statistic is computed on bins, and because
    rasterisation is about 1% of a permutation sweep's cost -- so these are effectively free for
    every gene even when permuting every gene is not.

    The choice of summaries is not arbitrary. :func:`moran_analytic_z` shows that for Moran the null
    variance depends on the gene ONLY through the kurtosis of the centred bin values; the others have
    no closed form, so scale, skew and sparsity are carried as well and the fit is left to decide.

    Parameters
    ----------
    grids : array-like of shape (H, W, n_genes) or (H, W)
        Binned values per gene (per-bin means, zero on empty bins), as the permutation nulls
        rasterise them. A 2-D grid is one gene.
    mask : array-like of bool, shape (H, W)
        Bins to summarise, normally the bins that hold at least one cell.

    Returns
    -------
    numpy.ndarray of shape (n_genes, 6)
        ``(n_genes, n_features)`` with a constant column first. Over the ``N`` masked bins, the
        columns are: a constant ``1``; the log variance (``ddof=0``); the log mean (a mean at or
        below zero is clipped to ``1e-300`` first); the kurtosis ``b2 = N * m4 / m2**2`` of the
        centred values (not the excess kurtosis); ``log1p(abs(skew))``; and the fraction of bins at
        or below zero.
    """
    mb = np.asarray(mask, bool)
    G = np.asarray(grids, float)
    if G.ndim == 2:
        G = G[..., None]
    v = G[mb]                                                   # (N, G)
    n = v.shape[0]
    mu = v.mean(0)
    c = v - mu
    m2 = (c ** 2).sum(0); m3 = (c ** 3).sum(0); m4 = (c ** 4).sum(0)
    safe2 = np.maximum(m2, 1e-300)
    b2 = n * m4 / safe2 ** 2                                    # kurtosis: the analytic predictor
    b1 = np.sqrt(n) * m3 / safe2 ** 1.5                         # skew
    return np.column_stack([
        np.ones_like(mu), np.log(np.maximum(m2 / n, 1e-300)), np.log(np.maximum(mu, 1e-300)),
        b2, np.log1p(np.abs(b1)), (v <= 0).mean(0)])


def fit_surrogate_null(values, row, col, HW, mask, radii, *, stat="moran", n_sample=200,
                       n_perm=100, seed=0, kind="disc"):
    """Fit a SURROGATE null: permute a sample of genes, predict the rest from their features.

    A full permutation sweep costs ~11 h for 10,000 genes and six statistics. The null moments,
    however, are largely a function of the lattice plus a few summaries of each gene's value
    distribution -- that is exactly what :func:`moran_analytic_z` exploits for Moran, where the only
    per-gene quantity is kurtosis. Where no closed form exists, the same structure can be fitted
    instead of derived: permute ``n_sample`` genes, regress the null moments on
    :func:`gene_features`, and predict for everything else.

    **DO NOT JUDGE THIS BY THE FIT'S R^2.** Measured on trem2_merfish, 60 genes fitted and 60
    HELD OUT with a real sweep for truth, the fit quality is anti-correlated with what matters:

    ============  ========  =============  ==============  ===========  ==================
    statistic     R^2(sd)   rho(z) held    rho(gene rank)  top-20 ovl   verdict
    ============  ========  =============  ==============  ===========  ==================
    moran            0.065         +0.997          +0.996         1.00  use
    geary            0.954         +0.997          +0.996         0.95  use
    general_g        0.983         +0.997          +0.991         0.95  use
    focality         0.737         +0.979          +0.946         0.95  use
    losh             0.966         +0.933          +0.885         0.85  borderline
    texture          0.991         +0.689          +0.648         0.75  NO -- real sweep
    ============  ========  =============  ==============  ===========  ==================

    Texture has the BEST fit and the WORST agreement; Moran has the worst fit and perfect agreement.
    Moran's null barely moves between genes, so there is nothing to predict and the sample mean
    serves. Texture's null sd spans orders of magnitude between genes, so a residual that is small in
    log terms is still large where it lands, and ``z = (obs - mu)/sd`` inherits all of it -- its z
    ratio has an IQR of 0.73 against 0.11 for Moran. An R^2 on the fitted quantity says nothing about
    the derived z, which is why the surrogate is judged by held-out agreement in z.

    **This is for RANKING, and not for texture.** It is a fitted approximation to a null, not a null.
    Use the real sweep for texture, for LOSH where the top set matters, and anywhere a calibrated tail
    probability is needed. The z ratio is centred on 1.00 for every statistic, so where the ranking
    holds the magnitude is roughly right too -- but that is a measured convenience, not a guarantee.

    Parameters
    ----------
    values : array-like of shape (n_cells, n_genes) or (n_cells,)
        All genes' per-cell values. A 1-D input is one gene.
    row, col, HW, mask, radii
        The lattice and radii, as in :func:`permutation_null`.
    stat : str, optional
        One of :data:`PERM_STATS`. Default ``"moran"``. This function does not refuse
        ``"texture"``; :func:`gene_inference` does.
    n_sample : int, optional
        Genes to permute, drawn at random without replacement. The rest are predicted. Default
        ``200``, capped at ``n_genes``.
    n_perm : int, optional
        Permutations per sampled gene. Default ``100``.
    seed : int, optional
        Seeds the gene sample and, through it, each sampled gene's permutations. Default ``0``.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``.

    Returns
    -------
    dict
        - ``stat`` : the statistic, as given.
        - ``obs`` : the observed curves, ``(R, G)``.
        - ``mu``, ``sd`` : null mean and standard deviation for every gene, ``(R, G)``. Sampled
          genes keep their measured values. For the rest, ``mu`` and ``log(sd)`` are predicted from
          :func:`gene_features` by a least-squares fit at each radius; where fewer than eight
          sampled genes have finite values at a radius, the sample median is used for every gene.
        - ``z`` : ``(obs - mu) / sd``, ``(R, G)``.
        - ``sample`` : sorted column indices of the genes actually permuted, ``(n_sample,)``.
        - ``r2_sd``, ``r2_mu`` : per-radius fit quality, the in-sample R^2 of the fits of
          ``log(sd)`` and ``mu``, ``(R,)``; NaN where the median was used.
        - ``n_perm`` : as given.
    """
    fn = _stat_fn(stat)
    mb = np.asarray(mask, bool)
    V = np.asarray(values, float)
    if V.ndim == 1:
        V = V[:, None]
    n_genes = V.shape[1]
    rng = np.random.default_rng(seed)

    grids = _rasterize_matrix(V, row, col, HW)                      # cheap: ~1% of a sweep
    obs = fn(grids, mb, list(radii), kind=kind)                     # (R, G)
    F = gene_features(grids, mb)                                    # (G, k)

    n_sample = int(min(n_sample, n_genes))
    sample = np.sort(rng.choice(n_genes, n_sample, replace=False))
    R = len(list(radii))
    mu_s = np.empty((R, n_sample)); sd_s = np.empty((R, n_sample))
    for i, j in enumerate(sample):
        r = permutation_null(V[:, j], row, col, HW, mb, radii, stat=stat, n_perm=n_perm,
                             seed=int(rng.integers(1 << 31)), kind=kind, keep_null=False)
        mu_s[:, i] = r["mu"]; sd_s[:, i] = r["sd"]

    A = F[sample]
    mu_hat = np.empty((R, n_genes)); sd_hat = np.empty((R, n_genes))
    r2_sd = np.full(R, np.nan); r2_mu = np.full(R, np.nan)
    for k in range(R):
        for tgt, out, r2 in ((np.log(np.maximum(sd_s[k], 1e-300)), sd_hat, r2_sd),
                             (mu_s[k], mu_hat, r2_mu)):
            good = np.isfinite(tgt) & np.isfinite(A).all(1)
            if good.sum() < A.shape[1] + 2:
                out[k] = np.nanmedian(tgt); continue
            beta, *_ = np.linalg.lstsq(A[good], tgt[good], rcond=None)
            pred_s = A[good] @ beta
            ss_res = ((tgt[good] - pred_s) ** 2).sum()
            ss_tot = ((tgt[good] - tgt[good].mean()) ** 2).sum()
            r2[k] = 1.0 - ss_res / max(ss_tot, 1e-300)
            out[k] = F @ beta
    sd_hat = np.exp(sd_hat)
    # the sampled genes keep their MEASURED moments: never predict what was computed
    mu_hat[:, sample] = mu_s; sd_hat[:, sample] = sd_s
    z = (obs - mu_hat) / np.maximum(sd_hat, 1e-300)
    return dict(stat=stat, obs=obs, mu=mu_hat, sd=sd_hat, z=z, sample=sample,
                r2_sd=r2_sd, r2_mu=r2_mu, n_perm=n_perm)


# --------------------------------------------------------------------------- A0
def moran_analytic_moments(mask, radii, kind="disc"):
    """Cliff & Ord's S0, S1, S2 for the engine's weights, shared across genes.

    The engine's weights are ``w_ij = 1/n_i`` for *j* in the disc of *i* with both bins occupied,
    zero on the diagonal, and zero rows where a bin has no occupied neighbour. For a symmetric
    support with row standardisation:

        S0 = Σ_i 1[n_i > 0]
        S1 = Σ_i 1/n_i + Σ_i c_i/n_i,   c_i = Σ_{j∈D(i)} 1/n_j = (K * (m/n))_i
        S2 = Σ_{n_i>0} (1 + c_i)²

    ``n`` and ``c`` are two FFT convolutions per radius. Checked against an explicit double sum in
    ``tests/test_null.py::test_G1_analytic_S_terms_match_explicit_sums``.

    Parameters
    ----------
    mask : array-like of bool, shape (H, W)
        The occupied bins: the same mask the scalograms are computed on.
    radii : sequence of int
        Disc radii in BINS; each is passed through ``int``. ``R = len(radii)``.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``. The S-terms are
        derived for 0/1 weights before row standardisation, which ``"disc"`` and ``"queen"`` give.
        ``"gaussian"`` is accepted, but its neighbour counts are rounded weight sums and the moments
        do not describe the engine's Gaussian weights.

    Returns
    -------
    dict
        - ``N`` : number of masked bins, int.
        - ``S0``, ``S1``, ``S2`` : Cliff & Ord's weight sums at each radius, ``(R,)`` float.
        - ``radii`` : the radii, as an array.

        Pass it to :func:`moran_analytic_z` or :func:`geary_analytic_z`.
    """
    from scipy.fft import next_fast_len
    from . import _fft
    mb = np.asarray(mask, bool)
    m = mb.astype(float)
    H, W = m.shape
    N = int(mb.sum())
    radii = list(radii)
    rmax = max(radii)
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W + 2 * rmax)
    Fm = _fft.rfft2(m, s=(fh, fw))
    S0 = np.empty(len(radii)); S1 = np.empty(len(radii)); S2 = np.empty(len(radii))
    for k, r in enumerate(radii):
        K = sp.neighbor_kernel(int(r), kind)
        off = (K.shape[0] - 1) // 2
        Fk = _fft.rfft2(K, s=(fh, fw))
        n = _fft.irfft2(Fm * Fk, s=(fh, fw))[off:off + H, off:off + W]
        n = np.where(mb, np.round(n), 0.0)                  # integer counts; kill FFT round-off
        has = n > 0
        inv = np.where(has, 1.0 / np.maximum(n, 1.0), 0.0)
        Fi = _fft.rfft2(inv, s=(fh, fw))
        c = _fft.irfft2(Fi * Fk, s=(fh, fw))[off:off + H, off:off + W]
        c = np.where(has, c, 0.0)
        S0[k] = has.sum()
        S1[k] = inv[has].sum() + (c[has] * inv[has]).sum()
        S2[k] = ((1.0 + c[has]) ** 2).sum()
    return dict(N=N, S0=S0, S1=S1, S2=S2, radii=np.asarray(radii))


def moran_analytic_z(scalogram, grids, mask, moments):
    """Per-gene analytic z at every radius from the shared moments and each gene's kurtosis.

    Cliff & Ord (1981) under randomisation, for ``I_CO = (N/S0) z'Wz / z'z``::

        E[I_CO]   = -1/(N-1)
        Var[I_CO] = [N((N²-3N+3)S1 - N S2 + 3S0²) - b2((N²-N)S1 - 2N S2 + 6S0²)]
                    / [(N-1)(N-2)(N-3)S0²]  -  1/(N-1)²

    The engine reports ``I = (S0/N) I_CO`` (row sums are 1 where a neighbour exists, so no N/S0
    prefactor), hence ``E[I] = (S0/N) E[I_CO]`` and ``Var[I] = (S0/N)² Var[I_CO]``. ``b2`` is the
    kurtosis of the centred bin values over the N occupied bins — the only per-gene quantity.

    Parameters
    ----------
    scalogram : array-like of shape (R, G)
        Observed Moran curves, from :func:`spectrast.spatial.morans_i_multiscale` on ``grids`` and
        ``mask`` at the radii the moments were built for.
    grids : array-like of shape (H, W, G) or (H, W)
        The binned values the scalogram was computed on. A 2-D grid is one gene.
    mask : array-like of bool, shape (H, W)
        The mask the moments were built from.
    moments : dict
        The output of :func:`moran_analytic_moments`.

    Returns
    -------
    E : numpy.ndarray of shape (R, G)
        Null mean of ``I`` at each radius; the same for every gene.
    Var : numpy.ndarray of shape (R, G)
        Null variance of ``I``.
    z : numpy.ndarray of shape (R, G)
        ``(scalogram - E) / sqrt(Var)``. NaN for a gene that is constant over the mask.
    """
    mb = np.asarray(mask, bool)
    G = np.asarray(grids, float)
    if G.ndim == 2:
        G = G[..., None]
    N = moments["N"]
    S0, S1, S2 = moments["S0"], moments["S1"], moments["S2"]
    zb = G[mb]                                              # (N, G)
    zc = zb - zb.mean(0, keepdims=True)
    m2 = (zc ** 2).sum(0)
    m4 = (zc ** 4).sum(0)
    b2 = np.where(m2 > 0, N * m4 / np.maximum(m2 ** 2, 1e-300), np.nan)   # (G,)
    n_ = float(N)
    E_co = -1.0 / (n_ - 1.0)
    num_a = n_ * ((n_ ** 2 - 3 * n_ + 3) * S1 - n_ * S2 + 3 * S0 ** 2)          # (R,)
    num_b = (n_ ** 2 - n_) * S1 - 2 * n_ * S2 + 6 * S0 ** 2                        # (R,)
    den = (n_ - 1) * (n_ - 2) * (n_ - 3) * S0 ** 2
    V_co = (num_a[:, None] - b2[None, :] * num_b[:, None]) / den[:, None] - 1.0 / (n_ - 1) ** 2
    f = (S0 / n_)[:, None]
    E = np.broadcast_to(f * E_co, V_co.shape).copy()
    V = f ** 2 * V_co
    A = np.asarray(scalogram, float)
    z = (A - E) / np.sqrt(np.maximum(V, 1e-300))
    return E, V, z


def geary_analytic_z(scalogram, grids, mask, moments):
    """Per-gene analytic z for Geary's C, from the SAME shared moments Moran uses.

    The engine computes ``C = ((N-1)/(2 S0)) * sum_ij w_ij (x_i-x_j)^2 / sum_i (x_i-xbar)^2`` with
    ``w_ij = 1/n_i`` on the centre-excluded disc -- which is exactly Cliff & Ord's row-standardised
    form, so their randomisation moments apply with the S-terms :func:`moran_analytic_moments`
    already builds. The derivation was checked rather than transcribed: for this weight support
    ``S1 = sum_i 1/n_i + sum_i c_i/n_i`` and ``S2 = sum_i (1 + c_i)^2`` reduce to the standard
    ``S1 = 1/2 sum_ij (w_ij + w_ji)^2`` and ``S2 = sum_i (sum_j w_ij + sum_j w_ji)^2`` because the
    disc support is symmetric.

    Under randomisation (Cliff & Ord 1981)::

        E[C]   = 1
        Var[C] = [ (N-1) S1 (N^2 - 3N + 3 - (N-1) b2)
                   - (1/4)(N-1) S2 (N^2 + 3N - 6 - (N^2 - N + 2) b2)
                   + S0^2 (N^2 - 3 - (N-1)^2 b2) ] / [ N (N-2)(N-3) S0^2 ]

    ``b2`` is the kurtosis of the centred bin values, the only per-gene quantity -- so this costs one
    pass over the occupied bins per gene once the S-terms exist, against a full permutation sweep.

    **MEASURED AGREEMENT.** Against the reference sweep (all 300 genes, n_perm=200): rho(z) +0.999,
    rho(gene rank) +0.998, top-25 overlap 0.96, median z ratio 1.07 (IQR 0.13). The mean offset below
    is real but propagates to only a 7% inflation in z -- an earlier draft of this docstring treated
    it as a heavier limitation than the reference shows it to be.

    On a 24-gene subsample, which is how the moments were first checked:

    ========================================  ========  ========
    quantity                                     Moran     Geary
    ========================================  ========  ========
    rho(z_analytic, z_permutation)             +0.9980   +0.9979
    median sd_analytic / sd_permutation          0.979     0.966
    median ``|E_analytic - E_perm|`` / sd        0.056     0.504
    ========================================  ========  ========

    The mean offset is the one real difference, and it is not a defect in the derivation: the two
    routes answer DIFFERENT nulls. The analytic moments are classical randomisation over fixed bin
    values; the permutation null permutes cell values and RE-RASTERISES, so bin re-averaging is part
    of it. For Moran these nearly coincide (E[I] is near zero and the offset is 0.06 sd); for Geary
    E[C] = 1 and the re-averaging shifts the permutation mean by 0.05 to 0.74 sd depending on radius,
    with a further +-0.28 sd spread BETWEEN genes -- so a single per-lattice calibration cannot
    remove it either.

    **Use this for ranking, not for an absolute p.** At rho 0.998 with the permutation z it orders
    genes essentially identically at a fraction of the cost. If a calibrated tail probability is
    wanted, run the permutation null.

    **C DECREASES with structure** (C ~ 1 - I for this implementation), so a structured gene has a
    NEGATIVE z here where Moran's is positive. The sign is returned as computed and is not flipped;
    callers ranking "most structured" must sort ascending, or flip the sign of ``z``.

    Parameters
    ----------
    scalogram : array-like of shape (R, G)
        Observed Geary curves, from :func:`spectrast.spatial.gearys_c_multiscale` on ``grids`` and
        ``mask`` at the radii the moments were built for.
    grids, mask
        As in :func:`moran_analytic_z`.
    moments : dict
        The output of :func:`moran_analytic_moments`, the same S-terms Moran uses.

    Returns
    -------
    E : numpy.ndarray of shape (R, G)
        Null mean of ``C``: ``1`` everywhere.
    Var : numpy.ndarray of shape (R, G)
        Null variance of ``C``.
    z : numpy.ndarray of shape (R, G)
        ``(scalogram - E) / sqrt(Var)``, negative for a structured gene. NaN for a gene that is
        constant over the mask.
    """
    mb = np.asarray(mask, bool)
    G = np.asarray(grids, float)
    if G.ndim == 2:
        G = G[..., None]
    N = float(moments["N"])
    S0, S1, S2 = moments["S0"], moments["S1"], moments["S2"]
    zb = G[mb]
    zc = zb - zb.mean(0, keepdims=True)
    m2 = (zc ** 2).sum(0)
    m4 = (zc ** 4).sum(0)
    b2 = np.where(m2 > 0, N * m4 / np.maximum(m2 ** 2, 1e-300), np.nan)      # (G,)

    t1 = (N - 1) * S1[:, None] * (N ** 2 - 3 * N + 3 - (N - 1) * b2[None, :])
    t2 = 0.25 * (N - 1) * S2[:, None] * (N ** 2 + 3 * N - 6 - (N ** 2 - N + 2) * b2[None, :])
    t3 = (S0 ** 2)[:, None] * (N ** 2 - 3 - (N - 1) ** 2 * b2[None, :])
    den = N * (N - 2) * (N - 3) * (S0 ** 2)[:, None]
    V = (t1 - t2 + t3) / den
    E = np.ones_like(V)
    A = np.asarray(scalogram, float)
    z = (A - E) / np.sqrt(np.maximum(V, 1e-300))
    return E, V, z


def general_g_binary_moments(mask, radii, kind="disc"):
    """Binary-weight sums W, S1, S2 for General G. NOT the row-standardised set Moran and Geary use.

    General G is the one statistic here whose weights are NOT row standardised: the engine computes
    ``G = sum_ij w_ij x_i x_j / sum_ij x_i x_j`` over ordered pairs i != j with ``w_ij`` the binary,
    centre-excluded disc. Feeding it :func:`moran_analytic_moments` would silently substitute the
    wrong weight convention -- which is exactly the trap the Getis-Ord paper's own notation invites.

    For a SYMMETRIC BINARY support the standard sums collapse, and this is why General G's moments
    are cheaper to build than Geary's rather than harder:

        W  = sum_i n_i                            n_i = neighbour count, one convolution per radius
        S1 = 1/2 sum_ij (w_ij + w_ji)^2 = 2 W     because w_ij = w_ji in {0, 1}
        S2 = sum_i (sum_j w_ij + sum_j w_ji)^2 = 4 sum_i n_i^2

    Only occupied bins contribute, and a bin with no occupied neighbour contributes nothing to any
    sum.

    Parameters
    ----------
    mask, radii
        As in :func:`moran_analytic_moments`.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``. The sums assume a
        0/1 support (``"disc"`` or ``"queen"``); with ``"gaussian"`` the neighbour counts are
        rounded weight sums and the sums do not describe the engine's Gaussian weights.

    Returns
    -------
    dict
        - ``N`` : number of masked bins, int.
        - ``W`` : sum of the neighbour counts ``n_i`` over the occupied bins, ``(R,)`` float.
        - ``S1`` : ``2 W``, ``(R,)``.
        - ``S2`` : ``4 * sum_i n_i^2``, ``(R,)``.
        - ``radii`` : the radii, as an array.

        Pass it to :func:`general_g_analytic_z`.
    """
    from scipy.fft import next_fast_len
    from . import _fft
    mb = np.asarray(mask, bool)
    m = mb.astype(float)
    H, W_ = m.shape
    N = int(mb.sum())
    radii = list(radii)
    rmax = max(radii)
    fh, fw = next_fast_len(H + 2 * rmax), next_fast_len(W_ + 2 * rmax)
    Fm = _fft.rfft2(m, s=(fh, fw))
    Wt = np.empty(len(radii)); S1 = np.empty(len(radii)); S2 = np.empty(len(radii))
    for k, r in enumerate(radii):
        K = sp.neighbor_kernel(int(r), kind)
        off = (K.shape[0] - 1) // 2
        Fk = _fft.rfft2(K, s=(fh, fw))
        n = _fft.irfft2(Fm * Fk, s=(fh, fw))[off:off + H, off:off + W_]
        n = np.where(mb, np.round(n), 0.0)                 # integer counts; kill FFT round-off
        Wt[k] = n.sum()
        S1[k] = 2.0 * Wt[k]
        S2[k] = 4.0 * (n ** 2).sum()
    return dict(N=N, W=Wt, S1=S1, S2=S2, radii=np.asarray(radii))


def general_g_analytic_z(scalogram, grids, mask, moments):
    """Per-gene analytic z for General G, from Getis & Ord (1992) randomisation moments.

    ``moments`` must come from :func:`general_g_binary_moments`, not from
    :func:`moran_analytic_moments` -- the weight conventions differ and mixing them is silent.

    With ``m_j = sum_i x_i^j`` over the N occupied bins and ``n^(4) = N(N-1)(N-2)(N-3)``::

        E[G]   = W / (N(N-1))
        Var[G] = [B0 m2^2 + B1 m4 + B2 m1^2 m2 + B3 m1 m3 + B4 m1^4] / [(m1^2 - m2)^2 n^(4)]
                 - E[G]^2

    with the B coefficients built from N, W, S1 and S2::

        B0 = (N^2 - 3N + 3) S1 - N S2 + 3 W^2
        B1 = -[(N^2 - N) S1 - 2 N S2 + 6 W^2]
        B2 = -[2 N S1 - (N + 3) S2 + 6 W^2]
        B3 = 4 (N - 1) S1 - 2 (N + 1) S2 + 8 W^2
        B4 = S1 - S2 + W^2

    **HIGH G means high values cluster**, and unlike Geary the direction here is not an inversion of
    Moran. Note that the curve RISES with radius because G is cumulative; that is a property of the
    construction, not of the direction of structure, and must not be used to infer a tail.

    **MEASURED AGREEMENT** against the reference sweep (trem2_merfish, all 300 genes, n_perm=200):

        rho(z_analytic, z_reference)       +0.999
        rho(gene rank)                     +0.997
        top-25 overlap                      0.96
        median z_analytic / z_reference     0.97   (IQR 0.11)

    The weight sums were checked first against an explicit double sum over the occupied bins and
    match exactly at every radius.

    **A caution about synthetic validation.** An earlier check on a
    synthetic 48x48 lattice at 2.45 cells per bin reported the analytic sd as 19% too large, giving a
    z ratio of 0.805, and that was written into this docstring as a limitation. On a real section the
    ratio is 0.97. The bias was a property of the test lattice's occupancy, not of the derivation.
    Validate against real tissue, or say which lattice the number came from.

    Parameters
    ----------
    scalogram : array-like of shape (R, G)
        Observed General G curves, from ``spectrast.spatial.general_g_multiscale`` on ``grids`` and
        ``mask`` at the radii the moments were built for.
    grids : array-like of shape (H, W, G) or (H, W)
        The binned values the scalogram was computed on, NOT centred; General G is read on
        non-negative values. A 2-D grid is one gene.
    mask : array-like of bool, shape (H, W)
        The mask the moments were built from.
    moments : dict
        The output of :func:`general_g_binary_moments`.

    Returns
    -------
    E : numpy.ndarray of shape (R, G)
        Null mean ``W / (N(N-1))`` at each radius; the same for every gene.
    Var : numpy.ndarray of shape (R, G)
        Null variance of ``G``.
    z : numpy.ndarray of shape (R, G)
        ``(scalogram - E) / sqrt(Var)``; positive where high values cluster.
    """
    mb = np.asarray(mask, bool)
    A3 = np.asarray(grids, float)
    if A3.ndim == 2:
        A3 = A3[..., None]
    v = A3[mb]                                              # (N, G) the values themselves, NOT centred
    N = float(moments["N"])
    W = moments["W"]; S1 = moments["S1"]; S2 = moments["S2"]
    m1 = v.sum(0); m2 = (v ** 2).sum(0); m3 = (v ** 3).sum(0); m4 = (v ** 4).sum(0)

    B0 = (N ** 2 - 3 * N + 3) * S1 - N * S2 + 3 * W ** 2
    B1 = -((N ** 2 - N) * S1 - 2 * N * S2 + 6 * W ** 2)
    B2 = -(2 * N * S1 - (N + 3) * S2 + 6 * W ** 2)
    B3 = 4 * (N - 1) * S1 - 2 * (N + 1) * S2 + 8 * W ** 2
    B4 = S1 - S2 + W ** 2

    n4 = N * (N - 1) * (N - 2) * (N - 3)
    den_pair = (m1 ** 2 - m2)                               # sum over ordered pairs i != j
    num = (B0[:, None] * (m2 ** 2)[None, :]
           + B1[:, None] * m4[None, :]
           + B2[:, None] * (m1 ** 2 * m2)[None, :]
           + B3[:, None] * (m1 * m3)[None, :]
           + B4[:, None] * (m1 ** 4)[None, :])
    EG = (W / (N * (N - 1)))[:, None] * np.ones((1, v.shape[1]))
    EG2 = num / (np.maximum(den_pair ** 2, 1e-300)[None, :] * n4)
    V = EG2 - EG ** 2
    z = (np.asarray(scalogram, float) - EG) / np.sqrt(np.maximum(V, 1e-300))
    return EG, V, z


#: What each statistic can be given, what it gets by DEFAULT, and why -- every entry measured
#: against the reference sweep (trem2_merfish, all 300 genes, n_perm=200).
#:
#: The default is the cheapest route that was measured to be trustworthy for RANKING genes, which is
#: what per-gene inference is used for here. It is never the cheapest route available: texture could
#: be given a surrogate and is not, because the surrogate was measured to be unstable for it.
ROUTES = {
    "moran": dict(analytic=True, surrogate=True, default="analytic",
                  note="analytic rho(z) +0.999 vs reference, z ratio 1.01"),
    "geary": dict(analytic=True, surrogate=True, default="analytic",
                  note="analytic rho(z) +0.999 vs reference, z ratio 1.07"),
    "general_g": dict(analytic=True, surrogate=True, default="analytic",
                      note="analytic rho(z) +0.999 vs reference, z ratio 0.97"),
    "focality": dict(analytic=False, surrogate=True, default="surrogate",
                     note="no closed form; surrogate rho(z) +0.989 at a 2% sample"),
    "losh": dict(analytic=False, surrogate=True, default="surrogate",
                 caveat=("RANKING ONLY. The surrogate reaches rho(gene rank) +0.92 at a 10% sample "
                         "but its top-25 overlap with the reference is 0.56-0.80 at EVERY sample "
                         "fraction and does not improve with more sampling -- it orders the bulk and "
                         "misses the top. Use method='permutation' when the top set matters. There "
                         "is no analytic route: the Ord & Getis (2012) chi-square result describes "
                         "the LOCAL H_i, while this scalogram reduces the whole H map to a Gini, and "
                         "a parametric chi-square null for that Gini was measured to be 5x too "
                         "narrow at r=2 bins and 140x too narrow at r=40, with a mean off by 3-4x."),
                 note="surrogate rho(gene rank) +0.92 at 10%; top-25 overlap 0.56-0.80"),
    "texture": dict(analytic=False, surrogate=False, default="permutation",
                    caveat=("The surrogate is REFUSED for texture, not merely discouraged: its "
                            "agreement with the reference is NON-MONOTONIC in sample fraction "
                            "(+0.625 at 2%, +0.757 at 5%, +0.389 at 10%, +0.656 at 20%) with a "
                            "z-ratio IQR up to 1.62. More data making a fit worse means it is not "
                            "converging, so a larger sample is not a remedy."),
                    note="permutation only; the surrogate does not converge"),
}


def _resolve_route(stat, method):
    if stat not in ROUTES:
        raise ValueError(f"unknown statistic {stat!r}; choose from {list(ROUTES)}")
    spec = ROUTES[stat]
    if method == "auto":
        return spec["default"], spec
    if method == "analytic" and not spec["analytic"]:
        raise ValueError(
            f"{stat!r} has no analytic null. "
            + spec.get("caveat", f"Use method='permutation' or 'surrogate' (default: "
                                 f"{spec['default']})."))
    if method == "surrogate" and not spec["surrogate"]:
        raise ValueError(f"the surrogate null is refused for {stat!r}. " + spec.get("caveat", ""))
    if method not in ("analytic", "surrogate", "permutation"):
        raise ValueError(f"unknown method {method!r}; choose auto, analytic, surrogate "
                         "or permutation")
    return method, spec


def gene_inference(values, row, col, HW, mask, radii, *, stat="moran", method="auto", n_perm=100,
                   n_sample=None, seed=0, kind="disc", genes=None):
    """Per-gene z for one statistic, by whichever null is appropriate. One entry point, three routes.

    ``method="auto"`` picks the cheapest route MEASURED to be trustworthy for ranking genes, which
    differs by statistic and is recorded in :data:`ROUTES`:

    =========  ============  ===========================================================
    statistic  default       why
    =========  ============  ===========================================================
    moran      analytic      closed form; rho(z) +0.999 against the reference sweep
    geary      analytic      closed form; rho(z) +0.999
    general_g  analytic      closed form; rho(z) +0.999
    focality   surrogate     no closed form; surrogate +0.989 at a 2% sample
    losh       surrogate     no closed form; RANKING ONLY -- see the warning it emits
    texture    permutation   the surrogate does not converge for it, and is refused
    =========  ============  ===========================================================

    Every statistic accepts ``method="permutation"``, which is always available and always the
    reference behaviour. ``method="analytic"`` and ``method="surrogate"`` raise where the route does
    not exist, with the measured reason rather than a bare NotImplemented.

    Parameters
    ----------
    values : array-like of shape (n_cells,) or (n_cells, n_genes)
        Per-cell values, one column per gene. A surrogate needs many genes to fit on: it is refused
        below 25 genes, so a 1-D input is accepted for the other two routes and refused for the
        surrogate.
    row, col, HW, mask, radii
        The lattice and radii, as in :func:`permutation_null`.
    stat : str, optional
        One of the keys of :data:`ROUTES`, the same six names as :data:`PERM_STATS`. Default
        ``"moran"``.
    method : str, optional
        ``"auto"`` (default), ``"analytic"``, ``"surrogate"`` or ``"permutation"``; see the table
        above. Where the statistic has a caveat in :data:`ROUTES` and the route taken is its
        default, a ``RuntimeWarning`` repeats the caveat.
    n_perm : int, optional
        Permutations per gene for the permutation route, and per sampled gene for the surrogate.
        Not used by the analytic route. Default ``100``.
    n_sample : int, optional
        Genes to permute when fitting a surrogate. Default is 2% of the panel, floored at 25,
        which is the fraction the benchmark found sufficient for the statistics that use one.
        ``None`` (or ``0``) gives that default. Used by the surrogate route only.
    seed : int, optional
        Default ``0``. The permutation route seeds gene ``j`` with ``seed + j``; the surrogate route
        passes it to :func:`fit_surrogate_null`. Not used by the analytic route.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``. The analytic
        moments assume a 0/1 kernel (see :func:`moran_analytic_moments`).
    genes : array-like, optional
        Gene names, one per column of ``values``. They are only copied into the result as
        ``genes``; they do not select columns.

    Returns
    -------
    dict
        - ``stat`` : the statistic, as given.
        - ``route`` : the route taken, ``"analytic"``, ``"surrogate"`` or ``"permutation"``.
        - ``z`` : signed z per radius and gene, ``(R, G)``.
        - ``obs`` : the observed curves, ``(R, G)``.
        - ``caveat`` : the statistic's caveat from :data:`ROUTES`, or ``None``.
        - ``n_sample``, ``sample`` : surrogate route only; the number and the column indices of the
          genes permuted.
        - ``p_two``, ``n_perm`` : permutation route only; the two-sided permutation p-value
          ``(R, G)``, as in :func:`permutation_null`, and the permutation count.
        - ``genes`` : only when ``genes`` is given.

        A calibrated tail probability is NOT returned for the analytic or surrogate routes: both
        were measured to rank well and neither is calibrated as a sigma count on real tissue.
    """
    route, spec = _resolve_route(stat, method)
    V = np.asarray(values, float)
    if V.ndim == 1:
        V = V[:, None]
    mb = np.asarray(mask, bool)
    radii = list(radii)
    caveat = spec.get("caveat")
    if caveat and route == spec["default"]:
        warnings.warn(f"{stat} via {route}: {caveat}", RuntimeWarning, stacklevel=2)

    if route == "analytic":
        grids = _rasterize_matrix(V, row, col, HW)
        obs = _stat_fn(stat)(grids, mb, radii, kind=kind)
        if stat == "general_g":
            mom = general_g_binary_moments(mb, radii, kind=kind)
            _E, _V, z = general_g_analytic_z(obs, grids, mb, mom)
        else:
            mom = moran_analytic_moments(mb, radii, kind=kind)
            fn = moran_analytic_z if stat == "moran" else geary_analytic_z
            _E, _V, z = fn(obs, grids, mb, mom)
        out = dict(stat=stat, route="analytic", z=z, obs=obs, caveat=caveat)

    elif route == "surrogate":
        if V.shape[1] < 25:
            raise ValueError(
                f"the surrogate null fits on a panel and got {V.shape[1]} gene(s); it needs at "
                "least 25. Use method='permutation' for a single gene.")
        ns = int(n_sample) if n_sample else max(25, int(round(0.02 * V.shape[1])))
        fit = fit_surrogate_null(V, row, col, HW, mb, radii, stat=stat, n_sample=ns,
                                 n_perm=n_perm, seed=seed, kind=kind)
        out = dict(stat=stat, route="surrogate", z=fit["z"], obs=fit["obs"], caveat=caveat,
                   n_sample=ns, sample=fit["sample"])

    else:
        R, G = len(radii), V.shape[1]
        z = np.empty((R, G)); obs = np.empty((R, G)); p2 = np.empty((R, G))
        for j in range(G):
            r = permutation_null(V[:, j], row, col, HW, mb, radii, stat=stat, n_perm=n_perm,
                                 seed=seed + j, kind=kind, keep_null=False)
            z[:, j] = r["z"]; obs[:, j] = r["obs"]; p2[:, j] = r["p_two"]
        out = dict(stat=stat, route="permutation", z=z, obs=obs, p_two=p2, caveat=caveat,
                   n_perm=n_perm)

    if genes is not None:
        out["genes"] = np.asarray(genes)
    return out


# --------------------------------------------------------------------------- strata from a covariate
def _quantile_labels(per_cell, n_strata):
    """Quantile-bin a per-cell scalar into ``n_strata`` labels, collapsing edges closer than a
    relative tolerance (a constant covariate comes back from the FFT varying at ~1e-16, and without
    this the quantiles slice that round-off into strata). Warns when fewer strata result."""
    import warnings
    per_cell = np.asarray(per_cell, float)
    span = float(np.ptp(per_cell))
    tol = 1e-9 * max(span, abs(float(per_cell.mean())), 1e-300)
    q = np.quantile(per_cell, np.linspace(0, 1, n_strata + 1)[1:-1])
    edges = [q[0]] if len(q) else []
    for v in q[1:]:
        if v - edges[-1] > tol:
            edges.append(v)
    edges = np.asarray(edges) if span > tol else np.asarray([])
    lab = np.searchsorted(edges, per_cell, side="right") if len(edges) else np.zeros(len(per_cell), int)
    k = len(np.unique(lab))
    if k < n_strata:
        warnings.warn(f"only {k} distinct strata from {n_strata} requested "
                      "(covariate has few distinct values after smoothing)", RuntimeWarning)
    return lab.astype(int)


def covariate_strata(xy, covariate, *, bin_um, radius_um, n_strata=10, HW=None, row=None, col=None):
    """Quantile strata of a per-cell covariate SMOOTHED at ``radius_um`` on the lattice.

    Smoothing first is the point: the strata should follow the section-wide trend (the thing that
    confers autocorrelation on every diffuse gene), not each cell's own noise.

    Parameters
    ----------
    xy : array-like of shape (n_cells, 2)
        Cell coordinates, x then y, in the units of ``bin_um`` (normally µm). Used to build the
        lattice when ``HW``, ``row`` and ``col`` are not all given.
    covariate : array-like of shape (n_cells,)
        The per-cell covariate, e.g. total counts or local cell density.
    bin_um : float
        Lattice bin side, in the units of ``xy``. Required even when the lattice is passed, because
        it converts ``radius_um`` to bins.
    radius_um : float
        Smoothing radius, in the units of ``xy``, rounded to whole bins (at least one). The
        covariate's bin means are averaged over a disc of this radius, centre included, over
        occupied bins only.
    n_strata : int, optional
        Number of quantile strata requested. Default ``10``.
    HW, row, col : optional
        A prebuilt lattice, as in :func:`permutation_null`. Used only when all three are given;
        otherwise the lattice is built from ``xy`` and ``bin_um``.

    Returns
    -------
    numpy.ndarray of int, shape (n_cells,)
        Int labels per cell in ``[0, n_strata)``; collapses to fewer labels when the covariate has
        fewer distinct quantiles and warns (``RuntimeWarning``).
    """
    xy = np.asarray(xy, float)
    cov = np.asarray(covariate, float)
    if row is None or col is None or HW is None:
        xyz = xy - xy.min(0)
        col = (xyz[:, 0] / bin_um).astype(int)
        row = (xyz[:, 1] / bin_um).astype(int)
        HW = (int(row.max()) + 1, int(col.max()) + 1)
    G = _rasterize_matrix(cov[:, None], row, col, HW)[..., 0]
    mask = np.zeros(HW, bool); mask[row, col] = True
    r_bin = max(1, int(round(radius_um / bin_um)))
    # disc, not spatial_smooth's Gaussian default, so the strata are smoothed with the same kernel
    # the statistic sweeps; [..., 0] because spatial_smooth returns (H, W, 1) for a 2-D input and a
    # bare np.where against the 2-D mask would broadcast that to (H, W, W)
    sm = sp.spatial_smooth(G, mask, radius=r_bin, kind="disc")[..., 0]
    sm = np.where(mask, sm, 0.0)                            # garbage outside the mask (toolkit skill)
    per_cell = sm[row, col]
    return _quantile_labels(per_cell, n_strata)


# --------------------------------------------------------------------------- density strata
#: Below this peak Moran the density curve has no characteristic scale (a half-max on a flat noise
#: curve is undefined); :func:`density_scale_um` returns NaN rather than reporting the grid floor.
DENSITY_SCALE_MIN_I = 0.05

#: Target cells per (possibly block-aggregated) bin for :func:`density_scale_um`. A literal per-bin
#: count field at ~0.2 cells/bin (real MERFISH at 10 um bins) is shot-noise dominated: its Moran
#: curve never clears ``DENSITY_SCALE_MIN_I`` even when the density genuinely has structure at a
#: coarser scale. Block-aggregating toward this many cells per bin before the scalogram is what
#: makes the readout robust to that, at the cost of coarsening the returned length's resolution.
DENSITY_SCALE_CELLS_PER_BIN = 4.0


def local_density(row, col, HW, *, radius_bins):
    """All-cell density per bin: the count field disc-summed at ``radius_bins`` and divided by the
    disc area in bins — cells per bin-area, a plain sum. NOT occupancy-normalised on purpose: a hole
    in the tissue IS low density for the purpose of where a cell type can sit, so the strata must
    see it. (``spatial_smooth`` averages over occupied bins only and would erase holes.)"""
    from scipy.signal import fftconvolve
    H, W = HW
    cnt = np.zeros((H, W))
    np.add.at(cnt, (np.asarray(row), np.asarray(col)), 1.0)
    r = max(1, int(radius_bins))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((yy ** 2 + xx ** 2) <= r ** 2).astype(float)
    raw = fftconvolve(cnt, disc, mode="same")
    # cnt and disc are both integer-valued, so the true disc sum is exactly an integer; round off
    # the ~1e-14 FFT noise before it survives into quantile edges downstream (density_strata), where
    # a uniform field otherwise scatters into spurious neighbouring strata across that noise floor
    raw = np.round(raw)
    d = raw / disc.sum()
    return np.maximum(d, 0.0)                     # FFT round-off can dip a true zero below zero


def density_strata(row, col, HW, *, radius_bins, n_strata=10):
    """Quantile strata of local cell density, one label per cell — the strata the cell-type null
    permutes labels within. ``radius_bins`` is the smoothing radius R_s in lattice bins. The null
    preserves ``P(type | local density)`` at that radius and tests arrangement finer than it."""
    d = local_density(row, col, HW, radius_bins=radius_bins)
    return _quantile_labels(d[np.asarray(row), np.asarray(col)], n_strata)


#: Block size (bins) used only to ESTIMATE tissue-level sparsity for :func:`_density_block_factor` —
#: independent of the aggregation factor ``f`` it produces.
_DENSITY_BLOCK_ESTIMATE_K = 8


def _density_block_factor(cnt):
    """The aggregation factor ``f`` :func:`density_scale_um` block-sums the count field by.

    Review fix (plan-mandated defect): the previous estimate was the mean count over FINE bins that
    already held >=1 cell, which is >= 1 by definition, so it could never see a true density below 1
    cell/bin and capped ``f`` at 2 — silently wrong exactly on the sparse sections the amendment was
    for. This instead measures cells per bin OVER THE TISSUE: block-SUM the count field into coarse
    ``k x k`` blocks (``k = _DENSITY_BLOCK_ESTIMATE_K = 8``, padding to a multiple of ``k``; a block
    is "occupied" if ANY tissue cell landed in it, so the estimate is not diluted by empty field
    outside the section), average the block sums over occupied blocks, and divide by the block area
    to get cells/bin: ``x = mean(block_sum[block_sum > 0]) / k**2``. ``f`` is then the smallest
    integer aggregation that brings that up to ``DENSITY_SCALE_CELLS_PER_BIN``:
    ``f = max(1, ceil(sqrt(DENSITY_SCALE_CELLS_PER_BIN / x)))``. Sanity: 0.2 cells/bin -> f=5 (~5
    cells per coarse bin); 2 cells/bin -> f=2; >=4 cells/bin -> f=1 (no aggregation needed).
    """
    H, W = cnt.shape
    k = _DENSITY_BLOCK_ESTIMATE_K
    Hp, Wp = int(np.ceil(H / k) * k), int(np.ceil(W / k) * k)
    cnt_p = np.zeros((Hp, Wp)); cnt_p[:H, :W] = cnt
    B = cnt_p.reshape(Hp // k, k, Wp // k, k).sum(axis=(1, 3))
    occ = B > 0
    if not occ.any():
        return 1
    x = float(B[occ].mean()) / (k * k)
    return max(1, int(np.ceil(np.sqrt(DENSITY_SCALE_CELLS_PER_BIN / x))))


def density_scale_um(row, col, HW, mask, *, bin_um, radii_bins=None):
    """Characteristic scale of the all-cell density field, in µm — the same half-max readout the
    toolkit reports for a gene (:func:`spectrast.scale.reduce_scale`), applied to the count field.
    This is the ``L_density`` the default stratification radius is stated against.

    **Why the block-aggregation**: the literal per-bin count field is shot-noise dominated on a
    sparse lattice — real MERFISH sections at 10 um bins run ~0.2 cells/bin — so its Moran curve
    never clears ``DENSITY_SCALE_MIN_I`` anywhere, even when the density genuinely has structure at
    a coarser scale, and the function would always return NaN exactly where it is needed most. Before
    the scalogram, the count field is block-SUMmED over ``f x f`` blocks (padding to a multiple of
    ``f``; the coarse mask is "any occupied fine bin in the block"), with ``f`` measured from the
    field's own tissue-level sparsity by :func:`_density_block_factor` (cells per bin averaged over
    OCCUPIED COARSE blocks, not fine bins — a fine bin that already holds >=1 cell is >=1 by
    definition and can never see a true density below 1 cell/bin). The default radii (in BINS on the
    coarse lattice) are unchanged; only the physical bin size fed to the reduction scales by ``f``
    (``radii_bins * (f * bin_um)``), so the returned length stays in µm on the ORIGINAL grid
    regardless of how coarse the aggregation had to go. ``f=1`` (already dense enough) reproduces the
    pre-amendment behaviour exactly.

    Returns NaN when there are no cells at all, or when the (possibly aggregated) density curve
    never rises above ``DENSITY_SCALE_MIN_I`` — a flat/noise curve has no half-max to report, and a
    low-amplitude curve's crossing is not a real length (the same NaN convention
    :func:`spectrast.scale.half_max` uses for a curve that never crosses its own mid-range, extended
    to curves too flat to define a mid-range worth crossing).
    """
    from . import scale as sc
    H, W = HW
    cnt = np.zeros((H, W))
    np.add.at(cnt, (np.asarray(row), np.asarray(col)), 1.0)
    m = np.asarray(mask, bool)
    if not (cnt > 0).any():
        return float("nan")
    f = _density_block_factor(cnt)
    if f > 1:
        Hp, Wp = int(np.ceil(H / f) * f), int(np.ceil(W / f) * f)
        cnt_p = np.zeros((Hp, Wp)); cnt_p[:H, :W] = cnt
        mask_p = np.zeros((Hp, Wp), bool); mask_p[:H, :W] = m
        cnt = cnt_p.reshape(Hp // f, f, Wp // f, f).sum(axis=(1, 3))
        m = mask_p.reshape(Hp // f, f, Wp // f, f).any(axis=(1, 3))
    if radii_bins is None:
        radii_bins = np.unique(np.round(np.geomspace(2, 40, 19)).astype(int))
    curve = sp.morans_i_multiscale(cnt[..., None], m, list(radii_bins))     # (R, 1)
    if curve.max() < DENSITY_SCALE_MIN_I:
        return float("nan")
    return float(sc.reduce_scale(curve, np.asarray(radii_bins, float) * (f * bin_um), "half_max",
                                 sign=1.0)[0])


# --------------------------------------------------------------------------- batch
def moran_null_batch(X, row, col, HW, mask, radii, *, genes=None, n_perm=200, strata=None, seed=0,
                     kind="disc", log=None):
    """The per-gene null over a matrix of genes, as one long table.

    Each gene is seeded ``seed + its column index`` so a single-gene call with that seed reproduces
    its rows bit for bit, and the batch does not depend on which genes were requested together.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n_cells, n_genes)
        Per-cell values, one column per gene, e.g. log-normalised expression.
    row, col, HW, mask, radii
        The lattice and radii, as in :func:`permutation_null`.
    genes : array-like of int, optional
        Column indices of ``X`` to test. ``None`` (default) tests every column. Gene names are not
        accepted.
    n_perm : int, optional
        Permutations per gene. Default ``200``.
    strata : array-like of shape (n_cells,), optional
        Stratum label per cell, used for every gene (N1). ``None`` (default) is N0.
    seed : int, optional
        Base seed; gene ``j`` uses ``seed + j``. Default ``0``.
    kind : str, optional
        Neighbourhood kernel, as in :func:`permutation_null`. Default ``"disc"``.
    log : logging.Logger, optional
        If given, progress is logged with ``log.info`` every 100 genes.

    Returns
    -------
    pandas.DataFrame
        One row per gene and radius. Columns: ``gene`` (int column index into ``X``), ``r_bin``
        (radius in bins), ``I``, ``mu``, ``sd``, ``p`` (one-sided upper) and ``z`` as in
        :func:`moran_permutation_null`, and ``null_kind`` (``"N0"`` without strata, ``"N1"`` with
        them).
    """
    import pandas as pd
    import scipy.sparse as sps
    Xc = X.tocsc() if sps.issparse(X) else np.asarray(X, float)
    n_genes = Xc.shape[1]
    idx = np.arange(n_genes) if genes is None else np.asarray(genes)
    rows = []
    radii = list(radii)
    for t, j in enumerate(idx):
        x = (np.asarray(Xc[:, j].todense()).ravel() if sps.issparse(Xc) else Xc[:, j])
        res = moran_permutation_null(x, row, col, HW, mask, radii, n_perm=n_perm, strata=strata,
                                     seed=seed + int(j), kind=kind, keep_null=False)
        for k, r in enumerate(radii):
            rows.append(dict(gene=int(j), r_bin=int(r), I=res["I"][k], mu=res["mu"][k],
                             sd=res["sd"][k], p=res["p"][k], z=res["z"][k]))
        if log is not None and (t + 1) % 100 == 0:
            log.info("    null: %d / %d genes", t + 1, len(idx))
    df = pd.DataFrame(rows)
    df["null_kind"] = "N0" if strata is None else "N1"
    return df


# --------------------------------------------------------------------------- typed point patterns
#: Two dense-loving (w=+1), two sparse-loving (w=-1), one indifferent (w=0) — a scalar slope on
#: log-DENSITY, so every type is independent of every other GIVEN density by construction (the null
#: hypothesis the stratified permutation is meant to hold under). Corrected during
#: implementation: the original draft put preferences on the two axes (nest,
#: gradient) directly, which are independent given the 2-D covariate but not given its 1-D product
#: (density) — the density strata then leave residual confounding no radius or stratum count removes
#: (T grows with N: a bias, not noise). That axis-preference model is kept, under ``axis_prefs``, as
#: the deliberately-broken case used to size that hidden-covariate limit (G2-H).
_DEFAULT_PREFS = [1.0, 1.0, -1.0, -1.0, 0.0]


def simulate_typed_points(*, n_types=5, n_cells=20000, L_um=3000.0, nest_um=100.0,
                          gradient_um=1000.0, prefs=None, axis_prefs=None, ring=None, seed=0):
    """A typed point pattern with density heterogeneity at two scales and, optionally, one planted
    interaction — the synthetic behind the stratification-radius sensitivity design.

    Cells are placed by thinning a dense uniform proposal with an intensity ``lambda(x)`` that is the
    product of a Matérn-nest field (nests of radius ``nest_um``) and a section-wide gradient (a
    logistic ramp of width ``gradient_um`` along x): ``lambda = (1 + 3*nest) * (0.5 + grad)``, so one
    density value can arise from two different geometries (a nest on the gradient's sparse side, open
    tissue on its dense side) — exactly the mixing a quantile density stratum must survive. Types are
    then drawn per cell with log-odds ``prefs[t] * log(lambda(x))``: a scalar SLOPE ON LOG-DENSITY
    per type, so every type is independent of every other type GIVEN DENSITY BY CONSTRUCTION — the
    null hypothesis the density-stratified permutation is meant to hold under, and the only thing an
    FP measurement against it is entitled to assume.

    ``axis_prefs`` (list of ``(w_nest, w_grad)``), when given, OVERRIDES ``prefs`` and draws log-odds
    ``w_nest * nest(x) + w_grad * grad(x)`` instead — independent of each other given the 2-D
    covariate ``(nest, grad)`` but NOT given its 1-D product ``lambda``, so it deliberately breaks the
    density-independence guarantee above. It exists to size the resulting hidden-covariate limit
    (gate G2-H), not as a general-purpose type model.

    ``ring=(a, b, r_um, n)`` adds ``n`` cells of type ``b`` at ``r_um`` (±15 %) around type-``a``
    cells: the one interaction the null must still detect. Anchors are restricted to type-``a`` cells
    at least ``1.15 * r_um`` from every edge of the field, so every ring cell lands strictly inside
    ``[0, L_um]`` at its intended distance — no clipping, which would otherwise silently compress the
    distance for anchors near a boundary. Raises ``ValueError`` if no type-``a`` cell is far enough
    from every edge to host a ring. ``density`` is the per-cell intensity the pattern was thinned
    with (for tests); ring cells carry NaN density (never thinned) and their anchor's index (into the
    returned ``xy``) is in ``ring_src``, present only when ``ring`` is given.
    """
    rng = np.random.default_rng(seed)
    if axis_prefs is not None:
        axis_prefs = list(axis_prefs)[:n_types]
        if len(axis_prefs) < n_types:
            axis_prefs += [(0.0, 0.0)] * (n_types - len(axis_prefs))
    else:
        prefs = list(prefs if prefs is not None else _DEFAULT_PREFS)[:n_types]
        if len(prefs) < n_types:
            prefs += [0.0] * (n_types - len(prefs))
    # proposal: 10x the target, thinned by the intensity (mean acceptance under the default
    # intensity is ~0.2, so a 4x proposal would leave fewer than n_cells surviving)
    m = 10 * int(n_cells)
    xy = rng.random((m, 2)) * L_um
    nest = np.zeros(m)
    if nest_um:
        n_nests = int((L_um / (3.0 * nest_um)) ** 2)   # nests cover pi/9 ~= 0.35 of the field
        centres = rng.random((n_nests, 2)) * L_um
        from scipy.spatial import cKDTree
        d, _ = cKDTree(centres).query(xy)
        nest = (d <= nest_um).astype(float)
    grad = np.zeros(m)
    if gradient_um:
        grad = 1.0 / (1.0 + np.exp(-(xy[:, 0] - 0.5 * L_um) / (0.25 * gradient_um)))
    intensity = (1.0 + 3.0 * nest) * (0.5 + grad)              # nests 4x, gradient 0.5..1.5
    p = intensity / intensity.max()
    keep = rng.random(m) < p
    xy, nest, grad, intensity = xy[keep], nest[keep], grad[keep], intensity[keep]
    if len(xy) < n_cells:
        raise ValueError(f"thinning kept only {len(xy)} of {m} proposed cells, fewer than the "
                         f"requested n_cells={n_cells}; raise the proposal multiplier")
    if len(xy) > n_cells:
        sel = rng.choice(len(xy), int(n_cells), replace=False)
        xy, nest, grad, intensity = xy[sel], nest[sel], grad[sel], intensity[sel]
    if axis_prefs is not None:
        logit = np.stack([w_n * nest + w_g * grad for (w_n, w_g) in axis_prefs], axis=1)
    else:
        loglam = np.log(intensity)                     # intensity >= 0.5 always: well-defined
        logit = np.stack([w * loglam for w in prefs], axis=1)
    pr = np.exp(logit - logit.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    labels = (rng.random(len(xy))[:, None] > np.cumsum(pr, 1)).sum(1).astype(int)
    labels = np.minimum(labels, n_types - 1)
    ring_src = None
    if ring is not None:
        a, b, r_um, n_ring = ring
        margin = 1.15 * r_um
        eligible = (labels == a) & (xy[:, 0] >= margin) & (xy[:, 0] <= L_um - margin) & \
                  (xy[:, 1] >= margin) & (xy[:, 1] <= L_um - margin)
        anchors = np.flatnonzero(eligible)
        if len(anchors) == 0:
            raise ValueError(f"no type-{a} cell sits at least {margin} um from every edge of a "
                             f"{L_um} um field; cannot place a ring at r_um={r_um} without "
                             f"clipping")
        ring_src = rng.choice(anchors, int(n_ring), replace=True)
        src = xy[ring_src]
        rad = r_um * rng.uniform(0.85, 1.15, int(n_ring))
        th = rng.uniform(0, 2 * np.pi, int(n_ring))
        ring_xy = src + np.c_[rad * np.cos(th), rad * np.sin(th)]     # inside [0, L_um] by
        xy = np.vstack([xy, ring_xy])                                  # construction -- no clip
        labels = np.concatenate([labels, np.full(int(n_ring), int(b))])
        intensity = np.concatenate([intensity, np.full(int(n_ring), np.nan)])
    out = dict(xy=xy, labels=labels, density=intensity)
    if ring_src is not None:
        out["ring_src"] = ring_src
    return out


# --------------------------------------------------------------------------- cross-PCF null
def _scatter(row, col, HW):
    """``(H*W, n)`` 0/1 scatter matrix — counts per bin are ``S @ onehot``."""
    import scipy.sparse as sps
    H, W = HW
    n = len(row)
    key = np.asarray(row, np.int64) * W + np.asarray(col, np.int64)
    return sps.csr_matrix((np.ones(n), (key, np.arange(n))), shape=(H * W, n))


def _type_fields(S, labels, n_types, HW):
    """Per-type count fields (density) and their 0/1 occupancy (mark), from one sparse matmul."""
    n = len(labels)
    typed = labels >= 0
    onehot = np.zeros((n, n_types))
    onehot[np.flatnonzero(typed), labels[typed]] = 1.0
    cnt = np.asarray(S @ onehot).reshape(HW[0], HW[1], n_types)
    dens = [cnt[..., t] for t in range(n_types)]
    marks = [(d > 0).astype(float) for d in dens]
    return marks, dens


def cross_pcf_strata_null(labels, row, col, HW, mask, edges, *, strata, n_types=None, n_perm=199,
                          seed=0, bin_size=1.0, self_pairs=True):
    """Cross pair-correlation for every unordered type pair with a label permutation WITHIN local-
    density strata as the null — design 2a-fix, built.

    ``labels`` are ints per cell, ``-1`` for a cell that carries no tested type: it stays in the
    tissue support and in the density, and is never relabelled. A draw permutes the whole label
    vector within ``(stratum, typed)`` groups, re-rasterises every type's count field with one
    sparse matmul, and recomputes every pair's curve from those fields — so one draw serves every
    pair and the batched screen equals looped single pairs. The observed curve takes the same path
    (draw 0 of nothing): ``g`` is exactly ``cross_pcf(null=None)``.

    What the null preserves is ``P(type | local density)`` at the smoothing radius the strata were
    built with; what it tests is arrangement finer than that. A
    majority type is not identifiable under it: such a type IS the local density.
    """
    from . import spatial as sp
    from .composition import stratified_permutation
    labels = np.asarray(labels, int)
    strata = np.asarray(strata, int)
    if labels.shape != strata.shape or labels.shape != (len(row),):
        raise ValueError("labels, strata and row/col must all have one entry per cell")
    T = int(labels.max()) + 1 if n_types is None else int(n_types)
    if T < 1:
        raise ValueError("no typed cells (every label is -1)")
    pairs = [(i, j) for i in range(T) for j in range(i, T) if self_pairs or i != j]
    edges = np.asarray(edges, float)
    H, W = HW
    from scipy.fft import next_fast_len
    lag = int(np.ceil(edges[-1] / bin_size)) + 1
    fh, fw = next_fast_len(H + lag), next_fast_len(W + lag)
    bn = sp._mcc_binner(edges, bin_size, fh, fw, False)
    m = np.asarray(mask, bool)
    S = _scatter(row, col, HW)

    marks, dens = _type_fields(S, labels, T, HW)
    g, n_pairs, _ = sp._mcc_curves(marks, dens, m, pairs, bn, "csr")
    r = 0.5 * (edges[:-1] + edges[1:])
    out = dict(r=r, edges=edges, pairs=pairs, g=g, n_pairs=n_pairs, n_types=T,
               n_strata_used=int(len(np.unique(strata))))
    if not n_perm:
        return out
    # permutation groups: stratum x typed. Untyped cells form their own groups and so never move.
    groups = np.where(labels >= 0, strata, -1 - strata)
    rng = np.random.default_rng(seed)
    draws = np.empty((int(n_perm), len(pairs), bn["n_bins"]))
    for d in range(int(n_perm)):
        perm = stratified_permutation(groups, rng)
        mk, dn = _type_fields(S, labels[perm], T, HW)
        draws[d] = sp._mcc_curves(mk, dn, m, pairs, bn, "csr")[0]
    out.update(draws=draws, **sp._mcc_score(g, draws, r), n_perm=int(n_perm), seed=int(seed))
    return out

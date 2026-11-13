"""AnnData-native spatial-statistics API (scverse / squidpy-parallel).

Thin, dependency-light wrappers so the FFT toolkit feels like :mod:`squidpy.gr`: read coordinates
from ``adata.obsm["spatial"]`` and expression from ``adata.X`` (or a ``layer``), write per-gene
results into ``adata.var``, per-cell maps into ``adata.obsm``, and matrices into ``adata.uns``.
Statistics take a physical ``radius`` in the same units as ``spatial`` (µm for Xenium/MERFISH); when
``radius`` is omitted it defaults to a multiple of the median nearest-neighbour spacing, so the tool
works without tuning.

Output keys are prefixed ``spectrast_`` (e.g. ``adata.var["spectrast_moranI"]``,
``adata.obsm["spectrast_local_moran"]``, ``adata.uns["spectrast_scalogram"]``).

Examples
--------
>>> import spectrast                                                        # doctest: +SKIP
>>> spectrast.gr.spatial_autocorr(adata, mode="moran")      # -> adata.var["spectrast_moranI"]  # doctest: +SKIP
>>> spectrast.gr.scale_spectrum(adata)                      # -> adata.var["spectrast_char_scale"]  # doctest: +SKIP
>>> spectrast.gr.local_moran(adata, genes=["EPCAM", "VWF"]) # -> adata.obsm["spectrast_local_moran"]  # doctest: +SKIP

Numeric cores live in :mod:`spectrast.spatial` / :mod:`spectrast.spatial_sc`; this module only
marshals AnnData.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- helpers
def _coords(adata, basis="spatial"):
    if basis not in adata.obsm:
        raise KeyError(f"adata.obsm['{basis}'] not found — need spatial coordinates")
    return np.asarray(adata.obsm[basis], float)


def _matrix(adata, layer=None, keep_sparse=True):
    """Expression matrix. Sparse input is kept SPARSE by default so the memory-safe path (sparse-aware SVD +
    gene chunking) engages automatically on large / whole-transcriptome data — the tool decides based on the
    data, no user tuning. Densifies only if ``keep_sparse=False`` is requested."""
    import scipy.sparse as sp  # lazy
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        return X if keep_sparse else X.toarray()
    return np.asarray(X)


def _spec_key(spec):
    """An EXACT registry key for a mark spec.

    ``repr`` of a numpy array is abbreviated (``[0. 0. 0. ... 0. 0. 0.]``), so two different
    per-cell fields that merely agree on their first and last three values produce the same string.
    Keying a field registry on that silently substitutes one field for the other, which turns a
    cross-pair into a self-pair and inflates it to the autocorrelation. Hash the buffer instead.
    Non-array specs (gene names, ``("cell_type", "T cell")`` tuples) are small and hash by ``repr``.
    """
    arr = getattr(spec, "values", spec) if hasattr(spec, "to_numpy") else spec
    if isinstance(arr, np.ndarray):
        import hashlib  # lazy: only array marks pay for it
        a = np.ascontiguousarray(arr)
        return ("ndarray", a.dtype.str, a.shape, hashlib.sha1(a.tobytes()).hexdigest())
    return repr(spec)


def _spec_label(spec):
    """A display label. ``str`` of an array is an abbreviated dump, useless in a results table."""
    arr = getattr(spec, "values", spec) if hasattr(spec, "to_numpy") else spec
    if isinstance(arr, np.ndarray):
        return f"array[{arr.shape[0]}]"
    return str(spec)


def _gene_idx(adata, genes):
    if genes is None:
        return np.arange(adata.n_vars), list(adata.var_names)
    genes = [genes] if isinstance(genes, str) else list(genes)
    pos = adata.var_names.get_indexer(genes)
    missing = [g for g, p in zip(genes, pos) if p < 0]
    if missing:
        raise KeyError(f"genes not in adata.var_names: {missing}")
    return np.asarray(pos), genes


def default_radius(adata, basis="spatial", k=6, factor=3.0, sample=5000):
    r"""Pick a sensible neighbourhood radius from the data's own cell spacing.

    Returns ``factor`` times the median of the distances from each cell to its ``k`` nearest
    neighbours, **pooled over cells and over those k neighbours** — the typical separation within
    the first neighbour ring, which on a regular lattice is the pitch. In the physical units of
    ``adata.obsm[basis]`` (µm for Xenium/MERFISH). Used as the default ``radius`` by every
    :mod:`spectrast.gr` statistic when none is given.

    .. warning::
       This is deliberately the **pooled** median over neighbours 1..``k``, *not* the distance to
       the ``k``-th neighbour. The two differ by a factor of about :math:`\sqrt2` on a square
       lattice (measured: a synthetic lattice of pitch 10 gives 10.000 pooled against 14.142 for
       the k-th; ``visiumHD_brain`` gives 29.20 against 41.29), and **every shipped default radius
       assumes the pooled form**. This docstring claimed the k-th until 2026-09-30; the code has
       always been pooled. Do not "fix" the code to match a description — changing it would move
       every default radius by ~40%.

    Parameters
    ----------
    adata : AnnData
        Must contain spatial coordinates in ``adata.obsm[basis]``.
    basis : str, optional
        Key in ``adata.obsm`` holding the ``(n_obs, 2)`` coordinates. Default ``"spatial"``.
    k : int, optional
        How many nearest neighbours each cell contributes distances for. All ``k`` are pooled;
        this is not a rank selector. Default ``6``.
    factor : float, optional
        Multiplier on that pooled median. Default ``3.0``.
    sample : int, optional
        Subsample this many cells for the neighbour search on large sections. Default ``5000``.

    Returns
    -------
    float
        The radius in the coordinate units of ``basis``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((400, 12))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((400, 2)) * 100
    >>> r = spectrast.gr.default_radius(adata)
    >>> r > 0
    True
    """
    return _radius_from_xy(_coords(adata, basis), k=k, factor=factor, sample=sample)


def _radius_from_xy(xy, k=6, factor=3.0, sample=5000):
    r""":func:`default_radius` on raw coordinates — used after a ``within=`` subset, where the
    spacing of the selected compartment, not of the whole section, sets the length scale.

    The neighbour search is FIT ON EVERY CELL and only QUERIED from a subsample. Fitting on the
    subsample instead — which this did until 2026-08-07 — measures the neighbour distance of a
    sparser point set and inflates the result by about sqrt(N / sample): 4.11x measured on an
    81,452-cell MERFISH section against 4.04x predicted. That made the returned radius a function of
    CELL COUNT rather than of cell spacing (about 2x at 19k cells, 12x at 718k), so no two sections of
    different size were being compared at the same physical neighbourhood, and ``default_radii`` fed
    the same distortion into the ``scale_spectrum`` grid. Querying a subsample costs little and
    changes nothing: the neighbours of a queried cell are found among ALL cells either way.

    THE STATISTIC IS POOLED, NOT A RANK. ``np.median(d[:, 1:])`` runs over the flattened
    ``(n_query, k)`` array, so the value returned is the median of every first-through-kth
    neighbour distance, not the median distance to the k-th neighbour. The two differ by about
    :math:`\sqrt2` on a square lattice and every shipped default radius assumes the pooled form —
    see :func:`default_radius` for the measured numbers. The public docstring said "k-th" until
    2026-09-30; the code has always been pooled, and it is the code that is right.
    """
    from sklearn.neighbors import NearestNeighbors
    xy = np.asarray(xy, float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)      # ALL cells define the density
    q = xy
    if len(xy) > sample:
        # order-INVARIANT deterministic subsample of the QUERY points: sort by coordinate then take an
        # even stride, so a row-reordered copy of the same tissue yields the SAME default radius (a
        # positional RNG choice did not).
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        q = xy[order[np.linspace(0, len(order) - 1, sample).astype(int)]]
    d, _ = nn.kneighbors(q)
    return float(factor * np.median(d[:, 1:]))


def _radii_from_xy(xy, n=19, lo_factor=2.0, hi_factor=40.0):
    """:func:`default_radii` on raw coordinates (see :func:`_radius_from_xy`)."""
    base = _radius_from_xy(xy, factor=1.0)
    return np.geomspace(lo_factor * base, hi_factor * base, n)


def default_radii(adata, basis="spatial", n=19, lo_factor=2.0, hi_factor=40.0):
    """Build a log-spaced radius sweep for the scale spectrum, anchored to cell spacing.

    Returns ``n`` radii geometrically spaced from ``lo_factor`` to ``hi_factor`` times the median
    nearest-neighbour spacing — the multiscale grid used by :func:`scale_spectrum`.

    Parameters
    ----------
    adata : AnnData
        Must contain spatial coordinates in ``adata.obsm[basis]``.
    basis : str, optional
        Coordinate key in ``adata.obsm``. Default ``"spatial"``.
    n : int, optional
        Number of radii (spectrum resolution). Default ``19``: 4.4 per octave over the 4.32-octave span,
        past every knee of a nine-section radius-count sweep and above
        ``curvefit.WEAK_RADII_STRETCHED``, so a default ``scale_fit_check`` identifies beta. It was ``6``
        (1.39 per octave) until 0.14.0.
    lo_factor, hi_factor : float, optional
        Smallest / largest radius as a multiple of median NN spacing. Defaults ``2.0`` / ``40.0``.

    Returns
    -------
    numpy.ndarray
        Radii ``(n,)`` in the coordinate units of ``basis``, increasing.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((400, 12))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((400, 2)) * 100
    >>> radii = spectrast.gr.default_radii(adata, n=5)
    >>> radii.shape
    (5,)
    """
    return _radii_from_xy(_coords(adata, basis), n=n, lo_factor=lo_factor, hi_factor=hi_factor)


def _within_mask(adata, within):
    """Resolve a ``within=`` compartment selector to a boolean cell mask ``(n_obs,)``.

    Accepts ``(obs_key, value)``, ``(obs_key, [values])``, ``{obs_key: value_or_values}``, or a
    boolean array. Returns ``None`` for ``within=None`` (use every cell).
    """
    if within is None:
        return None
    if isinstance(within, dict):
        if len(within) != 1:
            raise ValueError("within={...} takes exactly one obs key")
        within = next(iter(within.items()))
    if isinstance(within, tuple) and len(within) == 2 and isinstance(within[0], str):
        key, vals = within
        if key not in adata.obs:
            raise KeyError(f"adata.obs[{key!r}] not found")
        vals = [vals] if isinstance(vals, str) or not np.iterable(vals) else list(vals)
        col = adata.obs[key].astype(str).to_numpy()
        m = np.isin(col, [str(v) for v in vals])
        if not m.any():
            raise ValueError(f"within=({key!r}, {vals!r}) selected 0 cells; "
                             f"available: {sorted(set(col))[:12]}")
        return m
    m = np.asarray(within)
    if m.dtype != bool or m.shape != (adata.n_obs,):
        raise TypeError("within must be (obs_key, value(s)), {obs_key: value(s)}, or a boolean "
                        f"array of length n_obs={adata.n_obs}")
    if not m.any():
        raise ValueError("within mask selected 0 cells")
    return m


def _scale_um(scalogram, radii, how=None):
    """The µm characteristic scale of the public ``scale`` axis.

    The number :func:`scale_spectrum` writes to ``adata.var["spectrast_char_scale"]`` and that
    :func:`fingerprint` returns as ``scale_um``. **As of 2026-08-18 the reduction is the half-maximum,
    not the positive-scalogram centroid** — see :mod:`spectrast.scale` for the three measurements that
    demoted the centroid and for how to reproduce pre-flip numbers with ``how="centroid"``. Not to be
    confused with ``scale_peak_moran`` (the dimensionless peak of the multiscale Moran curve used
    internally by :func:`fingerprint_presence`).
    """
    from . import scale as _scale
    return _scale.reduce_scale(scalogram, radii, how or _scale.DEFAULT_REDUCTION)


def _scale_centroid(scalogram, radii):
    """Deprecated alias kept so pre-flip callers keep getting the centroid. Use :func:`_scale_um`."""
    from . import scale as _scale
    return _scale.centroid(scalogram, radii)


def _backend():
    from . import _fft
    return _fft.get_backend()


# --------------------------------------------------------------------------- global stats
def spatial_autocorr(adata, *, mode="moran", radius=None, reduced_rank=False, rank=30,
                     bin_size=None, layer=None, basis="spatial", key_added=None, copy=False):
    """Global spatial autocorrelation per gene, written to ``adata.var``.

    Computes Moran's I (or Geary's C) for every gene at a single neighbourhood radius, via the
    FFT engine (one batched transform over all genes). Higher Moran's I = the gene's expression is
    more spatially structured; Geary's C is its local-contrast complement.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    mode : {"moran", "geary"}, optional
        Which statistic to compute. Default ``"moran"``.
    radius : float, optional
        Neighbourhood radius in coordinate units. If ``None``, uses :func:`default_radius`.
    reduced_rank : bool, optional
        If ``False`` (default), compute the statistic on each measured gene. If ``True``, compute it
        on each gene's rank-``rank`` reconstruction from a truncated SVD of the centred expression
        matrix (all genes in ``adata``). That path is faster on wide panels and denoised, but its
        values describe the reconstructed gene, not the measured one (Moran's I runs higher on it,
        because the noise is gone). It also changes the default ``key_added``.
    rank : int, optional
        Number of SVD components for ``reduced_rank=True``; ignored otherwise. Capped at
        ``min(n_obs - 1, n_vars)``. :func:`autotune_rank` picks one from the data. Default ``30``.
    bin_size : float, optional
        Rasterization bin for the FFT neighbour sums, in coordinate units. ``None`` (default) picks
        it from the cell coordinates and ``radius``: the coarsest candidate between ``radius / 10``
        and ``4 * radius / 5`` that keeps a median of at most two cells per bin and at most a
        quarter of cells sensitive to the bin choice (if none qualifies, the candidate with the
        fewest such cells). The value passed (``None`` if automatic) is recorded in
        ``adata.uns[key_added + "_params"]``.
    layer : str, optional
        Expression layer; ``None`` uses ``adata.X``.
    basis : str, optional
        Coordinate key in ``adata.obsm``. Default ``"spatial"``.
    key_added : str, optional
        Destination column in ``adata.var``. Default ``"spectrast_moranI"`` / ``"spectrast_gearyC"``,
        or ``"spectrast_moranI_rr"`` / ``"spectrast_gearyC_rr"`` when ``reduced_rank=True``.
    copy : bool, optional
        If ``True`` return a modified copy instead of writing in place. Default ``False``.

    Returns
    -------
    numpy.ndarray or AnnData
        The per-gene values ``(n_vars,)``, or the modified copy if ``copy=True``. Also written to
        ``adata.var[key_added]``, with ``radius``, ``bin_size`` and the compute ``backend`` recorded
        in ``adata.uns[key_added + "_params"]``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.spatial_autocorr(adata, mode="moran")
    >>> adata.var["spectrast_moranI"].shape
    (15,)
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    r = radius if radius is not None else default_radius(adata, basis)
    if mode == "moran":
        val = sp.morans_global_genes(xy, X, r, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
        key = key_added or ("spectrast_moranI_rr" if reduced_rank else "spectrast_moranI")
    elif mode == "geary":
        val = sp.geary_c_genes(xy, X, r, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
        key = key_added or ("spectrast_gearyC_rr" if reduced_rank else "spectrast_gearyC")
    else:
        raise ValueError("mode must be 'moran' or 'geary'")
    adata.var[key] = np.asarray(val)
    adata.uns[key + "_params"] = dict(radius=r, bin_size=bin_size, backend=_backend())
    return adata if copy else val


def _scalogram_params(adata, *, layer, basis, bin_size, reduced_rank, rank, genes=None, within=None):
    """What a cached scalogram was computed FROM, as one JSON string, so a later call can tell whether the cache
    is its own (spectrast 0.14.1). Radii and gene count alone were matched before,
    so a different layer, basis, bin, rank, cell subset or gene subset silently reused the wrong curves. JSON,
    not a dict, because AnnData cannot write ``None`` into ``.uns``; the cell and gene NAMES enter as hashes."""
    import hashlib
    import json

    def h(values):
        return hashlib.sha1("\x1f".join(map(str, values)).encode()).hexdigest()

    if within is None:
        w = None
    elif isinstance(within, (np.ndarray, list)):
        w = hashlib.sha1(np.asarray(within).tobytes()).hexdigest()
    else:
        w = repr(within)
    bs = None if bin_size is None else np.atleast_1d(np.asarray(bin_size, float)).round(12).tolist()
    return json.dumps(dict(layer=layer, basis=basis, bin_size=bs, reduced_rank=bool(reduced_rank),
                           rank=int(rank) if reduced_rank else None,
                           cells=h(adata.obs_names), genes=h(adata.var_names if genes is None else genes),
                           within=w), sort_keys=True)


def scale_spectrum(adata, *, radii=None, reduced_rank=False, rank=30, bin_size=None, layer=None,
                   basis="spatial", key_added="spectrast_char_scale", reduction=None, copy=False):
    """Per-gene characteristic spatial scale from the Moran scalogram.

    Runs Moran's I across a sweep of radii (a "scalogram", one shared forward FFT) and reduces each
    gene to the radius at which its spatial structure concentrates. This is the length-scale axis of
    the gene fingerprint: small = fine/punctate patterns, large = broad domains.

    .. versionchanged:: 0.7.0
       The reduction is the **half-maximum** of each curve's own range. It was the positive-scalogram
       centroid, which is insensitive by three independent measurements and whose scramble null sits
       on the radius grid's centre of mass — see :mod:`spectrast.scale`. **This moves every reported
       micrometre value**, typically by 1.5–2x on strongly organized genes. Pass
       ``reduction="centroid"`` to reproduce pre-flip numbers. A curve with no crossing now returns
       NaN rather than a plausible value.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    radii : array-like, optional
        Radii to evaluate (coordinate units). If ``None``, uses :func:`default_radii`.
    reduced_rank, rank
        As in :func:`spatial_autocorr`: ``reduced_rank=True`` computes every gene's scalogram from
        a rank-``rank`` SVD reconstruction instead of the measured gene. Defaults ``False`` and
        ``30``.
    bin_size : float or sequence of float, optional
        Rasterization bin in coordinate units. ``None`` (default) picks a bin for each radius
        separately, by the rule given under :func:`spatial_autocorr` (about a fifth of the radius
        at small radii, finer at large ones). A scalar uses the same bin at every radius; a
        sequence with one value per radius sets each radius's bin (a sequence of any other length
        uses its first value everywhere).
    reduction : {"half_max", "centroid"}, optional
        How to reduce each curve. ``None`` uses the package default (``"half_max"``).
    layer : str, optional
        Expression layer; ``None`` uses ``adata.X``.
    basis : str, optional
        Coordinate key in ``adata.obsm``. Default ``"spatial"``.
    key_added : str, optional
        Destination column in ``adata.var``. Default ``"spectrast_char_scale"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of writing in place. Default ``False``.

    Returns
    -------
    numpy.ndarray or AnnData
        Per-gene characteristic scale ``(n_vars,)`` in coordinate units (NaN where the half-maximum
        reduction finds no crossing), or the modified copy. Also written to
        ``adata.var[key_added]``. The full ``(n_radii, n_vars)`` scalogram is stored in
        ``adata.uns["spectrast_scalogram"]`` (in
        ``adata.uns[key_added + "_scalogram"]`` when ``key_added`` is not the default) as a dict
        with ``scalogram``, ``radii``, ``genes``, ``reduction``, ``backend`` and ``params`` (the
        settings it was computed with, which :func:`scale_fit_check` checks before reusing it).

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.scale_spectrum(adata)
    >>> adata.uns["spectrast_scalogram"]["scalogram"].shape[1]
    15
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    rad = np.asarray(radii, float) if radii is not None else default_radii(adata, basis)
    scal = sp.morans_scalogram_genes(xy, X, rad, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
    scale = _scale_um(scal, rad, reduction)
    adata.var[key_added] = np.asarray(scale)
    # store the scalogram under a key tied to key_added so two calls with different key_added don't collide
    # (the default key_added keeps the documented "spectrast_scalogram" location).
    scal_key = "spectrast_scalogram" if key_added == "spectrast_char_scale" else f"{key_added}_scalogram"
    from . import scale as _scalemod
    adata.uns[scal_key] = dict(scalogram=np.asarray(scal), radii=rad, backend=_backend(),
                               genes=np.asarray(adata.var_names, dtype=str),   # column labels, for plotting
                               reduction=reduction or _scalemod.DEFAULT_REDUCTION,
                               params=_scalogram_params(adata, layer=layer, basis=basis, bin_size=bin_size,
                                                        reduced_rank=reduced_rank, rank=rank))
    return adata if copy else scale


#: Below this fraction of genes with a window-supported scale, :func:`scale_fit_check` warns that the
#: radius sweep, not the tissue, is setting the reported scales. Set from the measured separation, not
#: by taste: on three real sections 91-100% of strongly organized genes have a fitted length inside
#: the swept window, while their coordinate scrambles give 40-45%.
FIT_MIN_SUPPORTED = 0.5


def scale_fit_check(adata, *, radii=None, reduced_rank=False, rank=30, bin_size=None, layer=None,
                    basis="spatial", key_prefix="spectrast_scale_fit", copy=False):
    """Does a decay fit CORROBORATE each gene's reported characteristic scale?

    A diagnostic, not a fourth coordinate. :func:`scale_spectrum` reports a scale and
    :func:`fingerprint` reports it beside an amplitude; between them they describe these curves well,
    and nothing here replaces either number. What this adds is the question those two cannot answer on
    their own — whether a gene's scale is a length the tissue has, or an artefact of where the radius
    sweep stopped. Fit the single exponential, ask where its ``xi`` landed, and say so per gene.

    **Read two flags before any parameter.** ``scale_supported`` is false where the fit failed or its
    ``xi`` fell outside the swept window; those genes' scales should not be quoted, because the window
    is setting them. ``beta_pinned`` is true where the stretched exponent converged onto a bound,
    meaning the data do not identify it — a pinned ``beta`` is not a shape measurement, and on flat
    curves a third to three quarters of fits pin.

    **Why the reported scale is model-free, which this function is also here to evidence.** Fitting the
    stretched family to the strongly organized fifth of genes gives ``beta < 1`` for 66-99% of them
    across ten sections, four tissue types and a 3.6x range of cell pitch: a heavier tail than any
    single decay has. That separation is a property of tissue rather than of the criterion — on
    coordinate-scrambled curves the stretched family wins the same contest in 4-5% of cases against
    39-99% on real genes. A fitted ``xi`` misreports these curves, which is the reason the half-maximum
    is what gets reported, and the answer to "why not just fit an exponential?".

    The two-component family is deliberately not offered; see :mod:`spectrast.curvefit`.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
        If ``adata.uns["spectrast_scalogram"]`` already holds a scalogram on the requested radii it is
        reused, so the common ``scale_spectrum`` then ``scale_fit_check`` pair costs one sweep.
    radii : array-like, optional
        Radii to evaluate (coordinate units). ``None`` uses :func:`default_radii`, which returns nineteen —
        above the weak-identification threshold of eight, so both ``xi`` and ``beta`` are identified.
        Four parameters cannot be fitted to five radii; the stretched half is refused below six and warned
        about below eight.
    reduced_rank, rank, bin_size
        As in :func:`scale_spectrum`. They are also part of the cache check: a stored scalogram
        computed with different values is recomputed, not reused.
    layer, basis
        As in :func:`spatial_autocorr`. Also part of the cache check.
    key_prefix : str, optional
        Prefix for the ``adata.var`` columns written. Default ``"spectrast_scale_fit"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of writing in place. Default ``False``.

    Returns
    -------
    pandas.DataFrame or AnnData
        Per gene: ``xi_um``, ``xi_se_um``, ``xi_outside_window``, ``scale_supported``, ``beta``,
        ``beta_se``, ``beta_pinned``, ``heavier_than_exponential``, ``d_bic``, ``fit_failed``. Each
        column is also written to ``adata.var[key_prefix + "_" + column]``. The population summary
        that decides whether the per-gene numbers may be quoted, plus provenance, lands in
        ``adata.uns[key_prefix + "_summary"]``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((400, 8))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((400, 2)) * 200
    >>> out = spectrast.gr.scale_fit_check(adata, radii=np.geomspace(10, 160, 10))
    >>> "scale_supported" in out.columns
    True
    """
    import warnings
    import pandas as pd
    from . import curvefit as cfit
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    rad = (np.asarray(radii, float) if radii is not None
           else np.asarray(default_radii(adata, basis), float))
    cached = adata.uns.get("spectrast_scalogram")
    scal = None
    # reuse only a cache that PROVES it was computed on these settings: a cache without ``params`` (written
    # before 0.14.1) or with different ones is recomputed rather than trusted
    own = _scalogram_params(adata, layer=layer, basis=basis, bin_size=bin_size, reduced_rank=reduced_rank,
                            rank=rank)
    if (isinstance(cached, dict) and "scalogram" in cached and "radii" in cached
            and cached.get("params") == own):
        c_rad = np.asarray(cached["radii"], float)
        c_scal = np.asarray(cached["scalogram"], float)
        if c_rad.shape == rad.shape and np.allclose(c_rad, rad) and c_scal.shape[1] == adata.n_vars:
            scal = c_scal
    from_cache = scal is not None
    if not from_cache:
        xy = _coords(adata, basis); X = _matrix(adata, layer)
        scal = np.asarray(sp.morans_scalogram_genes(xy, X, rad, reduced_rank=reduced_rank, rank=rank,
                                                    bin_size=bin_size), float)
    # orient on the population, not per gene: a per-gene flip would make the sign of the statistic a
    # free parameter and destroy the comparison between genes
    A = cfit.orient(scal.T)
    fit = cfit.fit_decay(A, rad)
    summ = cfit.fit_summary(fit)
    if summ["frac_supported"] < FIT_MIN_SUPPORTED:
        warnings.warn(
            f"only {100 * summ['frac_supported']:.0f}% of genes have a fitted decay length inside the "
            f"swept window {rad[0]:.0f}-{rad[-1]:.0f} (threshold {100 * FIT_MIN_SUPPORTED:.0f}%). For "
            f"the rest the reported characteristic scale is being set by where the sweep stopped "
            f"rather than by the tissue, so those scales should not be quoted. Widen or refine the "
            f"radius grid before reading the scale axis on this section.", stacklevel=2)
    cols = ("xi_um", "xi_se_um", "xi_outside_window", "scale_supported", "beta", "beta_se",
            "beta_pinned", "heavier_than_exponential", "d_bic", "fit_failed")
    out = pd.DataFrame({c: fit[c] for c in cols}, index=adata.var_names)
    for c in out.columns:
        adata.var[f"{key_prefix}_{c}"] = out[c].to_numpy()
    adata.uns[f"{key_prefix}_summary"] = dict(
        summ, radii=rad, beta_bounds=(cfit.BETA_LO, cfit.BETA_HI), d_bic_threshold=cfit.D_BIC,
        scalogram_from_cache=from_cache, backend=_backend(),
        reported_scale="half-max / centroid — this is a CHECK on it, not a replacement",
        differential_axis=False)
    return adata if copy else out


# --------------------------------------------------------------------------- local maps
def _local(adata, func_name, genes, radius, reduced_rank, rank, bin_size, layer, basis, key_added, **kw):
    from . import spatial as sp
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    r = radius if radius is not None else default_radius(adata, basis)
    func = getattr(sp, func_name)
    M = func(xy, X, r, list(idx), reduced_rank=reduced_rank, rank=rank, bin_size=bin_size, **kw)  # (N, m)
    adata.obsm[key_added] = np.asarray(M)
    adata.uns[key_added + "_genes"] = list(names)
    return M


def local_moran(adata, *, genes=None, radius=None, reduced_rank=False, rank=30, bin_size=None,
                layer=None, basis="spatial", key_added="spectrast_local_moran", copy=False):
    """Per-cell local Moran's I (LISA) hotspot map for selected genes.

    For each cell and gene, a local indicator of spatial association: large positive = the cell sits
    in a same-signed cluster (a hot/cold spot), negative = a spatial outlier. Written to
    ``adata.obsm[key_added]`` as an ``(n_obs, n_genes)`` array; the gene order is recorded in
    ``adata.uns[key_added + "_genes"]``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to map. ``None`` uses all ``adata.var_names`` (one column per gene).
    radius : float, optional
        Neighbourhood radius in coordinate units. If ``None``, uses :func:`default_radius`.
    reduced_rank, rank, bin_size
        As in :func:`spatial_autocorr`. With ``reduced_rank=True`` the SVD runs over every gene in
        ``adata``, not only ``genes``. Defaults ``False``, ``30``, ``None``.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_local_moran"``.

    Returns
    -------
    numpy.ndarray or AnnData
        The ``(n_obs, n_genes)`` LISA map, or the modified copy if ``copy=True``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> adata.var_names = [f"g{i}" for i in range(15)]
    >>> _ = spectrast.gr.local_moran(adata, genes=["g0", "g1"])
    >>> adata.obsm["spectrast_local_moran"].shape
    (500, 2)
    """
    adata = adata.copy() if copy else adata
    _local(adata, "local_morans_i_genes", genes, radius, reduced_rank, rank, bin_size, layer, basis, key_added)
    return adata if copy else adata.obsm[key_added]


def getis_ord(adata, *, genes=None, radius=None, reduced_rank=False, rank=30, bin_size=None,
              layer=None, basis="spatial", key_added="spectrast_getis_ord", copy=False):
    """Per-cell Getis-Ord Gi* hot/cold-spot score for selected genes.

    A z-scored local statistic: large positive Gi* marks cells inside high-expression hotspots,
    large negative marks cold spots. Complements :func:`local_moran` (which also flags outliers).
    Written to ``adata.obsm[key_added]`` as ``(n_obs, n_genes)``; gene order in
    ``adata.uns[key_added + "_genes"]``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to map. ``None`` uses all ``adata.var_names``.
    radius : float, optional
        Neighbourhood radius; ``None`` uses :func:`default_radius`.
    reduced_rank, rank, bin_size
        As in :func:`local_moran`.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_getis_ord"``.

    Returns
    -------
    numpy.ndarray or AnnData
        The ``(n_obs, n_genes)`` Gi* map, or the modified copy if ``copy=True``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.getis_ord(adata)
    >>> adata.obsm["spectrast_getis_ord"].shape
    (500, 15)
    """
    adata = adata.copy() if copy else adata
    _local(adata, "getis_ord_genes", genes, radius, reduced_rank, rank, bin_size, layer, basis, key_added)
    return adata if copy else adata.obsm[key_added]


def local_geary(adata, *, genes=None, radius=None, reduced_rank=False, rank=30, bin_size=None,
                layer=None, basis="spatial", key_added="spectrast_local_geary", copy=False):
    """Per-cell local Geary's C (boundary / heterogeneity detector) for selected genes.

    A local-contrast statistic: large values mark cells where expression changes sharply between
    neighbours — tissue boundaries and interfaces, complementary to the cluster-focused
    :func:`local_moran`. Written to ``adata.obsm[key_added]`` as ``(n_obs, n_genes)``; gene order in
    ``adata.uns[key_added + "_genes"]``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to map. ``None`` uses all ``adata.var_names``.
    radius : float, optional
        Neighbourhood radius; ``None`` uses :func:`default_radius`.
    reduced_rank, rank, bin_size
        As in :func:`local_moran`.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_local_geary"``.

    Returns
    -------
    numpy.ndarray or AnnData
        The ``(n_obs, n_genes)`` local-Geary map, or the modified copy if ``copy=True``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.local_geary(adata)
    >>> adata.obsm["spectrast_local_geary"].shape
    (500, 15)
    """
    adata = adata.copy() if copy else adata
    _local(adata, "local_geary_genes", genes, radius, reduced_rank, rank, bin_size, layer, basis, key_added)
    return adata if copy else adata.obsm[key_added]


def losh(adata, *, genes=None, radius=None, a=2.0, reduced_rank=False, rank=30, bin_size=None,
         layer=None, basis="spatial", key_added="spectrast_losh", copy=False):
    """Per-cell LOSH — local spatial heteroscedasticity — for selected genes.

    The variance companion to :func:`local_moran` and :func:`getis_ord`, which locate where a gene is
    high or clustered but say nothing about how variable it is there. LOSH is 1 on average across the
    tissue: above 1 marks a neighbourhood noisier than the section as a whole (a speckled or
    transitional region), below 1 a locally smooth one. Two fields with the same local mean and the
    same global autocorrelation can differ entirely here. Written to ``adata.obsm[key_added]`` as
    ``(n_obs, n_genes)``; gene order in ``adata.uns[key_added + "_genes"]``.

    Ord & Getis, *Local spatial heteroscedasticity (LOSH)*, Annals of Regional Science 48:529-539
    (2012), with the residual formed against the neighbour mean at each cell before smoothing.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to map. ``None`` uses all ``adata.var_names``.
    radius : float, optional
        Neighbourhood radius; ``None`` uses :func:`default_radius`.
    a : float, optional
        Exponent on the residual: ``2`` (default) a variance measure, ``1`` an absolute-deviation
        measure that a single outlying cell cannot dominate.
    reduced_rank, rank, bin_size
        As in :func:`local_moran`.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_losh"``.

    Returns
    -------
    numpy.ndarray or AnnData
        The ``(n_obs, n_genes)`` LOSH map, or the modified copy if ``copy=True``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.losh(adata, genes=["0", "1"])
    >>> adata.obsm["spectrast_losh"].shape
    (500, 2)
    """
    adata = adata.copy() if copy else adata
    _local(adata, "losh_genes", genes, radius, reduced_rank, rank, bin_size, layer, basis, key_added, a=a)
    return adata if copy else adata.obsm[key_added]


def focality(adata, *, genes=None, radius=None, reduced_rank=False, rank=30, bin_size=None,
             layer=None, basis="spatial", key_added="spectrast_focality", copy=False):
    """Per-gene focality: how concentrated a gene's spatial signal is, written to ``adata.var``.

    Focality is the Gini coefficient of the gene's local Moran map — 0 when structure is spread
    diffusely across the tissue, approaching 1 when it is concentrated in a few sharp foci.
    :func:`fingerprint` carries a lattice version of the same idea as its ``focality`` axis.
    Streams over memory-sized gene chunks, so it scales to whole-transcriptome panels.

    This function maps local Moran over cells at ``radius``. The ``focality`` axis of
    :func:`fingerprint` is computed differently, on a lattice at a fixed reference radius of three
    bins, so the two give different numbers for the same gene.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to score. ``None`` scores all ``adata.var_names``.
    radius : float, optional
        Neighbourhood radius; ``None`` uses :func:`default_radius`.
    reduced_rank, rank, bin_size
        As in :func:`local_moran`.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_focality"``.

    Returns
    -------
    numpy.ndarray or AnnData
        Per-gene focality in ``[0, 1]`` for the selected genes, or the modified copy. Written to
        ``adata.var[key_added]`` (``NaN`` for genes not scored).

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 15))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.focality(adata)
    >>> adata.var["spectrast_focality"].shape
    (15,)
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    r = radius if radius is not None else default_radius(adata, basis)
    val = sp.focality_genes(xy, X, r, list(idx), reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
    out = np.full(adata.n_vars, np.nan); out[idx] = np.asarray(val)
    adata.var[key_added] = out
    return adata if copy else val


# --------------------------------------------------------------------------- texture + fingerprint presence
#: Rasterization bin as a multiple of median cell pitch, wherever a SINGLE lattice is needed. One
#: rule, one home — :mod:`spectrast.diff` reads it from here so the differential and the descriptive
#: grid axes cannot drift apart. Chosen from a nine-section parameter sweep.
#:
#: NOTE the scale axis does NOT use this. ``scale_spectrum``/``fingerprint``'s ``scale_um`` runs the
#: scalogram with ``bin_size=None``, which autotunes a bin PER RADIUS at r/b ~ 5 — a different regime,
#: kept because it holds disc quality constant across the sweep and is no less accurate (measured).
K_PITCH = 0.5

#: Grid cap. Pitch-anchoring does not bound the lattice the way the old extent-based rule did, so a
#: large section could otherwise allocate without limit. Above this the bin is coarsened and it WARNS —
#: a silently coarser grid is exactly the kind of thing that makes two sections incomparable.
MAX_GRID_BINS = 4_000_000


def _default_bin_um(xy):
    """The single-lattice bin: ``K_PITCH`` x median cell pitch, capped at :data:`MAX_GRID_BINS`.

    **Changed 2026-08-19.** Was ``extent / 400`` — a rule anchored to how big the section is rather
    than to how far apart its cells are. Measured across the nine-section panel that produced bins
    from **0.31x to 2.11x pitch, a 6.8x spread**: the same tissue imaged over a larger area was
    measured on a relatively coarser lattice, so between-section differences carried a grid artefact.
    It also disagreed with :func:`focality_landmark`, which already used pitch — the same statistic on
    two different lattices.

    Honest limit on the evidence: the convergence sweep behind ``K_PITCH`` was run on the SCALE axis,
    not on focality or texture. Adopting it here is justified by the internal inconsistency (one
    statistic, one lattice) rather than by a convergence measurement for those axes, which has not
    been made.
    """
    import warnings
    xy = np.asarray(xy, float)
    b = K_PITCH * _radius_from_xy(xy, factor=1.0)
    span = np.ptp(xy, 0)
    n = float(np.prod(np.maximum(span / max(b, 1e-9), 1.0)))
    if n > MAX_GRID_BINS:
        b_new = b * float(np.sqrt(n / MAX_GRID_BINS))
        warnings.warn(
            f"pitch-anchored bin {b:.2f} would need {n / 1e6:.1f}M lattice bins for this section, over "
            f"the {MAX_GRID_BINS / 1e6:.0f}M cap; coarsening to {b_new:.2f} ({b_new / max(b, 1e-9):.1f}x). "
            f"This section is now on a relatively coarser grid than others in the same analysis — pass "
            f"bin_um explicitly if the sections must be strictly comparable.", stacklevel=2)
        b = b_new
    return float(max(b, 1e-6))


def _gini(v):
    v = np.sort(np.abs(np.asarray(v, float))); n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum() + 1e-12))


#: Canonical magnitude-axis names COMPUTABLE on the shared fingerprint grid — the whitelist, not the
#: default vector. ``texture`` belongs here (it computes, and the presence test reports it) but not in
#: ``_FINGERPRINT_DEFAULT``; keep the two distinct.
#: ``scale_peak_moran`` is DIMENSIONLESS (the peak of the multiscale Moran curve) and is NOT the
#: public µm scale axis — that is ``scale_um`` (see :func:`scale_spectrum` / :func:`fingerprint`).
_ISO_AXES = ("scale_peak_moran", "focality", "texture")

#: Retired axis spellings -> canonical name. ``"scale"`` used to mean the dimensionless peak-Moran
#: statistic here while ``gr.scale_spectrum`` used it for the µm centroid; the two are different
#: quantities in different units, so the bare name is no longer accepted anywhere.
_AXIS_ALIASES = {"scale": "scale_peak_moran"}


def _canon_axes(axes, valid, *, where):
    """Map retired axis spellings onto the canonical names, warning once per retired name."""
    import warnings
    out = []
    for ax in axes:
        if ax in _AXIS_ALIASES and ax not in valid:
            new = _AXIS_ALIASES[ax]
            warnings.warn(
                f"{where}: axis 'scale' is ambiguous and has been renamed. It meant the DIMENSIONLESS "
                f"peak multiscale Moran, which is now '{new}'; the µm characteristic scale (from "
                f"gr.scale_spectrum) is 'scale_um'. Using '{new}'.",
                DeprecationWarning, stacklevel=3)
            out.append(new)
        else:
            out.append(ax)
    bad = [a for a in out if a not in valid]
    if bad:
        raise ValueError(f"{where}: unknown axes {bad}; choose from {list(valid)}")
    return tuple(out)


def _texture_bins(H, W, bin_um, NB=24):
    """Radial-frequency binning for the texture spectrum, on the HALF (rfft2) grid.

    Returns ``(which, mult, ck, kc)``. ``mult`` is the Hermitian multiplicity of each half-grid
    column (1 for DC and, on even ``W``, Nyquist; 2 elsewhere), so a real-input ``rfft2`` reproduces
    the full-grid radial sums exactly: |F(k)|² = |F(-k)|² and the mirror lands in the same radial
    bin. ``ck`` is therefore the full-grid bin count, identical to binning ``fft2`` output.
    """
    ky = np.fft.fftfreq(H, d=bin_um); kx = np.fft.rfftfreq(W, d=bin_um)
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    # K.max() equals the full-grid maximum: max|rfftfreq| == max|fftfreq| for both parities, and the
    # grid is a full outer product so both coordinate maxima co-occur.
    kb = np.linspace(0, K.max() * 0.7, NB + 1); kc = 0.5 * (kb[:-1] + kb[1:])
    which = np.clip(np.digitize(K.ravel(), kb) - 1, 0, NB - 1)
    mult = np.full(kx.size, 2.0); mult[0] = 1.0
    if W % 2 == 0:
        mult[-1] = 1.0
    multf = np.broadcast_to(mult[None, :], K.shape).ravel()
    ck = np.maximum(np.bincount(which, weights=multf, minlength=NB), 1.0)
    return which, multf, ck, kc


def _texture_power_device(F, which, multf, ck):
    """Radial power profile of an ``(H, W, n)`` stack, computed WITHOUT leaving the device.

    ``_fft.rfft2`` given a host array does a full numpy->device->numpy round trip per call, and for
    a transform this size that costs more than the GPU saves: texture measured **0.89x** against the
    CPU on MPS that way, i.e. the accelerator made it slower. Pushing the stack over once, binning
    on-device, and returning only the ``(NB, n)`` profile makes the transfer proportional to the
    OUTPUT (24 x n floats) instead of to the field (H x W x n).

    Returns ``None`` if the device path is unavailable, so the caller falls back to the host path.
    The bin accumulation runs in the device's float32, so texture from a GPU backend differs from the
    float64 host result at ~1e-5 relative — within the float32 contract the GPU backends already
    carry, and far below anything the rank-based presence test can see.
    """
    from . import _fft
    try:
        import torch
        Ft = _fft.to_device(F)
        S = _fft.rfft2(Ft, axes=(0, 1))                       # tensor in -> stays on device
        Pw = (S.real ** 2 + S.imag ** 2).reshape(-1, S.shape[2])
        dev = Pw.device
        w = torch.as_tensor(np.ascontiguousarray(multf, np.float32), device=dev)[:, None]
        idx = torch.as_tensor(np.ascontiguousarray(which, np.int64), device=dev)
        acc = torch.zeros((len(ck), Pw.shape[1]), device=dev, dtype=Pw.dtype)
        acc.index_add_(0, idx, Pw * w)
        return _fft.from_device(acc).astype(np.float64) / np.asarray(ck, float)[:, None]
    except Exception:
        return None


def _texture_from_power(P, kc):
    """Spectral centroid of a radial power profile ``P`` ``(NB,)`` or ``(NB, n)``, skipping the DC bin."""
    Pc = np.clip(P[1:], 0, None)
    return (Pc * kc[1:, None]).sum(0) / (Pc.sum(0) + 1e-12) if P.ndim == 2 \
        else (Pc * kc[1:]).sum() / (Pc.sum() + 1e-12)


def _isotropic_fingerprint(xy, Z, bin_um, axes=_ISO_AXES, texture_impl="batched", texture_chunk=96,
                           focality_impl="batched", focality_chunk=64):
    """Per-gene magnitude-axis statistics on a bin grid (shared by :func:`texture`,
    :func:`fingerprint` and :func:`fingerprint_presence`).

    ``Z`` is the ``(N, G)`` per-gene z-scored expression. Returns a dict with the requested keys among
    ``scale_peak_moran`` (peak of the multiscale Moran curve — DIMENSIONLESS, not a length),
    ``focality`` (Gini of the local Moran map) and ``texture`` (2-D radial-power spectral centroid;
    LOW = smooth/structured). The math is the validated fingerprint definition; chunked over genes so the ``(H, W, chunk)`` stack
    stays bounded on whole-transcriptome panels.

    ``texture_impl="batched"`` (default) runs one real-input ``rfft2`` over a ``(H, W, texture_chunk)``
    stack through the pluggable :mod:`spectrast._fft` backend. ``focality_impl="batched"`` (default)
    likewise runs one batched convolution per ``focality_chunk`` genes instead of one per gene, on the
    float64 host path. The ``"legacy"`` settings restore the original per-gene loops; they are kept
    only so the two can be compared (``tests/test_gr.py::test_texture_batched_matches_legacy``,
    ``::test_focality_batched_matches_legacy``) and are several times slower.
    """
    from scipy.signal import fftconvolve
    from . import spatial as sp
    from . import _fft
    axes = tuple(axes); want = set(axes); chunk = 600
    bad = [a for a in axes if a not in _ISO_AXES]
    if bad:
        raise ValueError(f"unknown magnitude axes {bad}; choose from {list(_ISO_AXES)}")
    xy = np.asarray(xy, float); Z = np.asarray(Z, float)
    import scipy.sparse as _sps
    xyz = xy - xy.min(0); col = (xyz[:, 0] / bin_um).astype(int); row = (xyz[:, 1] / bin_um).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    G = Z.shape[1]

    # Rasterize with ONE sparse scatter matrix rather than np.add.at per gene. Same arithmetic (a
    # cell contributes to exactly one bin), ~19x faster on a real section — and it is pure host
    # work, identical on every backend, so it was silently capping how much the GPU could ever help.
    lin = row * W + col
    _SCAT = _sps.csr_matrix((np.ones(lin.size), (lin, np.arange(lin.size))), shape=(H * W, lin.size))
    cnt = np.bincount(lin, minlength=H * W).astype(float).reshape(H, W)
    mk = cnt > 0
    _inv_cnt = np.maximum(cnt[mk], 1)[:, None]

    def grid_stack(cols):
        g = np.asarray(_SCAT @ Z[:, cols]).reshape(H, W, len(cols))
        g[mk] /= _inv_cnt
        return g, mk

    mkf = mk.astype(float)
    out = {}
    # The multiscale sweep spans 2 to 16 BINS, and is built in bin units. It used to be built as a
    # physical length, ``np.round(np.geomspace(2 * bin_um, 16 * bin_um, 12))``, which silently
    # assumes the coordinate unit is about a micrometre: on a section stored in MILLIMETRES the bin
    # is ~0.015 and np.round sends all twelve radii to zero, so ``RB`` becomes [1]*12 and the whole
    # multiscale Moran curve collapses onto one radius. Ratios of bins carry no unit, so this form
    # is exact for every coordinate system — and identical to the old one wherever a bin is at least
    # a few coordinate units, which is every micrometre-scale dataset (tests/test_gr.py).
    RB = [max(int(round(b)), 1) for b in np.geomspace(2.0, 16.0, 12)]
    if "scale_peak_moran" in want:
        out["scale_peak_moran"] = np.zeros(G)
    if "focality" in want:
        RREF = RB[2]                                   # the single reference radius of the sweep
        d = sp.neighbor_kernel(RREF, "disc"); den = fftconvolve(mkf, d, "same"); nz = den > 0
        out["focality"] = np.zeros(G)
    if "texture" in want:
        NB = 24
        which, multf, ck, kc = _texture_bins(H, W, bin_um, NB)
        if texture_impl == "batched":
            import scipy.sparse as _sps
            B = _sps.csr_matrix((multf, (which, np.arange(which.size))), shape=(NB, which.size))
            # `which`/`multf` are also handed to the device path, which bins without a round trip
        elif texture_impl == "legacy":
            lky = np.fft.fftfreq(H, d=bin_um); lkx = np.fft.fftfreq(W, d=bin_um)
            lK = np.sqrt(lkx[None, :] ** 2 + lky[:, None] ** 2)
            lkb = np.linspace(0, lK.max() * 0.7, NB + 1); lkc = 0.5 * (lkb[:-1] + lkb[1:])
            lwhich = np.clip(np.digitize(lK.ravel(), lkb) - 1, 0, NB - 1)
            lck = np.maximum(np.bincount(lwhich, minlength=NB), 1)
        else:
            raise ValueError(f"texture_impl must be 'batched' or 'legacy', got {texture_impl!r}")
        out["texture"] = np.zeros(G)
    for s in range(0, G, chunk):
        cols = list(range(s, min(s + chunk, G))); grids, _ = grid_stack(cols)
        if "scale_peak_moran" in want:
            sg = sp.morans_i_multiscale(grids, mk, RB, kind="disc")
            out["scale_peak_moran"][s:s + len(cols)] = sg.max(0)
        if "focality" in want and focality_impl == "batched":
            # ONE batched convolution per sub-chunk instead of one per gene. Stays on the float64
            # HOST path (sp._fftconv_same -> _fft.rfft2_hp) on purpose: this single-radius neighbour
            # sum is paired with the float64 `den` degree field below, and running the numerator on a
            # float32 GPU backend would mix precision inside one statistic.
            for t in range(0, len(cols), focality_chunk):
                sub = grids[..., t:t + focality_chunk]
                conv = sp._fftconv_same(sub * mkf[..., None], d, method="fft")
                mn = np.where(nz[..., None], conv / np.maximum(den, 1e-9)[..., None], 0.0)
                prod = sub * mn
                for k in range(sub.shape[2]):
                    out["focality"][cols[t + k]] = _gini(prod[..., k][mk])
        if "texture" in want and texture_impl == "batched":
            # ONE batched real-input transform per sub-chunk, through the pluggable backend. The
            # sub-chunk bounds the (H, W//2+1, n) complex spectrum; the outer chunk only bounds the
            # real grid stack, which is half the size.
            for t in range(0, len(cols), texture_chunk):
                sub = grids[..., t:t + texture_chunk]
                mu = sub[mk].mean(0)                                    # mean over occupied bins, per gene
                F = (sub - mu) * mkf[..., None]
                P = None
                if _fft.get_backend() != "cpu":
                    P = _texture_power_device(F, which, multf, ck)      # no host round trip
                if P is None:
                    S = _fft.rfft2(F, axes=(0, 1))
                    Pw = (S.real ** 2 + S.imag ** 2).reshape(-1, S.shape[2])
                    P = (B @ Pw) / ck[:, None]
                out["texture"][s + t:s + t + sub.shape[2]] = _texture_from_power(P, kc)
        legacy_foc = "focality" in want and focality_impl == "legacy"
        if legacy_foc or ("texture" in want and texture_impl == "legacy"):
            for k, gi in enumerate(cols):
                zg = grids[..., k]
                if legacy_foc:
                    mn = np.where(nz, fftconvolve(zg * mkf, d, "same") / np.maximum(den, 1e-9), 0)
                    out["focality"][gi] = _gini((zg * mn)[mk])
                if "texture" in want and texture_impl == "legacy":
                    f = (zg - zg[mk].mean()) * mkf
                    P = np.bincount(lwhich, weights=(np.abs(np.fft.fft2(f)) ** 2).ravel(), minlength=NB) / lck
                    out["texture"][gi] = _texture_from_power(P, lkc)
    return {ax: out[ax] for ax in axes}


LANDMARK_MIN_COVERAGE = 1.0 / 3.0     # below this the landmark degenerates onto the scale axis


def focality_landmark(adata, *, radii=None, bin_um=None, layer=None, basis="spatial",
                      prominence=None, key_prefix="spectrast_focality", copy=False):
    """Where a gene's focality curve turns over, and what SHAPE that curve has.

    A third per-gene coordinate, independent of the two the fingerprint reports. The focality curve
    rises for some genes, falls for others and peaks inside the measured window for most, so no single
    decay family fits the population — four attempts to reduce it with one formula each collapsed onto
    the amplitude axis. This classifies the curve first (monotone up/down, one max, one min, two or
    more, flat) and then applies the reduction that fits that class: the position of the dominant
    extremum where one exists, the monotone slope where it does not.

    **Scope, which is a measured constraint and not a caveat.** This is a per-section DESCRIPTIVE
    coordinate. It is near-orthogonal to the characteristic scale within a section (``|rho| <= 0.13`` on
    five of seven supported sections) and independent of both the amplitude and scale shifts in a
    differential (rho -0.05, -0.10), so it says something the other axes do not. It is **not** a
    differential axis: across nineteen replicate sections its between-condition shift is 0.223 octaves
    against a within-condition scatter of 0.663, so a between-arm test cannot resolve it at that
    design. Report it per section; do not difference it between conditions.

    Coverage is returned and warned on, because it is the quantity that decides whether the coordinate
    means anything: where few curves turn over, the ones that do are the extreme-scale genes and the
    third coordinate degenerates into the first. Below roughly a third, treat it as undefined rather
    than small.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    radii : array-like, optional
        Radii to evaluate (coordinate units). ``None`` uses :func:`default_radii`. A COARSE grid is
        the main thing that suppresses coverage: five radii resolve a landmark for 47% of genes on a
        section where nineteen resolve 80%.
    bin_um : float, optional
        Raster bin in coordinate units. ``None`` derives it from the median cell spacing, as in
        :func:`texture`. Each radius is rounded to a whole number of bins, at least one.
    layer, basis
        As in :func:`spatial_autocorr`.
    prominence : float, optional
        Fraction of a curve's own range an extremum must clear to count. ``None`` uses the module
        default (``spectrast.curveshape.PROM``, 0.12). Scaled by the range on purpose — an absolute
        threshold would be amplitude by another name.
    key_prefix : str, optional
        Prefix for the ``adata.var`` columns written. Default ``"spectrast_focality"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of writing in place. Default ``False``.

    Returns
    -------
    pandas.DataFrame or AnnData
        Per gene: ``shape_class``, ``landmark_um``, ``landmark_kind``, ``prominence``, ``n_extrema``,
        ``monotone_rho``, ``roughness``. ``landmark_um`` is in coordinate units and is NaN where the
        curve has no landmark. Each column is also written to
        ``adata.var[key_prefix + "_" + column]``. Class fractions, coverage and provenance land in
        ``adata.uns[key_prefix + "_landmark"]``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((400, 8))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((400, 2)) * 200
    >>> out = spectrast.gr.focality_landmark(adata)
    >>> "landmark_um" in out.columns
    True
    """
    import warnings
    import pandas as pd
    import scipy.sparse as _sps
    from . import curveshape as cshape
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    rad = np.asarray(radii, float) if radii is not None else np.asarray(default_radii(adata, basis),
                                                                        float)
    b = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    Z = X.toarray() if _sps.issparse(X) else np.asarray(X)
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / b).astype(int); row = (xyz[:, 1] / b).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    lin = row * W + col
    scat = _sps.csr_matrix((np.ones(lin.size), (lin, np.arange(lin.size))), shape=(H * W, lin.size))
    cnt = np.bincount(lin, minlength=H * W).astype(float).reshape(H, W)
    mask = cnt > 0
    inv = np.maximum(cnt[mask], 1)[:, None]
    r_bin = [max(int(round(x / b)), 1) for x in rad]
    G = Z.shape[1]
    F = np.zeros((len(r_bin), G))
    for s in range(0, G, 256):
        cols = list(range(s, min(s + 256, G)))
        g = np.asarray(scat @ Z[:, cols]).reshape(H, W, len(cols))
        g[mask] /= inv
        F[:, s:s + len(cols)] = sp.focality_scalogram(g, mask, r_bin, kind="disc")
    A = F.T
    ok = np.isfinite(A).all(1) & (np.nanstd(A, 1) > 1e-12)
    n = len(A)
    cls = np.full(n, len(cshape.CLASSES) - 1, int); kind = np.zeros(n, int)
    radius = np.full(n, np.nan); prom = np.full(n, np.nan)
    rho = np.full(n, np.nan); nex = np.zeros(n, int); rough = np.full(n, np.nan)
    if ok.any():
        kw = {} if prominence is None else dict(prom=float(prominence))
        L = cshape.landmark(A[ok], rad, **kw)
        cls[ok] = L["cls"]; kind[ok] = L["kind"]; radius[ok] = L["radius_um"]
        prom[ok] = L["prominence"]; rho[ok] = L["rho_radius"]; nex[ok] = L["n_extrema"]
        rough[ok] = cshape.roughness(A[ok])
    cov = float(np.isfinite(radius).mean())
    if cov < LANDMARK_MIN_COVERAGE:
        warnings.warn(
            f"only {100 * cov:.0f}% of genes yield a focality landmark (threshold "
            f"{100 * LANDMARK_MIN_COVERAGE:.0f}%). Where few curves turn over, the ones that do are "
            f"the genes with extreme characteristic scale, and this coordinate degenerates onto the "
            f"scale axis rather than adding to it — treat it as undefined here, not as small. A finer "
            f"radius grid is the first thing to try: coverage runs 47% on five radii against 80% on "
            f"nineteen for the same section.", stacklevel=2)
    out = pd.DataFrame(
        {"shape_class": [cshape.CLASSES[c] for c in cls],
         "landmark_um": radius,
         "landmark_kind": [cshape.KIND[k] for k in kind],
         "prominence": prom, "n_extrema": nex, "monotone_rho": rho, "roughness": rough},
        index=adata.var_names)
    for c in out.columns:
        adata.var[f"{key_prefix}_{c}"] = out[c].to_numpy()
    adata.uns[f"{key_prefix}_landmark"] = dict(
        radii=rad, bin_um=b, coverage=cov,
        class_fractions={c: float((cls == k).mean()) for k, c in enumerate(cshape.CLASSES)},
        prominence=cshape.PROM if prominence is None else float(prominence),
        backend=_backend(), differential_axis=False)
    return adata if copy else out


def texture(adata, *, genes=None, bin_um=None, layer=None, basis="spatial",
            key_added="spectrast_texture", copy=False):
    """Per-gene spatial texture: the radial-power spectral centroid, written to ``adata.var``.

    Texture is the third magnitude axis of the gene fingerprint, alongside characteristic scale
    (:func:`scale_spectrum`) and focality (:func:`focality`). It bins each gene onto a grid, takes the
    2-D power spectrum, and reduces it to the centroid of its radial (isotropic) profile — a **LOW**
    centroid means the signal's power sits at low spatial frequency (smooth, coarse-grained texture),
    a **HIGH** centroid means fine/grainy texture. Because it is a normalised spectral moment it is
    invariant to a gene's overall level, so it separates *pattern* from *magnitude*.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to score. ``None`` scores all ``adata.var_names``.
    bin_um : float, optional
        Grid bin size (coordinate units). ``None`` uses ``K_PITCH`` (0.5) times the median cell
        spacing (:func:`default_radius` with ``factor=1``), coarsened with a warning if the grid
        would exceed ``MAX_GRID_BINS`` bins.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_texture"``.

    Returns
    -------
    numpy.ndarray or AnnData
        Per-gene texture centroid (cycles per coordinate unit) for the selected genes, or the modified
        copy. Written to ``adata.var[key_added]`` (``NaN`` for genes not scored). Note that structure =
        a LOW value, so rank ascending (or pass ``higher=False`` to :func:`fingerprint_presence`).

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 6))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> _ = spectrast.gr.texture(adata)
    >>> adata.var["spectrast_texture"].shape
    (6,)
    """
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    Xg = (X[:, idx].toarray() if hasattr(X, "toarray") else np.asarray(X)[:, idx]).astype("float32")
    Z = (Xg - Xg.mean(0)) / (Xg.std(0) + 1e-9)
    bu = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    tex = _isotropic_fingerprint(xy, Z, bu, axes=("texture",))["texture"]
    out = np.full(adata.n_vars, np.nan); out[idx] = np.asarray(tex)
    adata.var[key_added] = out
    return adata if copy else tex


# --------------------------------------------------------------------------- fused fingerprint
#: Axes :func:`fingerprint` can return, and the ``adata.var`` column each is written to.
#: ``scale_um`` is the public µm scale axis and deliberately shares ``scale_spectrum``'s key —
#: it is the same quantity, computed by the same code path.
_FINGERPRINT_KEYS = {"scale_um": "spectrast_char_scale",
                     "focality": "spectrast_focality",
                     "texture": "spectrast_texture",
                     "scale_peak_moran": "spectrast_scale_peak_moran"}
#: The DEFAULT fingerprint vector is two-dimensional. ``texture`` is computed, shipped and still
#: reachable through ``axes=`` or :func:`texture`, but it is not a default coordinate: adding it to
#: (``scale_um``, ``focality``) changes co-expression programme recovery by **-0.000, CI
#: [-0.020, +0.020]** on the MERFISH sagittal panel across every label definition tested.
#: The same fingerprint DOES beat a
#: texture-shuffled control, so the axis carries real information — information ``scale_um`` and
#: ``focality`` already hold. A per-gene spectral slope / entropy / bandwidth family was measured
#: separately and found redundant WITH texture (rho 0.89-0.97), so the whole spectral branch
#: collapses onto an axis that adds nothing to a comparison.
#:
#: It stays a PRESENCE axis (see ``_PRESENCE_AXES``) because "near-saturated in almost every tissue"
#: is a measured result worth reporting. Presence-tested and
#: fingerprint-coordinate are different jobs; texture passes one and fails the other.
_FINGERPRINT_DEFAULT = ("scale_um", "focality")


def _fingerprint_core(xy, V, axes, *, radii, bin_um, reduced_rank=False, rank=30, bin_size=None,
                      reduction=None):
    """Compute the requested fingerprint axes for the ``(n_obs, k)`` raw value matrix ``V``.

    ``scale_um`` runs the same scalogram path as :func:`scale_spectrum` (so the two agree exactly);
    the grid axes share one rasterization and one batched forward transform via
    :func:`_isotropic_fingerprint`. Returns ``(values_dict, scalogram_or_None)``.
    """
    from . import spatial as sp
    out = {}; scal = None
    if "scale_um" in axes:
        scal = sp.morans_scalogram_genes(xy, V, radii, reduced_rank=reduced_rank, rank=rank,
                                         bin_size=bin_size)
        out["scale_um"] = _scale_um(scal, radii, reduction)
    iso = tuple(a for a in axes if a in _ISO_AXES)
    if iso:
        Vd = (V.toarray() if hasattr(V, "toarray") else np.asarray(V)).astype("float32")
        Z = (Vd - Vd.mean(0)) / (Vd.std(0) + 1e-9)
        out.update(_isotropic_fingerprint(xy, Z, bin_um, axes=iso))
    return out, scal


def fingerprint(adata, *, axes=_FINGERPRINT_DEFAULT, genes=None, within=None, radii=None,
                bin_um=None, layer=None, basis="spatial", reduced_rank=False, rank=30,
                bin_size=None, reduction=None, copy=False):
    """All magnitude fingerprint axes for every gene in one pass.

    The fused entry point: instead of calling :func:`scale_spectrum` and :func:`texture`
    separately — each re-reading the matrix and re-rasterizing — this computes the requested axes
    from a shared rasterization and a shared batched forward transform. Its ``focality`` axis is a
    lattice version of :func:`focality`, not the same number (see below).

    The axes are **different statistics**, not moments of one spectrum; what they share is the
    lattice and the FFT engine:

    ``scale_um``
        Characteristic spatial scale in coordinate units (µm): the half-maximum radius of the Moran
        scalogram, or its centroid with ``reduction="centroid"``. **This is the public scale
        axis** — identical to what :func:`scale_spectrum` writes to
        ``adata.var["spectrast_char_scale"]``, computed by the same code path.
    ``focality``
        Gini coefficient of the local Moran map: 0 = structure spread diffusely, →1 = concentrated
        in a few sharp foci. Computed on the shared lattice at a fixed reference radius of three
        bins, so it is not the same number as :func:`focality`, which maps cells at ``radius``.
    ``texture``
        *Optional, not a default coordinate.* Radial power-spectrum centroid. **LOW =
        smooth/coarse**, high = fine/grainy. Real but redundant with the two axes above for
        comparing genes or tissues, and bin-sensitive (rho 0.468 on ``visiumHD_brain`` for a 1.25x
        bin change), so quote it only against a stated lattice.
    ``scale_peak_moran``
        *Optional, dimensionless.* The peak of the multiscale Moran curve — "how strongly organized",
        not "at what length". This is the statistic :func:`fingerprint_presence` ranks against its
        null. It is **not** a length and must not be reported as one; that is ``scale_um``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    axes : sequence, optional
        Any of ``"scale_um"``, ``"focality"``, ``"texture"``, ``"scale_peak_moran"``. Default
        ``("scale_um", "focality")`` — texture is opt-in because it adds nothing measurable to a
        comparison built on the other two (see ``_FINGERPRINT_DEFAULT``).
    genes : str or list of str, optional
        Genes to fingerprint. ``None`` uses all ``adata.var_names``.
    within : tuple, dict, or bool array, optional
        Restrict the fingerprint to one compartment — ``("cell_type", "stroma")``,
        ``("cell_type", ["stroma", "fibroblast"])``, ``{"cell_type": "stroma"}``, or a boolean cell
        mask. The statistic is then computed on **that compartment's own lattice**: the other cells
        are absent, not zero, so the neighbourhood mask, the grid and the null all follow the
        compartment. That is the intended meaning of "how is stroma organized", but it also means
        compartment fingerprints are not directly comparable to whole-tissue ones — compare like
        with like. Default ``None`` (every cell).
    radii : array-like, optional
        Radii for the ``scale_um`` scalogram (coordinate units). ``None`` uses :func:`default_radii`,
        computed from the selected cells when ``within`` is given.
    bin_um : float, optional
        Grid bin size for the grid axes (``focality``, ``texture``, ``scale_peak_moran``), in
        coordinate units. ``None`` uses the rule in :func:`texture`, applied to the selected cells.
        Not used by ``scale_um``.
    layer, basis, copy
        As in :func:`spatial_autocorr`.
    reduced_rank, rank, bin_size
        As in :func:`scale_spectrum`; they apply to the ``scale_um`` axis only, and the grid axes
        ignore them. With ``reduced_rank=True`` the SVD runs over the selected genes and cells.
        Defaults ``False``, ``30``, ``None``.
    reduction : {"half_max", "centroid"}, optional
        How ``scale_um`` reduces each curve, as in :func:`scale_spectrum`. ``None`` uses the package
        default (``"half_max"``).

    Returns
    -------
    pandas.DataFrame or AnnData
        A ``(n_genes, n_axes)`` frame indexed by gene name, or the modified copy if ``copy=True``.
        Each axis is also written to ``adata.var`` (``spectrast_char_scale``, ``spectrast_focality``,
        ``spectrast_texture``, ``spectrast_scale_peak_moran``; NaN for genes not scored), with the
        resolved compute backend and parameters recorded in
        ``adata.uns["spectrast_fingerprint_params"]``. When ``scale_um`` is computed, its scalogram
        is stored in ``adata.uns["spectrast_scalogram"]`` (``scalogram``, ``radii``, ``genes``,
        ``reduction``, ``params``).

    See Also
    --------
    fingerprint_signature : the same axes for a pre-scored module / signature vector.
    fingerprint_presence : does the tissue express each axis at all, against a null.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 8))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> fp = spectrast.gr.fingerprint(adata)
    >>> list(fp.columns)
    ['scale_um', 'focality']
    """
    import pandas as pd
    adata = adata.copy() if copy else adata
    axes = _canon_axes(tuple(axes), tuple(_FINGERPRINT_KEYS), where="gr.fingerprint")
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    V = X if genes is None else (X[:, idx] if hasattr(X, "toarray") else np.asarray(X)[:, idx])
    cells = _within_mask(adata, within)
    if cells is not None:
        xy = xy[cells]
        V = V[cells] if hasattr(V, "toarray") else np.asarray(V)[cells]
    rad = (np.asarray(radii, float) if radii is not None
           else default_radii(adata, basis) if cells is None else _radii_from_xy(xy))
    bu = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    vals, scal = _fingerprint_core(xy, V, axes, radii=rad, bin_um=bu, reduced_rank=reduced_rank,
                                   rank=rank, bin_size=bin_size, reduction=reduction)
    for ax in axes:
        col = np.full(adata.n_vars, np.nan); col[idx] = np.asarray(vals[ax])
        adata.var[_FINGERPRINT_KEYS[ax]] = col
    from . import scale as _scalemod
    if scal is not None:
        adata.uns["spectrast_scalogram"] = dict(
            scalogram=np.asarray(scal), radii=rad,
            genes=np.asarray(names, dtype=str),       # column labels: a genes= subset is not var_names
            reduction=reduction or _scalemod.DEFAULT_REDUCTION,
            params=_scalogram_params(adata, layer=layer, basis=basis, bin_size=bin_size,
                                     reduced_rank=reduced_rank, rank=rank,
                                     genes=None if genes is None else names, within=within))
    adata.uns["spectrast_fingerprint_params"] = dict(axes=list(axes), radii=rad, bin_um=bu,
                                                     n_genes=len(names), n_obs=int(len(xy)),
                                                     within=str(within) if within is not None else None,
                                                     reduction=reduction or _scalemod.DEFAULT_REDUCTION,
                                                     backend=_backend())
    df = pd.DataFrame({ax: np.asarray(vals[ax]) for ax in axes}, index=pd.Index(names, name="gene"))
    return adata if copy else df


def fingerprint_signature(adata, values, *, names=None, axes=_FINGERPRINT_DEFAULT, within=None,
                          radii=None, bin_um=None, basis="spatial", reduced_rank=False, rank=30,
                          bin_size=None, reduction=None):
    """Fingerprint a **pre-scored** module / signature field instead of a single gene.

    Spectrast does not score gene sets — it fingerprints whatever continuous field you hand it. Score
    your modules with an established tool, then pass the resulting vector here:

    * :func:`scanpy.tl.score_genes` for simple gene-set module scores
      (``sc.tl.score_genes(adata, gene_list, score_name="myeloid")`` → ``adata.obs["myeloid"]``);
    * `decoupler <https://decoupler-py.readthedocs.io>`_ for pathway / regulon / signature scoring
      (AUCell, ULM, MLM, …) → an ``obsm`` matrix of activities.

    Keeping the scoring outside means the fingerprint never depends on our choice of scoring method,
    and any field you can define per cell — a module score, a regulon activity, a pseudotime, a
    morphology measure — is fingerprintable on the same axes as a gene.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``.
    values : str, tuple, or array-like
        The pre-scored field(s). One of:

        * ``str`` — an ``adata.obs`` column, or an ``adata.obsm`` key (all its columns are scored);
        * ``(obsm_key, columns)`` — selected columns of an ``obsm`` matrix (int, list of int, or
          list of str when the matrix is a DataFrame);
        * array-like of shape ``(n_obs,)`` or ``(n_obs, k)``.
    names : list of str, optional
        Row labels for the result. Defaults to the ``obs``/``obsm`` column names, or
        ``signature0…`` for a bare array.
    axes, within, radii, bin_um, basis, reduced_rank, rank, bin_size, reduction
        As in :func:`fingerprint`. ``within`` restricts the field to one compartment — score the
        signature on all cells first, then restrict here, so the score itself is unaffected by the
        subsetting. With ``reduced_rank=True`` the SVD runs over the ``k`` signature columns, so
        ``rank`` is capped at ``k``.

    Returns
    -------
    pandas.DataFrame
        A ``(k, n_axes)`` frame indexed by signature name. Nothing is written to ``adata`` — a
        signature is not a gene, so it has no ``var`` row to write to. The compute backend is in
        ``df.attrs["spectrast_backend"]``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 8))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> adata.obs["myeloid"] = adata.X[:, :3].mean(1)      # e.g. from sc.tl.score_genes
    >>> fp = spectrast.gr.fingerprint_signature(adata, "myeloid")
    >>> list(fp.index)
    ['myeloid']
    """
    import pandas as pd
    axes = _canon_axes(tuple(axes), tuple(_FINGERPRINT_KEYS), where="gr.fingerprint_signature")
    xy = _coords(adata, basis)
    V, auto = _signature_matrix(adata, values)
    if V.shape[0] != adata.n_obs:
        raise ValueError(f"values has {V.shape[0]} rows but adata has {adata.n_obs} observations")
    if not np.isfinite(V).all():
        raise ValueError("values contains NaN/inf — score the signature on all cells before "
                         "fingerprinting (a spatial statistic cannot skip cells)")
    lab = list(names) if names is not None else auto
    if len(lab) != V.shape[1]:
        raise ValueError(f"names has {len(lab)} entries for {V.shape[1]} signature column(s)")
    cells = _within_mask(adata, within)
    if cells is not None:
        xy = xy[cells]; V = V[cells]
    rad = (np.asarray(radii, float) if radii is not None
           else default_radii(adata, basis) if cells is None else _radii_from_xy(xy))
    bu = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    vals, _ = _fingerprint_core(xy, V.astype(float), axes, radii=rad, bin_um=bu, reduction=reduction,
                                reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
    df = pd.DataFrame({ax: np.asarray(vals[ax]) for ax in axes},
                      index=pd.Index(lab, name="signature"))
    df.attrs["spectrast_backend"] = _backend()
    return df


def fingerprint_by(adata, key, *, axes=_FINGERPRINT_DEFAULT, genes=None, within=None,
                   radii=None, bin_um=None, shared_grid=True, min_cells=50, layer=None,
                   basis="spatial", reduced_rank=False, rank=30, bin_size=None, reduction=None,
                   progress=False):
    """Fingerprint every section separately and return one tidy ``(feature × section)`` table.

    Spatial statistics are **not meaningful pooled across sections** — cells from two slides have no
    shared coordinate frame — so every real analysis is per-section. This loops :func:`fingerprint`
    over the groups of ``adata.obs[key]`` and stacks the results long-form, which is the shape
    downstream comparison and plotting want.

    **Sections are made comparable by default.** ``radii`` and ``bin_um`` normally adapt to each
    section's own cell spacing and extent, which would silently give every section a *different*
    measuring stick — differences between sections would then be partly an artefact of the grid.
    With ``shared_grid=True`` (default) the per-section defaults are computed, reduced to their
    median, and that single grid is used for every section. Pass ``shared_grid=False`` to restore
    per-section adaptation, or give ``radii``/``bin_um`` explicitly to fix them yourself.

    Parameters
    ----------
    adata : AnnData
        Concatenated sections, with coordinates in ``adata.obsm[basis]``.
    key : str
        ``adata.obs`` column identifying the section / batch.
    axes, genes, within, layer, basis, reduced_rank, rank, bin_size, reduction
        As in :func:`fingerprint`. ``within`` is applied inside each section.
    radii, bin_um : optional
        Fix the grid explicitly (coordinate units). If given, ``shared_grid`` is irrelevant for that
        quantity.
    shared_grid : bool, optional
        Use one median grid for every section so the numbers are comparable. Default ``True``.
    min_cells : int, optional
        Sections with fewer cells than this are skipped and reported in ``df.attrs["skipped"]``,
        rather than returning a fingerprint from too few cells to mean anything. Default ``50``.
    progress : bool, optional
        Print a line per section. Default ``False``.

    Returns
    -------
    pandas.DataFrame
        Long-form, one row per (feature, section): columns ``gene``, ``<key>``, ``n_obs``, and one
        per axis. ``df.attrs`` carries ``radii``, ``bin_um``, ``shared_grid``, ``skipped``,
        ``within`` and ``backend``. Nothing is written to ``adata`` — per-section results do not
        fit one ``var`` column.

    See Also
    --------
    aggregate_fingerprint : reduce this table from sections to experimental units.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((600, 5))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((600, 2)) * 100
    >>> adata.obs["section"] = ["s0"] * 300 + ["s1"] * 300
    >>> tidy = spectrast.gr.fingerprint_by(adata, "section", min_cells=10)
    >>> sorted(tidy["section"].unique())
    ['s0', 's1']
    """
    import pandas as pd
    if key not in adata.obs:
        raise KeyError(f"adata.obs[{key!r}] not found")
    lab = adata.obs[key].astype(str).to_numpy()
    groups = list(pd.unique(lab))
    xy_all = _coords(adata, basis)

    keep, skipped = [], {}
    for g in groups:
        m = lab == g
        if within is not None:
            wm = _within_mask(adata, within)
            m = m & wm
        n = int(m.sum())
        if n < min_cells:
            skipped[g] = n
        else:
            keep.append((g, m, n))
    if not keep:
        raise ValueError(f"every group of {key!r} has fewer than min_cells={min_cells} cells")

    # one measuring stick for all sections (see the docstring) — median of the per-section defaults
    rad_shared, bu_shared = radii, bin_um
    if shared_grid:
        if rad_shared is None:
            rad_shared = np.median(np.stack([_radii_from_xy(xy_all[m]) for _, m, _ in keep]), 0)
        if bu_shared is None:
            bu_shared = float(np.median([_default_bin_um(xy_all[m]) for _, m, _ in keep]))

    frames = []
    for g, m, n in keep:
        sub = adata[m]
        fp = fingerprint(sub.to_memory() if hasattr(sub, "to_memory") and sub.isbacked else sub.copy(),
                         axes=axes, genes=genes, radii=rad_shared, bin_um=bu_shared, layer=layer,
                         basis=basis, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size,
                         reduction=reduction)
        fp = fp.reset_index()
        fp.insert(1, key, g); fp.insert(2, "n_obs", n)
        frames.append(fp)
        if progress:
            print(f"  {key}={g}: {n} cells, {len(fp)} features", flush=True)
    out = pd.concat(frames, ignore_index=True)
    # plain lists in attrs: pandas compares attrs with == when it truncates a table for display,
    # and an ndarray there makes the table unprintable
    out.attrs.update(radii=[float(v) for v in rad_shared] if rad_shared is not None else None,
                     bin_um=bu_shared, shared_grid=bool(shared_grid), skipped=skipped,
                     within=str(within) if within is not None else None, backend=_backend())
    if skipped:
        import warnings
        warnings.warn(f"fingerprint_by: skipped {len(skipped)} group(s) with < {min_cells} cells "
                      f"({skipped}); see df.attrs['skipped'].")
    return out


def aggregate_fingerprint(tidy, by, *, unit=None, axes=None, reducer="mean", feature="gene",
                          min_sections=1):
    """Pool a per-section fingerprint table to one row per experimental unit.

    The pool-without-concatenating-fields pattern: each physical section keeps its own spatial field
    (they are never merged into one lattice, which would be meaningless), and only the resulting
    *numbers* are averaged within a donor / patient / mouse. Use this rather than concatenating
    sections before fingerprinting.

    Parameters
    ----------
    tidy : pandas.DataFrame
        Output of :func:`fingerprint_by`.
    by : str
        Column identifying the section in ``tidy`` (the ``key`` used for :func:`fingerprint_by`).
    unit : str or dict or pandas.Series, optional
        The unit each section belongs to: a column already in ``tidy``, or a mapping
        ``{section: unit}``. ``None`` treats each section as its own unit (a no-op regroup).
    axes : sequence of str, optional
        Axis columns to reduce. ``None`` reduces every numeric column except ``n_obs``.
    reducer : str or callable, optional
        Any pandas aggregation — ``"mean"`` (default), ``"median"``, or a callable.
    feature : str, optional
        Feature column name. Default ``"gene"``.
    min_sections : int, optional
        Units contributing fewer sections than this are dropped. Default ``1``.

    Returns
    -------
    pandas.DataFrame
        One row per (feature, unit): the reduced axes plus ``n_sections`` and total ``n_obs``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((600, 5))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((600, 2)) * 100
    >>> adata.obs["section"] = ["s0"] * 300 + ["s1"] * 300
    >>> tidy = spectrast.gr.fingerprint_by(adata, "section", min_cells=10)
    >>> agg = spectrast.gr.aggregate_fingerprint(tidy, "section", unit={"s0": "d0", "s1": "d0"})
    >>> list(agg["unit"].unique())
    ['d0']
    """
    import pandas as pd
    t = tidy.copy()
    if unit is None:
        t["unit"] = t[by].astype(str)
    elif isinstance(unit, str):
        if unit not in t:
            raise KeyError(f"{unit!r} is not a column of the table; pass a {{section: unit}} mapping "
                           f"instead, or add the column first")
        t["unit"] = t[unit].astype(str)
    else:
        mp = dict(unit) if not isinstance(unit, pd.Series) else unit.to_dict()
        missing = sorted(set(t[by].astype(str)) - {str(k) for k in mp})
        if missing:
            raise KeyError(f"no unit given for section(s) {missing}")
        t["unit"] = t[by].astype(str).map({str(k): str(v) for k, v in mp.items()})
    if axes is None:
        axes = [c for c in t.columns
                if c not in (feature, by, "unit", "n_obs") and pd.api.types.is_numeric_dtype(t[c])]
    agg = (t.groupby([feature, "unit"], sort=False)
             .agg(**{a: (a, reducer) for a in axes},
                  n_sections=(by, "nunique"), n_obs=("n_obs", "sum"))
             .reset_index())
    if min_sections > 1:
        agg = agg[agg.n_sections >= min_sections].reset_index(drop=True)
    return agg


def _signature_matrix(adata, values):
    """Resolve the ``values`` argument of :func:`fingerprint_signature` to ``((n_obs, k), labels)``."""
    import pandas as pd
    if isinstance(values, str):
        if values in adata.obs:
            return np.asarray(adata.obs[values], float)[:, None], [values]
        if values in adata.obsm:
            M = adata.obsm[values]
            if isinstance(M, pd.DataFrame):
                return M.to_numpy(float), list(M.columns)
            M = np.asarray(M, float)
            return M, [f"{values}{i}" for i in range(M.shape[1])]
        raise KeyError(f"{values!r} is in neither adata.obs nor adata.obsm. Score the signature "
                       f"first (e.g. scanpy.tl.score_genes or decoupler), then pass it here.")
    if isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], str):
        key, cols = values
        if key not in adata.obsm:
            raise KeyError(f"adata.obsm[{key!r}] not found")
        M = adata.obsm[key]
        cols = [cols] if isinstance(cols, (str, int, np.integer)) else list(cols)
        if isinstance(M, pd.DataFrame):
            return M[cols].to_numpy(float), [str(c) for c in cols]
        return np.asarray(M, float)[:, cols], [f"{key}{c}" for c in cols]
    V = np.asarray(values, float)
    V = V[:, None] if V.ndim == 1 else V
    return V, [f"signature{i}" for i in range(V.shape[1])]


_PRESENCE_AXES = {"scale_peak_moran": True, "focality": True, "texture": False}  # value = ``higher`` (structure = high?)


# --------------------------------------------------------------------------- per-gene null, one section
NULL_RADII_LO, NULL_RADII_PITCH_HI, NULL_RADII_N = 2.0, 40.0, 19      # the bank's / diff's convention
NULL_CELLS_PER_BIN_WARN = 2.0


def moran_null(adata, *, genes=None, method="analytic", n_perm=199, strata_key=None, covariate=None,
               covariate_radius_um=None, n_strata=10, radii=None, bin_um=None, layer=None,
               basis="spatial", seed=0, key_added="spectrast_null", copy=False):
    """Per-gene, per-radius significance of the multiscale Moran curve on ONE section.

    The toolkit's other nulls answer other questions — :func:`fingerprint_presence` is one call per
    axis per *tissue* (a rank shift across genes), :func:`spectrast.diff.differential` tests a
    *difference* between two arms. This is the single-section, single-gene question: is this gene's
    autocorrelation at this radius beyond what a coordinate scramble of ITS OWN values would give?

    Three nulls (:mod:`spectrast.null`):

    ``method="analytic"`` (A0, default)
        Cliff & Ord's exact permutation mean and variance of I for the engine's own row-standardised
        disc weights. Two FFTs per radius shared by every gene plus one kurtosis per gene, so a
        whole transcriptome costs about one scalogram. Validated against permutation at rho 0.999
        on three lattices. **Its sigma runs 8-12 % low where there are ~2 cells per bin** (the
        analytic null permutes bins; the permutation null permutes cells and re-averages), which is
        the whole source of its mild anticonservatism at fine radii; it warns above
        ``NULL_CELLS_PER_BIN_WARN``. Cannot condition on strata.
    ``method="permutation"`` (N0)
        ``n_perm`` permutations of the gene's values across cells, rasterised through one scatter
        and swept in one batched call with the observed curve as column 0. Exact; 2-10 s per gene.
    ``method="permutation"`` with ``strata_key`` or ``covariate`` (N1)
        The permutation stays inside strata. ``strata_key`` names a categorical ``obs`` column (the
        composition null's mechanism); ``covariate`` is a per-cell continuous value or the name of an
        ``obs`` column, smoothed at ``covariate_radius_um`` (default: the largest swept radius) and
        cut into ``n_strata`` quantile strata. The covariate's spatial pattern survives in the null,
        so the test asks for organisation BEYOND a section-wide trend.

    **Read this before thresholding.** On sections of 10^5 cells, 95-99 % of gene x radius cells are
    significant under this calibrated null (measured). Significance is
    a floor, not a ranking: rank by amplitude, gate by this.

    Returns a long table (gene, r_bin, r_um, I, mu, sd, z, p) with provenance in ``attrs``, and
    writes ``<key_added>_zmax`` and ``<key_added>_r_zmax_um`` to ``adata.var``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : list of str or int, optional
        Genes to test, as names in ``adata.var_names`` or integer column positions. ``None`` tests
        every gene. Pass a list even for one gene: a bare string is read character by character.
    method : {"analytic", "permutation"}, optional
        The null, as described above. Default ``"analytic"``.
    n_perm : int, optional
        Permutations per gene for ``method="permutation"``; the analytic null ignores it. Default
        ``199``.
    strata_key : str, optional
        Categorical ``adata.obs`` column; permutations stay within its levels. Every cell needs a
        label, and there must be at least two levels. Permutation null only; cannot be combined
        with ``covariate``.
    covariate : str or array-like, optional
        A per-cell continuous value, or the name of a numeric ``adata.obs`` column. It is smoothed
        on the lattice with a disc of radius ``covariate_radius_um`` and cut into ``n_strata``
        quantile strata; permutations stay within them. Permutation null only; cannot be combined
        with ``strata_key``.
    covariate_radius_um : float, optional
        Smoothing radius for ``covariate``, in coordinate units. ``None`` uses the largest swept
        radius. On a small section that can exceed the section and collapse the strata; the
        number actually used is reported in ``df.attrs["strata"]``.
    n_strata : int, optional
        Number of quantile strata cut from the smoothed covariate. Fewer are used, with a warning,
        when the covariate has fewer distinct quantiles. Default ``10``.
    radii : sequence of int, optional
        Radii in **bins** of the lattice, not in coordinate units; each value is truncated to an
        integer. ``None`` uses ``NULL_RADII_N`` (19) log-spaced radii from ``NULL_RADII_LO`` (2)
        bins to ``NULL_RADII_PITCH_HI`` (40) times the median cell spacing, rounded to whole bins
        with duplicates removed, so there can be fewer than 19. The ``r_um`` column gives each
        radius in coordinate units.
    bin_um : float, optional
        Lattice bin in coordinate units. ``None`` uses the rule in :func:`texture`.
    layer, basis
        As in :func:`spatial_autocorr`.
    seed : int, optional
        Base seed for the permutations; the gene in column ``j`` of ``adata`` uses ``seed + j``.
        Default ``0``.
    key_added : str, optional
        Prefix for the ``adata.var`` and ``adata.uns`` keys written. Default ``"spectrast_null"``.
    copy : bool, optional
        If ``True``, work on a copy of ``adata`` and return it together with the table. Default
        ``False``.

    Returns
    -------
    pandas.DataFrame or tuple of (AnnData, pandas.DataFrame)
        A long table with one row per gene and radius: ``gene``, ``r_bin`` (radius in bins),
        ``r_um`` (radius in coordinate units), ``I`` (observed Moran's I), ``mu`` and ``sd`` (null
        mean and standard deviation), ``z``, and ``p``. ``p`` is one-sided (organization above the
        null): a normal tail probability for the analytic null, and
        ``(1 + #{draws >= I}) / (n + 1)`` over the ``n`` finite draws for the permutation null
        (``n = n_perm`` unless a draw is undefined). Where ``I`` is undefined, e.g. at every radius
        of a gene with no counts, ``z`` and ``p`` are NaN under either null.

        ``df.attrs`` holds ``spectrast_backend``, ``spectrast_null_method``, ``spectrast_null_kind``
        (``"A0"``, ``"N0"`` or ``"N1"``), ``n_perm``, ``seed``, ``bin_um``, ``radii_bins``,
        ``radii_um``, ``n_cells``, ``n_bins``, ``cells_per_bin``, ``strata``, ``layer`` and
        ``basis``; the same dict is written to ``adata.uns[key_added + "_params"]``. Per gene, the
        largest ``z`` over the sweep and the radius it sits at go to
        ``adata.var[key_added + "_zmax"]`` and ``adata.var[key_added + "_r_zmax_um"]`` (NaN for
        genes not tested, and for a gene whose ``z`` is NaN at every radius). With ``copy=True``
        the return value is ``(adata_copy, df)``.
    """
    import warnings
    import pandas as pd
    import scipy.sparse as sps
    from scipy.stats import norm
    from . import null as _null
    from . import spatial as sp
    from . import composition as _comp
    if copy:
        adata = adata.copy()
    if strata_key is not None and covariate is not None:
        raise ValueError("strata_key and covariate are mutually exclusive: they define two different "
                         "stratifications and stacking them conditions on a product nobody asked for.")
    if method not in ("analytic", "permutation"):
        raise ValueError(method)
    if method == "analytic" and (strata_key is not None or covariate is not None):
        raise ValueError("the analytic null cannot condition on strata; use method='permutation' "
                         "for strata_key= or covariate=.")

    xy = np.asarray(adata.obsm[basis], float)
    X = _matrix(adata, layer)
    names = np.asarray(adata.var_names).astype(str)
    if genes is None:
        idx = np.arange(X.shape[1])
    else:
        pos = {g: i for i, g in enumerate(names)}
        idx = np.asarray([pos[g] if isinstance(g, str) else int(g) for g in genes])

    # lattice and radii: the single-lattice regime diff and the curve bank use
    b = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    pitch = _radius_from_xy(xy, factor=1.0)
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / b).astype(int); row = (xyz[:, 1] / b).astype(int)
    HW = (int(row.max()) + 1, int(col.max()) + 1)
    mask = np.zeros(HW, bool); mask[row, col] = True
    if radii is None:
        hi = NULL_RADII_PITCH_HI * pitch / b
        r_bins = np.unique(np.maximum(np.round(np.geomspace(NULL_RADII_LO, hi, NULL_RADII_N))
                                      .astype(int), 1)).tolist()
    else:
        r_bins = [int(r) for r in radii]
    r_um = [r * b for r in r_bins]
    cells_per_bin = float(len(xy) / max(int(mask.sum()), 1))

    strata = None; strata_prov = None
    if strata_key is not None:
        strata = _comp.strata_from_obs(adata, strata_key)
        strata_prov = dict(source=f"obs:{strata_key}", n_strata=int(len(np.unique(strata))))
    elif covariate is not None:
        cov = (np.asarray(adata.obs[covariate], float) if isinstance(covariate, str)
               else np.asarray(covariate, float))
        r_cov = float(covariate_radius_um) if covariate_radius_um is not None else float(r_um[-1])
        strata = _null.covariate_strata(xy, cov, bin_um=b, radius_um=r_cov, n_strata=n_strata,
                                        HW=HW, row=row, col=col)
        strata_prov = dict(source=("obs:" + covariate) if isinstance(covariate, str) else "array",
                           n_strata=int(len(np.unique(strata))), smoothed_at_um=r_cov)

    rows = []
    if method == "analytic":
        if cells_per_bin > NULL_CELLS_PER_BIN_WARN:
            warnings.warn(f"moran_null: {cells_per_bin:.1f} cells per bin; the analytic null's sigma "
                          f"runs ~10% low at 2 cells/bin and the gap grows with density. Prefer "
                          f"method='permutation' for marginal calls.", stacklevel=2)
        M = _null.moran_analytic_moments(mask, r_bins)
        B = 256
        for s0 in range(0, len(idx), B):
            cols = idx[s0:s0 + B]
            V = (np.asarray(X[:, cols].todense(), float) if sps.issparse(X)
                 else np.asarray(X[:, cols], float))
            G = _null._rasterize_matrix(V, row, col, HW)
            A = sp.morans_i_multiscale(G, mask, r_bins)
            E, Vr, z = _null.moran_analytic_z(A, G, mask, M)
            sd = np.sqrt(np.maximum(Vr, 0.0))
            pz = norm.sf(z)
            for k in range(len(r_bins)):
                for t, j in enumerate(cols):
                    rows.append(dict(gene=names[j], r_bin=r_bins[k], r_um=r_um[k], I=A[k, t],
                                     mu=E[k, t], sd=sd[k, t], z=z[k, t], p=pz[k, t]))
        kind = "A0"
    else:
        for j in idx:
            x = (np.asarray(X[:, j].todense()).ravel() if sps.issparse(X)
                 else np.asarray(X[:, j], float).ravel())
            res = _null.moran_permutation_null(x, row, col, HW, mask, r_bins, n_perm=n_perm,
                                               strata=strata, seed=seed + int(j), keep_null=False)
            for k in range(len(r_bins)):
                rows.append(dict(gene=names[j], r_bin=r_bins[k], r_um=r_um[k], I=res["I"][k],
                                 mu=res["mu"][k], sd=res["sd"][k], z=res["z"][k], p=res["p"][k]))
        kind = "N1" if strata is not None else "N0"

    df = pd.DataFrame(rows)
    df.attrs.update({
        "spectrast_backend": _backend(), "spectrast_null_method": method,
        "spectrast_null_kind": kind, "n_perm": (None if method == "analytic" else int(n_perm)),
        "seed": int(seed), "bin_um": b, "radii_bins": r_bins, "radii_um": r_um,
        "n_cells": int(len(xy)), "n_bins": int(mask.sum()), "cells_per_bin": cells_per_bin,
        "strata": strata_prov, "layer": layer, "basis": basis,
    })
    # per-gene summary to var: the largest z on the sweep and the radius it sits at
    g = df.groupby("gene")
    zmax = g.z.max()
    # a gene whose z is NaN at every radius (e.g. no counts) has no radius of max z: leave it out
    # here, so reindexing gives it NaN, rather than look up the NaN label idxmax returns for it
    zd = df.dropna(subset=["z"])
    r_at = zd.loc[zd.groupby("gene").z.idxmax(), ["gene", "r_um"]].set_index("gene").r_um
    adata.var[f"{key_added}_zmax"] = pd.Series(zmax).reindex(adata.var_names).to_numpy()
    adata.var[f"{key_added}_r_zmax_um"] = pd.Series(r_at).reindex(adata.var_names).to_numpy()
    adata.uns[f"{key_added}_params"] = {k: v for k, v in df.attrs.items()}
    return (adata, df) if copy else df


def fingerprint_presence(adata, *, axes=tuple(_PRESENCE_AXES), bin_um=None, n_perm=1,
                         orientation=False, orient_n_perm=100, genes=None, layer=None, basis="spatial",
                         seed=0, alpha=0.05, effect_pct=95, strata_key=None,
                         min_cells_per_stratum=None, small_strata="pool", markers=None,
                         **orient_kw):
    """Tissue-level PRESENCE test for the magnitude fingerprint axes.

    Answers "does this tissue exhibit organized spatial structure on each axis at all?": compute each
    per-gene axis statistic on the observed tissue and on a **coordinate scramble** (permuting
    ``adata.obsm[basis]`` rows preserves the expression marginal, panel and density but destroys
    location), then apply a one-sided **Mann-Whitney** rank shift test, calibrated at any panel size (the old binomial-vs-95th-percentile count test was
    anti-conservative below ~300 genes). ``texture`` is tested with ``higher=False`` because structure
    there is a LOW spectral centroid.

    The scale axis is tested as ``scale_peak_moran`` — the DIMENSIONLESS peak of the multiscale Moran
    curve. A presence test only needs a statistic that is monotone in "how organized", not a physical
    length, so the µm characteristic scale (``scale_um``, from :func:`scale_spectrum` /
    :func:`fingerprint`) is deliberately not what is ranked here. The two are different quantities;
    the bare name ``"scale"`` is retired to keep them from being confused.

    Report the panel size ``n_tested`` on every call as the power figure and treat a "weak/absent" call
    as meaningful only against a tissue where the axis is strong (matched-power framing). Fewer than 30
    tested genes is flagged ``underpowered``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    axes : sequence of {"scale_peak_moran", "focality", "texture"}, optional
        Magnitude axes to test. Default all three.
    bin_um : float, optional
        Grid bin size for the isotropic fingerprint, in coordinate units. ``None`` uses the rule in
        :func:`texture`.
    n_perm : int, optional
        Number of coordinate scrambles pooled into the null (``1`` reproduces the validated single-
        scramble study; more stabilises the null at higher cost). Default ``1``.
    genes : str or list of str, optional
        Genes to test. ``None`` uses all ``adata.var_names``.
    layer, basis
        As in :func:`spatial_autocorr`.
    seed : int, optional
        Base RNG seed for the coordinate scramble(s); scramble ``j`` uses ``seed + j``. Default ``0``.
    alpha, effect_pct : float, optional
        Per-gene rate for the descriptive ``score``/``enrichment`` and the percentile reported as the
        panel-independent ``effect_size`` (p95 of the observed statistic for ``higher`` axes; the
        complementary low percentile for texture). Defaults ``0.05`` and ``95``.
    strata_key : str, optional
        ``obs`` column of cell-type (or other) labels. When given, the null permutes **within** each
        stratum instead of across the whole section, so composition, per-type abundance and each
        type's spatial footprint are preserved and only cross-type arrangement is destroyed. The test
        then asks "is there organization BEYOND what cell-type placement explains?".

        Use it when the tissues being compared differ compositionally. **Do not** use it when the
        composition change *is* the finding — it conditions the mechanism away. See
        :mod:`spectrast.composition`, and check :func:`spectrast.composition.resolution_flip_report`
        across two label resolutions before trusting any individual call.
    min_cells_per_stratum : int, optional
        Strata smaller than this have too little permutation entropy for a usable null. Default 50.
    small_strata : {"pool", "drop", "error"}, optional
        What to do with them. Default ``"pool"``.
    markers : dict or list, optional
        Markers used to annotate ``strata_key``. Triggers a circularity warning for tested genes that
        appear in it — conditioning on labels derived from a gene removes that gene's signal by
        construction.

    Returns
    -------
    pandas.DataFrame
        One row per axis with columns ``axis, n_tested, n_sig, n_expected, enrichment, presence_p,
        score, effect_size, underpowered, present``. ``df.attrs["values"][axis]`` holds the vectors the
        call was made from (plain lists): ``genes``, ``observed`` (the per-gene statistic on the
        tissue, in ``genes`` order), ``null`` (the same statistic on the coordinate scramble(s):
        ``n_perm`` consecutive blocks in ``genes`` order — exactly what the rank test compared
        against), ``higher``, ``n_perm`` and ``alpha``. :func:`spectrast.pl.presence` draws them.
        ``df.attrs`` also holds ``spectrast_backend`` and ``spectrast_composition`` (the strata
        provenance, ``conditioned=False`` without ``strata_key``). Nothing is written to ``adata``.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast                            # doctest: +SKIP
    >>> spectrast.gr.fingerprint_presence(adata, axes=("scale_peak_moran", "focality"))  # doctest: +SKIP
    """
    import pandas as pd
    from . import orient as _orient
    from . import composition as _comp
    axes = _canon_axes(tuple(axes), tuple(_PRESENCE_AXES), where="gr.fingerprint_presence")
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    Xg = (X[:, idx].toarray() if hasattr(X, "toarray") else np.asarray(X)[:, idx]).astype("float32")
    Z = (Xg - Xg.mean(0)) / (Xg.std(0) + 1e-9)
    bu = float(bin_um) if bin_um is not None else _default_bin_um(xy)
    n_tested = len(names)

    # composition-aware null: permute WITHIN cell-type strata, so composition, per-type abundance and
    # each type's spatial footprint survive and only cross-type arrangement is destroyed.
    strata = prov = None
    if strata_key is not None:
        lab = _comp.strata_from_obs(adata, strata_key)
        mc = _comp.DEFAULT_MIN_CELLS if min_cells_per_stratum is None else int(min_cells_per_stratum)
        strata, keep, rep = _comp.apply_min_cells(lab, mc, small_strata)
        _comp.check_circularity(names, markers, strata_key)
        if not keep.all():                       # policy='drop' removes cells from the test entirely
            xy = xy[keep]; Z = Z[keep]; strata = strata[keep]
            bu = float(bin_um) if bin_um is not None else _default_bin_um(xy)
        prov = dict(strata_key=strata_key, conditioned=True, **rep)

    obs = _isotropic_fingerprint(xy, Z, bu, axes=axes)
    null = {ax: [] for ax in axes}
    for j in range(max(int(n_perm), 1)):
        rng = np.random.default_rng(seed + j)
        perm = (_comp.stratified_permutation(strata, rng) if strata is not None
                else rng.permutation(len(xy)))
        nf = _isotropic_fingerprint(xy[perm], Z, bu, axes=axes)
        for ax in axes:
            null[ax].append(nf[ax])
    rows = []
    values = {}
    for ax in axes:
        hi = _PRESENCE_AXES[ax]
        nl_ax = np.concatenate(null[ax])
        pr = _orient.axis_presence(obs[ax], nl_ax, alpha=alpha, higher=hi, n_tested=n_tested)
        eff = float(np.nanpercentile(obs[ax], effect_pct if hi else 100 - effect_pct))
        rows.append(dict(axis=ax, effect_size=eff, **pr))
        values[ax] = _presence_values(names, obs[ax], nl_ax, hi, len(null[ax]), alpha)
    if orientation:
        okw = {k: v for k, v in orient_kw.items() if k != "return_screens"}
        op = _orient.orientation_presence(adata, basis=basis, layer=layer, seed=seed, alpha=alpha,
                                          effect_pct=effect_pct, n_perm=orient_n_perm,
                                          return_screens=True, **okw)
        rows.append(dict(axis="orientation", **{k: op[k] for k in
                    ("n_tested", "n_sig", "n_expected", "enrichment", "presence_p", "score",
                     "underpowered", "present", "effect_size")}))
        # the same two vectors orientation_presence hands to axis_presence
        o_obs = op["screen"].set_index("gene")["frac_oriented"]
        o_null = op["null_screen"].set_index("gene")["frac_oriented"].reindex(o_obs.index).fillna(0.0)
        values["orientation"] = _presence_values(o_obs.index, o_obs.to_numpy(), o_null.to_numpy(), True, 1,
                                                 alpha)
    cols = ["axis", "n_tested", "n_sig", "n_expected", "enrichment", "presence_p", "score",
            "effect_size", "underpowered", "present"]
    df = pd.DataFrame(rows)[cols]
    df.attrs["spectrast_backend"] = _backend()
    df.attrs["spectrast_composition"] = prov or dict(conditioned=False)   # provenance, always stamped
    df.attrs["values"] = values
    return df


def _presence_values(genes, observed, null, higher, n_perm, alpha):
    """The per-gene vectors behind one presence call, as plain lists: numpy arrays in ``DataFrame.attrs``
    make ``pd.concat`` of two such tables raise (pandas compares attrs with ``==``)."""
    return dict(genes=[str(g) for g in genes], observed=np.asarray(observed, float).tolist(),
                null=np.asarray(null, float).tolist(), higher=bool(higher), n_perm=int(n_perm),
                alpha=float(alpha))


# --------------------------------------------------------------------------- matrices / tuning
def cross_correlation(adata, *, genes=None, radius=None, reduced_rank=False, rank=30, bin_size=None,
                      layer=None, basis="spatial", key_added="spectrast_cross_corr", copy=False):
    """All-pairs gene–gene spatial cross-correlation (bivariate Moran) among selected genes.

    The ``(m, m)`` matrix of spatial co-organization: entry ``(i, j)`` is high when genes ``i`` and
    ``j`` are expressed in the same neighbourhoods, negative when they spatially exclude. The diagonal
    equals each gene's Moran's I. Stored in ``adata.uns[key_added]`` as a dict with ``matrix``,
    ``genes``, and ``radius``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    genes : str or list of str, optional
        Genes to correlate. ``None`` uses all ``adata.var_names`` (an ``n_vars × n_vars`` matrix).
    radius : float, optional
        Neighbourhood radius; ``None`` uses :func:`default_radius`.
    reduced_rank, rank, bin_size
        As in :func:`local_moran`. With ``reduced_rank=True`` the matrix is rebuilt from ``rank``
        neighbourhood convolutions instead of one per gene.
    layer, basis, key_added, copy
        As in :func:`spatial_autocorr`. ``key_added`` defaults to ``"spectrast_cross_corr"``.

    Returns
    -------
    tuple(numpy.ndarray, list) or AnnData
        ``(matrix, gene_names)`` where ``matrix`` is ``(m, m)``, or the modified copy if ``copy=True``.
        Also written to ``adata.uns[key_added]`` as a dict with ``matrix``, ``genes`` and ``radius``
        (the radius used, in coordinate units).

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> adata = ad.AnnData(np.abs(rng.standard_normal((500, 8))).astype("float32"))
    >>> adata.obsm["spatial"] = rng.random((500, 2)) * 100
    >>> C, names = spectrast.gr.cross_correlation(adata)
    >>> C.shape
    (8, 8)
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    idx, names = _gene_idx(adata, genes)
    r = radius if radius is not None else default_radius(adata, basis)
    C = sp.bivariate_spatial_matrix(xy, X, list(idx), r, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
    adata.uns[key_added] = dict(matrix=np.asarray(C), genes=list(names), radius=r)
    return adata if copy else (C, names)


def autotune_rank(adata, *, radius=None, candidate_ranks=None, bin_size=None, layer=None,
                  basis="spatial", key_added="spectrast_autotune_rank", copy=False):
    """Data-driven embedding dimension for the optional advanced reconstruction path.

    Advanced/experimental; the default statistics do not require it. Returns a diagnostics dict (chosen
    dimension under key ``"rank"``). With ``copy=False`` (default) the dict is also written to
    ``adata.uns[key_added]`` in place; with ``copy=True`` a modified copy of ``adata`` is returned instead
    (the in-place write is the only reason to opt out with ``copy=True``).

    The cells are split at random (with a fixed seed) into two halves, and for each candidate rank
    the per-gene reduced-rank Moran's I is computed on both. The chosen rank is the smallest
    candidate whose split-half Spearman correlation reaches 99% of the best one. Pass it as
    ``rank=`` together with ``reduced_rank=True`` to the other functions in this module.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    radius : float, optional
        Radius of the Moran's I used for the split-half comparison, in coordinate units. ``None``
        uses :func:`default_radius`.
    candidate_ranks : sequence of int, optional
        Ranks to compare, in increasing order (the first that qualifies is chosen). Values above
        ``min(min(n_obs, n_vars) - 1, 64)`` are dropped. ``None`` uses 2, 4, 8, 12, 16, 24, 32, 48
        and 64, up to that cap.
    bin_size : float, optional
        Rasterization bin in coordinate units, shared by every candidate. ``None`` picks one at
        ``radius`` as in :func:`spatial_autocorr`.
    layer, basis
        As in :func:`spatial_autocorr`.
    key_added : str, optional
        Key in ``adata.uns`` for the result. Default ``"spectrast_autotune_rank"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of writing in place. Default ``False``.

    Returns
    -------
    dict or AnnData
        ``rank`` (the chosen rank), ``candidate_ranks``, ``reproducibility`` (the split-half
        Spearman correlation for each candidate), ``gd_rank`` (the Gavish-Donoho estimate of the
        expression matrix's signal rank, capped at 64, for comparison) and ``method``
        (``"reproducibility"``). Also written to ``adata.uns[key_added]``. With ``copy=True`` the
        modified copy is returned instead.
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    xy = _coords(adata, basis); X = _matrix(adata, layer)
    r = radius if radius is not None else default_radius(adata, basis)
    res = sp.autotune_rank_genes(xy, X, r, candidate_ranks=candidate_ranks, bin_size=bin_size)
    adata.uns[key_added] = res
    return adata if copy else res


# --------------------------------------------------------------------------- bivariate marks
def _mark_grid(adata, spec, mark_type, *, layer, basis, bin_um, extent):
    """Rasterize one mark onto the shared lattice. Returns a (H, W) field.

    ``mark_type`` selects where the mark comes from:
      ``expr``      a gene name  -> MEAN expression per bin
      ``obs``       an ``adata.obs`` column -> MEAN per bin
      ``obsm``      ``("key", col)`` in ``adata.obsm`` -> MEAN per bin
      ``indicator`` ``("obs_col", value)`` -> COUNT of cells of that class per bin
      ``array``     a per-cell vector, length ``n_obs`` -> MEAN per bin (how a derived field such
                    as a fingerprint axis is fed in without writing it to ``obs`` first)

    Indicators are counted, not averaged, and that distinction is not cosmetic. A per-bin FRACTION
    makes the cell-type marks compositional — the fractions sum to one, so two types are negatively
    coupled by construction and a dense nest of type A suppresses every other type's mark in the
    same bin. Measured on a planted control where A and C were placed independently, the fraction
    form reported |z| up to 5.1 and g down to 0.43; the count form (a density, which is what a
    cross-PCF is defined on) returns the flat curve it should.
    """
    xy = _coords(adata, basis)
    (x0, y0), (H, W) = extent
    col = np.clip(((xy[:, 0] - x0) / bin_um).astype(int), 0, W - 1)
    row = np.clip(((xy[:, 1] - y0) / bin_um).astype(int), 0, H - 1)

    if mark_type == "expr":
        idx, _ = _gene_idx(adata, [spec] if isinstance(spec, str) else spec)
        X = _matrix(adata, layer)
        v = X[:, idx[0]]
        v = np.asarray(v.todense()).ravel() if hasattr(v, "todense") else np.asarray(v).ravel()
    elif mark_type == "obs":
        v = np.asarray(adata.obs[spec].values, float)
    elif mark_type == "obsm":
        key, c = spec
        v = np.asarray(adata.obsm[key], float)[:, int(c)]
    elif mark_type == "indicator":
        key, value = spec
        v = (np.asarray(adata.obs[key].values).astype(str) == str(value)).astype(float)
    elif mark_type == "array":
        v = np.asarray(spec, float).ravel()
        if v.shape != (adata.n_obs,):
            raise ValueError(f"an 'array' mark must be one value per cell (n_obs={adata.n_obs}), "
                             f"got shape {np.shape(spec)}")
    else:
        raise ValueError(f"mark_type must be expr|obs|obsm|indicator|array, got {mark_type!r}")

    return np.nan_to_num(np.asarray(v, float)), (row, col)


def _rasterize_mark(v, rc, HW, mark_type, keep=None):
    """Per-cell mark values -> (mark field, density field, occupancy of the cells used).

    The three are kept apart on purpose. ``mark`` is the MEAN value per bin, ``density`` is how many
    cells contribute to that bin, and the statistic needs both: cross-correlating mean-expression
    fields alone conflates "A and B are co-expressed" with "many cells sit here", which is exactly
    what the density-conditioned normalizer exists to remove.

    For an indicator the mark is identically 1 and the density IS the class count, so a cell-type
    pair reduces to the classical cross-PCF with no separate code path.

    ``keep`` restricts which CELLS contribute this side's mark and density — the mechanism behind
    the cell-type-resolved view (O4): one side is confined to a sender or receiver population while
    the tissue support, and therefore the pair count the statistic normalizes against, stays the
    whole section.
    """
    row, col = rc
    H, W = HW
    if keep is not None:
        keep = np.asarray(keep, bool)
        v, row, col = v[keep], row[keep], col[keep]
    num = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(num, (row, col), v)
    np.add.at(cnt, (row, col), 1.0)
    occ = cnt > 0                             # tissue support: every cell, whatever its mark
    if mark_type == "indicator":
        return np.where(num > 0, 1.0, 0.0), num, occ     # mark == 1, density == class count
    mark = np.zeros((H, W))
    mark[occ] = num[occ] / cnt[occ]
    return mark, cnt, occ


def _tissue_mask(rc, HW, close_um=None, bin_um=None):
    """Occupied bins over EVERY cell — the support the pair count is normalized against.

    ``close_um`` switches the support from OCCUPIED BINS to the FILLED TISSUE OUTLINE, by closing
    the occupancy with a disc of that physical radius and filling the interior holes. This matters
    only at fine bin sizes. Once a bin is smaller than the typical cell spacing most bins are empty,
    and the occupancy mask stops being "the tissue" and becomes a scatter of isolated bins inside
    it. That breaks the toroidal-shift null, which needs the support to overlap its own translate:
    measured on a 169k-cell breast section, the median overlap falls from 50% at 20 µm bins to 24%
    at 10 µm and 10% at 6 µm, against a 25% floor — the same degeneracy that blocks a
    toroidal-shift null on point patterns.

    Under ``normalize="density"`` this changes the reported curve NOT AT ALL. The added bins hold no
    cells, so they contribute zero to both the mark numerator and the density denominator and drop
    out of the marginal means; only the shift null's overlap test and the reported ``n_pairs`` see
    them. Under ``normalize="csr"`` it is a real change of the null expectation — there it says
    "random placement anywhere in the tissue" rather than "anywhere a cell was actually seen",
    which is the more defensible reading but is not the same number.

    Deliberately not taken from either side's rasterization: restricting a mark to a sender
    population must not shrink the tissue, or the statistic would silently renormalize against that
    population's own footprint instead of the section.
    """
    occ = np.zeros(HW, bool)
    occ[rc[0], rc[1]] = True
    if not close_um:
        return occ
    if not bin_um:
        raise ValueError("close_um needs bin_um to convert the closing radius to bins")
    from scipy import ndimage                                              # lazy
    r = max(1, int(round(float(close_um) / float(bin_um))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2) <= r ** 2
    # OR with the raw occupancy: binary_closing erodes against the array border, so a cell sitting
    # in the outermost rows would otherwise be dropped from its own tissue — which would change the
    # marks, not just the support.
    return ndimage.binary_fill_holes(ndimage.binary_closing(occ, structure=disc)) | occ


def _mark_extent(adata, basis, bin_um):
    xy = _coords(adata, basis)
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    W = int(np.ceil((xy[:, 0].max() - x0) / bin_um)) + 1
    H = int(np.ceil((xy[:, 1].max() - y0) / bin_um)) + 1
    return (x0, y0), (H, W)


def mark_cross_correlation(adata, a, b, *, a_type="expr", b_type="expr", radii=None,
                           normalize=None, n_bins=12, r_max=None, bin_um=None, null="shift", n_perm=0,
                           seed=0, layer=None, basis="spatial", a_within=None, b_within=None,
                           mask_close_um=None, density_bw=None,
                           key_added="spectrast_mark_cross", copy=False):
    """Distance-resolved co-organization of two mark fields — g_AB(r) — on an AnnData.

    The AnnData-native form of :func:`spectrast.spatial.mark_cross_correlation`. Rasterizes the two
    marks onto a shared lattice and reports, per annulus, how much more A-weight and B-weight
    co-occur at that separation than a random arrangement of the same marks over the same tissue
    would give. ``g > 1`` co-localization, ``g < 1`` exclusion.

    ``a`` and ``b`` are interpreted by ``a_type`` / ``b_type``: a gene name (``"expr"``), an
    ``adata.obs`` column (``"obs"``), an ``(obsm_key, column)`` pair (``"obsm"``), or an
    ``("obs_col", value)`` pair for a categorical indicator (``"indicator"``). Any pair of fields is
    valid — the named wrappers below only assemble the fields and call this.

    ``null`` defaults to ``"shift"`` (toroidal), which preserves each field's own autocorrelation.
    ``"reshuffle"`` is the CSR null and is correct for indicator/point-like marks, but on smooth
    fields it is badly anti-conservative — measured 88% false positives at alpha=0.05 versus 4% for
    the shift null.

    ``mask_close_um`` replaces the occupied-bin support with the filled tissue outline, closing
    the occupancy with a disc of that radius. Needed once ``bin_um`` drops below the typical cell
    spacing, where most bins are empty and the shift null loses the overlap it needs; under
    ``normalize="density"`` it leaves the curve unchanged and only repairs the null (see
    :func:`_tissue_mask`).

    ``a_within`` / ``b_within`` confine ONE side to a cell population — ``("cell_type", "T cell")``,
    a list of values, or a boolean cell mask — while the tissue support stays the whole section.
    That is the cell-type-resolved reading: "is this pair carried by senders of a particular type?"

    Returns the result dict; also stored in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; marks as selected by ``a_type`` / ``b_type``.
    a, b
        The two mark specs, read according to ``a_type`` and ``b_type``.
    a_type, b_type : {"expr", "obs", "obsm", "indicator", "array"}, optional
        How to read ``a`` and ``b``:

        * ``"expr"`` — a gene name; the mark is the mean expression per bin.
        * ``"obs"`` — a numeric ``adata.obs`` column; mean per bin.
        * ``"obsm"`` — an ``(obsm_key, column)`` pair with an integer column; mean per bin.
        * ``"indicator"`` — an ``(obs_column, value)`` pair; the mark is the COUNT of cells of
          that class per bin, not a fraction.
        * ``"array"`` — one value per cell, length ``n_obs``; mean per bin.

        NaN marks are set to zero. Marks must be non-negative. Default ``"expr"``.
    radii : array-like, optional
        Annulus EDGES in coordinate units, increasing (``n + 1`` edges give ``n`` annuli). When
        given, ``n_bins`` and ``r_max`` are ignored. ``None`` builds ``n_bins`` equal-width annuli
        from one bin to ``r_max``.
    normalize : {"density", "csr", "coherence", "density_robust"}, optional
        Which question the curve answers; see :func:`spectrast.spatial.mark_cross_correlation`.
        ``None`` (default) uses ``"csr"`` (the cross pair-correlation function) when both sides
        are indicators and ``"density"`` (Stoyan's mark correlation) otherwise.
    n_bins : int, optional
        Number of annuli when ``radii`` is ``None``. Default ``12``.
    r_max : float, optional
        Outer edge of the last annulus, in coordinate units, when ``radii`` is ``None``. ``None``
        uses a quarter of the lattice's shorter side. Never less than two bins.
    bin_um : float, optional
        Lattice bin in coordinate units. ``None`` uses :func:`default_radius` divided by 5, which
        is 0.6 times the median cell spacing.
    null : {"shift", "reshuffle"}, optional
        Null for ``z`` and p-values, run only when ``n_perm > 0``. ``"shift"`` (default) shifts B
        toroidally over the tissue; ``"reshuffle"`` permutes B's marks over occupied bins.
    n_perm : int, optional
        Number of null draws. ``0`` (default) runs no null and returns the curve only.
    seed : int, optional
        Seed for the null draws. Default ``0``.
    layer : str, optional
        Expression layer for ``"expr"`` marks; ``None`` uses ``adata.X``.
    basis : str, optional
        Coordinate key in ``adata.obsm``. Default ``"spatial"``.
    a_within, b_within : tuple, dict or bool array, optional
        Confine one side's mark and density to a cell population: ``(obs_key, value)``,
        ``(obs_key, [values])``, ``{obs_key: value_or_values}`` or a boolean cell mask. The tissue
        support stays the whole section. ``None`` (default) uses every cell.
    mask_close_um : float, optional
        Closing radius in coordinate units that turns the occupied-bin support into the filled
        tissue outline. ``None`` (default) keeps the occupied bins.
    density_bw : float, optional
        Bandwidth, in coordinate units, of the local-mean density used by
        ``normalize="density_robust"``; ignored otherwise. ``None`` uses half the outer edge of
        the last annulus.
    key_added : str, optional
        Key in ``adata.uns`` for the result. Default ``"spectrast_mark_cross"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of the result dict. Default ``False``.

    Returns
    -------
    dict or AnnData
        ``r`` (annulus midpoints, coordinate units), ``g`` (the curve, one value per annulus),
        ``n_pairs`` (bin pairs per annulus), ``edges``. Under ``"density"`` and
        ``"density_robust"`` also ``g_geometry``, the cross-PCF of the same pair with no null.
        When a null is run: ``z``, ``pval`` (per annulus), ``null_mean``, ``null_sd``, ``n_valid``
        (usable draws), ``T_max``, ``p_max`` and ``r_at_max_um`` (the max-over-annuli test, which
        prices in choosing the peak), ``null`` and ``n_perm``. Always ``a``, ``b`` (as strings),
        ``a_type``, ``b_type``, ``bin_um``, ``normalize`` and ``backend``. The dict is also stored
        in ``adata.uns[key_added]``; with ``copy=True`` the modified copy is returned instead.
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    if bin_um is None:
        bin_um = default_radius(adata, basis) / 5.0
    extent = _mark_extent(adata, basis, bin_um)
    HW = extent[1]
    va, rc = _mark_grid(adata, a, a_type, layer=layer, basis=basis, bin_um=bin_um, extent=extent)
    vb, _ = _mark_grid(adata, b, b_type, layer=layer, basis=basis, bin_um=bin_um, extent=extent)
    ma, na, _ = _rasterize_mark(va, rc, HW, a_type, keep=_within_mask(adata, a_within))
    mb, nb, _ = _rasterize_mark(vb, rc, HW, b_type, keep=_within_mask(adata, b_within))
    mask = _tissue_mask(rc, HW, mask_close_um, bin_um)   # tissue support, not either mark's
    if normalize is None:
        # cell types are a geometry question (cross-PCF); everything else is an expression question
        # asked at fixed geometry (Stoyan), so it must divide the density out.
        normalize = "csr" if (a_type == "indicator" and b_type == "indicator") else "density"

    res = sp.mark_cross_correlation(ma, mb, mask, density_a=na, density_b=nb,
                                    normalize=normalize, bin_size=bin_um, n_bins=n_bins,
                                    r_max=r_max, annuli=radii,
                                    null=(null if n_perm else None), n_perm=n_perm, seed=seed,
                                    density_bw=density_bw)
    res = {k: (np.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    res.update(a=str(a), b=str(b), a_type=a_type, b_type=b_type, bin_um=float(bin_um),
               normalize=normalize, backend=_backend())
    adata.uns[key_added] = res
    return adata if copy else res


def mark_cross_correlation_matrix(adata, pairs, *, a_type="expr", b_type="expr", radii=None,
                                  normalize=None, n_bins=12, r_max=None, bin_um=None,
                                  null="shift", n_perm=0, seed=0, layer=None, basis="spatial",
                                  a_within=None, b_within=None, mask_close_um=None,
                                  density_bw=None,
                                  key_added="spectrast_mark_cross_matrix", copy=False):
    """:func:`mark_cross_correlation` over a list of pairs — one transform per distinct field.

    Screening pairs is the normal use: a resource hands you hundreds of ligand-receptor pairs drawn
    from a few hundred distinct genes, so every gene appears in many pairs. Each distinct field is
    rasterized and transformed once and reused wherever it appears, and each null draw's geometry is
    built once and applied to every pair. The output matches looping
    :func:`mark_cross_correlation` over the same pairs at the same ``seed``.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    pairs : sequence of tuple
        ``(a, b)`` mark specs, each interpreted by ``a_type`` / ``b_type`` exactly as in the
        single-pair call.
    a_type, b_type, radii, normalize, n_bins, r_max, bin_um, null, n_perm, seed, layer, basis
        As in :func:`mark_cross_correlation`. One type applies to every A side and one to every B
        side.
    a_within, b_within, mask_close_um, density_bw
        As in :func:`mark_cross_correlation`.
    key_added : str, optional
        Key in ``adata.uns`` for the result. Default ``"spectrast_mark_cross_matrix"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of the result dict. Default ``False``.

    Returns
    -------
    dict or AnnData
        ``r``, ``g`` — ``(n_pairs_tested, n_bins)`` — ``pairs``, ``pair_names``, ``n_pairs``,
        ``edges``, ``normalize``; plus ``z`` and ``pval`` when a null is run. Also stored in
        ``adata.uns[key_added]``.

        In detail: ``n_pairs`` is the number of bin pairs per annulus (shared by every mark pair),
        ``pairs`` holds display labels (an array mark shows as ``array[n]``) and ``pair_names``
        joins them as ``"a~b"``. Under ``"density"`` and ``"density_robust"`` also ``g_geometry``
        ``(n_pairs_tested, n_bins)``. When a null is run, also ``null_mean``, ``null_sd`` (per pair
        and annulus), ``T_max``, ``p_max``, ``r_at_max_um``, ``n_valid`` (per pair), ``null`` and
        ``n_perm``. Always ``a_type``, ``b_type``, ``bin_um``, ``n_fields`` (distinct fields
        transformed) and ``backend``. The returned dict alone also carries ``pair_specs``, the
        original specs, which are left out of ``adata.uns``. With ``copy=True`` the modified copy
        is returned instead.

    See Also
    --------
    spectrast.bivariate.pair_table : rank the result by peak signal and interaction distance.
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    pairs = [tuple(p) for p in pairs]
    if not pairs:
        raise ValueError("pairs is empty")
    if bin_um is None:
        bin_um = default_radius(adata, basis) / 5.0
    extent = _mark_extent(adata, basis, bin_um)
    HW = extent[1]
    keep_a, keep_b = _within_mask(adata, a_within), _within_mask(adata, b_within)
    # Sides rasterized by the same rule share one registry, so a gene used as both ligand and
    # receptor is transformed once; a different type or a different `within` makes it a new field.
    tag_a = (a_type, None if a_within is None else "a")
    tag_b = (b_type, None if b_within is None else "b")

    reg, marks, dens, rc = {}, [], [], None

    def field(spec, mark_type, keep, tag):
        nonlocal rc
        key = (tag, _spec_key(spec))
        if key in reg:
            return reg[key]
        v, rc_ = _mark_grid(adata, spec, mark_type, layer=layer, basis=basis, bin_um=bin_um,
                            extent=extent)
        rc = rc_ if rc is None else rc
        mk, dn, _ = _rasterize_mark(v, rc_, HW, mark_type, keep=keep)
        reg[key] = len(marks)
        marks.append(mk); dens.append(dn)
        return reg[key]

    pairs_idx = [(field(a, a_type, keep_a, tag_a), field(b, b_type, keep_b, tag_b))
                 for a, b in pairs]
    mask = _tissue_mask(rc, HW, mask_close_um, bin_um)
    if normalize is None:
        normalize = "csr" if (a_type == "indicator" and b_type == "indicator") else "density"

    res = sp.mark_cross_correlation_batch(marks, mask, pairs_idx, densities=dens,
                                          normalize=normalize, bin_size=bin_um, n_bins=n_bins,
                                          r_max=r_max, annuli=radii,
                                          null=(null if n_perm else None), n_perm=n_perm,
                                          seed=seed, density_bw=density_bw)
    res["pairs"] = [(_spec_label(a), _spec_label(b)) for a, b in pairs]
    res["pair_names"] = [f"{a}~{b}" for a, b in res["pairs"]]
    res.update(a_type=a_type, b_type=b_type, bin_um=float(bin_um), normalize=normalize,
               n_fields=len(marks), backend=_backend())
    # uns gets the display form; the RETURNED dict also carries the original specs, because
    # str() destroys anything that is not already a gene name — ("cell_type", "T cell") becomes an
    # unparseable string and a per-cell array becomes its own repr — and the first-order map has to
    # rebuild the very fields the curve was computed from. Kept out of uns so an AnnData written to
    # disk does not carry a copy of every mark array.
    adata.uns[key_added] = dict(res)
    res["pair_specs"] = list(pairs)
    return adata if copy else res


def mark_cross_map(adata, pairs, *, a_type="expr", b_type="expr", r_lo=None, r_hi=None,
                   bin_um=None, mask_close_um=None, normalize=None, kind="contribution",
                   layer=None, basis="spatial", a_within=None, b_within=None,
                   key_added="spectrast_mark_cross_map", copy=False):
    """WHERE each pair's coupling lives — the per-cell first-order field behind the curve.

    :func:`mark_cross_correlation` answers at what RANGE two marks are coupled. It cannot answer
    where in the tissue, and a range without a map is not an interpretable result: two sections with
    identical curves can have the signal coming from a tumour margin in one and scattered through
    the stroma in the other. This returns the per-location integrand of the curve's own numerator at
    a chosen annulus, so summing it reproduces that numerator exactly (verified to ~1e-16).

    Pass ``r_lo``/``r_hi`` from a curve's ``edges`` at its peak — see
    :func:`spectrast.bivariate.first_order_map`, which does that for you — to get "where does the
    interaction at the optimal distance actually happen".

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    pairs : tuple or sequence of tuple
        One ``(a, b)`` pair or a list of them, interpreted by ``a_type`` / ``b_type`` as elsewhere.
    a_type, b_type
        As in :func:`mark_cross_correlation`. Default ``"expr"``.
    r_lo, r_hi : float
        Inner and outer edge of the annulus, in coordinate units. Both are required; the default
        ``None`` raises.
    bin_um, mask_close_um, layer, basis, a_within, b_within
        As in :func:`mark_cross_correlation`. Use the same ``bin_um`` as the curve being explained.
    normalize : {"density", "csr", "coherence"}, optional
        As in :func:`mark_cross_correlation`; ``"density_robust"`` is not available here. ``None``
        (default) uses ``"csr"`` when both sides are indicators and ``"density"`` otherwise.
    kind : {"contribution", "local"}, optional
        ``contribution`` sums to the numerator — overlay this on tissue. ``local`` is a locally
        normalized analogue of ``k`` (~1 = tissue-wide expectation), better when comparing regions
        of different cell density. Default ``"contribution"``.
    key_added : str, optional
        Key in ``adata.obsm`` for the per-cell map. Default ``"spectrast_mark_cross_map"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of the result dict. Default ``False``.

    Returns
    -------
    dict or AnnData
        ``per_cell`` ``(n_obs, n_pairs)``, ``fields`` ``(n_pairs, H, W)``, ``numerator`` per pair,
        ``pairs``, ``pair_names``, ``r_lo``, ``r_hi``. ``per_cell`` is also written to
        ``adata.obsm[key_added]`` with the pair order in ``adata.uns[key_added + "_pairs"]``.
        The dict also holds ``kind``, ``normalize``, ``bin_um`` and ``backend``. ``per_cell`` reads
        each cell's bin of ``fields``; under ``kind="local"`` bins with no pairs in the annulus are
        NaN. With ``copy=True`` the modified copy is returned instead.
    """
    from . import spatial as sp
    adata = adata.copy() if copy else adata
    if isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], (list, tuple)):
        pairs = [pairs]
    pairs = [tuple(p) for p in pairs]
    if not pairs:
        raise ValueError("pairs is empty")
    if r_lo is None or r_hi is None:
        raise ValueError("r_lo and r_hi are required — take them from the curve's edges at the "
                         "annulus you want to explain (bivariate.first_order_map does this)")
    if bin_um is None:
        bin_um = default_radius(adata, basis) / 5.0
    extent = _mark_extent(adata, basis, bin_um)
    HW = extent[1]
    keep_a, keep_b = _within_mask(adata, a_within), _within_mask(adata, b_within)
    if normalize is None:
        normalize = "csr" if (a_type == "indicator" and b_type == "indicator") else "density"

    fields, nums, rc = [], [], None
    for a, b in pairs:
        va, rc = _mark_grid(adata, a, a_type, layer=layer, basis=basis, bin_um=bin_um,
                            extent=extent)
        vb, _ = _mark_grid(adata, b, b_type, layer=layer, basis=basis, bin_um=bin_um, extent=extent)
        ma, na, _ = _rasterize_mark(va, rc, HW, a_type, keep=keep_a)
        mb, nb, _ = _rasterize_mark(vb, rc, HW, b_type, keep=keep_b)
        mask = _tissue_mask(rc, HW, mask_close_um, bin_um)
        out = sp.mark_cross_map(ma, mb, mask, r_lo=r_lo, r_hi=r_hi, density_a=na, density_b=nb,
                                normalize=normalize, bin_size=bin_um, kind=kind)
        fields.append(out["map"])
        nums.append(out["numerator"])
    per_cell = np.column_stack([f[rc[0], rc[1]] for f in fields])
    adata.obsm[key_added] = per_cell
    adata.uns[key_added + "_pairs"] = [f"{a}~{b}" for a, b in pairs]
    res = dict(per_cell=per_cell, fields=np.asarray(fields), numerator=np.asarray(nums),
               pairs=[(str(a), str(b)) for a, b in pairs],
               pair_names=[f"{a}~{b}" for a, b in pairs], r_lo=float(r_lo), r_hi=float(r_hi),
               kind=kind, normalize=normalize, bin_um=float(bin_um), backend=_backend())
    return adata if copy else res


def lric(adata, ligand, receptor, **kw):
    """Ligand-receptor interaction curve: g(r) for ligand expression against receptor expression.

    Thin wrapper on :func:`mark_cross_correlation` — it only names the marks. Answers not just
    *whether* a ligand-receptor pair is co-organized but over what RANGE, which is the part a
    single co-expression scalar cannot report.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    ligand, receptor : str
        Gene names in ``adata.var_names``; the A and B marks, both ``"expr"``.
    **kw
        Forwarded to :func:`mark_cross_correlation` (``radii``, ``null``, ``n_perm``,
        ``a_within``, ``layer``, ...). ``a_type`` and ``b_type`` are fixed and cannot be passed.

    Returns
    -------
    dict or AnnData
        As :func:`mark_cross_correlation`; written to ``adata.uns["spectrast_mark_cross"]`` unless
        ``key_added`` is passed.
    """
    return mark_cross_correlation(adata, ligand, receptor, a_type="expr", b_type="expr", **kw)


# None: the G2 sensitivity design found no calibrated stratification radius.
STRATA_RADIUS_FACTOR = None      # c in R_s = c * L_density; set by the G2 sensitivity design
MAJORITY_WARN = 0.5              # a type above this fraction of cells IS the density
RANGE_FLOOR = 0.05               # |g - 1| below this at the peak annulus is too small to call a
                                  # direction; r_cross_um is NaN rather than a crossing of noise
G_ZERO = 1e-9                    # a cross-PCF below this is a zero-pair annulus (FFT residue), missing


def _range_readouts(g, r):
    """``(r_peak_um, r_cross_um, r_half_um)`` per pair from a ``(P, nb)`` OBSERVED cross-PCF matrix.

    Descriptive, not a significance call (the strata null does not
    calibrate, so range is read off ``g`` itself rather than a null z). ``r_peak`` is the annulus
    of the largest |log g| (``g <= G_ZERO`` or NaN treated as missing; all NaN if every annulus is
    missing). ``r_cross`` is the first annulus beyond the peak where ``g - 1`` changes sign — where
    enrichment turns to exclusion or the reverse — NaN if the peak's |g - 1| is below
    ``RANGE_FLOOR`` (too small a deviation to have a direction to cross from) or if no sign change
    follows the peak. ``r_half`` (the PRIMARY range) is the half-decay length:
    with ``y = |log g|`` and the peak ``y_k``, the radius beyond ``r_k`` at which ``y`` first falls
    to ``y_k / 2``, interpolated in log r between the last valid annulus above the level and the
    first at or below it; NaN if it never does inside the window or the peak is below the floor.
    It exists for every pair with a peak, which the crossing does not (on real tissue most curves
    decay toward 1 without crossing it). A peak at the last valid annulus is the edge of the
    window, not a peak: all three lengths are NaN for that pair.
    """
    g = np.asarray(g, float); r = np.asarray(r, float)
    # An annulus with no pairs comes back from the FFT path as ~1e-16, not 0 (Task 7b, real tissue:
    # log g = -37 at such an annulus was read as the peak). Anything below G_ZERO is a zero.
    valid = np.isfinite(g) & (g > G_ZERO)
    dev = np.where(valid, g - 1.0, np.nan)          # g <= 0 or NaN is a missing annulus, not a
                                                      # crossing candidate, in BOTH the peak search
                                                      # and the sign-change scan (review fix: a
                                                      # zero-pair annulus read as dev=-1 could
                                                      # otherwise masquerade as r_cross_um)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_g = np.where(valid, np.log(g), np.nan)
    peak = np.full(len(g), np.nan); cross = np.full(len(g), np.nan); half = np.full(len(g), np.nan)
    lr = np.log(r)
    for p, lg in enumerate(log_g):
        if not np.isfinite(lg).any():
            continue
        y = np.abs(lg)
        k = int(np.nanargmax(y))
        if k == int(np.flatnonzero(np.isfinite(y))[-1]):
            continue                  # a peak at the last VALID annulus is the edge of the window,
                                      # not a peak: no length can be read from it (review 7b-2)
        peak[p] = r[k]
        dk = dev[p, k]
        assert np.isfinite(dk), "peak annulus must be a valid (finite, > 0) g by construction"
        if abs(dk) < RANGE_FLOOR:
            continue
        s = np.sign(dk)
        after = np.flatnonzero(np.sign(dev[p, k + 1:]) == -s)   # sign(nan) is nan, so a masked
                                                                  # (missing) annulus never matches
        if after.size:
            cross[p] = r[k + 1 + after[0]]
        level = 0.5 * y[k]
        prev = k                                                  # last valid annulus above the level
        for j in range(k + 1, len(r)):
            if not np.isfinite(y[j]):
                continue                                          # a missing annulus is skipped
            if y[j] <= level:
                f = (y[prev] - level) / max(y[prev] - y[j], 1e-12)
                half[p] = float(np.exp(lr[prev] + f * (lr[j] - lr[prev])))
                break
            prev = j
    return peak, cross, half


def _resolve_strata_radius(strata_radius_um, L_density):
    if strata_radius_um is not None:
        return float(strata_radius_um), "given"
    if STRATA_RADIUS_FACTOR is None:
        raise ValueError("strata_radius_um is required: the default rule (R_s = c * L_density) is "
                         "unset: no stratification radius has yet been found to calibrate")
    if not np.isfinite(L_density):
        raise ValueError("the density field has no characteristic scale on this section; pass "
                         "strata_radius_um explicitly")
    return float(STRATA_RADIUS_FACTOR * L_density), "estimated"


def celltype_range(adata, *, cell_type_key, types=None, min_cells=200, max_types=8, radii=None,
                   n_bins=12, r_max=None, bin_um=None, null="strata", strata_radius_um=None,
                   n_strata=10, n_perm=0, within=None, self_pairs=True, basis="spatial", seed=0,
                   mask_close_um=None, key_added="spectrast_celltype_range", copy=False):
    """Interaction RANGE for every unordered cell-type pair — descriptive by default.

    Per pair: the cross-PCF curve ``g(r)`` and three lengths read off it — ``r_peak_um`` (the
    annulus of the largest ``|log g|``), ``r_half_um`` (the PRIMARY range: the log-interpolated radius
    beyond the peak at which ``|log g|`` first falls to half its peak value) and ``r_cross_um`` (the
    secondary: the first annulus beyond the peak where enrichment turns to exclusion, or the
    reverse — which most real curves never do within the window). Both lengths are NaN when the
    peak's ``|g - 1|`` is below ``RANGE_FLOOR`` (too small a deviation to call a direction) or when
    the curve never reaches the level within the window. None is a number a single-graph
    neighbourhood enrichment can produce.

    A sensitivity study found the density-stratified label-permutation
    null (``null="strata"``; see :func:`spectrast.null.cross_pcf_strata_null`) does NOT calibrate on
    structured tissue — ring-free false-positive rate >= 0.47 at every stratification radius tried,
    and ``|T|`` on independent pairs grows with N. It ships **opt-in and uncalibrated**: the default is
    ``n_perm=0`` (no null is built, table is purely descriptive), and any ``n_perm > 0`` warns —
    ``z``, ``T_max`` and ``p_max`` are still computed and returned when requested, but as indicative
    numbers, not a significance test. Range as reported here is a descriptive length; its
    reliability comes from reproducibility across sections and stability across annotation
    resolution, not from a p-value.

    ``strata_radius_um`` is the smoothing radius the density strata are built with, only relevant
    once a permutation is actually requested (``n_perm > 0``). ``None`` estimates it from the
    section's own density scale once the sensitivity design has set a rule (``STRATA_RADIUS_FACTOR``
    — currently unset). ``within=(obs_key, value)`` restricts which cells carry a type
    label; the tissue support and the density stay whole-section.

    The majority-type check divides each candidate type's count by the number
    of TYPED cells (``within``-restricted, if given), not by ``adata.n_obs`` — the fraction the
    null's own identifiability limit is about.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; labels in ``adata.obs[cell_type_key]``.
    cell_type_key : str
        ``adata.obs`` column of cell-type labels (compared as strings). Required.
    types : list of str, optional
        Types to pair, used as given and in this order; each must have at least one typed cell.
        ``None`` keeps the ``max_types`` most abundant types with at least ``min_cells`` typed
        cells, sorted by name.
    min_cells : int, optional
        Smallest typed-cell count for automatic type selection; ignored when ``types`` is given.
        Default ``200``.
    max_types : int, optional
        Most types kept by automatic selection; ignored when ``types`` is given. Default ``8``.
    radii : array-like, optional
        Annulus EDGES in coordinate units, as in :func:`mark_cross_correlation`. The table's curve
        columns are named after the rounded annulus midpoints, so the midpoints must be at least one
        unit apart (coordinates in µm are expected). ``None`` builds ``n_bins`` equal-width annuli.
    n_bins, r_max, bin_um, mask_close_um
        As in :func:`mark_cross_correlation`. Defaults ``12``, ``None``, ``None``, ``None``.
    null : {"strata"}, optional
        The only null offered; any other value raises. Default ``"strata"``.
    strata_radius_um : float, optional
        Smoothing radius, in coordinate units, of the local-density field the strata are cut from.
        Used only when ``n_perm > 0``, where it is required: no default rule is set, so ``None``
        then raises.
    n_strata : int, optional
        Number of local-density quantile strata. Default ``10``.
    n_perm : int, optional
        Label permutations for the uncalibrated null. ``0`` (default) builds no null.
    within : tuple, dict or bool array, optional
        Which cells carry a type label, in any form :func:`fingerprint` accepts. Unselected cells
        stay in the tissue support and the density but are never paired or relabelled.
    self_pairs : bool, optional
        Include each type paired with itself. Default ``True``.
    basis : str, optional
        Coordinate key in ``adata.obsm``. Default ``"spatial"``.
    seed : int, optional
        Seed for the permutations. Default ``0``.
    key_added : str, optional
        Key in ``adata.uns`` for the table's provenance. Default ``"spectrast_celltype_range"``.
    copy : bool, optional
        If ``True`` return a modified copy instead of the table. Default ``False``.

    Returns
    -------
    pandas.DataFrame or AnnData
        One row per unordered type pair: ``type_a``, ``type_b``, ``n_a``, ``n_b`` (typed-cell
        counts), ``T_max`` and ``p_max`` (NaN without a null), ``r_peak_um``, ``r_half_um``,
        ``r_cross_um``, then ``g_<r>`` per annulus, where ``<r>`` is the rounded midpoint. With a
        null, also ``z_<r>`` and ``null_<r>`` (the null mean) per annulus. ``df.attrs`` holds
        ``radii_um``, ``edges_um``, ``bin_um``, ``strata_radius_um``, ``strata_radius_source``,
        ``L_density_um``, ``n_strata``, ``n_strata_used``, ``n_perm``, ``seed``, ``backend``,
        ``levels``, ``dropped_levels``, ``majority_warned``, ``null``, ``n_pairs``,
        ``cell_type_key`` and ``within``; that dict, not the table, is written to
        ``adata.uns[key_added]``. With ``copy=True`` the modified copy is returned instead.
    """
    import warnings
    import pandas as pd
    from . import null as nl
    from . import spatial as sp
    if null not in ("strata",):
        raise ValueError("celltype_range supports null='strata' only; 'labels' (random labelling) "
                         "was measured at 50-100% false positives on clustered tissue and removed, "
                         "and 'shift' is degenerate on point patterns")
    if null == "strata" and n_perm > 0:
        warnings.warn(
            "celltype_range: null='strata' is not calibrated (sensitivity study: "
            "false-positive rate >= 0.47 at every stratification radius); z, T_max and p_max are "
            "indicative only, not a significance test", RuntimeWarning, stacklevel=2)
    adata = adata.copy() if copy else adata
    if bin_um is None:
        bin_um = default_radius(adata, basis) / 5.0
    extent = _mark_extent(adata, basis, bin_um)
    (x0, y0), HW = extent
    xy = _coords(adata, basis)
    col = np.clip(((xy[:, 0] - x0) / bin_um).astype(int), 0, HW[1] - 1)
    row = np.clip(((xy[:, 1] - y0) / bin_um).astype(int), 0, HW[0] - 1)
    mask = _tissue_mask((row, col), HW, mask_close_um, bin_um)

    lab_str = adata.obs[cell_type_key].astype(str).to_numpy()
    keep = _within_mask(adata, within)
    typed = np.ones(adata.n_obs, bool) if keep is None else np.asarray(keep, bool)
    counts = pd.Series(lab_str[typed]).value_counts()
    if types is not None:
        levels = [str(t) for t in types]
        missing = [t for t in levels if t not in counts.index]
        if missing:
            raise ValueError(f"types {missing} have no typed cells for {cell_type_key!r} "
                             f"(within={within!r}); check spelling or the within= filter")
    else:
        # auto-selection ranks by abundance (min_cells filter, max_types cap) but the table's
        # type_a/type_b order must not depend on which type happened to have more cells this
        # section — sort the survivors by name so it agrees with an explicit types=[a, b] call.
        levels = sorted(counts[counts >= int(min_cells)].index[:int(max_types)])
    if len(levels) < 1:
        raise ValueError(f"no level of {cell_type_key!r} has >= {min_cells} cells")
    dropped = [c for c in counts.index if c not in levels]
    codes = pd.Categorical(lab_str, categories=levels).codes.astype(int)
    codes[~typed] = -1
    labels = codes
    majority = [t for t in levels if counts[t] / typed.sum() > MAJORITY_WARN]
    if majority:
        warnings.warn(f"celltype_range: {majority} exceed {MAJORITY_WARN:.0%} of cells — a "
                      "majority type IS the density, so 'near it' and 'in dense tissue' are one "
                      "geometry; the null cannot separate them", RuntimeWarning)

    L_density = nl.density_scale_um(row, col, HW, mask, bin_um=bin_um)
    # No null is built at n_perm=0, so an unresolved default radius (STRATA_RADIUS_FACTOR unset,
    # strata_radius_um not given) must not block a plain g(r) read — only skip resolving/building
    # strata in that specific case; an explicitly given strata_radius_um is still honored.
    skip_strata = (n_perm == 0 and strata_radius_um is None)
    if skip_strata:
        R_s, source = float("nan"), "unused"
        strata = np.zeros(len(row), int)
    else:
        R_s, source = _resolve_strata_radius(strata_radius_um, L_density)
        strata = nl.density_strata(row, col, HW, radius_bins=max(1, int(round(R_s / bin_um))),
                                   n_strata=n_strata)
    edges = (np.asarray(radii, float) if radii is not None
             else sp._mcc_annuli(bin_um, n_bins, r_max, HW[0], HW[1]))
    res = nl.cross_pcf_strata_null(labels, row, col, HW, mask, edges, strata=strata,
                                   n_types=len(levels), n_perm=n_perm, seed=seed,
                                   bin_size=bin_um, self_pairs=self_pairs)
    n_strata_used = 0 if skip_strata else int(res["n_strata_used"])
    r = np.asarray(res["r"])
    names = [f"g_{int(round(x))}" for x in r]
    if len(set(names)) < len(r):
        raise ValueError(
            "celltype_range: two or more annuli round to the same integer micrometre column "
            f"name ({r.tolist()!r} -> {names!r}) — check that coordinates are in micrometres "
            "(a mm/um mix-up is the usual cause), or pass radii at least 1 um apart")
    rows = []
    have_null = "z" in res
    peak, cross, half = _range_readouts(res["g"], r)
    for p, (i, j) in enumerate(res["pairs"]):
        d = dict(type_a=levels[i], type_b=levels[j], n_a=int(counts[levels[i]]),
                 n_b=int(counts[levels[j]]),
                 T_max=res["T_max"][p] if have_null else np.nan,
                 p_max=res["p_max"][p] if have_null else np.nan,
                 r_peak_um=peak[p], r_half_um=half[p], r_cross_um=cross[p])
        for k, rk in enumerate(r):
            d[f"g_{int(round(rk))}"] = res["g"][p, k]
            if have_null:
                d[f"z_{int(round(rk))}"] = res["z"][p, k]
                d[f"null_{int(round(rk))}"] = res["null_mean"][p, k]
        rows.append(d)
    tab = pd.DataFrame(rows)
    within_out = list(within) if isinstance(within, tuple) else within
    tab.attrs.update(radii_um=r.tolist(), edges_um=np.asarray(edges).tolist(), bin_um=float(bin_um),
                     strata_radius_um=R_s, strata_radius_source=source,
                     L_density_um=float(L_density), n_strata=int(n_strata),
                     n_strata_used=n_strata_used, n_perm=int(n_perm), seed=int(seed),
                     backend=_backend(), levels=levels, dropped_levels=dropped,
                     majority_warned=majority, null=null, n_pairs=np.asarray(res["n_pairs"]).tolist(),
                     cell_type_key=cell_type_key, within=within_out)
    adata.uns[key_added] = dict(tab.attrs)
    return adata if copy else tab


def celltype_range_matrix(table, value="r_half_um"):
    """Symmetric type x type matrix of one column of :func:`celltype_range`'s table.

    Parameters
    ----------
    table : pandas.DataFrame
        The table returned by :func:`celltype_range`.
    value : str, optional
        Column to place in the matrix, e.g. ``"r_half_um"`` (default), ``"r_peak_um"``,
        ``"r_cross_um"``, ``"p_max"`` or one ``"g_<r>"`` column.

    Returns
    -------
    pandas.DataFrame
        Square, with the types as both index and columns in order of first appearance in
        ``type_a`` then ``type_b``. Entry ``(i, j)`` equals entry ``(j, i)``; pairs absent from
        the table are NaN.
    """
    import pandas as pd
    lv = list(dict.fromkeys(list(table.type_a) + list(table.type_b)))
    M = pd.DataFrame(np.nan, index=lv, columns=lv)
    for _, rw in table.iterrows():
        M.loc[rw.type_a, rw.type_b] = rw[value]
        M.loc[rw.type_b, rw.type_a] = rw[value]
    return M


def cross_pcf(adata, type_a, type_b, *, cell_type_key="cell_type", null="strata",
              strata_radius_um=None, n_strata=10, n_perm=0, **kw):
    """Cross pair-correlation function between two cell types — infiltration vs exclusion by scale.

    Indicator marks, so the mean-product-over-product-of-means reduces exactly to the classical
    cross-PCF (verified against direct pair counting). The null is the label
    permutation within local-density strata (``"strata"``, :func:`celltype_range`) — opt-in and
    NOT calibrated on clustered point patterns (ring-free false-positive
    rate >= 0.47 at every stratification radius tried). ``"shift"`` and ``"reshuffle"`` remain
    available for the record but are degenerate/uncalibrated too, and ``"labels"`` is refused.

    The default ``n_perm=0`` builds no null, so the returned ``g`` is purely descriptive — this is
    the intended, quiet everyday call. Passing ``n_perm > 0`` under ``null="strata"`` warns (see
    :func:`celltype_range`): ``z``, ``T_max`` and ``p_max`` are still returned but are indicative
    only, not a significance test.

    ``null="strata"`` writes ``adata.uns["spectrast_cross_pcf"]`` (not
    ``"spectrast_celltype_range"``, which a separate :func:`celltype_range` call on this same
    ``adata`` keeps as its own record) unless a caller passes its own ``key_added=``. ``type_a ==
    type_b`` is a valid self-pair. At the default ``n_perm=0`` no null is built, so an unresolved
    ``STRATA_RADIUS_FACTOR``/``strata_radius_um`` does not block reading ``g``; ``copy=True`` is
    refused here — this function always returns a dict, never an AnnData copy.

    **The 0.11.0 surface, under the (now default) ``null="strata"``.** The call delegates entirely
    to :func:`celltype_range` — it builds a one-pair (or, for a self-pair, one-type) call, pulls
    the single resulting row back out and reshapes it into a dict. The returned keys are always
    ``r, g, a, b, null, strata_radius_um, L_density_um, n_strata, n_perm, bin_um, backend``, plus
    ``z, null_mean, T_max, p_max, r_peak_um, r_half_um, r_cross_um`` only when ``n_perm > 0``. Under
    this path ``cross_pcf`` does **not** accept ``a_within``, ``b_within``, ``layer``,
    ``normalize``, ``density_bw`` or ``copy`` — those belong to the
    :func:`mark_cross_correlation`-backed continuous-mark surface below and are silently absent
    from ``celltype_range``'s signature (an unrecognised keyword raises a ``TypeError`` from
    ``celltype_range``, not from here). To reach the old surface — ``a_within``/``b_within``,
    ``normalize``, ``density_bw``, ``copy``, and a real p-value from the ``shift``/``reshuffle``
    null on continuous marks — pass ``null="shift"`` or ``null="reshuffle"``, which falls through to
    :func:`mark_cross_correlation` with cell-type indicator marks exactly as before. **The default
    changed from ``"shift"`` to ``"strata"`` at 0.11.0**: a caller that has not been updated and
    relied on the implicit ``shift`` null (and its ``a_within``/``b_within``/``normalize`` kwargs)
    will now silently get the descriptive, unpermuted ``strata`` path at ``n_perm=0`` unless it
    passes ``null="shift"`` explicitly.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; labels in ``adata.obs[cell_type_key]``.
    type_a, type_b
        The two cell-type labels (compared as strings). Equal labels give a self-pair.
    cell_type_key : str, optional
        ``adata.obs`` column of cell-type labels. Default ``"cell_type"``.
    null : {"strata", "shift", "reshuffle"}, optional
        ``"strata"`` (default) goes through :func:`celltype_range`; ``"shift"`` and
        ``"reshuffle"`` go through :func:`mark_cross_correlation` with indicator marks.
        ``"labels"`` raises.
    strata_radius_um, n_strata
        As in :func:`celltype_range`; used only under ``null="strata"``. Defaults ``None`` and
        ``10``.
    n_perm : int, optional
        Number of null draws. ``0`` (default) builds no null.
    **kw
        Forwarded to :func:`celltype_range` under ``null="strata"`` (``radii``, ``n_bins``,
        ``r_max``, ``bin_um``, ``within``, ``seed``, ``mask_close_um``, ``basis``, ``key_added``)
        or to :func:`mark_cross_correlation` otherwise.

    Returns
    -------
    dict or AnnData
        Under ``null="strata"``, the dict described above: ``r`` (annulus midpoints, coordinate
        units), ``g``, ``a``, ``b``, ``null``, ``strata_radius_um``, ``L_density_um``,
        ``n_strata``, ``n_perm``, ``bin_um`` and ``backend``, plus ``z``, ``null_mean``,
        ``T_max``, ``p_max``, ``r_peak_um``, ``r_half_um`` and ``r_cross_um`` when ``n_perm > 0``;
        the table's provenance goes to ``adata.uns["spectrast_cross_pcf"]`` unless ``key_added`` is
        passed. Under ``"shift"`` or ``"reshuffle"``, the return value of
        :func:`mark_cross_correlation` (an AnnData only if ``copy=True`` is passed), stored in
        ``adata.uns["spectrast_mark_cross"]`` unless ``key_added`` is passed.
    """
    if null == "labels":
        raise ValueError("null='labels' (random labelling) was removed: 50-100% false positives on "
                         "clustered tissue. Use null='strata'.")
    if null == "strata":
        if kw.get("copy"):
            raise ValueError("cross_pcf(null='strata') does not support copy=True: cross_pcf "
                             "returns a dict, never an AnnData copy — call celltype_range directly "
                             "for copy=True's behavior.")
        kw.setdefault("key_added", "spectrast_cross_pcf")
        same_type = str(type_a) == str(type_b)
        tab = celltype_range(adata, cell_type_key=cell_type_key,
                             types=[type_a] if same_type else [type_a, type_b],
                             strata_radius_um=strata_radius_um, n_strata=n_strata, n_perm=n_perm,
                             self_pairs=same_type, **kw)
        rw = tab.iloc[0]
        r = np.asarray(tab.attrs["radii_um"])
        res = dict(r=r, g=rw[[f"g_{int(round(x))}" for x in r]].to_numpy(float),
                   a=str(type_a), b=str(type_b), null="strata", **{k: tab.attrs[k] for k in
                   ("strata_radius_um", "L_density_um", "n_strata", "n_perm", "bin_um", "backend")})
        if n_perm:
            res.update(z=rw[[f"z_{int(round(x))}" for x in r]].to_numpy(float),
                       null_mean=rw[[f"null_{int(round(x))}" for x in r]].to_numpy(float),
                       T_max=float(rw.T_max), p_max=float(rw.p_max),
                       r_peak_um=float(rw.r_peak_um), r_half_um=float(rw.r_half_um),
                       r_cross_um=float(rw.r_cross_um))
        return res
    return mark_cross_correlation(adata, (cell_type_key, type_a), (cell_type_key, type_b),
                                  a_type="indicator", b_type="indicator", null=null,
                                  n_perm=n_perm, **kw)


def gene_niche(adata, gene, cell_type, *, cell_type_key="cell_type", **kw):
    """Is a gene organized AROUND a cell type, and how tightly? Mixed continuous x indicator marks.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    gene : str
        Gene name in ``adata.var_names``; the A mark (``"expr"``).
    cell_type
        Label of the cell type; the B mark is the count of cells with this label per bin.
    cell_type_key : str, optional
        ``adata.obs`` column holding the labels. Default ``"cell_type"``.
    **kw
        Forwarded to :func:`mark_cross_correlation`. ``normalize`` defaults to ``"density"`` here,
        because only one side is an indicator.

    Returns
    -------
    dict or AnnData
        As :func:`mark_cross_correlation`.
    """
    return mark_cross_correlation(adata, gene, (cell_type_key, cell_type),
                                  a_type="expr", b_type="indicator", **kw)


def program_cross(adata, score_a, score_b, **kw):
    """Co-localization of two program scores (``adata.obs`` columns) and the scale it holds over.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    score_a, score_b : str
        Numeric ``adata.obs`` columns with no negative values. Scores that can be negative, such
        as those of :func:`scanpy.tl.score_genes`, must be shifted first.
    **kw
        Forwarded to :func:`mark_cross_correlation`.

    Returns
    -------
    dict or AnnData
        As :func:`mark_cross_correlation`.
    """
    return mark_cross_correlation(adata, score_a, score_b, a_type="obs", b_type="obs", **kw)


def covariate_cross(adata, field, covariate, *, field_type="expr", **kw):
    """A gene or program against a continuous ``obs`` covariate — organization WITH architecture.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    field
        The A mark, read according to ``field_type``.
    covariate : str
        Numeric, non-negative ``adata.obs`` column; the B mark.
    field_type : {"expr", "obs", "obsm", "indicator", "array"}, optional
        How to read ``field``, as ``a_type`` in :func:`mark_cross_correlation`. Default ``"expr"``.
    **kw
        Forwarded to :func:`mark_cross_correlation`.

    Returns
    -------
    dict or AnnData
        As :func:`mark_cross_correlation`.
    """
    return mark_cross_correlation(adata, field, covariate, a_type=field_type, b_type="obs", **kw)


def _gene_set_mean(adata, genes, layer):
    """Per-cell mean expression over a gene set — the mark for a regulon or a small program."""
    idx, names = _gene_idx(adata, list(genes))
    if not len(idx):
        raise ValueError("no gene in the set was found in adata.var_names")
    X = _matrix(adata, layer)[:, list(idx)]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    return X.mean(axis=1), list(names)


def regulon_cross(adata, tf, targets, *, layer=None, **kw):
    """A transcription factor against its target set — spatial coherence of a regulon.

    The B mark is the per-cell MEAN over the targets, so the question is whether the TF's field and
    its targets' field share territory and over what range — a spatial readout of regulatory
    activity that a bulk TF-target correlation cannot give, because that correlation is the same
    number whether the two are interleaved cell by cell or sit in separate regions.

    Parameters
    ----------
    adata : AnnData
        As in :func:`mark_cross_correlation`.
    tf : str
        Gene name of the transcription factor; the A mark (``"expr"``).
    targets : iterable of str
        Target gene names. Every one must be in ``adata.var_names``; a missing name raises
        ``KeyError``.
    layer : str, optional
        Expression layer for both the TF and the target mean; ``None`` uses ``adata.X``.
    **kw
        Forwarded to :func:`mark_cross_correlation`. ``copy=True`` is not supported.

    Returns
    -------
    dict
        As :func:`mark_cross_correlation`, with ``a`` set to the TF name, ``b`` to
        ``"regulon[<n targets>]"`` and an extra ``targets`` list. The same dict is stored in
        ``adata.uns[key_added]`` (default ``"spectrast_mark_cross"``).
    """
    v, names = _gene_set_mean(adata, targets, layer)
    res = mark_cross_correlation(adata, tf, v, a_type="expr", b_type="array", layer=layer, **kw)
    res.update(a=str(tf), b=f"regulon[{len(names)}]", targets=names)
    return res


_FP_CROSS_AXES = {"local_moran": "local_morans_i_genes", "getis": "getis_ord_genes"}


def fingerprint_cross(adata, gene_a, gene_b, *, axis="local_moran", radius=None, sign="positive",
                      layer=None, basis="spatial", **kw):
    """Do two genes share the same organizational HOTSPOTS — a second-order statistic ON the
    fingerprint.

    Every other bivariate mark asks whether the EXPRESSION of two genes occupies the same territory.
    This one asks whether their ORGANIZATION does: is the region where gene A is spatially clustered
    the same region where gene B is, whether or not the two are ever expressed in the same cells. A
    pair can be expression-anticorrelated (mutually exclusive cells) and organization-correlated
    (both clustered in the same territory) — evidence of a shared organizing field, and a
    distinction no co-expression measure can draw.

    The marks are per-cell fingerprint fields, so the statistic is second-order on a first-order
    organizational quantity. That is the case that cannot exist without the fingerprint machinery.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or ``adata.layers[layer]``.
    gene_a, gene_b : str
        Gene names in ``adata.var_names``. Their per-cell fields are the A and B marks.
    axis : {"local_moran", "getis"}, optional
        Which organizational field. ``local_moran`` is clustered-ness (a cell in a same-signed
        cluster); ``getis`` is hotspot intensity. A LOCAL focality field is deliberately not offered:
        focality is a Gini over a whole map, and a windowed version of it is a primitive this package
        does not have — substituting the global scalar would be a different statistic wearing the
        same name. Default ``"local_moran"``.
    radius : float, optional
        Neighbourhood radius for the fingerprint field, in coordinate units. Default
        :func:`default_radius`. The fields are computed as :func:`local_moran` / :func:`getis_ord`
        compute them with their other defaults.
    sign : {"positive", "abs"}, optional
        Both fields are signed, and the statistic needs a non-negative mark. ``"positive"`` (default)
        clips at zero, keeping clustered-ness and discarding spatial outliers — "is A organized
        here". ``"abs"`` keeps outlier structure as organization too.
    layer : str, optional
        Expression layer the fingerprint fields are computed from; ``None`` uses ``adata.X``.
    basis : str, optional
        Coordinate key in ``adata.obsm``, used for the fields and the cross-correlation. Default
        ``"spatial"``.
    **kw
        Forwarded to :func:`mark_cross_correlation` (``radii``, ``null``, ``n_perm``, ...).
        ``copy=True`` is not supported.

    Returns
    -------
    dict
        As :func:`mark_cross_correlation`, with ``a`` and ``b`` set to the gene names, plus
        ``fingerprint_axis``, ``sign`` and ``radius_um`` (the field radius, in coordinate units).
        The same dict is stored in ``adata.uns[key_added]`` (default ``"spectrast_mark_cross"``).

    Notes
    -----
    The fingerprint fields are themselves neighbourhood-smoothed, so ``null="reshuffle"`` is badly
    anti-conservative on them; the default ``null="shift"`` preserves each field's own
    autocorrelation and is the calibrated choice here.
    """
    from . import spatial as sp
    if axis not in _FP_CROSS_AXES:
        raise ValueError(f"axis must be one of {sorted(_FP_CROSS_AXES)}, got {axis!r}")
    if sign not in ("positive", "abs"):
        raise ValueError("sign must be 'positive' or 'abs'")
    xy = _coords(adata, basis)
    idx, names = _gene_idx(adata, [gene_a, gene_b])
    r = float(radius) if radius is not None else default_radius(adata, basis)
    M = np.asarray(getattr(sp, _FP_CROSS_AXES[axis])(xy, _matrix(adata, layer), r, list(idx)))
    M = np.abs(M) if sign == "abs" else np.clip(M, 0.0, None)
    res = mark_cross_correlation(adata, M[:, 0], M[:, 1], a_type="array", b_type="array",
                                 basis=basis, **kw)
    res.update(a=str(names[0]), b=str(names[1]), fingerprint_axis=axis, sign=sign, radius_um=r)
    return res


# The unified organization API. Imported at the end so the lazy
# `from .gr import ...` calls inside organization.py resolve against a fully-built module.
# `fingerprint` is deliberately NOT redefined in terms of `organization`: it returns a gene x axis
# table, `organization` returns a gene x statistic tidy, and four figure generators consume the
# former -- aliasing would break exactly what the alias was meant to protect.
from .organization import (ORG_STATS, organization, presence,  # noqa: E402,F401
                           presence_variation, celltype_scale)

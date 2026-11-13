"""spectrast.bivariate — reading biology off a mark cross-correlation curve.

:func:`spectrast.gr.mark_cross_correlation` returns a curve per pair; a curve is not an answer. This
module holds the four reductions that are:

  O1  :func:`interaction_distance` — the distance at which a pair most co-organizes, its sign
      (attraction or exclusion), and whether the sign FLIPS across scale.
  O2  :func:`pair_table` — every pair ranked by peak signal, with a peak-selection-corrected p and
      an FDR over pairs. The discovery table.
  O3  :func:`annulus_matrix` / :func:`heatmap` — pairs x distance, so strength and range are read in
      one glance and bands of pairs sharing an interaction scale become visible.
  O4  :func:`stratified_pair_table` — the same ranking with ONE side confined to a cell
      population, answering which population carries an interaction; and
      :func:`directed_pair_table`, which confines BOTH sides to give the directional
      sender -> receiver view (the statistic is genuinely asymmetric, so A -> B and B -> A are
      different numbers, not a relabelling).

Plus the FIRST-ORDER half, without which none of the above is interpretable:
  :func:`first_order_map` — the per-location field whose sum IS the curve's numerator at the pair's
  peak annulus. A curve gives the RANGE; the map gives the PLACE. Report them together.

All four are MARK-AGNOSTIC: they consume a curve and a pair index, nothing ligand-receptor specific.
The same table that ranks LR pairs by interaction range ranks program pairs by the scale at which
two pathways share territory, or fingerprint pairs by the scale at which two genes share
organizational hotspots.

Read the sign against the normalizer. Under ``normalize="density"`` (the default, Stoyan's mark
correlation) the curve is centred on 1 with geometry divided out, so a peak IS an interaction range.
Under ``normalize="coherence"`` it is descriptive only and its peak must not be read as a distance.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["autotune_bin", "interaction_distance", "pair_table", "annulus_matrix", "heatmap",
           "first_order_map", "stratified_pair_table", "directed_pair_table",
           "directed_matrix"]

FLIP_MIN = 0.05        # |k-1| a lobe must reach on both sides before a sign change counts as a flip


def _curves(res):
    """Normalize a single-pair or batched result to ``(names, pairs, r, g, z, pval)`` 2-D form."""
    g = np.asarray(res["g"], float)
    single = g.ndim == 1
    g = np.atleast_2d(g)
    r = np.asarray(res["r"], float)
    if "pair_names" in res:
        names = list(res["pair_names"])
        pairs = [tuple(p) for p in res.get("pairs", [(n, "") for n in names])]
    else:
        a, b = str(res.get("a", "A")), str(res.get("b", "B"))
        names, pairs = [f"{a}~{b}"], [(a, b)]

    def col(key, scalar=False):
        if key not in res:
            return None
        v = np.asarray(res[key], float)
        return v.reshape(1, -1) if (single and not scalar) else np.atleast_1d(v)

    return (names, pairs, r, g, col("z"), col("pval"),
            col("p_max", True), col("T_max", True), col("r_at_max_um", True))


def autotune_bin(adata, *, basis="spatial", nn_quantile=0.5, bins_per_nn=2.0, n_annuli=12,
                 r_max_frac=0.15, max_pixels=4_000_000, sample=50_000, seed=0, verbose=False):
    """Choose the lattice bin and the annuli FROM THE CELL SPACING, instead of by hand.

    A hand-picked bin silently decides what the analysis can see. At 20 µm bins the innermost
    annulus lands near 46 µm, and on breast sections 57-64% of pairs then "peak" there — the number
    reported is the binning choice, not an interaction range. The floor is not a free parameter: it
    is set by how far apart the cells actually are.

    Three constraints, all measured rather than assumed:

    1. **The physical floor.** Cell centroids cannot be closer than roughly one cell diameter, so the
       nearest-neighbour distance is the smallest separation that carries real pairs. It anchors the
       innermost annulus (breast: median 8.2 µm, with 71% of cells having a partner within 10 µm —
       which is why 8 µm annuli resolve and 46 µm cannot).
    2. **The lattice must be able to represent it.** Displacements come in multiples of the bin, so a
       bin coarser than the nearest-neighbour distance cannot place an annulus there at all.
       ``bins_per_nn`` sets how many bins span that distance.
    3. **The toroidal shift null needs the support to overlap its own translate** (25%, see
       :data:`spectrast.spatial.MCC_MIN_SHIFT_OVERLAP`). Below roughly one cell spacing most bins are
       empty, the occupancy mask stops being "the tissue", and the null dies — measured 50% overlap
       at 20 µm falling to 10% at 6 µm. The fix is a filled tissue support, so this returns the
       ``mask_close_um`` needed rather than leaving the caller to discover the failure.

    All lengths are in the units of ``adata.obsm[basis]`` (µm for coordinates in µm).

    Parameters
    ----------
    adata : AnnData
        Cell coordinates in ``adata.obsm[basis]``; only the first two columns are used.
    basis : str, optional
        Key in ``adata.obsm`` holding the coordinates. Default ``"spatial"``.
    nn_quantile : float, optional
        Quantile of the per-cell nearest-neighbour distance taken as the cell spacing ``nn_um``.
        Default ``0.5`` (the median).
    bins_per_nn : float, optional
        Bins across one nearest-neighbour distance: the bin is ``nn_um / bins_per_nn``. Default
        ``2.0``.
    n_annuli : int, optional
        Number of annuli. ``radii`` has ``n_annuli + 1`` edges, spaced geometrically from ``nn_um``
        to the outer edge. Default ``12``.
    r_max_frac : float, optional
        Outer annulus edge as a fraction of the shorter side of the coordinate bounding box. The
        edge is capped at 600 and raised to at least ``8 * nn_um``. Default ``0.15``.
    max_pixels : int, optional
        Largest lattice allowed. If the spacing-derived bin would give more pixels than this, the
        bin is widened to ``sqrt(x_span * y_span / max_pixels)`` and a note is added. Default
        ``4_000_000``.
    sample : int, optional
        Number of cells, drawn at random, whose nearest-neighbour distance is measured. The
        neighbour search still runs over every cell. Default ``50_000``.
    seed : int, optional
        Seed for the cell subsample and for the random toroidal shifts of the overlap test.
        Default ``0``.
    verbose : bool, optional
        Print a one-line summary and any notes. Default ``False``.

    Returns
    -------
    dict
        ``bin_um``, ``radii`` (annulus EDGES, pass straight to ``radii=``), ``mask_close_um``
        (``None`` when the raw occupancy already overlaps enough, and also when none of the closing
        radii tried, 2, 4, 6 and 8 bins, reaches the 25% floor; ``notes`` then says so), and the
        diagnostics behind them:

        - ``nn_um`` : float, the cell spacing (``nn_quantile`` of the nearest-neighbour distance).
        - ``median_occupancy`` : float, median number of cells per occupied bin.
        - ``shift_overlap`` : float, median fraction of the support that overlaps its own
          translate over 8 random toroidal shifts; measured on the closed support when
          ``mask_close_um`` is set.
        - ``grid`` : tuple ``(H, W)`` of the lattice at ``bin_um``; ``pixels`` is ``H * W``.
        - ``notes`` : list of str, why the bin was widened or a closing is needed.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> a = ad.AnnData(np.abs(rng.standard_normal((800, 4))).astype("float32"))
    >>> a.obsm["spatial"] = rng.random((800, 2)) * 400
    >>> tune = spectrast.bivariate.autotune_bin(a)
    >>> tune["bin_um"] > 0 and len(tune["radii"]) == 13
    True
    """
    from .spatial import MCC_MIN_SHIFT_OVERLAP
    xy = np.asarray(adata.obsm[basis], float)[:, :2]
    rng = np.random.default_rng(seed)
    notes = []

    # 1. nearest-neighbour spacing, on a subsample when the section is large
    sub = xy if len(xy) <= sample else xy[rng.choice(len(xy), sample, replace=False)]
    from sklearn.neighbors import NearestNeighbors           # lazy
    d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(sub)
    nn = float(np.quantile(d[:, 1], nn_quantile))
    if not np.isfinite(nn) or nn <= 0:
        raise ValueError(
            "nearest-neighbour distance is zero or non-finite: coordinates are duplicated or "
            "degenerate, so no bin can be derived from them. Check that obsm holds distinct "
            "positions and that the units are what you think they are.")

    # 2. bin from the spacing, then widened if the grid would be unreasonable
    bin_um = nn / float(bins_per_nn)
    span = np.ptp(xy, axis=0)
    px = (span[1] / bin_um + 1) * (span[0] / bin_um + 1)
    if px > max_pixels:
        bin_um = float(np.sqrt(span[0] * span[1] / max_pixels))
        notes.append(f"bin widened to {bin_um:.1f} µm to keep the grid under "
                     f"{max_pixels:,} pixels; the spacing alone would allow {nn / bins_per_nn:.1f}")

    def grid_stats(b, close_um=None):
        from . import gr as _gr
        x0, y0 = xy.min(0)
        col = ((xy[:, 0] - x0) / b).astype(int)
        row = ((xy[:, 1] - y0) / b).astype(int)
        H, W = int(row.max()) + 1, int(col.max()) + 1
        occ = _gr._tissue_mask((row, col), (H, W), close_um, b)
        cnt = np.zeros((H, W))
        np.add.at(cnt, (row, col), 1.0)
        ov = []
        for _ in range(8):
            cy, cx = int(rng.integers(1, H)), int(rng.integers(1, W))
            ov.append((np.roll(np.roll(occ, cy, 0), cx, 1) & occ).sum() / max(occ.sum(), 1))
        return H, W, float(np.median(cnt[cnt > 0])), float(np.median(ov))

    H, W, occup, overlap = grid_stats(bin_um)

    # 3. does the shift null survive here? if not, find the closing radius that rescues it
    close = None
    if overlap < MCC_MIN_SHIFT_OVERLAP:
        for k in (2.0, 4.0, 6.0, 8.0):
            cand = k * bin_um
            _, _, _, ov2 = grid_stats(bin_um, cand)
            if ov2 >= MCC_MIN_SHIFT_OVERLAP:
                close, overlap = cand, ov2
                break
        notes.append(
            f"occupancy overlaps its own translate only {overlap:.0%} at this bin"
            + (f", so a filled support with a {close:.0f} µm closing is REQUIRED for the shift null"
               if close else " and no closing radius tried reached the 25% floor — treat any "
                             "p-value here as unusable"))

    lo = nn                                    # innermost annulus edge = the physical floor
    hi = max(min(r_max_frac * float(span.min()), 600.0), lo * 8)
    radii = np.geomspace(lo, hi, int(n_annuli) + 1)
    out = dict(bin_um=float(bin_um), radii=radii, mask_close_um=close, nn_um=nn,
               median_occupancy=occup, shift_overlap=overlap, grid=(H, W), pixels=int(H * W),
               notes=notes)
    if verbose:
        print(f"  bin {bin_um:.1f} µm | NN {nn:.1f} µm | grid {H}x{W} | occ {occup:.1f} cells/bin "
              f"| shift overlap {overlap:.0%}" + (f" | close {close:.0f} µm" if close else ""))
        for n in notes:
            print(f"    note: {n}")
    return out


#: Warn from :func:`pair_table` when at least this share of pairs peak on the innermost annulus.
#: 0.25 because the measured rate on real ligand-receptor panels is 38-67 %, so anything at or above
#: a quarter is the regime where a between-group distance comparison is confounded by censoring.
CENSOR_WARN = 0.25


def interaction_distance(res, pair=0, *, flip_min=FLIP_MIN):
    """O1 — the distance at which one pair most co-organizes, and the shape of its sign structure.

    The peak is taken on ``|k(r) - 1|``, not on ``k`` itself: exclusion (``k < 1``) is as real a
    result as attraction and must not be ranked below a weaker attraction. A pair can also do both —
    co-localize at one distance and exclude at another — and that CROSS-SCALE SIGN FLIP is reported
    as a first-class field rather than averaged away, which is what a single scalar co-expression
    number does to it.

    Parameters
    ----------
    res : dict
        A result from :func:`spectrast.gr.mark_cross_correlation` (single pair) or
        :func:`spectrast.gr.mark_cross_correlation_matrix` (batched).
    pair : int, optional
        Row to read when ``res`` is batched. Default ``0``.
    flip_min : float, optional
        Minimum ``|k - 1|`` a lobe must reach on BOTH sides of 1 for the flip to count, so noise
        crossing the line does not read as biology. Default ``0.05``.

    Returns
    -------
    dict
        ``pair`` (the pair name), ``a``, ``b`` (its two sides), ``peak_r_um`` (annulus centre of the
        peak, in coordinate units), ``peak_k``, ``peak_dev`` (``|k - 1|``), ``sign``
        (``attraction`` / ``exclusion`` / ``none``), ``flip``, ``r_attract_um``, ``r_exclude_um``;
        plus ``peak_z``, ``p_at_peak``, ``p_max``, ``T_max``, ``r_at_max_um`` when a null was run.

        **Censoring flags**, which a range claim is not auditable without: ``peak_index``,
        ``n_annuli``, ``peak_at_floor`` (the peak sits on the innermost annulus, so the sweep ran out
        of resolution before the curve turned over -- the reported distance is the binning choice),
        ``peak_at_ceiling`` (the outermost), and ``range_resolved`` (strictly interior, the only case
        in which ``peak_r_um`` locates a length). On real ligand-receptor panels 38-67 % of pairs come
        back with ``peak_at_floor`` -- see the note in the source.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:64, 0:64]
    >>> blob = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 8.0 ** 2))
    >>> res = spectrast.spatial.mark_cross_correlation(blob, blob, np.ones((64, 64), bool), n_bins=8)
    >>> spectrast.bivariate.interaction_distance(res)["sign"]
    'attraction'
    """
    names, pairs, r, g, z, p, p_max, T_max, r_max = _curves(res)
    i = int(pair)
    k = g[i]
    dev = np.abs(k - 1.0)
    if not np.isfinite(dev).any():
        raise ValueError(f"pair {names[i]!r} has no finite annulus; widen r_max or coarsen bin_um")
    j = int(np.nanargmax(dev))
    up = np.where(np.isfinite(k) & (k - 1.0 >= flip_min))[0]
    dn = np.where(np.isfinite(k) & (1.0 - k >= flip_min))[0]
    # IS THE PEAK AT THE EDGE OF WHAT WAS SWEPT? A peak on the FIRST annulus does not locate an
    # interaction distance -- it says the curve was still rising when the sweep ran out of
    # resolution, and the reported `peak_r_um` is then the binning choice rather than a length
    # (`autotune_bin`'s docstring names this failure mode directly). It has to be visible in the
    # output, because a censored pair otherwise looks exactly like a measured one and comes back
    # carrying a plausible number instead of a NaN.
    #
    # MEASURED, and the reason this field exists (2026-09-29): across six autotuned sections (mark
    # `m1_lr`), 38-67 % of the 24 ligand-receptor pairs peak
    # ON the innermost annulus, and the per-pair distance takes only 2-8 distinct values among 24
    # pairs (on dcis_idc, TWO). Any per-pair range claim on these platforms has to be read against
    # that censoring rate, and any comparison BETWEEN groups of pairs has to establish that the
    # groups are not merely differently censored.
    first = int(np.flatnonzero(np.isfinite(dev))[0])
    last = int(np.flatnonzero(np.isfinite(dev))[-1])
    out = dict(pair=names[i], a=pairs[i][0], b=pairs[i][1],
               peak_r_um=float(r[j]), peak_k=float(k[j]), peak_dev=float(dev[j]),
               peak_index=j, n_annuli=int(np.isfinite(dev).sum()),
               peak_at_floor=bool(j == first), peak_at_ceiling=bool(j == last),
               range_resolved=bool(first < j < last),
               sign=("attraction" if k[j] > 1 else "exclusion") if dev[j] >= flip_min else "none",
               flip=bool(len(up) and len(dn)),
               r_attract_um=float(r[up[np.argmax(k[up])]]) if len(up) else np.nan,
               r_exclude_um=float(r[dn[np.argmin(k[dn])]]) if len(dn) else np.nan)
    if z is not None:
        out["peak_z"] = float(z[i, j])
    if p is not None:
        out["p_at_peak"] = float(p[i, j])
    for key, v in (("p_max", p_max), ("T_max", T_max), ("r_at_max_um", r_max)):
        if v is not None:
            out[key] = float(v[i])
    return out


def pair_table(res, *, flip_min=FLIP_MIN, sort=True):
    """O2 — every pair ranked by peak signal: the discovery table.

    One row per pair, sorted strongest first. When a null was run the ranking uses the
    peak-selection-corrected max-statistic (``T_max``) rather than the raw deviation, and ``fdr``
    is Benjamini-Hochberg over pairs on ``p_max``; without a null it falls back to ``peak_dev`` and
    reports no p-value rather than inventing one.

    The ``sign`` and ``flip`` columns matter as much as the ranking. A pair with ``flip=True`` is
    co-localized at one distance and excluded at another — the structure a scalar co-expression
    score cannot represent.

    Parameters
    ----------
    res : dict
        A batched result from :func:`spectrast.gr.mark_cross_correlation_matrix` (a single-pair
        result gives a one-row table).
    flip_min : float, optional
        Passed to :func:`interaction_distance`. Default ``0.05``.
    sort : bool, optional
        Sort strongest first. ``False`` keeps the input pair order. Default ``True``.

    Returns
    -------
    pandas.DataFrame
        ``pair``, ``a``, ``b``, ``peak_r_um``, ``peak_k``, ``peak_dev``, ``sign``, ``flip``,
        ``r_attract_um``, ``r_exclude_um``, the censoring flags from
        :func:`interaction_distance` (``peak_at_floor``, ``peak_at_ceiling``, ``range_resolved``,
        ``peak_index``, ``n_annuli``); plus ``peak_z``, ``p_at_peak``, ``T_max``, ``p_max``,
        ``r_at_max_um``, ``fdr`` when a null was run.

    Warns
    -----
    UserWarning
        When a large share of pairs peak on the innermost annulus. A table in that state still
        ranks pairs honestly -- the ranking is on ``T_max`` or ``peak_dev``, not on distance -- but
        its ``peak_r_um`` column is partly a record of where the sweep stopped, and **a comparison
        of distances between two GROUPS of pairs is confounded by how differently they are
        censored**, not merely weakened by it.
    """
    import pandas as pd
    names = _curves(res)[0]
    rows = [interaction_distance(res, i, flip_min=flip_min) for i in range(len(names))]
    df = pd.DataFrame(rows)
    if len(df) and "peak_at_floor" in df:
        frac = float(df["peak_at_floor"].mean())
        if frac >= CENSOR_WARN:
            n_dist = int(df["peak_r_um"].nunique())
            warnings.warn(
                f"{df['peak_at_floor'].sum()} of {len(df)} pairs ({frac:.0%}) peak on the INNERMOST "
                f"annulus, and the distance column takes only {n_dist} distinct value(s) over "
                f"{len(df)} pairs. For those pairs `peak_r_um` is the binning choice, not an "
                f"interaction range: the curve was still rising where the sweep ran out of "
                f"resolution. The ranking (T_max / peak_dev) is unaffected. Do NOT compare distances "
                f"between groups of pairs from this table without first showing the groups are "
                f"censored at the same rate -- `peak_at_floor` is the column to test. Finer annuli "
                f"do not fix this below the tissue's own spacing.", stacklevel=2)
    
    if "p_max" in df:
        from .diff import bh_fdr
        df["fdr"] = bh_fdr(df["p_max"].to_numpy())
    if sort:
        key = "T_max" if "T_max" in df else "peak_dev"
        df = df.sort_values(key, ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def annulus_matrix(res, *, order=None, flip_min=FLIP_MIN):
    """O3 (data) — the pairs x distance matrix behind :func:`heatmap`.

    Parameters
    ----------
    res : dict
        A result from :func:`spectrast.gr.mark_cross_correlation` or
        :func:`spectrast.gr.mark_cross_correlation_matrix`.
    order : sequence of str or int, optional
        Rows to return, in this order: pair names as in ``res["pair_names"]``, or row indices. A
        subset is allowed. ``None`` (default) returns every pair in :func:`pair_table` order,
        strongest first.
    flip_min : float, optional
        Passed to :func:`pair_table` when ``order`` is ``None``. It changes only the ``sign`` and
        ``flip`` columns, not the ranking, so it does not change the output. Default ``0.05``.

    Returns
    -------
    tuple
        ``(names, r, M)`` — row labels (list of str), annulus centres in µm (coordinate units of
        the curve; ``(n_bins,)`` array), and the ``(n_rows, n_bins)`` matrix of ``k(r)``. Rows
        follow :func:`pair_table` order unless ``order`` is given.
    """
    names, _, r, g = _curves(res)[:4]
    if order is None:
        idx = [names.index(n) for n in pair_table(res, flip_min=flip_min)["pair"]]
    else:
        idx = [names.index(n) if isinstance(n, str) else int(n) for n in order]
    return [names[i] for i in idx], r, g[idx]


def heatmap(res, *, ax=None, order=None, flip_min=FLIP_MIN, vmax=None, top=None,
            cmap="RdBu_r", cbar=True):
    """O3 — pairs x distance, diverging about 1: red co-localization, blue exclusion.

    Centring the colour scale on 1 (not on the data mean) is what makes the picture readable: the
    statistic's null value is 1, so a symmetric scale about it puts attraction and exclusion on the
    same footing and leaves uncoupled pairs white. Rows come sorted by :func:`pair_table`, so bands
    of pairs sharing an interaction scale line up.

    Columns are one annulus wide and the ticks carry the true annulus centres. Annuli are usually
    spaced geometrically, so drawing them on a linear physical axis would put each label at the
    wrong place; equal-width columns with honest labels is the same convention as a scalogram.

    Parameters
    ----------
    res : dict
        A batched result from :func:`spectrast.gr.mark_cross_correlation_matrix`.
    ax : matplotlib.axes.Axes, optional
        Draw here; a new figure is made if omitted, sized to the number of pairs and annuli.
    order, flip_min
        As in :func:`annulus_matrix`.
    vmax : float, optional
        Colour limit as ``|k - 1|``: the colour scale runs from ``1 - vmax`` to ``1 + vmax``.
        Default: the 98th percentile of ``|k - 1|`` over the rows drawn (``1.0`` if no value is
        finite).
    top : int, optional
        Keep only the strongest ``top`` pairs. With ``order`` given, keeps the first ``top`` rows
        of ``order`` instead. ``None`` (default) draws every row.
    cmap : str or matplotlib.colors.Colormap, optional
        Diverging colour map. Default ``"RdBu_r"``.
    cbar : bool, optional
        Add a colour bar labelled ``k(r)``. Default ``True``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn into. Rows are pairs, columns are annuli, tick labels are annulus centres.
    """
    import matplotlib.pyplot as plt
    names, r, M = annulus_matrix(res, order=order, flip_min=flip_min)
    if top is not None:
        names, M = names[:int(top)], M[:int(top)]
    if vmax is None:
        d = np.abs(M - 1.0)
        vmax = float(np.nanpercentile(d, 98)) if np.isfinite(d).any() else 1.0
    vmax = max(vmax, 1e-6)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.09 * len(r) + 3.2, 0.16 * len(names) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=1 - vmax, vmax=1 + vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    step = max(1, len(r) // 6)
    ax.set_xticks(np.arange(0, len(r), step))
    ax.set_xticklabels([f"{v:.0f}" for v in r[::step]])
    ax.set_xlabel("separation r (µm)")
    if cbar:
        ax.figure.colorbar(im, ax=ax, label="k(r)", fraction=0.03, pad=0.02)
    return ax


def first_order_map(adata, res, pair=0, *, kind="contribution", **kw):
    """The map that goes with the curve — evaluated at the pair's OWN peak annulus.

    A curve says at what range a pair couples; this says where in the tissue that coupling is. The
    two belong together: identical curves can arise from signal concentrated at one margin or spread
    through the whole stroma, and only the map distinguishes them. The annulus is read from the
    result's own ``edges`` at the peak, so the map explains the number the table reports rather than
    some other distance chosen by hand.

    Parameters
    ----------
    adata : AnnData
        The AnnData the curve was computed on. :func:`spectrast.gr.mark_cross_map` writes the
        per-cell map to ``adata.obsm[key_added]`` (default ``"spectrast_mark_cross_map"``) and the
        pair name to ``adata.uns[key_added + "_pairs"]``.
    res : dict
        A result from :func:`spectrast.gr.mark_cross_correlation` or ``..._matrix``. The map is
        drawn for the annulus where ``|k - 1|`` peaks, between ``res["edges"][j]`` and
        ``res["edges"][j + 1]``.
    pair : int or str, optional
        Row index, or a pair name as it appears in the table (``res["pair_names"]``). Default ``0``.
    kind : {"contribution", "local"}, optional
        As in :func:`spectrast.gr.mark_cross_map`. ``"contribution"`` (default) sums to the
        curve's numerator; ``"local"`` is a local analogue of ``k`` (~1 = tissue-wide
        expectation).
    **kw
        Forwarded to :func:`spectrast.gr.mark_cross_map` (``bin_um``, ``mask_close_um``,
        ``a_type``/``b_type``, ``a_within``/``b_within``, ...). Pass the SAME ``bin_um`` and
        ``mask_close_um`` used for the curve, or the map will not decompose it. ``a_type``,
        ``b_type``, ``normalize`` and ``bin_um`` default to the values stored in ``res``;
        ``mask_close_um``, ``a_within``, ``b_within``, ``layer`` and ``basis`` are not stored there
        and must be passed again. Results computed with ``normalize="density_robust"`` are not
        supported by :func:`spectrast.gr.mark_cross_map`.

    Returns
    -------
    dict
        As :func:`spectrast.gr.mark_cross_map` for the one pair (``per_cell`` ``(n_obs, 1)``,
        ``fields`` ``(1, H, W)``, ``numerator``, ``r_lo``, ``r_hi``, ...), plus ``peak_r_um``,
        the centre of the annulus mapped.
    """
    from . import gr
    names, pairs, r, g = _curves(res)[:4]
    i = names.index(pair) if isinstance(pair, str) else int(pair)
    # prefer the ORIGINAL specs when the result carries them: `pairs` is the display form, in which
    # a cell-type indicator or a raw array has been stringified beyond recovery.
    specs = res.get("pair_specs")
    if specs is not None:
        pairs = list(specs)
    dev = np.abs(g[i] - 1.0)
    if not np.isfinite(dev).any():
        raise ValueError(f"pair {names[i]!r} has no finite annulus to explain")
    j = int(np.nanargmax(dev))
    edges = np.asarray(res["edges"], float)
    kw.setdefault("a_type", res.get("a_type", "expr"))
    kw.setdefault("b_type", res.get("b_type", "expr"))
    kw.setdefault("normalize", res.get("normalize"))
    kw.setdefault("bin_um", res.get("bin_um"))
    out = gr.mark_cross_map(adata, [pairs[i]], r_lo=edges[j], r_hi=edges[j + 1], kind=kind, **kw)
    out["peak_r_um"] = float(r[j])
    return out


def directed_pair_table(adata, pairs, by, *, values=None, min_cells=200, max_types=6,
                        include_self=True, **kw):
    """O4 directed — every SENDER -> RECEIVER ordered cell-type combination.

    :func:`stratified_pair_table` confines one side and asks "which population carries this pair".
    That is weaker than the question a cell-cell-interaction reader actually wants, which is
    directional: does the signal run from fibroblasts to tumour, or the reverse? Confining BOTH
    sides answers it, and the statistic is genuinely asymmetric — ``A -> B`` and ``B -> A`` weight
    different cells as senders, so they are different numbers rather than a relabelling.

    The tissue support stays the whole section throughout, so every cell arrives on a common scale
    and the resulting matrix can be compared cell by cell.

    Cost grows as the SQUARE of the number of types, so the level list is capped: types are ranked by
    abundance, those below ``min_cells`` dropped, and the top ``max_types`` kept. What was dropped is
    reported in the returned frame's ``attrs`` rather than silently discarded.

    Parameters
    ----------
    adata : AnnData
        Coordinates, marks and the ``by`` column.
    pairs : sequence of tuple
        ``(a, b)`` mark specs, as in :func:`spectrast.gr.mark_cross_correlation_matrix`. ``a`` is
        the sender side, ``b`` the receiver side.
    by : str
        ``adata.obs`` column of cell-type labels.
    values : sequence, optional
        Use exactly these levels, bypassing the abundance ranking and ``min_cells``.
    min_cells : int, optional
        Levels with fewer cells are dropped. Default ``200``.
    max_types : int, optional
        Keep at most this many levels. Default ``6`` (36 ordered combinations).
    include_self : bool, optional
        Keep ``sender == receiver`` combinations, which are the within-population interactions.
        Default ``True``.
    **kw
        Forwarded to :func:`spectrast.gr.mark_cross_correlation_matrix` (``bin_um``, ``radii``,
        ``n_perm``, ``mask_close_um``, ``a_type``/``b_type``, ...). ``a_within`` and ``b_within``
        are set here and cannot be passed. Each combination overwrites
        ``adata.uns[key_added]``, so only the last one is left there.

    Returns
    -------
    pandas.DataFrame
        :func:`pair_table` columns plus ``sender``, ``receiver``, ``n_sender``, ``n_receiver``
        (cells in each level). One block of rows per combination, each sorted strongest first.
        ``attrs["levels"]`` lists the levels used, in abundance order, and
        ``attrs["dropped_levels"]`` the levels left out.
    """
    import pandas as pd
    from . import gr
    if by not in adata.obs:
        raise KeyError(f"adata.obs[{by!r}] not found")
    col = adata.obs[by].astype(str).to_numpy()
    counts = pd.Series(col).value_counts()
    if values is not None:
        levels = [str(v) for v in values]
    else:
        keep = counts[counts >= int(min_cells)]
        levels = list(keep.index[:int(max_types)])
    if not levels:
        raise ValueError(f"no level of {by!r} has >= {min_cells} cells")
    dropped = [c for c in counts.index if c not in levels]

    out = []
    for s in levels:
        for r in levels:
            if s == r and not include_self:
                continue
            res = gr.mark_cross_correlation_matrix(adata, pairs, a_within=(by, s),
                                                   b_within=(by, r), **kw)
            df = pair_table(res)
            df.insert(0, "receiver", r)
            df.insert(0, "sender", s)
            df["n_sender"] = int(counts[s])
            df["n_receiver"] = int(counts[r])
            out.append(df)
    res = pd.concat(out, ignore_index=True)
    res.attrs["levels"] = levels
    res.attrs["dropped_levels"] = dropped
    return res


def directed_matrix(table, pair=None, value="peak_k"):
    """The sender x receiver matrix for one pair, from :func:`directed_pair_table`.

    Parameters
    ----------
    table : pandas.DataFrame
        The table returned by :func:`directed_pair_table`.
    pair : str, optional
        Pair name, as in the ``pair`` column. ``None`` (default) uses every row. If the table
        holds more than one pair, each cell then takes the first row for that sender and
        receiver, which is the top-ranked pair of that combination, so pass ``pair`` unless the
        table holds a single pair.
    value : str, optional
        Numeric column to place in the matrix, e.g. ``"peak_k"``, ``"peak_dev"``, ``"T_max"`` or
        ``"peak_r_um"``. Default ``"peak_k"``.

    Returns
    -------
    tuple
        ``(senders, receivers, M)`` with ``M[i, j]`` the value for sender ``i`` -> receiver ``j``.
        ``senders`` and ``receivers`` are lists of level names in the order of
        ``table.attrs["levels"]``; ``M`` is a float array of shape
        ``(n_senders, n_receivers)``, NaN where a combination was not computed.
    """
    import pandas as pd
    t = table if pair is None else table[table["pair"] == pair]
    if not len(t):
        raise ValueError(f"no rows for pair {pair!r}")
    piv = t.pivot_table(index="sender", columns="receiver", values=value, aggfunc="first")
    order = [c for c in table.attrs.get("levels", list(piv.index)) if c in piv.index]
    piv = piv.reindex(index=order, columns=[c for c in order if c in piv.columns])
    return list(piv.index), list(piv.columns), piv.to_numpy(float)


def stratified_pair_table(adata, pairs, by, *, values=None, side="b", **kw):
    """O4 — the ranked table recomputed per cell-type stratum, so interactions get attributed.

    One side's mark is confined to each population in turn while the tissue support stays the whole
    section, so the strata are on a common footing and a pair's rank can be compared across them.
    That answers the question a section-wide curve cannot: not "is this pair co-organized" but "which
    population carries it".

    Parameters
    ----------
    adata : AnnData
        Coordinates, marks and the ``by`` column.
    pairs : sequence of tuple
        As :func:`spectrast.gr.mark_cross_correlation_matrix`.
    by : str
        ``adata.obs`` column defining the strata.
    values : sequence, optional
        Strata to run. Default: every level with at least one cell.
    side : {"b", "a"}, optional
        Which side to confine. Default ``"b"`` (the receiver).
    **kw
        Forwarded to :func:`spectrast.gr.mark_cross_correlation_matrix`.

    Returns
    -------
    pandas.DataFrame
        :func:`pair_table` columns plus ``stratum`` and ``n_cells``, sorted within stratum.
    """
    import pandas as pd
    from . import gr
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    if by not in adata.obs:
        raise KeyError(f"adata.obs[{by!r}] not found")
    col = adata.obs[by].astype(str).to_numpy()
    levels = [str(v) for v in values] if values is not None else sorted(set(col))
    out = []
    for v in levels:
        n = int((col == v).sum())
        if not n:
            continue
        res = gr.mark_cross_correlation_matrix(adata, pairs, **{f"{side}_within": (by, v)}, **kw)
        df = pair_table(res)
        df.insert(0, "stratum", v)
        df["n_cells"] = n
        out.append(df)
    if not out:
        raise ValueError(f"no cells in any level of {by!r}")
    return pd.concat(out, ignore_index=True)

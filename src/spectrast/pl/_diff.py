"""Two conditions, many sections: plots of :func:`spectrast.diff.differential`
results and of :func:`spectrast.gr.fingerprint_by` tables.

The table plots (:func:`volcano`, :func:`effect_plane`) take the result table as returned. The curve
plots (:func:`diff_curves`, :func:`unit_curves`, :func:`diff_heatmap`) take the dict that
``differential(..., return_curves=True)`` returns beside it. :func:`diff_fingerprint` and
:func:`scale_shift` take one result per tested axis, as a dict keyed by axis. :func:`neighborhood_maps`
computes one per-cell field (the neighbourhood mean of the standardised feature) on the differential's
lattice.
"""
from __future__ import annotations

import numpy as np

from . import _utils as U

_TABLE_HINT = "spectrast.diff.differential(...)"
_CURVES_HINT = "spectrast.diff.differential(..., return_curves=True)"
_CURVE_KEYS = ("radii_um", "ids", "meanA", "meanB", "diff", "sd", "band", "exceed")
_NO_UNITS = ("{what}: this result has no per-unit curves. It came from the matched (cell-repartition) "
             "null, which has no units; per-unit curves exist under the replicate null. Run "
             "spectrast.diff.differential with >= 2 sections per arm (null='replicate') and "
             "return_curves=True.")

#: y-axis label of each differential axis's organization curve.
_AXIS_LABEL = {"moran": "Moran's I", "geary": "1 − Geary's C", "focality": "focality",
               "general_g": "General G", "losh": "LOSH concentration", "mean": "mean level",
               "coloc": "co-expression", "cross": "mark correlation − 1"}
_X_LABEL = {"dI_at_max": "difference at the tested radius (A − B)",
            "dI_raw_at_peak": "largest raw difference (A − B)",
            "delta_scale_um": "Δ scale (µm, A − B)",
            "T_max": "T_max (studentised maximum)"}
_AMPLITUDE_COLS = ("scale_peak_moran", "amplitude", "peak_moran")


# ---------------------------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------------------------
def _check_table(df, cols, what):
    """Validate a differential result table; return its feature-name array."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{what} takes the table returned by {_TABLE_HINT}; got {type(df).__name__}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what}: column(s) {missing} not in the table; pass the table returned by "
                       f"{_TABLE_HINT}")
    for c in ("gene", "pair"):
        if c in df.columns:
            return np.asarray(df[c], dtype=str)
    return np.asarray(df.index, dtype=str)


def _check_curves(curves, what):
    if not isinstance(curves, dict) or any(k not in curves for k in _CURVE_KEYS):
        got = type(curves).__name__
        raise TypeError(f"{what} takes the curves dict returned by {_CURVES_HINT} (the last element "
                        f"of its return value); got {got}")
    return curves


def _feature_idx(curves, wanted):
    ids = list(map(str, curves["ids"]))
    try:
        return U.index_of(ids, wanted)
    except KeyError as e:
        raise KeyError(f"{e.args[0]} among the {len(ids)} features of this result "
                       f"(first: {ids[:5]})") from None


def _arm_labels(curves, labels):
    if labels is not None:
        labels = list(labels)
        if len(labels) != 2:
            raise ValueError(f"labels must name the two arms (A, B); got {labels}")
        return str(labels[0]), str(labels[1])
    lab = curves.get("labels") if curves is not None else None
    return (str(lab[0]), str(lab[1])) if lab is not None else ("A", "B")


_log_axis = U.radius_axis           # shared: ticks at 1, 2, 5 per decade


def _legend_above(ax, ncol=2, **kw):
    """Legend in a band above the Axes, so it never covers data."""
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol, frameon=False,
                     fontsize=7, borderaxespad=0.0, handletextpad=0.4, columnspacing=1.0, **kw)


_place_labels = U.place_labels      # shared: see _utils.place_labels


def _top_called(names, q, effect, called, n):
    """Indices of up to ``n`` called features, by FDR then by |effect| (largest first)."""
    idx = np.flatnonzero(called)
    if n is None or n <= 0 or idx.size == 0:
        return idx[:0]
    order = np.lexsort((-np.abs(np.nan_to_num(effect[idx])), q[idx]))
    return idx[order][:n]


# ---------------------------------------------------------------------------------------------
# 1. volcano
# ---------------------------------------------------------------------------------------------
def volcano(df, *, x="dI_at_max", fdr=0.05, n_label=10, ax=None, show=None, save=None):
    """Signed effect against −log10 FDR for every feature of a differential.

    Called features (``fdr < fdr``) are highlighted and the top ``n_label`` of them named (smallest
    FDR first, then largest ``|x|``); the dashed line is the FDR threshold.

    At the permutation floor many hits share one p-value, and so one FDR: their vertical positions
    are then equal by construction, and the effect axis is what separates them.

    Parameters
    ----------
    df : pandas.DataFrame
        The table returned by :func:`spectrast.diff.differential` (or
        :func:`~spectrast.diff.differential_cohort`).
    x : str, optional
        Column on the horizontal axis. Default ``"dI_at_max"``, the signed difference (A − B) at
        the radius where the studentised difference is largest.
    fdr : float, optional
        Threshold for a feature to count as called. Default ``0.05``.
    n_label : int, optional
        How many called features to name. Default ``10``.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        ``show=True`` shows the figure and returns ``None``; ``save="path.png"`` writes it.

    Returns
    -------
    matplotlib.axes.Axes
        Not-called features are the first scatter collection, called ones the second. Features
        whose ``x`` is undefined (NaN) are not drawn; their count is given in the legend.

    Examples
    --------
    >>> df = spectrast.diff.differential(A, B, features=genes)        # doctest: +SKIP
    >>> spectrast.pl.volcano(df, fdr=0.05, n_label=10)                 # doctest: +SKIP
    """
    names = _check_table(df, [x, "fdr"], "volcano")
    xv = df[x].to_numpy(float)
    q = df["fdr"].to_numpy(float)
    y = -np.log10(np.clip(q, 1e-300, 1.0))
    ok = np.isfinite(xv) & np.isfinite(y)
    called = ok & (q < fdr)
    rest = ok & ~called
    fig, ax = U.new_axes(ax)
    ax.scatter(xv[rest], y[rest], s=9, c=U.MUTED, linewidths=0, zorder=2,
               label=f"not called (n = {int(rest.sum())})")
    ax.scatter(xv[called], y[called], s=16, c=U.HIGHLIGHT, linewidths=0, zorder=3,
               label=f"FDR < {fdr:g} (n = {int(called.sum())})")
    if (~ok).any():
        ax.plot([], [], ls="none", label=f"{x} undefined, not drawn (n = {int((~ok).sum())})")
    ax.axhline(-np.log10(fdr), ls="--", lw=0.8, c="0.3", zorder=1)
    ax.axvline(0.0, lw=0.5, c="0.7", zorder=1)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(_X_LABEL.get(x, x))
    ax.set_ylabel("−log10 FDR")
    _legend_above(ax)
    top = _top_called(names, q, xv, called, n_label)
    _place_labels(ax, xv[top], y[top], names[top])
    return U.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 2. diff_curves
# ---------------------------------------------------------------------------------------------
def diff_curves(curves, feature, *, labels=None, show_units=False, show=None, save=None):
    """One feature's two arm-mean curves and their difference against the envelope.

    Top: the mean organization curve of each arm (optionally every unit's curve, faint). Bottom:
    the tested difference (A − B) with the simultaneous envelope ``±T_alpha·sd(r)`` shaded; radii
    where the difference leaves the envelope are marked. A feature is significant at
    ``envelope_alpha`` exactly when it leaves the band somewhere, so the marked radii say where the
    arms differ without changing whether they do.

    Parameters
    ----------
    curves : dict
        Returned by :func:`spectrast.diff.differential` with ``return_curves=True``.
    feature : str
        A gene name (one of ``curves["ids"]``).
    labels : (str, str), optional
        Arm names. Default ``curves["labels"]``, else ``("A", "B")``.
    show_units : bool, optional
        Also draw every unit's curve, faint. Needs the replicate null. Default ``False``.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.figure.Figure
        Two Axes, top then bottom.

    Examples
    --------
    >>> df, curves = spectrast.diff.differential(A, B, features=genes,
    ...                                          return_curves=True)  # doctest: +SKIP
    >>> spectrast.pl.diff_curves(curves, "Trem2", show_units=True)     # doctest: +SKIP
    """
    c = _check_curves(curves, "diff_curves")
    g = int(_feature_idx(c, feature)[0])
    la, lb = _arm_labels(c, labels)
    r = np.asarray(c["radii_um"], float)
    if show_units and (c.get("unit_curves_A") is None or c.get("unit_curves_B") is None):
        raise ValueError(_NO_UNITS.format(what="diff_curves(show_units=True)"))
    fig = U.new_figure((4.4, 4.8))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=[1.15, 1.0])
    if show_units:
        for stack, col in ((c["unit_curves_A"], U.ARM_COLORS[0]), (c["unit_curves_B"], U.ARM_COLORS[1])):
            for u in range(stack.shape[0]):
                ax1.plot(r, stack[u, :, g], c=col, lw=0.6, alpha=0.3, zorder=1)
    ax1.plot(r, c["meanA"][:, g], c=U.ARM_COLORS[0], lw=1.8, marker="o", ms=3, label=la, zorder=3)
    ax1.plot(r, c["meanB"][:, g], c=U.ARM_COLORS[1], lw=1.8, marker="o", ms=3, label=lb, zorder=3)
    ax1.set_ylabel(_AXIS_LABEL.get(c.get("axis"), str(c.get("axis", "organization"))))
    ax1.legend(frameon=False, fontsize=7)
    d = np.asarray(c["diff"][:, g], float)
    b = np.asarray(c["band"][:, g], float)
    ex = np.asarray(c["exceed"][:, g], bool)
    alpha = c.get("envelope_alpha", 0.05)
    ax2.fill_between(r, -b, b, color="0.86", lw=0, zorder=1,
                     label=f"simultaneous envelope (α = {alpha:g})")
    ax2.axhline(0.0, c="0.5", lw=0.6, zorder=1)
    ax2.plot(r, d, c="k", lw=1.4, marker="o", ms=2.5, zorder=2, label=f"{la} − {lb}")
    ax2.scatter(r[ex], d[ex], c=U.HIGHLIGHT, s=24, zorder=3, linewidths=0,
                label=f"outside the envelope ({int(ex.sum())} of {len(r)} radii)")
    _log_axis(ax2, r)
    ax2.set_ylabel("difference")
    ax2.legend(frameon=False, fontsize=7)
    fig.suptitle(str(feature))
    return U.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# 3. unit_curves
# ---------------------------------------------------------------------------------------------
def unit_curves(curves, feature, *, labels=None, ax=None, show=None, save=None):
    """Every unit's organization curve for one feature, coloured by arm, with the arm means bold.

    The units are what the replicate null relabels (sections, or donors when ``units_A`` /
    ``units_B`` were given; under the paired null only the units present in both arms).

    Parameters
    ----------
    curves : dict
        Returned by :func:`spectrast.diff.differential` with ``return_curves=True``, under the
        replicate null.
    feature : str
        A feature name in ``curves["ids"]``.
    labels : (str, str), optional
        Arm names. Default ``curves["labels"]``, else ``("A", "B")``.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.axes.Axes
        Lines in order: arm A's units, arm A's mean, arm B's units, arm B's mean.

    Examples
    --------
    >>> df, curves = spectrast.diff.differential_cohort(
    ...     adata, "amyloid", ["amyloid-"], ["amyloid+"], sample_key="section_id",
    ...     features=genes, return_curves=True)                         # doctest: +SKIP
    >>> spectrast.pl.unit_curves(curves, "Trem2")                      # doctest: +SKIP
    """
    c = _check_curves(curves, "unit_curves")
    g = int(_feature_idx(c, feature)[0])
    if c.get("unit_curves_A") is None or c.get("unit_curves_B") is None:
        raise ValueError(_NO_UNITS.format(what="unit_curves"))
    la, lb = _arm_labels(c, labels)
    r = np.asarray(c["radii_um"], float)
    fig, ax = U.new_axes(ax, figsize=(4.4, 3.4))
    for stack, mean, col, lab in ((c["unit_curves_A"], c["meanA"], U.ARM_COLORS[0], la),
                                  (c["unit_curves_B"], c["meanB"], U.ARM_COLORS[1], lb)):
        for u in range(stack.shape[0]):
            ax.plot(r, stack[u, :, g], c=col, lw=0.7, alpha=0.45, zorder=1)
        ax.plot(r, mean[:, g], c=col, lw=2.2, zorder=3, label=f"{lab} (n = {stack.shape[0]})")
    _log_axis(ax, r)
    ax.set_ylabel(_AXIS_LABEL.get(c.get("axis"), str(c.get("axis", "organization"))))
    ax.set_title(str(feature))
    ax.legend(frameon=False, fontsize=7)
    return U.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 4. diff_heatmap
# ---------------------------------------------------------------------------------------------
def _radius_columns(ax, radii_um):
    """Label heatmap columns (one per radius, equal width) in µm. When the radii span enough of a
    log scale, ticks sit at 1, 2 and 5 per decade, placed by log interpolation between columns (the
    columns of a log-spaced sweep are a log axis); otherwise each column carries its own radius."""
    r = np.asarray(radii_um, float)
    pos = np.arange(len(r))
    nice = []
    if len(r) > 1 and np.all(np.diff(r) > 0) and r[0] > 0:
        for e in range(int(np.floor(np.log10(r[0]))), int(np.ceil(np.log10(r[-1]))) + 1):
            nice += [m * 10.0 ** e for m in (1, 2, 5) if r[0] <= m * 10.0 ** e <= r[-1]]
    if len(nice) >= 3:
        ax.set_xticks(np.interp(np.log(nice), np.log(r), pos))
        ax.set_xticklabels([f"{v:g}" for v in nice], fontsize=7)
    else:
        step = max(1, int(np.ceil(len(r) / 10)))
        ax.set_xticks(pos[::step])
        ax.set_xticklabels([f"{v:.0f}" for v in r[::step]], fontsize=7)
    ax.set_xlabel("radius (µm)")


def diff_heatmap(curves, *, df, fdr=0.05, n_label=None, ax=None, show=None, save=None):
    """Called features × radius: the raw difference between the two arm-mean curves.

    One row per feature called at ``fdr``; each cell is ``curves["diff"]``, the arm A mean curve
    minus the arm B mean curve at that radius, on a diverging colour scale symmetric about zero
    (clipped at the 98th percentile of ``|diff|`` over the drawn rows). Rows are split into two
    blocks by the sign of the change at the tested radius (``dI_at_max``): the statistic higher in A
    on top, lower in A below, with a white rule between them. Within a block, rows run from the
    finest tested radius (``r_at_max_um``) to the coarsest, then by ``|dI_at_max|`` (largest first).
    A dot marks each row's tested radius.

    The cells are raw differences, not the studentised ones the test maximises: the tested radius is
    where ``|diff| / sd`` peaks, and because the null spread falls with radius it often sits coarser
    than the largest raw difference. Read the dot as where the test found the change, not as the
    scale of the change.

    Parameters
    ----------
    curves : dict
        Returned by :func:`spectrast.diff.differential` with ``return_curves=True``.
    df : pandas.DataFrame
        The result table of the same call (needs ``fdr``, ``dI_at_max`` and ``r_at_max_um``).
    fdr : float, optional
        Threshold for a feature to be drawn. Default ``0.05``.
    n_label : int, optional
        Name every row when at most ``n_label`` rows are drawn, and every other row above that.
        Default ``None``, meaning 40.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.axes.Axes
        The image is ``ax.images[0]`` (rows = features in the plotted order, columns = radii, values
        = ``curves["diff"]``); the dots are ``ax.collections[0]``, at (column, row). The colour bar
        label names the arms from ``curves["labels"]`` (``A``, ``B`` when the curves carry none).

    Examples
    --------
    >>> df, curves = spectrast.diff.differential(A, B, features=genes,
    ...                                          return_curves=True)  # doctest: +SKIP
    >>> spectrast.pl.diff_heatmap(curves, df=df, fdr=0.05)            # doctest: +SKIP
    """
    c = _check_curves(curves, "diff_heatmap")
    names = _check_table(df, ["fdr", "dI_at_max", "r_at_max_um"], "diff_heatmap")
    q = df["fdr"].to_numpy(float)
    called = np.nan_to_num(q, nan=1.0) < fdr
    if not called.any():
        raise ValueError(f"diff_heatmap: no feature has FDR < {fdr:g} in df; pass a larger fdr=")
    dI = df["dI_at_max"].to_numpy(float)[called]
    r_t = df["r_at_max_um"].to_numpy(float)[called]
    hit = names[called]
    up = dI > 0
    rank = np.lexsort((-np.abs(np.nan_to_num(dI)), r_t, ~up))      # higher in A first, then by radius
    hit, dI, r_t, up = hit[rank], dI[rank], r_t[rank], up[rank]
    idx = _feature_idx(c, list(hit))
    Z = np.asarray(c["diff"], float)[:, idx].T                      # (F, n_radii)
    r = np.asarray(c["radii_um"], float)
    col = np.array([int(np.argmin(np.abs(np.log(r) - np.log(x)))) if np.isfinite(x) and x > 0 else -1
                    for x in r_t])
    la, lb = _arm_labels(c, None)
    F, nR = Z.shape
    n_up = int(up.sum())
    vmin, vmax = U.robust_limits(Z, pct=98.0, symmetric=True)
    fig, ax = U.new_axes(ax, figsize=(max(3.6, 0.18 * nR + 2.4), min(max(2.4, 0.14 * F + 1.4), 16.0)))
    import matplotlib
    cmap = matplotlib.colormaps[U.DIVERGING].copy()
    cmap.set_bad("0.88")
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ok = col >= 0
    ax.scatter(col[ok], np.arange(F)[ok], s=9, c="k", linewidths=0, zorder=3)
    if 0 < n_up < F:
        ax.axhline(n_up - 0.5, color="white", lw=2.5, zorder=2)
    _radius_columns(ax, r)
    limit = 40 if n_label is None else int(n_label)
    rows = np.arange(F) if F <= limit else np.arange(F)[::2]
    ax.set_yticks(rows)
    ax.set_yticklabels(hit[rows], fontsize=7)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    what = _AXIS_LABEL.get(c.get("axis"), str(c.get("axis", "organization")))
    cb.set_label(f"difference in {what} ({la} − {lb})", fontsize=8)
    ax.set_title(f"top: higher in {la} (n = {n_up}); bottom: lower (n = {F - n_up})\n"
                 f"dot: tested radius; FDR < {fdr:g}", fontsize=8)
    return U.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 4b. diff_fingerprint
# ---------------------------------------------------------------------------------------------
#: Column titles of :func:`diff_fingerprint` and panel titles of :func:`scale_shift`.
_AXIS_TITLE = {"moran": "Moran's I", "geary": "Geary (1 − C)", "losh": "LOSH", "focality": "Focality",
               "general_g": "General G", "mean": "mean level", "coloc": "co-expression",
               "cross": "mark correlation"}
_MANY_ROWS = 40                     # above this, row names thin out (heatmap) or alternate sides


def _results_tables(results, what):
    """``results`` as ``[(axis, table, curves or None)]``; values may be a table or ``(df, curves)``."""
    if not isinstance(results, dict) or not results:
        raise TypeError(f"{what} takes a dict mapping each axis name to the result of "
                        f"{_TABLE_HINT} on that axis, e.g. {{'moran': df_moran, 'geary': df_geary}}; "
                        f"got {type(results).__name__}")
    out = []
    for a, v in results.items():
        if isinstance(v, tuple):
            out.append((str(a), v[0], v[-1] if len(v) > 1 else None))
        else:
            out.append((str(a), v, None))
    return out


def diff_fingerprint(results, *, fdr=0.05, fdr_scope="pooled", cluster=True, ax=None, show=None,
                     save=None):
    """Genes × axes: how each called gene's organization changes on every tested statistic.

    Each cell is the signed studentised change, ``T_max`` (the largest studentised difference over
    radii, the tested statistic) times the sign of ``dI_at_max`` (the difference A − B at that
    radius). Every column rises with clustering: Moran's I and LOSH concentration and focality do by
    construction, and the Geary axis of :func:`spectrast.diff.differential` already tests 1 − C, so
    its column is titled "Geary (1 − C)" and its sign is used as returned (a positive value is more
    clustering in arm A, i.e. a lower Geary's C). Red cells: the statistic is higher in arm A; blue:
    lower. The colour scale is symmetric about zero, clipped at the 98th percentile of ``|value|``.

    Rows are the genes called on at least one axis; a small square marks each called cell (black on
    light cells, white on dark ones). A gene missing from a table is a grey cell. With more than 40
    rows the gene names alternate between the left and right sides of the rows. The title counts
    the genes called on exactly 1, 2, ... axes.

    Parameters
    ----------
    results : dict
        Axis name → the table :func:`spectrast.diff.differential` returned on that axis, e.g.
        ``{"moran": df_m, "geary": df_g, "losh": df_l, "focality": df_f}``. A ``(df, curves)``
        pair from ``return_curves=True`` is accepted in place of a table. Columns follow the
        dict's order.
    fdr : float, optional
        Threshold for a cell to count as called. Default ``0.05``.
    fdr_scope : {"pooled", "per_axis"}, optional
        ``"pooled"`` (default) recomputes Benjamini–Hochberg over every gene × axis from the
        tables' ``p`` (:func:`spectrast.diff.bh_fdr`), so the threshold controls the false
        discovery rate of the whole matrix; ``"per_axis"`` uses each table's own ``fdr``, which
        controls it within each axis only.
    cluster : bool, optional
        Order rows by Ward clustering (scipy) of the signed matrix. Default ``True``. ``False``
        orders by the number of axes called (most first), then by the largest ``|value|`` among
        the called cells.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.axes.Axes
        The image is ``ax.images[0]`` (rows = genes in the plotted order, columns = axes). Called
        cells are ``ax.collections[0]`` (black squares) and ``ax.collections[1]`` (white squares),
        at (column, row). With more than 40 rows the right-hand names are on ``ax.child_axes[0]``.

    Examples
    --------
    >>> res = {a: spectrast.diff.differential(A, B, axis=a, features=genes)
    ...        for a in ("moran", "geary", "losh", "focality")}       # doctest: +SKIP
    >>> spectrast.pl.diff_fingerprint(res, fdr=0.05)                  # doctest: +SKIP
    """
    from .. import diff as _d
    if fdr_scope not in ("pooled", "per_axis"):
        raise ValueError(f"fdr_scope must be 'pooled' or 'per_axis'; got {fdr_scope!r}")
    pooled = fdr_scope == "pooled"
    entries = _results_tables(results, "diff_fingerprint")
    need = ["T_max", "dI_at_max", "p" if pooled else "fdr"]
    tabs = [(a, _check_table(t, need, f"diff_fingerprint ({a})"), t) for a, t, _ in entries]
    genes = list(dict.fromkeys(n for _, names, _ in tabs for n in names))
    pos = {g: i for i, g in enumerate(genes)}
    G, K = len(genes), len(tabs)
    M = np.full((G, K), np.nan)
    P = np.full((G, K), np.nan)
    present = np.zeros((G, K), bool)
    for j, (a, names, t) in enumerate(tabs):
        ii = np.array([pos[n] for n in names], int)
        sgn = np.sign(np.nan_to_num(t["dI_at_max"].to_numpy(float)))
        M[ii, j] = t["T_max"].to_numpy(float) * sgn
        P[ii, j] = t["p" if pooled else "fdr"].to_numpy(float)
        present[ii, j] = True
    Q = np.full((G, K), np.nan)
    Q[present] = _d.bh_fdr(P[present]) if pooled else P[present]
    called = present & (np.nan_to_num(Q, nan=1.0) < fdr)
    if not called.any():
        raise ValueError(f"diff_fingerprint: no feature is called at FDR < {fdr:g} "
                         f"({'pooled over genes × axes' if pooled else 'per axis'}) on any axis")
    sel = np.flatnonzero(called.any(1))
    ncall = called[sel].sum(1)
    strength = np.max(np.where(called[sel], np.abs(np.nan_to_num(M[sel])), -np.inf), axis=1)
    sel = sel[np.lexsort((np.arange(len(sel)), -strength, -ncall))]
    if cluster and len(sel) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage
        sel = sel[leaves_list(linkage(np.nan_to_num(M[sel]), method="ward"))]
    Z, C, names = M[sel], called[sel], np.asarray(genes, dtype=object)[sel]
    F = len(sel)
    counts = np.bincount(called[called.any(1)].sum(1), minlength=K + 1)[1:]
    vmin, vmax = U.robust_limits(Z, pct=98.0, symmetric=True)
    fig, ax = U.new_axes(ax, figsize=(max(3.0, 0.55 * K + 2.2), min(max(3.2, 0.12 * F + 2.4), 18.0)))
    import matplotlib
    cmap = matplotlib.colormaps[U.DIVERGING].copy()
    cmap.set_bad("0.88")
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    rr, cc = np.nonzero(C)
    dark = np.abs(Z[rr, cc]) > 0.6 * vmax
    ax.scatter(cc[~dark], rr[~dark], marker="s", s=10, c="k", linewidths=0, zorder=3)
    ax.scatter(cc[dark], rr[dark], marker="s", s=10, c="white", linewidths=0, zorder=3)
    ax.set_xticks(np.arange(K + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(F + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([_AXIS_TITLE.get(a, a) for a, _, _ in tabs], rotation=90, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if F <= _MANY_ROWS:
        ax.set_yticks(np.arange(F))
        ax.set_yticklabels(names, fontsize=7)
    else:
        ev, od = np.arange(0, F, 2), np.arange(1, F, 2)
        ax.set_yticks(ev)
        ax.set_yticklabels(names[ev], fontsize=7)
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(od)
        sec.set_yticklabels(names[od], fontsize=7)
        sec.tick_params(length=0)
        for sp in sec.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.8, pad=0.02, aspect=25)
    cb.set_label("T_max × sign of the change (A − B)", fontsize=8)
    ks = ", ".join(str(k) for k in range(1, K + 1))
    ax.set_title(f"genes called on {ks} axes: {', '.join(str(int(n)) for n in counts)}\n"
                 f"FDR < {fdr:g}, {'pooled over genes × axes' if pooled else 'per axis'}; "
                 f"square: called", fontsize=8)
    return U.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 4c. scale_shift
# ---------------------------------------------------------------------------------------------
_RISE, _FALL = "#B2182B", "#2166AC"     # the ends of the diverging map: statistic higher / lower in A


def scale_shift(results, *, fdr=0.05, show=None, save=None):
    """Each gene's characteristic scale in one arm against the other, one panel per axis.

    x is the characteristic scale of arm B's mean curve and y that of arm A's
    (``curves["scale_meanB_um"]`` and ``curves["scale_meanA_um"]``: the half-max of each arm's mean
    organization curve, whose difference is the table's ``delta_scale_um``). Which arm is A and which
    is B is fixed by the call, and the axis labels name them from ``curves["labels"]``; put the
    reference (control) arm in B and the treatment in A, so a point above the dashed identity line is
    a pattern that is coarser under treatment. Both axes are logarithmic and share one range across
    panels.

    Genes called at ``fdr`` on that axis are coloured red when the statistic is higher in arm A at
    the tested radius (``dI_at_max > 0``) and blue when it is lower; filled when the shift is
    resolvable (``delta_scale_resolvable``: larger than the mean of the two arms' between-unit scale
    spreads), open otherwise. Genes not called are grey. A gene with no defined scale in one or both
    arms (the arm's curve never falls to half its own range inside the radius window) cannot be
    drawn; each panel's title counts them, and how many of them are called.

    The scale is descriptive: the test compares whole curves, not the scale, so a called gene can sit
    on the identity line and a large shift can belong to a gene that is not called.

    Parameters
    ----------
    results : dict
        Axis name → the ``(df, curves)`` pair returned by :func:`spectrast.diff.differential` with
        ``return_curves=True`` on that axis, e.g. ``{"moran": (df_m, c_m), "geary": (df_g, c_g)}``.
        One panel per entry, in the dict's order.
    fdr : float, optional
        Threshold for a gene to count as called. Default ``0.05``.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.figure.Figure
        One Axes per axis. In each, the scatter collections are, in order: not called, called with
        the statistic higher in A, called with it lower.

    Examples
    --------
    >>> res = {a: spectrast.diff.differential(treated, control, axis=a, features=genes,
    ...                                       return_curves=True)
    ...        for a in ("moran", "geary", "losh", "focality")}       # doctest: +SKIP
    >>> spectrast.pl.scale_shift(res)                                 # doctest: +SKIP
    """
    entries = _results_tables(results, "scale_shift")
    panels = []
    for a, t, c in entries:
        if c is None or not isinstance(c, dict):
            raise TypeError(f"scale_shift takes, for each axis, the (df, curves) pair returned by "
                            f"{_CURVES_HINT}; axis {a!r} has no curves")
        _check_curves(c, f"scale_shift ({a})")
        names = _check_table(t, ["fdr", "dI_at_max", "delta_scale_resolvable"], f"scale_shift ({a})")
        for k in ("scale_meanA_um", "scale_meanB_um"):
            if k not in c:
                raise KeyError(f"scale_shift ({a}): the curves dict has no {k!r}; recompute it with this "
                               f"version of spectrast ({_CURVES_HINT})")
        if list(map(str, c["ids"])) != list(names):
            raise ValueError(f"scale_shift ({a}): the table and the curves are not from the same call "
                             f"(their features differ, or are in a different order)")
        panels.append((a, t, c))
    vals = np.concatenate([np.r_[np.asarray(c["scale_meanA_um"], float), np.asarray(c["scale_meanB_um"], float)]
                           for _, _, c in panels])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lo, hi = (float(vals.min()) / 1.25, float(vals.max()) * 1.25) if vals.size else (1.0, 10.0)
    n = len(panels)
    ncols = 2 if n == 4 else min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig = U.new_figure((3.2 * ncols + 0.2, 3.7 * nrows + 0.7))
    grid = fig.subplots(nrows, ncols, squeeze=False)
    for extra in grid.ravel()[n:]:
        fig.delaxes(extra)
    la = lb = None
    for ax, (a, t, c) in zip(grid.ravel(), panels):
        la, lb = _arm_labels(c, None)
        x = np.asarray(c["scale_meanB_um"], float)
        y = np.asarray(c["scale_meanA_um"], float)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        called = np.nan_to_num(t["fdr"].to_numpy(float), nan=1.0) < fdr
        rise = np.nan_to_num(t["dI_at_max"].to_numpy(float)) > 0
        res = np.asarray(t["delta_scale_resolvable"].eq(True).fillna(False), bool)
        grey, up, dn = ok & ~called, ok & called & rise, ok & called & ~rise
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, c="0.35", zorder=1)
        ax.scatter(x[grey], y[grey], s=6, c=U.MUTED, linewidths=0, zorder=2)
        for sel, col in ((up, _RISE), (dn, _FALL)):
            fcs = [col if r else "none" for r in res[sel]]
            ax.scatter(x[sel], y[sel], s=22, facecolors=fcs, edgecolors=col, linewidths=0.9, zorder=3)
        U.radius_axis(ax, label=f"scale in {lb} (µm)", axis="x")
        U.radius_axis(ax, label=f"scale in {la} (µm)", axis="y")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        cs = ok & called
        ax.set_title(f"{_AXIS_TITLE.get(a, a)}\n"
                     f"called with a scale: {int(cs.sum())} ({int((cs & (y > x)).sum())} coarser, "
                     f"{int((cs & (y < x)).sum())} finer)\n"
                     f"resolvable shift: {int((cs & res).sum())}\n"
                     f"no scale in one or both arms: {int((~ok).sum())} ({int((~ok & called).sum())} called)",
                     loc="left", fontsize=7)
    from matplotlib.lines import Line2D
    mk = dict(ls="none", marker="o", ms=5, mew=0.9)
    handles = [Line2D([], [], mfc=_RISE, mec=_RISE, label=f"higher in {la}, resolvable", **mk),
               Line2D([], [], mfc="none", mec=_RISE, label=f"higher in {la}, not resolvable", **mk),
               Line2D([], [], mfc=_FALL, mec=_FALL, label=f"lower in {la}, resolvable", **mk),
               Line2D([], [], mfc="none", mec=_FALL, label=f"lower in {la}, not resolvable", **mk),
               Line2D([], [], ls="none", marker="o", ms=3, mfc=U.MUTED, mec=U.MUTED,
                      label=f"not called (FDR ≥ {fdr:g})"),
               Line2D([], [], ls="--", lw=0.8, c="0.35", label=f"no change (above: coarser in {la})")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    return U.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# 5. effect_plane
# ---------------------------------------------------------------------------------------------
def effect_plane(df, *, fdr=0.05, n_label=10, ax=None, show=None, save=None):
    """Difference in organization against shift in characteristic scale, called features highlighted.

    x is ``dI_at_max`` (the tested difference at its radius), y is ``delta_scale_um``. Features
    whose ``delta_scale_um`` is undefined (NaN: one arm has no characteristic length, common for
    hits where one arm is flat) are not dropped: they sit in a band at the bottom labelled "no
    scale shift defined", with their count. When the table carries ``delta_scale_resolvable``, a
    hollow marker is a shift no larger than the arms' between-unit scale spread.

    Parameters
    ----------
    df : pandas.DataFrame
        The table returned by :func:`spectrast.diff.differential`.
    fdr : float, optional
        Threshold for a feature to count as called. Default ``0.05``.
    n_label : int, optional
        How many called features to name (smallest FDR, then largest ``|dI_at_max|``). Default ``10``.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.axes.Axes
        Scatter collections in order: not called with a scale shift, called with a scale shift,
        not called without one (band), called without one (band).

    Examples
    --------
    >>> df = spectrast.diff.differential(A, B, features=genes)         # doctest: +SKIP
    >>> spectrast.pl.effect_plane(df)                                  # doctest: +SKIP
    """
    names = _check_table(df, ["dI_at_max", "delta_scale_um", "fdr"], "effect_plane")
    x = df["dI_at_max"].to_numpy(float)
    y = df["delta_scale_um"].to_numpy(float)
    q = df["fdr"].to_numpy(float)
    called = np.nan_to_num(q, nan=1.0) < fdr
    hollow = np.zeros(len(df), bool)
    if "delta_scale_resolvable" in df.columns:
        res = df["delta_scale_resolvable"]
        hollow = np.asarray(res.eq(False).fillna(False), bool)        # explicitly not resolvable
    fin = np.isfinite(y)
    okx = np.isfinite(x)
    lo, hi = (float(np.min(y[fin & okx])), float(np.max(y[fin & okx]))) if (fin & okx).any() else (-1.0, 1.0)
    span = (hi - lo) or 1.0
    band_top, band_bot = lo - 0.08 * span, lo - 0.42 * span
    bh = band_top - band_bot
    y_band = band_top - 0.38 * bh                                       # points in the upper part, the
    rng = np.random.default_rng(0)                                      # band's label below them;
    jit = (rng.random(len(df)) - 0.5) * 0.4 * bh                        # deterministic jitter
    yy = np.where(fin, y, y_band + jit)
    fig, ax = U.new_axes(ax, figsize=(4.6, 3.8))
    groups = ((fin & ~called, U.MUTED, 9), (fin & called, U.HIGHLIGHT, 18),
              (~fin & ~called, U.MUTED, 9), (~fin & called, U.HIGHLIGHT, 18))
    for sel, col, s in groups:
        sel = sel & okx
        fcs = ["none" if hh else col for hh in hollow[sel]]
        ax.scatter(x[sel], yy[sel], s=s, facecolors=fcs, edgecolors=col, linewidths=0.8, zorder=3)
    ax.axhspan(band_bot, band_top, color="0.93", zorder=0, lw=0)
    n_nan = int((~fin & okx).sum())
    band_txt = ax.text(0.02, band_bot + 0.04 * bh, f"no scale shift defined (n = {n_nan}; "
                       f"{int((~fin & okx & called).sum())} called)",
                       transform=ax.get_yaxis_transform(), fontsize=7, va="bottom", ha="left",
                       color="0.25")
    ax.set_ylim(band_bot, hi + 0.08 * span)
    ax.axhline(0.0, lw=0.5, c="0.7", zorder=1)
    ax.axvline(0.0, lw=0.5, c="0.7", zorder=1)
    ticks = [t for t in ax.get_yticks() if band_top < t <= hi + 0.08 * span]
    ax.set_yticks(ticks)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(_X_LABEL["dI_at_max"])
    ax.set_ylabel(_X_LABEL["delta_scale_um"])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", mfc=U.HIGHLIGHT, mec=U.HIGHLIGHT, ms=5,
                      label=f"FDR < {fdr:g} (n = {int((called & okx).sum())})"),
               Line2D([], [], ls="none", marker="o", mfc=U.MUTED, mec=U.MUTED, ms=4,
                      label=f"not called (n = {int((~called & okx).sum())})")]
    if hollow.any():
        handles.append(Line2D([], [], ls="none", marker="o", mfc="none", mec="0.3", ms=5,
                              label="shift within between-unit spread"))
    if (~okx).any():
        handles.append(Line2D([], [], ls="none", label=f"dI_at_max undefined, not drawn "
                                                          f"(n = {int((~okx).sum())})"))
    _legend_above(ax, handles=handles)
    top = _top_called(names, q, x, called & okx, n_label)
    _place_labels(ax, x[top], yy[top], names[top], obstacles=(band_txt,))
    return U.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 6. neighborhood_maps
# ---------------------------------------------------------------------------------------------
def _neighborhood_field(adata, feature, radii_bins, bin_size, layer=None, basis="spatial"):
    """Per-cell neighbourhood mean of the standardised feature, on the differential's lattice.

    The feature is binned to the mean per lattice bin, standardised over the occupied bins (mean 0,
    sd 1 there; Moran's I is invariant to this choice of units), and averaged over each bin's disc
    of neighbours (centre excluded, row-standardised). Each cell reads its bin's value; bins with no
    occupied neighbour are NaN. Summed over occupied bins, ``z * lag`` divided by ``z**2`` is the
    differential's Moran's I at that radius.

    Returns ``(xy, values (n_radii, n_cells), extra)`` with ``extra`` holding the lattice ``z``,
    ``mask``, the per-radius ``lag`` grids and each cell's ``binidx``. Nothing is written to ``adata``.
    """
    import scipy.sparse as sp
    from .. import diff as _d
    from ..spatial import _fftconv_same, neighbor_kernel
    xy = U.coords(adata, basis)
    v = np.nan_to_num(U.feature_vector(adata, feature, layer), nan=0.0, posinf=0.0, neginf=0.0)
    Xg = sp.csr_matrix(v.astype(np.float32)[:, None])                 # the differential's precision
    fld, mask, H, W, binidx = _d._bin_fields(xy, Xg, bin_size)
    # Standardised over the OCCUPIED bins, in float64. The differential's own z-stack divides by the
    # spread over the whole bounding box (empty bins included), so its units would depend on how much
    # empty space a section's box holds, and two sections could not share a colour scale. Moran's I is
    # invariant to that rescaling, and centring in float64 matches the engine, which re-centres
    # (centring in float32 alone leaves a ~1e-9 relative gap to the curve).
    f = np.asarray(fld[..., 0], np.float64)
    mu = f[mask].mean()
    sd = f[mask].std()
    z = (np.where(mask, f - mu, 0.0) / (sd if sd > 0 else np.nan))[..., None]   # (H, W, 1)
    m = mask.astype(float)[..., None]
    out = np.empty((len(radii_bins), len(xy)))
    lags = []
    for i, rb in enumerate(radii_bins):
        k = neighbor_kernel(int(rb), "disc")
        deg = _fftconv_same(m, k)[..., 0]
        with np.errstate(divide="ignore"):
            inv = np.where(deg > 0.5, 1.0 / np.maximum(deg, 1e-12), np.nan)
        lag = _fftconv_same(z, k)[..., 0] * inv
        lag[~mask] = np.nan
        out[i] = lag.ravel()[binidx]
        lags.append(lag)
    return xy, out, dict(z=z[..., 0], mask=mask, lag=lags, binidx=binidx)


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _pick_section(curves, arm, g, n_sections):
    """Position of the section whose unit curve is closest (L2 over radii) to its arm's mean."""
    stack = curves.get(f"unit_curves_{arm}")
    if stack is None:
        return 0
    mean = curves[f"mean{arm}"][:, g]
    d = np.sqrt(np.nansum((stack[:, :, g] - mean[None, :]) ** 2, axis=1))
    u = int(np.nanargmin(np.where(np.isfinite(d), d, np.inf)))
    secs = curves.get(f"unit_sections_{arm}")
    s = int(secs[u][0]) if secs else u
    if s >= n_sections:
        raise ValueError(f"curves name section {s} of arm {arm} but only {n_sections} were passed; "
                         f"pass the same section lists the differential was run on, or sections=")
    return s


def neighborhood_maps(adatas_A, adatas_B, feature, radii, *, curves=None, sections=None,
                      labels=None, bin_um=None, layer=None, basis="spatial", show=None, save=None):
    """One section per arm, mapped as the neighbourhood mean of the standardised feature.

    For each radius, every cell shows the row-standardised mean, over its disc of neighbouring
    lattice bins, of the feature after standardising it over the section's occupied bins. That is
    the quantity Moran's I correlates the field against, so a difference in expression level alone
    cannot produce a contrast here; positive values are neighbourhoods above the section mean. It is
    computed on the same lattice as :func:`spectrast.diff.differential` (same bin, same disc
    kernel), only for this one field, and nothing is written to the inputs. Each radius has one
    colour scale shared by the two arms, symmetric and clipped at the 98th percentile of
    ``|value|`` over both sections.

    Parameters
    ----------
    adatas_A, adatas_B : list of AnnData or AnnData
        The two arms' sections, as passed to :func:`spectrast.diff.differential`.
    feature : str
        A gene, an ``obs`` column, or ``"<obs_key>==<value>"``.
    radii : float or sequence of float
        One to three radii in µm (coordinate units). Each is rounded to whole lattice bins; the
        radius actually used is in the column title.
    curves : dict, optional
        From ``differential(..., return_curves=True)`` on these sections. Supplies the lattice bin
        and, under the replicate null, chooses each arm's section: the unit whose curve lies
        closest to its arm's mean curve (its first section when a unit holds several).
    sections : (int or str, int or str), optional
        Explicit section per arm: positions in ``adatas_A`` / ``adatas_B``, or names from
        ``curves["section_labels_A"/"_B"]``. Overrides the choice above. Default: as above, else
        the first section of each arm.
    labels : (str, str), optional
        Arm names. Default ``curves["labels"]``, else ``("A", "B")``.
    bin_um : float, optional
        Lattice bin. Default ``curves["bin_size"]``, else the differential's default
        (``K_PITCH`` × the median cell pitch pooled over both arms).
    layer : str, optional
        Expression layer. Default ``X``.
    basis : str, optional
        ``obsm`` key of the coordinates. Default ``"spatial"``.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.figure.Figure
        Rows = arms (A, B), columns = radii.

    Examples
    --------
    >>> df, curves = spectrast.diff.differential(A, B, features=genes,
    ...                                          return_curves=True)  # doctest: +SKIP
    >>> spectrast.pl.neighborhood_maps(A, B, "Trem2", [77, 230], curves=curves)  # doctest: +SKIP
    """
    A, B = _as_list(adatas_A), _as_list(adatas_B)
    if not A or not B:
        raise ValueError("neighborhood_maps needs at least one section in each arm")
    rad = np.atleast_1d(np.asarray(radii, float))
    if not 1 <= rad.size <= 3 or not np.all(rad > 0):
        raise ValueError(f"radii must be one to three positive radii in µm; got {list(rad)}")
    if curves is not None:
        _check_curves(curves, "neighborhood_maps")
    la, lb = _arm_labels(curves, labels)
    if bin_um is None:
        if curves is not None and curves.get("bin_size"):
            bin_um = float(curves["bin_size"])
        else:
            from .. import diff as _d
            bin_um = _d.K_PITCH * _d._median_pitch(A + B, basis)
    bin_um = float(bin_um)
    rbins = [max(1, int(round(x / bin_um))) for x in rad]

    names = {}
    for arm, lst in (("A", A), ("B", B)):
        given = curves.get(f"section_labels_{arm}") if curves is not None else None
        if given is not None and len(given) != len(lst):
            raise ValueError(f"curves were computed on {len(given)} section(s) in arm {arm} but "
                             f"{len(lst)} were passed; pass the same section lists")
        names[arm] = list(given) if given is not None else [f"{arm}{i}" for i in range(len(lst))]

    def _resolve(arm, lst, s):
        if isinstance(s, str):
            if s not in names[arm]:
                raise KeyError(f"section {s!r} not in arm {arm}: {names[arm]}")
            return names[arm].index(s)
        s = int(s)
        if not 0 <= s < len(lst):
            raise IndexError(f"section {s} out of range for arm {arm} ({len(lst)} sections)")
        return s

    if sections is not None:
        if len(sections) != 2:
            raise ValueError("sections must be a pair (section in arm A, section in arm B)")
        iA, iB = _resolve("A", A, sections[0]), _resolve("B", B, sections[1])
    elif curves is not None and curves.get("unit_curves_A") is not None:
        g = int(_feature_idx(curves, feature)[0])
        iA, iB = _pick_section(curves, "A", g, len(A)), _pick_section(curves, "B", g, len(B))
    else:
        iA = iB = 0

    fields = []
    for ad in (A[iA], B[iB]):
        xy, vals, _ = _neighborhood_field(ad, feature, rbins, bin_um, layer, basis)
        fields.append((xy, vals))
    nr = len(rbins)
    fig = U.new_figure((3.0 * nr + 0.6, 5.6))
    axes = np.atleast_2d(fig.subplots(2, nr, squeeze=False))
    for j, rb in enumerate(rbins):
        both = np.concatenate([fields[0][1][j], fields[1][1][j]])
        vmin, vmax = U.robust_limits(both, pct=98.0, symmetric=True)
        sc = None
        for row, ((xy, vals), arm, lab, i) in enumerate(zip(fields, ("A", "B"), (la, lb), (iA, iB))):
            ax = axes[row, j]
            sc = U.spatial_scatter(ax, xy, vals[j], cmap=U.DIVERGING, vmin=vmin, vmax=vmax,
                                   colorbar=False)
            if row == 0:
                ax.set_title(f"r = {rb * bin_um:.0f} µm", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{lab}\n{names[arm][i]}", fontsize=8)
        cb = fig.colorbar(sc, ax=list(axes[:, j]), location="bottom", shrink=0.8, pad=0.02)
        cb.set_label("neighbourhood mean of z", fontsize=7)
    fig.suptitle(str(feature))
    return U.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# 7. section_scales
# ---------------------------------------------------------------------------------------------
def _section_col(tidy):
    cols = list(tidy.columns)
    if len(cols) > 2 and cols[0] == "gene" and cols[2] == "n_obs":
        return cols[1]                                                # gr.fingerprint_by's layout
    for c in cols:
        if c != "gene" and not np.issubdtype(np.asarray(tidy[c]).dtype, np.number):
            return c
    raise KeyError("cannot find the section column; pass the table from spectrast.gr.fingerprint_by")


def _group_colors(levels):
    import matplotlib
    if len(levels) <= 2:
        return {lv: U.ARM_COLORS[k] for k, lv in enumerate(levels)}
    cmap = matplotlib.colormaps["tab10"]
    return {lv: cmap(k % 10) for k, lv in enumerate(levels)}


def section_scales(tidy, *, genes=None, groups=None, kind="strip", sections=None, axis="scale_um",
                   ax=None, show=None, save=None):
    """Per-section values of one fingerprint axis, from :func:`spectrast.gr.fingerprint_by`.

    ``kind="strip"``: genes on y, each section a point at its value (log µm axis for ``scale_um``),
    a black tick at each gene's median, optionally coloured by a section → group mapping. Rows are
    ordered by median value. ``kind="pairs"``: one section's values against another's, with the
    identity line and the Spearman correlation in the title.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The long table returned by :func:`spectrast.gr.fingerprint_by` (columns ``gene``, the
        section key, ``n_obs``, one per axis).
    genes : list of str, optional
        Genes to show. Default for ``"strip"``: up to 30 genes with a finite value in the most
        sections, ties broken by the highest median amplitude when the table has one
        (``scale_peak_moran``, from ``fingerprint_by(axes=(..., "scale_peak_moran"))``) and by
        table order otherwise. Default for ``"pairs"``: every gene.
    groups : dict or pandas.Series, optional
        Section → group (e.g. condition), for colour in ``"strip"``. Groups take colours in the
        order they first appear in ``groups`` (two groups: the first gets arm A's colour in the
        other differential plots, the second arm B's); sections missing from it are "other".
    kind : {"strip", "pairs"}, optional
        Default ``"strip"``.
    sections : (str, str), optional
        The two sections compared by ``kind="pairs"`` (required there).
    axis : str, optional
        The fingerprint column to show. Default ``"scale_um"``.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show, save
        As in :func:`volcano`.

    Returns
    -------
    matplotlib.axes.Axes
        ``"strip"``: one scatter collection per group (in the order of ``groups``; one in all
        when ``groups`` is None), then the medians. ``"pairs"``: one scatter collection.

    Examples
    --------
    >>> tidy = spectrast.gr.fingerprint_by(adata, "section_id")       # doctest: +SKIP
    >>> spectrast.pl.section_scales(tidy, groups=section_to_condition)  # doctest: +SKIP
    >>> spectrast.pl.section_scales(tidy, kind="pairs", sections=("WT1", "WT2"))  # doctest: +SKIP
    """
    import pandas as pd
    if not isinstance(tidy, pd.DataFrame) or "gene" not in tidy.columns:
        raise TypeError("section_scales takes the table returned by spectrast.gr.fingerprint_by(adata, key)")
    if axis not in tidy.columns:
        num = [c for c in tidy.columns if c != "n_obs" and pd.api.types.is_numeric_dtype(tidy[c])]
        raise KeyError(f"axis {axis!r} not in the table; available: {num}")
    key = _section_col(tidy)
    tab = tidy.assign(_sec=tidy[key].astype(str), _gene=tidy["gene"].astype(str))
    piv = tab.pivot_table(index="_gene", columns="_sec", values=axis, aggfunc="first", dropna=False)
    piv = piv.reindex(index=pd.unique(tab["_gene"]), columns=pd.unique(tab["_sec"]))
    log_axis = axis.endswith("_um")

    if kind == "pairs":
        if sections is None or len(sections) != 2:
            raise ValueError(f"kind='pairs' needs sections=(s1, s2); available: {list(piv.columns)}")
        s1, s2 = map(str, sections)
        miss = [s for s in (s1, s2) if s not in piv.columns]
        if miss:
            raise KeyError(f"section(s) {miss} not in the table; available: {list(piv.columns)}")
        sub = piv if genes is None else piv.loc[[g for g in map(str, genes) if g in piv.index]]
        a, b = sub[s1].to_numpy(float), sub[s2].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if log_axis:
            ok &= (a > 0) & (b > 0)
        from scipy.stats import spearmanr
        rho = float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan
        fig, ax = U.new_axes(ax, figsize=(3.8, 3.6))
        ax.scatter(a[ok], b[ok], s=10, c="0.25", linewidths=0, alpha=0.8)
        lo, hi = (min(a[ok].min(), b[ok].min()), max(a[ok].max(), b[ok].max())) if ok.any() else (0, 1)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, c=U.HIGHLIGHT)
        if log_axis:
            _log_axis(ax, label=f"{axis} in {s1}", axis="x")
            _log_axis(ax, label=f"{axis} in {s2}", axis="y")
        else:
            ax.set_xlabel(f"{axis} in {s1}")
            ax.set_ylabel(f"{axis} in {s2}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Spearman ρ = {rho:.2f} (n = {int(ok.sum())} genes)", fontsize=9)
        return U.finish(fig, ax, show, save)
    if kind != "strip":
        raise ValueError(f"kind must be 'strip' or 'pairs'; got {kind!r}")

    if genes is None:
        n_fin = piv.notna().sum(1).to_numpy()
        amp_col = next((c for c in _AMPLITUDE_COLS if c in tidy.columns), None)
        if amp_col is not None:
            amp = tab.groupby("_gene", sort=False)[amp_col].median().reindex(piv.index).to_numpy(float)
        else:
            amp = np.zeros(len(piv))
        o = np.lexsort((np.arange(len(piv)), -np.nan_to_num(amp, nan=-np.inf), -n_fin))
        chosen = list(piv.index[o][:30])
    else:
        chosen = [str(g) for g in genes]
        miss = [g for g in chosen if g not in piv.index]
        if miss:
            raise KeyError(f"gene(s) {miss[:10]} not in the table")
    sub = piv.loc[chosen]
    med = sub.median(axis=1, skipna=True).to_numpy(float)
    order = np.argsort(np.where(np.isfinite(med), med, np.inf), kind="stable")
    sub, med = sub.iloc[order], med[order]
    secs = list(sub.columns)
    if groups is not None:
        gmap = {str(k): str(v) for k, v in (groups.items() if hasattr(groups, "items") else groups)}
        sec_group = [gmap.get(s, "other") for s in secs]
        present = set(sec_group)
        levels = [g for g in dict.fromkeys(list(gmap.values()) + ["other"]) if g in present]
    else:
        sec_group = ["all sections"] * len(secs)
        levels = ["all sections"]
    colors = _group_colors(levels)
    nG = len(sub)
    fig, ax = U.new_axes(ax, figsize=(4.6, max(2.6, 0.18 * nG + 1.0)))
    offs = (np.arange(len(secs)) / max(len(secs) - 1, 1) - 0.5) * 0.5 if len(secs) > 1 else np.zeros(1)
    V = sub.to_numpy(float)
    for lv in levels:
        cols = [k for k, gg in enumerate(sec_group) if gg == lv]
        xs, ys = [], []
        for k in cols:
            v = V[:, k]
            ok = np.isfinite(v)
            xs.append(v[ok]); ys.append(np.arange(nG)[ok] + offs[k])
        xs = np.concatenate(xs) if xs else np.empty(0)
        ys = np.concatenate(ys) if ys else np.empty(0)
        n_sec = len(cols)
        ax.scatter(xs, ys, s=9, color=colors[lv], linewidths=0, alpha=0.85,
                   label=f"{lv} (n = {n_sec} sections)")
    ax.scatter(med, np.arange(nG), marker="|", s=90, c="k", linewidths=1.2, zorder=3, label="median")
    ax.set_yticks(np.arange(nG))
    ax.set_yticklabels(list(sub.index), fontsize=7)
    ax.set_ylim(nG - 0.5, -0.5)
    if log_axis:
        _log_axis(ax, label=f"{axis} (µm)")
    else:
        ax.set_xlabel(axis)
    _legend_above(ax)
    return U.finish(fig, ax, show, save)

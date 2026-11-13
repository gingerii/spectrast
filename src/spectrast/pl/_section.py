"""One section: how its genes and cell types are organized.

Inputs come from three computing calls:

* :func:`spectrast.organization.organization` -- the tidy (gene, statistic) table of characteristic
  scale and amplitude, and with ``return_curves=True`` the curves it was reduced from;
* :func:`spectrast.gr.scale_spectrum` / :func:`spectrast.gr.fingerprint` -- the stored Moran
  scalogram ``adata.uns["spectrast_scalogram"]`` and ``adata.var["spectrast_char_scale"]``
  (plus ``adata.var["spectrast_focality"]`` from ``fingerprint``);
* :func:`spectrast.gr.moran_null` -- the per-gene, per-radius significance table;
* :func:`spectrast.organization.celltype_scale` -- the same reduction for cell-type indicators.

The local-Moran maps are the one place a plot computes a field: the per-cell local Moran's I at
one radius, ``z_i * mean_{j in N_r(i)} z_j`` on the standardised field, with the row-standardised
disc neighbourhood and the per-radius lattice that :func:`spectrast.gr.scale_spectrum` uses. Its
mean over cells IS the stored scalogram's value at that radius (the global statistic is defined as
that mean), so a map is its curve's point resolved over the tissue. Nothing is written to ``adata``.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

from . import _utils as _u

#: Axis labels for the six organization statistics.
_STAT_LABEL = {"moran": "Moran's I", "geary": "Geary's C", "general_g": "General G",
               "losh": "LOSH (Gini of H)", "focality": "focality (Gini)",
               "texture": "texture power"}

#: Above this many rows a heatmap drops its gene tick labels.
_MAX_ROW_LABELS = 60


# ---------------------------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------------------------
def _gate_mask(amp, gate):
    """``(read mask, threshold, description)`` for an amplitude gate.

    ``gate`` is a quantile in [0, 1] (``0.8`` reads the top 20%), the tuple ``("quantile", q)``,
    the tuple ``("abs", value)`` for an absolute amplitude threshold, or ``None`` to read every
    gene with a finite amplitude. The quantile is taken over every gene passed in."""
    amp = np.asarray(amp, float)
    fin = np.isfinite(amp)
    if gate is None:
        return fin, -np.inf, "no amplitude gate"
    if isinstance(gate, tuple):
        if len(gate) != 2 or gate[0] not in ("quantile", "abs"):
            raise ValueError("gate must be a quantile in [0, 1], ('quantile', q), ('abs', value) or None; "
                             f"got {gate!r}")
        kind, v = gate[0], float(gate[1])
    else:
        kind, v = "quantile", float(gate)
    if kind == "quantile":
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"gate quantile {v} is outside [0, 1]; pass ('abs', value) for an absolute "
                             "amplitude threshold")
        thr = float(np.quantile(amp[fin], v)) if fin.any() else np.inf
        desc = f"top {100 * (1 - v):.0f}% by amplitude" if v > 0 else "all genes"
    else:
        thr = v
        desc = f"amplitude >= {v:g}"
    return fin & (amp >= thr), thr, desc


def _var_scale_col(key):
    """The ``adata.var`` column a scalogram key's reduction was written to."""
    if key == _u.SCALOGRAM_KEY:
        return "spectrast_char_scale"
    if key.endswith("_scalogram"):
        return key[: -len("_scalogram")]
    return None


def _pop_sign(S):
    """Population orientation (+1 decaying, -1 rising), NaN-tolerant, as the reductions take it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1.0 if np.nanmedian(S[0]) >= np.nanmedian(S[-1]) else -1.0


def _curve_source(data, key=_u.SCALOGRAM_KEY, stat="moran"):
    """``(S, radii, names, scale, amplitude, sign)`` from an AnnData (stored scalogram) or from the
    dict :func:`spectrast.organization.organization` returns with ``return_curves=True``."""
    from ..scale import half_max
    if isinstance(data, dict):
        from ..organization import _reduce_curve
        missing = [k for k in ("radii_um", "genes", stat) if k not in data]
        if missing:
            raise KeyError(f"the curve dict lacks {missing}: pass the dict returned by "
                           f"spectrast.organization.organization(adata, stats=(..., {stat!r}), "
                           "return_curves=True)")
        S = np.asarray(data[stat], float)
        r = np.asarray(data["radii_um"], float)
        names = np.asarray(data["genes"], dtype=str)
        R = _reduce_curve(S, r)
        return S, r, names, R["scale_um"], R["amplitude"], R["orientation"]
    S, r, names = _u.scalogram(data, key)
    sign = _pop_sign(S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amp = np.nanmax(sign * S, axis=0)
    col = _var_scale_col(key)
    if col is not None and col in data.var:
        scale = data.var[col].reindex(names).to_numpy(float)
    else:
        scale = half_max(S, r, sign=sign)
    return S, r, names, scale, amp, sign


def _interp_log(r, y, at):
    """``y`` at radius ``at``, linear in log radius (the reductions interpolate the same way)."""
    return float(np.interp(np.log(at), np.log(np.asarray(r, float)), np.asarray(y, float)))


def _scalogram_params(adata, key):
    """The stored scalogram's provenance (layer, basis, bin_size, ...) as a dict; empty if absent."""
    st = adata.uns.get(key) if hasattr(adata, "uns") else None
    if st is None or "params" not in st:
        return {}
    p = st["params"]
    try:
        return json.loads(p) if isinstance(p, str) else dict(p)
    except (TypeError, ValueError):
        return {}


def _bin_for(params, radii_stored, r):
    """The lattice bin the stored scalogram used at radius ``r``; ``None`` means autotuned per radius,
    which is the default of :func:`spectrast.gr.scale_spectrum`."""
    bs = params.get("bin_size")
    if bs is None:
        return None
    bs = np.atleast_1d(np.asarray(bs, float))
    if bs.size == 1:
        return float(bs[0])
    if radii_stored is not None and bs.size == len(radii_stored):
        return float(bs[int(np.argmin(np.abs(np.log(np.asarray(radii_stored, float) / r))))])
    return float(bs[0])


def _local_moran(xy, v, r, bin_size=None):
    """Per-cell local Moran's I of ``v`` at radius ``r``: ``z_i * mean_{j in disc(i)} z_j``.

    ``z`` is ``v`` standardised over cells; the neighbourhood is the disc of radius ``r`` with the
    cell itself excluded, row-standardised, evaluated by the package's FFT neighbour reduction on
    the lattice ``bin_size`` (``None``: autotuned for this radius, as the scalogram does). The mean
    of the returned map over cells is the mean-neighbour global Moran's I at ``r``. Returns
    ``(local, neighbour mean of z, neighbour count, bin used)``."""
    from ..spatial_sc import _neighbor_sum_count, _standardize_cols
    z = _standardize_cols(np.asarray(v, float))
    nsum, ncnt, b = _neighbor_sum_count(np.asarray(xy, float), z, float(r), bin_size)
    mean_nb = nsum[:, 0] / np.maximum(ncnt, 1)
    return z[:, 0] * mean_nb, mean_nb, ncnt, b


def _circle_centre(xy, v, mean_nb, ncnt):
    """Centre of the drawn r* circle: among cells with a positive standardised value and at least
    the median neighbour count, the one whose neighbourhood mean is largest -- the core of the
    strongest high-valued patch (for an indicator, the cell of the type most surrounded by it)."""
    v = np.asarray(v, float)
    z = (v - v.mean()) / (v.std() + 1e-12)
    ok = ncnt >= np.median(ncnt[ncnt > 0]) if (ncnt > 0).any() else np.ones(len(v), bool)
    cand = ok & (z > 0)
    if not cand.any():
        cand = ok
    i = int(np.flatnonzero(cand)[np.argmax(mean_nb[cand])])
    return xy[i]


def _feature_curve(adata, feature, key, *, need=True):
    """``(curve, radii, r_star, source)`` for one feature.

    A gene held in the stored scalogram reads its curve from there and its r* from ``adata.var``
    (or the half-max of the stored curve). Any other feature -- a gene outside the stored set, an
    obs column, a ``"<obs_key>==<value>"`` indicator -- is measured here: ONE field, the same
    per-cell Moran scalogram and radii as the stored one (or :func:`spectrast.gr.default_radii`
    when nothing is stored). ``need=False`` returns ``(None, radii, nan, "absent")`` instead of
    computing."""
    from ..scale import half_max
    params = _scalogram_params(adata, key)
    sign = 1.0
    if key in adata.uns:
        S, r, names = _u.scalogram(adata, key)
        sign = _pop_sign(S)
        if "==" not in feature and feature in set(names):
            j = int(np.flatnonzero(names == feature)[0])
            col = _var_scale_col(key)
            rstar = (float(adata.var.loc[feature, col]) if col in adata.var and feature in adata.var_names
                     else float(half_max(S[:, [j]], r, sign=sign)[0]))
            return S[:, j], r, rstar, "stored"
        radii = r
    else:
        from ..gr import default_radii
        radii = default_radii(adata, params.get("basis") or "spatial")
    if not need:
        return None, np.asarray(radii, float), np.nan, "absent"
    from ..spatial_sc import morans_scalogram
    xy = _u.coords(adata, params.get("basis") or "spatial")
    v = _u.feature_vector(adata, feature, layer=params.get("layer"))
    bins = [_bin_for(params, radii, float(rr)) for rr in radii]
    bins = None if all(b is None for b in bins) else bins
    curve = morans_scalogram(xy, v[:, None], np.asarray(radii, float), bin_size=bins)[:, 0]
    rstar = float(half_max(curve[:, None], radii, sign=sign)[0])
    return curve, np.asarray(radii, float), rstar, "computed"


def _draw_curve(ax, r, y, *, rstar=np.nan, r_map=None, map_mean=None, color="0.15",
                ylabel="Moran's I", sign=1.0, note=None):
    """One curve on a log-radius axis, with r* (dashed) and its half-range level marked."""
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    ax.plot(r, y, "-o", color=color, ms=2.5, lw=1.2)
    _radius_axis(ax, r)
    ax.set_ylabel(ylabel)
    fin = np.isfinite(y)
    if fin.any():
        A = sign * y[fin]
        lvl = sign * (A.min() + 0.5 * (A.max() - A.min()))
        ax.axhline(lvl, color="0.5", ls=":", lw=0.8)
    if np.isfinite(rstar):
        ax.axvline(rstar, color=_u.HIGHLIGHT, ls="--", lw=1.0)
        ax.plot([rstar], [_interp_log(r, y, rstar)], "o", color=_u.HIGHLIGHT, ms=4)
    if r_map is not None and map_mean is not None:
        ax.plot([r_map], [map_mean], "s", mfc="none", mec="k", ms=6)
    if note:
        ax.set_title(note, fontsize=8)


def _draw_local(ax, xy, L, r, *, title=None, centre=None, circle=True, vlim=None, colorbar=True):
    """A local-Moran map: diverging, symmetric about 0, clipped at the 99.5th percentile of ``|I|``."""
    vmin, vmax = vlim if vlim is not None else _u.robust_limits(L, 99.5, symmetric=True)
    sc = _u.spatial_scatter(ax, xy, L, cmap=_u.DIVERGING, vmin=vmin, vmax=vmax, title=title,
                            colorbar=colorbar, cbar_label="local Moran's I", order_by_value=True)
    if circle and centre is not None:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((float(centre[0]), float(centre[1])), float(r), fill=False,
                            ec="k", lw=1.0))
    return sc


_radius_axis = _u.radius_axis       # shared: ticks at 1, 2, 5 per decade


def _radius_ticks(ax, r, n=6, axis="x"):
    """Tick a heatmap's column index with rounded radii (columns are the radii, in order)."""
    r = np.asarray(r, float)
    pos = np.unique(np.linspace(0, len(r) - 1, min(n, len(r))).round().astype(int))
    lab = [f"{r[i]:.0f}" for i in pos]
    if axis == "x":
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
    else:
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)


def _require(table, cols, call):
    missing = [c for c in cols if c not in getattr(table, "columns", [])]
    if missing:
        raise KeyError(f"table lacks columns {missing}: pass the table returned by {call}")


# ---------------------------------------------------------------------------------------------
# 1. the scale x amplitude plane
# ---------------------------------------------------------------------------------------------
def scale_plane(table, *, stat="moran", color=None, adata=None, gate=0.8, scramble=None, n_label=5,
                ax=None, show=None, save=None):
    """Every gene placed by its characteristic scale and amplitude for one statistic.

    x is ``scale_um`` (log, µm), the radius at which the gene's curve falls to half its own range;
    y is ``amplitude``, that range's top. Genes at or above the amplitude gate are *read*: coloured
    by ``color`` (or one highlight colour), with the ``n_label`` finest and coarsest named. The rest
    are drawn grey. A gene whose curve never crosses its half-range has no scale and cannot be
    placed; how many is said in the title.

    Parameters
    ----------
    table : DataFrame
        From :func:`spectrast.organization.organization` (one row per gene and statistic).
    stat : str
        Which statistic's rows to draw. Default ``"moran"``.
    color : str or array-like, optional
        A column of ``adata.var`` (e.g. ``"spectrast_focality"``, written by
        :func:`spectrast.gr.fingerprint`; needs ``adata``) or values aligned to this statistic's
        rows of ``table``. ``None`` draws read genes in one colour.
    adata : AnnData, optional
        Source of a named ``color`` column.
    gate : float, tuple or None
        The amplitude gate. A float in [0, 1] is a quantile of this statistic's amplitudes (0.8
        reads the top 20%); ``("abs", value)`` is an absolute amplitude threshold; ``None`` reads
        every gene. Drawn as a dashed horizontal line.
    scramble : DataFrame, optional
        The same table computed on a coordinate-permuted copy of the section (permute the rows of
        ``adata.obsm["spatial"]`` and rerun ``organization``), drawn behind as density contours
        enclosing 50, 80 and 95% of its placed genes.
    n_label : int
        Name this many of the finest and of the coarsest read genes.
    ax, show, save
        The scanpy plotting convention (see :mod:`spectrast.pl`).

    Returns
    -------
    matplotlib.axes.Axes, or None if ``show=True``.

    Notes
    -----
    The amplitude axis is logarithmic when at least 95% of the placed amplitudes are positive
    (Moran, General G, LOSH, focality, texture); genes with a non-positive amplitude are then left
    out and counted in the title. Geary's amplitude is the top of the sign-flipped curve, which is
    negative, so its axis is linear.

    Examples
    --------
    >>> tab = spectrast.organization.organization(adata)                         # doctest: +SKIP
    >>> spectrast.gr.fingerprint(adata)        # adata.var["spectrast_focality"]  # doctest: +SKIP
    >>> spectrast.pl.scale_plane(tab, color="spectrast_focality", adata=adata)   # doctest: +SKIP
    """
    import pandas as pd
    call = "spectrast.organization.organization(adata)"
    _require(table, ["gene", "statistic", "scale_um", "amplitude"], call)
    t = table[table["statistic"] == stat]
    if t.empty:
        raise ValueError(f"no rows for statistic {stat!r} in the table (has "
                         f"{sorted(table['statistic'].unique())}); run {call} with stats including it")
    genes = t["gene"].astype(str).to_numpy()
    x = t["scale_um"].to_numpy(float)
    y = t["amplitude"].to_numpy(float)
    read, thr, desc = _gate_mask(y, gate)

    cvals = None
    clabel = None
    if color is not None:
        if isinstance(color, str):
            if adata is None:
                raise ValueError(f"color={color!r} names an adata.var column; pass adata=")
            if color not in adata.var:
                hint = (" (run spectrast.gr.fingerprint(adata) or spectrast.gr.focality(adata))"
                        if "focality" in color else "")
                raise KeyError(f"adata.var[{color!r}] not found{hint}")
            cvals = pd.Series(adata.var[color].to_numpy(float),
                              index=np.asarray(adata.var_names, str)).reindex(genes).to_numpy(float)
            clabel = color.replace("spectrast_", "")
        else:
            cvals = np.asarray(color, float).ravel()
            if cvals.size != len(t):
                raise ValueError(f"color has {cvals.size} values; this statistic has {len(t)} rows")
            clabel = "colour"

    placed = np.isfinite(x) & (x > 0) & np.isfinite(y)
    n_noscale = int((~np.isfinite(x) & np.isfinite(y)).sum())
    logy = placed.any() and (y[placed] > 0).mean() >= 0.95
    n_nonpos = 0
    if logy:
        n_nonpos = int((placed & (y <= 0)).sum())
        placed &= y > 0

    fig, ax = _u.new_axes(ax, figsize=(5.0, 4.0))
    if scramble is not None:
        _require(scramble, ["statistic", "scale_um", "amplitude"], call + " on a coordinate-permuted copy")
        s = scramble[scramble["statistic"] == stat]
        sx, sy = s["scale_um"].to_numpy(float), s["amplitude"].to_numpy(float)
        ok = np.isfinite(sx) & (sx > 0) & np.isfinite(sy) & ((sy > 0) if logy else True)
        _density_contours(ax, sx[ok], sy[ok], logy=logy)

    grey = placed & ~read
    ax.scatter(x[grey], y[grey], s=4, c=_u.MUTED, linewidths=0, rasterized=True, label="not read",
               zorder=2)
    rd = placed & read
    if cvals is None:
        ax.scatter(x[rd], y[rd], s=9, c=_u.HIGHLIGHT, linewidths=0, label="read", zorder=3)
    else:
        cr = cvals[rd]
        okc = np.isfinite(cr)
        if (~okc).any():
            ax.scatter(x[rd][~okc], y[rd][~okc], s=9, facecolors="none", edgecolors="0.4",
                       linewidths=0.6, label="read, no colour value", zorder=3)
        sc = ax.scatter(x[rd][okc], y[rd][okc], s=9, c=cr[okc], cmap=_u.SEQUENTIAL,
                        vmin=_u.robust_limits(cr[okc], 98)[0], vmax=_u.robust_limits(cr[okc], 98)[1],
                        linewidths=0, label="read", zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label(clabel)
    if np.isfinite(thr):
        ax.axhline(thr, color="k", ls="--", lw=0.8, label=f"gate: {desc}", zorder=4)

    if n_label and rd.any():
        idx = np.flatnonzero(rd)
        order = idx[np.argsort(x[idx])]
        lo = list(order[:n_label])
        hi = [i for i in order[::-1][:n_label] if i not in set(lo)]
        # two stacked columns (finest at left, coarsest at right) with leader lines, highest first
        for side, sel in ((0, lo), (1, hi)):
            for m, i in enumerate(sorted(sel, key=lambda q: -y[q])):
                ax.annotate(genes[i], (x[i], y[i]), xycoords="data",
                            xytext=(0.02 if side == 0 else 0.98, 0.96 - 0.055 * m),
                            textcoords="axes fraction", ha="left" if side == 0 else "right",
                            va="center", fontsize=7,
                            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5,
                                            shrinkA=1, shrinkB=2))

    _radius_axis(ax, label="characteristic scale (µm)")
    if placed.any():
        ax.set_xlim(np.nanmin(x[placed]) / 1.2, np.nanmax(x[placed]) * 1.2)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(f"amplitude ({_STAT_LABEL.get(stat, stat)})")
    note = f"{stat}: {int(placed.sum())} placed, {int(rd.sum())} read"
    if n_noscale:
        note += f"; {n_noscale} with no scale not placed"
    if n_nonpos:
        note += f"; {n_nonpos} with amplitude <= 0 off the log axis"
    ax.set_title(note, fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    if scramble is not None:
        from matplotlib.patches import Patch
        handles.append(Patch(color="#E8A27C", alpha=0.6))
        labels.append("coordinate scramble")
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, markerscale=1.5)
    return _u.finish(fig, ax, show, save)


def _density_contours(ax, x, y, *, logy=True, color="#E8A27C"):
    """Filled contours enclosing 50, 80 and 95% of the points, in log-x (and log-y) space."""
    from scipy.ndimage import gaussian_filter
    if x.size < 20:
        ax.scatter(x, y, s=3, c=color, linewidths=0, alpha=0.6, zorder=0)
        return
    lx = np.log10(x)
    ly = np.log10(y) if logy else y
    pad = lambda v: (v.min() - 0.25 * np.ptp(v) - 1e-9, v.max() + 0.25 * np.ptp(v) + 1e-9)  # noqa: E731
    H, xe, ye = np.histogram2d(lx, ly, bins=80, range=[pad(lx), pad(ly)])
    H = gaussian_filter(H, 1.5)
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    levels = sorted({float(h[min(np.searchsorted(c, f), h.size - 1)]) for f in (0.95, 0.8, 0.5)})
    levels = [lv for lv in levels if lv > 0] + [float(H.max()) * 1.0001]
    if len(levels) < 2:
        return
    xc = 10 ** ((xe[:-1] + xe[1:]) / 2)
    yc = 10 ** ((ye[:-1] + ye[1:]) / 2) if logy else (ye[:-1] + ye[1:]) / 2
    alphas = np.linspace(0.25, 0.6, len(levels) - 1)
    for k in range(len(levels) - 1):
        ax.contourf(xc, yc, H.T, levels=[levels[k], levels[-1]], colors=[color], alpha=alphas[k],
                    zorder=0)


# ---------------------------------------------------------------------------------------------
# 2. a few genes' curves, with their null bands
# ---------------------------------------------------------------------------------------------
def curves(adata, genes, *, null=None, alpha=0.05, mark_scale=True, key=_u.SCALOGRAM_KEY, ax=None,
           show=None, save=None):
    """Moran curves of a few genes from the stored scalogram, with r* and optional null bands.

    Each gene's curve is drawn in its own colour on a log-radius axis. ``mark_scale`` marks r*
    (``adata.var["spectrast_char_scale"]`` when present, else the half-max of the stored curve)
    with a dot on the curve and a dashed line. With ``null``, each gene's null band
    ``mu +/- z_{1-alpha/2} sd`` is shaded in the gene's colour at the null's own radii, and the
    Moran's I the null tested is drawn as markers there: filled where ``p < alpha``, open otherwise.

    Parameters
    ----------
    adata : AnnData
        After :func:`spectrast.gr.scale_spectrum` or :func:`spectrast.gr.fingerprint`.
    genes : str or sequence of str
        Genes to draw (from the stored scalogram's genes).
    null : DataFrame, optional
        From :func:`spectrast.gr.moran_null` (columns ``gene, r_um, I, mu, sd, p``).
    alpha : float
        Significance level for the band width and the filled markers.
    mark_scale : bool
        Mark each gene's r*.
    key : str
        ``adata.uns`` key of the scalogram. Default ``"spectrast_scalogram"``.
    ax, show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.axes.Axes, or None if ``show=True``.

    Notes
    -----
    The stored curve and the null are measured on different lattices -- the scalogram autotunes a
    bin per radius, :func:`spectrast.gr.moran_null` uses one pitch-anchored bin -- so the markers
    (the null's own observed I) can sit slightly off the line. The band is two-sided at
    ``alpha``; ``p`` is the table's own (upper-tail) p-value.

    Examples
    --------
    >>> spectrast.gr.scale_spectrum(adata)                                       # doctest: +SKIP
    >>> null = spectrast.gr.moran_null(adata, genes=["Gfap", "Mbp"])              # doctest: +SKIP
    >>> spectrast.pl.curves(adata, ["Gfap", "Mbp"], null=null)                   # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from ..scale import half_max
    S, r, names = _u.scalogram(adata, key)
    genes = [genes] if isinstance(genes, str) else list(genes)
    idx = _u.index_of(names, genes)
    sign = _pop_sign(S)
    if null is not None:
        _require(null, ["gene", "r_um", "I", "mu", "sd", "p"], "spectrast.gr.moran_null(adata)")
        absent = [g for g in genes if g not in set(null["gene"].astype(str))]
        if len(absent) == len(genes):
            raise KeyError(f"none of {genes} is in the null table; run spectrast.gr.moran_null(adata, "
                           f"genes={genes!r})")
        if absent:
            warnings.warn(f"no null rows for {absent}; their bands are not drawn", stacklevel=2)
        zc = float(norm.ppf(1 - alpha / 2))

    fig, ax = _u.new_axes(ax, figsize=(4.6, 3.4))
    cmap = plt.get_cmap("tab10")
    col = _var_scale_col(key)
    marks = []
    for k, (g, j) in enumerate(zip(genes, idx)):
        c = cmap(k % 10)
        y = S[:, j]
        ax.plot(r, y, "-", color=c, lw=1.4, label=g, zorder=3)
        if null is not None and g not in absent:
            d = null[null["gene"].astype(str) == g].sort_values("r_um")
            rr = d["r_um"].to_numpy(float)
            mu, sd = d["mu"].to_numpy(float), d["sd"].to_numpy(float)
            ax.fill_between(rr, mu - zc * sd, mu + zc * sd, color=c, alpha=0.18, lw=0, zorder=1)
            sig = d["p"].to_numpy(float) < alpha
            I = d["I"].to_numpy(float)
            ax.scatter(rr[sig], I[sig], s=14, color=c, zorder=4)
            ax.scatter(rr[~sig], I[~sig], s=14, facecolors="white", edgecolors=c, linewidths=0.9,
                       zorder=4)
        if mark_scale:
            if col is not None and col in adata.var and g in adata.var_names:
                rs = float(adata.var.loc[g, col])
            else:
                rs = float(half_max(S[:, [j]], r, sign=sign)[0])
            if np.isfinite(rs):
                yv = _interp_log(r, y, rs)
                ax.plot([rs], [yv], "D", color=c, ms=4, zorder=5)
                marks.append((rs, yv, c))
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    lo, hi = ax.get_ylim()
    for rs, yv, c in marks:                      # dashed drop line from r* on the curve to the axis
        ax.plot([rs, rs], [lo, yv], ls="--", color=c, lw=0.8, zorder=2)
    ax.set_ylim(lo, hi)
    _radius_axis(ax, r)
    ax.set_ylabel("Moran's I")
    ax.legend(fontsize=7, frameon=False)
    if null is not None:
        ax.set_title(f"band: null mu +/- {zc:.2f} sd; filled: p < {alpha:g}", fontsize=8)
    return _u.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 3. significance over genes x radii
# ---------------------------------------------------------------------------------------------
def significance_heatmap(null, *, value="z", adata=None, genes=None, order="scale", max_genes=None,
                         ax=None, show=None, save=None):
    """Genes x radius heatmap of the per-gene, per-radius Moran null.

    Parameters
    ----------
    null : DataFrame
        From :func:`spectrast.gr.moran_null` (columns ``gene, r_um, z, p``).
    value : {"z", "-log10p"}
        What to colour by. ``-log10p`` clips ``p`` at 1e-300 so an underflowed zero stays finite.
    adata : AnnData, optional
        Needed for ``order="scale"``: rows are ordered by ``adata.var["spectrast_char_scale"]``
        (from :func:`spectrast.gr.scale_spectrum` / :func:`spectrast.gr.fingerprint`), genes with
        no scale last, and each row's r* is marked.
    genes : sequence of str, optional
        Restrict to these genes.
    order : {"scale", "zmax", None}
        Row order: by r* (fine at top), by the radius of each gene's largest z, or the table's.
    max_genes : int, optional
        Keep the genes with the largest maximum z.
    ax, show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.axes.Axes, or None if ``show=True``.

    Notes
    -----
    Gene names are written only when at most 60 rows are drawn. On large sections nearly every
    gene x radius cell is significant under this null; read the heatmap for where z is largest,
    not for which cells pass.

    Examples
    --------
    >>> null = spectrast.gr.moran_null(adata)                                    # doctest: +SKIP
    >>> spectrast.pl.significance_heatmap(null, adata=adata)                     # doctest: +SKIP
    """
    import pandas as pd
    if value not in ("z", "-log10p"):
        raise ValueError(f"value must be 'z' or '-log10p', got {value!r}")
    _require(null, ["gene", "r_um", "z", "p"], "spectrast.gr.moran_null(adata)")
    if order not in ("scale", "zmax", None):
        raise ValueError(f"order must be 'scale', 'zmax' or None, got {order!r}")
    d = null.copy()
    d["gene"] = d["gene"].astype(str)
    if genes is not None:
        genes = [genes] if isinstance(genes, str) else list(genes)
        missing = sorted(set(genes) - set(d["gene"]))
        if missing:
            raise KeyError(f"genes not in the null table: {missing[:10]}")
        d = d[d["gene"].isin(genes)]
    clipped = False
    if value == "z":
        d["_v"] = d["z"].astype(float)
    else:
        p = d["p"].astype(float).to_numpy()
        clipped = bool((p < 1e-300).any())
        d["_v"] = -np.log10(np.clip(p, 1e-300, None))
    M = d.pivot_table(index="gene", columns="r_um", values="_v", aggfunc="first")
    Z = d.pivot_table(index="gene", columns="r_um", values="z", aggfunc="first")
    M = M.reindex(columns=sorted(M.columns))
    Z = Z.reindex(index=M.index, columns=M.columns)
    r = np.asarray(M.columns, float)
    if max_genes is not None and len(M) > max_genes:
        keep = Z.max(axis=1).sort_values(ascending=False).index[:int(max_genes)]
        M, Z = M.loc[keep], Z.loc[keep]

    rstar = None
    if order == "scale":
        if adata is None:
            raise ValueError("order='scale' orders rows by r*, which needs adata= (after "
                             "spectrast.gr.scale_spectrum(adata)); or pass order='zmax'")
        if "spectrast_char_scale" not in adata.var:
            raise KeyError("adata.var['spectrast_char_scale'] not found: run "
                           "spectrast.gr.scale_spectrum(adata) or spectrast.gr.fingerprint(adata)")
        rs = pd.Series(adata.var["spectrast_char_scale"].to_numpy(float),
                       index=np.asarray(adata.var_names, str)).reindex(M.index)
        o = np.argsort(np.where(np.isfinite(rs.to_numpy()), rs.to_numpy(), np.inf), kind="stable")
        M, Z = M.iloc[o], Z.iloc[o]
        rstar = rs.iloc[o].to_numpy(float)
    elif order == "zmax":
        Zv = np.nan_to_num(Z.to_numpy(float), nan=-np.inf)
        o = np.lexsort((-Zv.max(1), Zv.argmax(1)))
        M, Z = M.iloc[o], Z.iloc[o]

    fig, ax = _u.new_axes(ax, figsize=(4.8, max(3.0, min(9.0, 0.12 * len(M) + 1.5))))
    V = M.to_numpy(float)
    if value == "z":
        vmin, vmax = _u.robust_limits(V, 99, symmetric=True)
        cmap = _u.DIVERGING
    else:
        vmin, vmax = 0.0, _u.robust_limits(V, 99)[1]
        cmap = _u.SEQUENTIAL
    im = ax.imshow(V, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    if rstar is not None:
        fin = np.isfinite(rstar) & (rstar >= r.min()) & (rstar <= r.max())
        cpos = np.interp(np.log(rstar[fin]), np.log(r), np.arange(len(r)))
        ax.scatter(cpos, np.flatnonzero(fin), s=6, c="k", marker="|", linewidths=0.8)
        ax.set_title("black tick: each gene's r*", fontsize=8)
    _radius_ticks(ax, r)
    ax.set_xlabel("radius (µm)")
    if len(M) <= _MAX_ROW_LABELS:
        ax.set_yticks(np.arange(len(M)))
        ax.set_yticklabels(M.index, fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_ylabel(f"{len(M)} genes" + (", ordered by r*" if order == "scale" else
                                       ", ordered by radius of max z" if order == "zmax" else ""))
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("z" if value == "z" else "-log10 p" + (" (p clipped at 1e-300)" if clipped else ""))
    return _u.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# 4. one gene, every statistic
# ---------------------------------------------------------------------------------------------
def gene_card(adata, gene, *, stats=None, radius=None, curves=None, show=None, save=None):
    """One gene on one page: its expression, its six organization curves and its local Moran map.

    Top: the expression map and the per-cell local Moran's I at the gene's Moran r* (or at
    ``radius``), with a circle of that radius on the gene's strongest patch. Bottom: one small panel
    per statistic, the curve on a log-radius axis with r* dashed and its flags in the title.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm["spatial"]``, expression in ``adata.X``.
    gene : str
    stats : sequence of str, optional
        Statistics to draw; default all six of :data:`spectrast.organization.ORG_STATS`.
    radius : float, optional
        Radius of the local map (µm). Default: the gene's Moran r*.
    curves : dict, optional
        The dict from ``spectrast.organization.organization(adata, return_curves=True)``; the
        gene's curves and r* are read from it, reduced against the whole panel exactly as the table
        was. When absent, :func:`~spectrast.organization.organization` is run here on this ONE gene
        (light: one rasterisation of one field); its curves' orientation is then taken from that
        gene alone, which can differ from a whole-panel table for a curve that rises.
    show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.figure.Figure, or None if ``show=True``.

    Notes
    -----
    The local map is the per-cell form on the scalogram's per-radius lattice (see
    :mod:`spectrast.pl`); the curves are on :func:`~spectrast.organization.organization`'s single
    pitch-anchored lattice, so the map's mean sits near, not exactly on, the Moran curve at r*.

    Examples
    --------
    >>> tab, cur = spectrast.organization.organization(adata, return_curves=True)  # doctest: +SKIP
    >>> spectrast.pl.gene_card(adata, "Gfap", curves=cur)                           # doctest: +SKIP
    """
    from ..organization import ORG_STATS, _reduce_curve, organization
    stats = tuple(ORG_STATS if stats is None else ([stats] if isinstance(stats, str) else stats))
    bad = [s for s in stats if s not in ORG_STATS]
    if bad:
        raise ValueError(f"unknown statistics {bad}; choose from {list(ORG_STATS)}")
    if gene not in adata.var_names:
        raise KeyError(f"{gene!r} not in adata.var_names")
    if curves is None:
        _, curves = organization(adata, stats=stats, genes=[gene], return_curves=True)
        src = "computed for this gene"
    else:
        missing = [s for s in stats if s not in curves]
        if missing or "radii_um" not in curves or "genes" not in curves:
            raise KeyError(f"curves lacks {missing or ['radii_um/genes']}: pass the dict from "
                           f"spectrast.organization.organization(adata, stats={stats!r}, return_curves=True)")
        src = "whole-panel curves"
    names = np.asarray(curves["genes"], dtype=str)
    j = int(_u.index_of(names, gene)[0])
    r = np.asarray(curves["radii_um"], float)
    red = {s: _reduce_curve(np.asarray(curves[s], float), r) for s in stats}

    xy = _u.coords(adata)
    v = _u.feature_vector(adata, gene)
    rs_moran = float(red["moran"]["scale_um"][j]) if "moran" in red else np.nan
    if radius is not None:
        rmap, why = float(radius), "given radius"
    elif np.isfinite(rs_moran):
        rmap, why = rs_moran, "Moran r*"
    else:
        rmap = float(2.0 ** np.mean(np.log2(r)))
        why = "no Moran r*; grid centre"
    L, mean_nb, ncnt, _ = _local_moran(xy, v, rmap)

    fig = _u.new_figure((11.0, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])
    top = gs[0].subgridspec(1, 2)
    bot = gs[1].subgridspec(1, len(stats))
    ax_e = fig.add_subplot(top[0])
    ax_l = fig.add_subplot(top[1])
    vmax = float(np.percentile(v[v > 0], 99)) if (v > 0).any() else 1.0
    _u.spatial_scatter(ax_e, xy, v, vmin=0.0, vmax=vmax, title=f"{gene} expression",
                       cbar_label="expression", order_by_value=True)
    _draw_local(ax_l, xy, L, rmap, centre=_circle_centre(xy, v, mean_nb, ncnt),
                title=f"local Moran's I at r = {rmap:.0f} µm ({why}); mean {np.mean(L):.3f}")
    for k, s in enumerate(stats):
        a = fig.add_subplot(bot[k])
        R = red[s]
        rs = float(R["scale_um"][j])
        flags = []
        if not np.isfinite(rs):
            flags.append("no scale")
        if bool(R["window_limited"][j]):
            flags.append("window-limited")
        if bool(R["grid_suspect"][j]):
            flags.append("grid-suspect")
        head = f"{s}: r* {rs:.0f} µm" if np.isfinite(rs) else f"{s}"
        _draw_curve(a, r, np.asarray(curves[s], float)[:, j], rstar=rs, sign=R["orientation"],
                    ylabel=_STAT_LABEL.get(s, s),
                    note=head + (f"\n({', '.join(flags)})" if flags else ""))
        a.yaxis.label.set_size(7)
        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7)
    fig.suptitle(f"{gene}: organization curves ({src})", fontsize=10)
    return _u.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# 5. local Moran map and 6. its filmstrip across radii
# ---------------------------------------------------------------------------------------------
def local_map(adata, feature, radius=None, *, show_curve=True, circle=True, key=_u.SCALOGRAM_KEY,
              ax=None, show=None, save=None):
    """Per-cell local Moran's I of one feature at one radius, with its curve beneath.

    The map is ``z_i * mean_{j in N_r(i)} z_j`` over cells, on the standardised field, with the
    row-standardised disc neighbourhood of radius ``r`` (the cell itself excluded) evaluated on the
    per-radius lattice :func:`spectrast.gr.scale_spectrum` uses. Its mean over cells is the global
    Moran's I of the curve at that radius, so the map shows where in the tissue the curve's value
    comes from. Colour is symmetric about 0 and saturates at the 99.5th percentile of ``|I|``. Nothing
    is written to ``adata``.

    Parameters
    ----------
    adata : AnnData
    feature : str
        A gene, a numeric ``obs`` column, or ``"<obs_key>==<value>"`` for the 0/1 indicator of one
        category (e.g. ``"cell_type==B cell"``).
    radius : float, optional
        Radius in µm. Default: the feature's r*. For a gene in the stored scalogram that is
        ``adata.var["spectrast_char_scale"]``; any other feature has its Moran curve measured here
        (one field, the stored scalogram's radii, or :func:`spectrast.gr.default_radii`).
    show_curve : bool
        Draw the curve beneath the map with r* (dashed), its half-range level (dotted) and the
        map's mean at its radius (open square).
    circle : bool
        Draw a circle of radius ``r`` on the feature's strongest patch: among cells with a
        positive standardised value and at least the median neighbour count, the one whose
        neighbourhood mean is largest.
    key : str
        ``adata.uns`` key of the stored scalogram.
    ax, show, save
        The scanpy plotting convention. With ``ax`` the curve goes into an inset beneath it.

    Returns
    -------
    matplotlib.axes.Axes (the map), or None if ``show=True``.

    Examples
    --------
    >>> spectrast.gr.scale_spectrum(adata)                                       # doctest: +SKIP
    >>> spectrast.pl.local_map(adata, "Gfap")                                   # doctest: +SKIP
    >>> spectrast.pl.local_map(adata, "cell_type==B cell")                      # doctest: +SKIP
    """
    params = _scalogram_params(adata, key)
    need = radius is None or show_curve
    curve, rr, rstar, srcname = _feature_curve(adata, feature, key, need=need)
    if radius is None:
        if not np.isfinite(rstar):
            raise ValueError(f"{feature!r} has no r*: its Moran curve never falls to half its own range "
                             "inside the swept radii; pass radius=")
        r = float(rstar)
    else:
        r = float(radius)
    xy = _u.coords(adata, params.get("basis") or "spatial")
    v = _u.feature_vector(adata, feature, layer=params.get("layer"))
    L, mean_nb, ncnt, _ = _local_moran(xy, v, r, _bin_for(params, rr, r))

    if ax is None:
        import matplotlib.pyplot as plt
        if show_curve:
            fig, (ax, axc) = plt.subplots(2, 1, figsize=(4.6, 5.6), height_ratios=[4, 1.3],
                                          layout="constrained")
        else:
            fig, ax = _u.new_axes(None, figsize=(4.6, 4.2))
            axc = None
    else:
        fig = ax.figure
        axc = ax.inset_axes([0.05, -0.42, 0.9, 0.28]) if show_curve else None
    tag = " (r*)" if radius is None else ""
    _draw_local(ax, xy, L, r, centre=_circle_centre(xy, v, mean_nb, ncnt) if circle else None,
                circle=circle, title=f"{feature}: local Moran's I at r = {r:.0f} µm{tag}")
    if axc is not None:
        _draw_curve(axc, rr, curve, rstar=rstar, r_map=r, map_mean=float(np.mean(L)),
                    note=f"curve ({srcname}); square: map mean {np.mean(L):.3f}")
    return _u.finish(fig, ax, show, save)


def local_filmstrip(adata, feature, radii=None, *, shared_scale=False, key=_u.SCALOGRAM_KEY,
                    show=None, save=None):
    """The same local Moran map at several radii, side by side.

    Parameters
    ----------
    adata : AnnData
    feature : str
        A gene, a numeric ``obs`` column, or ``"<obs_key>==<value>"`` (see :func:`local_map`).
    radii : sequence of float, optional
        Radii in µm. Default: 5 log-spaced radii across the stored scalogram's range (or across
        :func:`spectrast.gr.default_radii` when nothing is stored).
    shared_scale : bool
        One colour scale for every panel (the 99.5th percentile of ``|I|`` over all of them), so the
        panels compare in magnitude; default: each panel its own, so each shows its own pattern.
    key : str
        ``adata.uns`` key of the stored scalogram (for the default radii and lattice).
    show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.figure.Figure, or None if ``show=True``.

    Examples
    --------
    >>> spectrast.pl.local_filmstrip(adata, "Gfap")                              # doctest: +SKIP
    """
    params = _scalogram_params(adata, key)
    stored_r = None
    if key in adata.uns:
        _, stored_r, _ = _u.scalogram(adata, key)
    if radii is None:
        if stored_r is None:
            from ..gr import default_radii
            stored_r = default_radii(adata, params.get("basis") or "spatial")
        radii = np.geomspace(float(np.min(stored_r)), float(np.max(stored_r)), 5)
    radii = np.atleast_1d(np.asarray(radii, float))
    xy = _u.coords(adata, params.get("basis") or "spatial")
    v = _u.feature_vector(adata, feature, layer=params.get("layer"))
    maps = [_local_moran(xy, v, float(r), _bin_for(params, stored_r, float(r)))[0] for r in radii]
    vlim = _u.robust_limits(np.concatenate(maps), 99.5, symmetric=True) if shared_scale else None

    n = len(radii)
    fig = _u.new_figure((3.0 * n + (0.6 if shared_scale else 0.0), 3.4))
    axes = [fig.add_subplot(1, n, k + 1) for k in range(n)]
    sc = None
    for a, r, L in zip(axes, radii, maps):
        sc = _draw_local(a, xy, L, r, circle=False, vlim=vlim, colorbar=not shared_scale,
                         title=f"r = {r:.0f} µm; mean {np.mean(L):.3f}")
    if shared_scale and sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01)
        cb.set_label("local Moran's I")
    fig.suptitle(f"{feature}: local Moran's I across radii", fontsize=10)
    return _u.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# 7. genes embedded by curve shape, 8. curve families
# ---------------------------------------------------------------------------------------------
def _read_curves(adata, gate, genes=None, key=_u.SCALOGRAM_KEY):
    """Range-normalised, oriented curves of the read genes: ``(C (n_genes, n_radii), r, names,
    scale, amp, desc)``; genes with a flat or non-finite curve are dropped."""
    S, r, names, scale, amp, sign = _curve_source(adata, key)
    read, _, desc = _gate_mask(amp, gate)
    sel = read.copy()
    if genes is not None:
        want = np.zeros(len(names), bool)
        want[_u.index_of(names, genes)] = True
        sel &= want
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = _u.range_normalise(sign * S[:, sel]).T
    ok = np.isfinite(C).all(1)
    idx = np.flatnonzero(sel)[ok]
    return C[ok], r, names[idx], np.asarray(scale, float)[idx], np.asarray(amp, float)[idx], desc, idx


def curve_embedding(adata, *, color="scale", method="pca", gate=0.8, genes=None, families=None,
                    ax=None, show=None, save=None):
    """Read genes embedded by the shape of their range-normalised Moran curves.

    Each read gene's curve is rescaled to [0, 1] over its own range, so genes sit together when
    their curves have the same shape whatever their amplitude, and embedded in two dimensions.

    Parameters
    ----------
    adata : AnnData or dict
        After :func:`spectrast.gr.scale_spectrum` / :func:`spectrast.gr.fingerprint` (the stored
        scalogram), or the dict from ``spectrast.organization.organization(adata,
        return_curves=True)`` (its Moran curves).
    color : str or array-like
        ``"scale"`` (r*, log colour scale), ``"amplitude"``, ``"focality"``
        (``adata.var["spectrast_focality"]``), ``"family"`` (needs ``families``), any other
        ``adata.var`` column, or values aligned to all of the source's genes.
    method : {"pca", "umap"}
        PCA by numpy SVD (default), or UMAP when the optional ``umap-learn`` package is installed.
    gate : float, tuple or None
        Amplitude gate; see :func:`scale_plane`.
    genes : sequence of str, optional
        Restrict to these genes (the gate still applies).
    families : Series or array-like, optional
        Family labels, e.g. from ``curve_families(..., return_labels=True)``: a Series indexed by
        gene, or values aligned to the embedded genes.
    ax, show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.axes.Axes, or None if ``show=True``.

    Examples
    --------
    >>> fig, fam = spectrast.pl.curve_families(adata, return_labels=True)        # doctest: +SKIP
    >>> spectrast.pl.curve_embedding(adata, color="family", families=fam)         # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LogNorm
    if method not in ("pca", "umap"):
        raise ValueError(f"method must be 'pca' or 'umap', got {method!r}")
    C, r, names, scale, amp, desc, idx = _read_curves(adata, gate, genes)
    n = len(names)
    if n < 3:
        raise ValueError(f"only {n} read genes with a non-flat curve; lower the gate or pass more genes")
    if method == "pca":
        M = C - C.mean(0)
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        E = U[:, :2] * s[:2]
        ev = s ** 2 / max(float((s ** 2).sum()), 1e-300)
        xl, yl = f"PC1 ({100 * ev[0]:.0f}% of variance)", f"PC2 ({100 * ev[1]:.0f}%)"
    else:
        try:
            import umap
        except ImportError as e:
            raise ImportError("method='umap' needs the optional umap-learn package "
                              "(pip install umap-learn); method='pca' needs nothing extra") from e
        E = umap.UMAP(n_neighbors=min(15, n - 1), random_state=0).fit_transform(C)
        xl, yl = "UMAP 1", "UMAP 2"

    fig, ax = _u.new_axes(ax, figsize=(4.6, 3.8))
    cat = None
    if isinstance(color, str) and color == "family":
        if families is None:
            raise ValueError("color='family' needs families= (e.g. the labels from "
                             "curve_families(..., return_labels=True))")
        cat = (families.reindex(names).to_numpy() if isinstance(families, pd.Series)
               else np.asarray(families))
        if len(cat) != n:
            raise ValueError(f"families has {len(cat)} values; {n} genes are embedded")
    elif isinstance(color, str):
        if color == "scale":
            cv, lab = scale, "r* (µm)"
        elif color == "amplitude":
            cv, lab = amp, "amplitude"
        else:
            col = "spectrast_focality" if color == "focality" else color
            if isinstance(adata, dict) or col not in adata.var:
                hint = (" (run spectrast.gr.fingerprint(adata) or spectrast.gr.focality(adata))"
                        if color == "focality" else "")
                raise KeyError(f"adata.var[{col!r}] not found{hint}")
            cv = pd.Series(adata.var[col].to_numpy(float),
                           index=np.asarray(adata.var_names, str)).reindex(names).to_numpy(float)
            lab = color
    else:
        cv = np.asarray(color, float).ravel()
        total = len(_curve_source(adata)[2])
        if cv.size == total:
            cv = cv[idx]
        elif cv.size != n:
            raise ValueError(f"color has {cv.size} values; expected {total} (all genes) or {n} (embedded)")
        lab = "colour"

    if cat is not None:
        levels = [c for c in pd.unique(cat) if pd.notna(c)]
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        cmap = plt.get_cmap("tab10")
        for k, lv in enumerate(levels):
            m = cat == lv
            ax.scatter(E[m, 0], E[m, 1], s=10, color=cmap(k % 10), linewidths=0, label=str(lv))
        ax.legend(fontsize=7, frameon=False, title="family", title_fontsize=7, markerscale=1.5)
    else:
        ok = np.isfinite(cv)
        if (~ok).any():
            ax.scatter(E[~ok, 0], E[~ok, 1], s=8, c=_u.MUTED, linewidths=0, label="no value")
        norm = None
        if color == "scale" and (cv[ok] > 0).all() and ok.any():
            norm = LogNorm(vmin=float(np.min(cv[ok])), vmax=float(np.max(cv[ok])))
        sc = ax.scatter(E[ok, 0], E[ok, 1], s=10, c=cv[ok], cmap=_u.SEQUENTIAL, norm=norm,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
        cb.set_label(lab)
        if norm is not None:                      # plain-number µm ticks on the log colour scale
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cb.ax.yaxis.set_minor_formatter(NullFormatter())
        if (~ok).any():
            ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"{n} read genes by curve shape ({desc})", fontsize=8)
    return _u.finish(fig, ax, show, save)


def curve_families(adata, *, k=6, gate=0.8, terms=None, seed=0, return_labels=False, show=None,
                   save=None):
    """Read genes grouped into k families of Moran-curve shape, numbered fine to coarse.

    The read genes' curves, each range-normalised to [0, 1], are clustered by k-means
    (:func:`scipy.cluster.vq.kmeans2`, k-means++ start, seeded). Families are numbered 1..k by
    their median r*, finest first. Left: one curve per row, grouped by family and sorted by r*
    within it. Middle: each family's r* distribution (median dot, interquartile bar, 5th-95th
    percentile line). Right, if ``terms`` is given: up to three enrichment terms per family.

    Parameters
    ----------
    adata : AnnData or dict
        After :func:`spectrast.gr.scale_spectrum` / :func:`spectrast.gr.fingerprint`, or the dict
        from ``spectrast.organization.organization(adata, return_curves=True)``.
    k : int
        Number of families. A display choice: the families are close to a binning of r*.
    gate : float, tuple or None
        Amplitude gate; see :func:`scale_plane`.
    terms : DataFrame, optional
        Columns ``family`` (the numbers this function assigns, 1 = finest), ``term``, and ``q``
        and/or ``fold``. Bars show ``fold`` when present (largest three), else ``-log10 q``
        (smallest three q). Compute it on the labels from ``return_labels=True``; the clustering
        is seeded, so a second call with the same arguments reproduces them.
    seed : int
        k-means seed.
    return_labels : bool
        Also return the labels as a Series indexed by gene (name ``"family"``).
    show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.figure.Figure, or ``(Figure, Series)`` with ``return_labels``; None for the
    figure if ``show=True``.

    Examples
    --------
    >>> fig, fam = spectrast.pl.curve_families(adata, k=6, return_labels=True)   # doctest: +SKIP
    >>> spectrast.pl.curve_families(adata, k=6, terms=enrichment_table)          # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.cluster.vq import kmeans2
    C, r, names, scale, amp, desc, _ = _read_curves(adata, gate)
    n = len(names)
    if n < k:
        raise ValueError(f"{n} read genes with a non-flat curve but k={k}; lower k or the gate")
    if terms is not None:
        _require(terms, ["family", "term"], "an enrichment of the families from "
                                            "curve_families(..., return_labels=True)")
        if "q" not in terms.columns and "fold" not in terms.columns:
            raise KeyError("terms needs a 'q' or a 'fold' column")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lab = kmeans2(C, int(k), minit="++", seed=np.random.default_rng(seed))
    present = np.unique(lab)
    med = np.array([np.nanmedian(scale[lab == c]) if np.isfinite(scale[lab == c]).any() else np.inf
                    for c in present])
    rank = {c: i + 1 for i, c in enumerate(present[np.argsort(med, kind="stable")])}
    fam = np.array([rank[c] for c in lab])
    kk = len(present)
    if kk < k:
        warnings.warn(f"k-means left {k - kk} families empty; drawing {kk}", stacklevel=2)
    key_s = np.where(np.isfinite(scale), scale, np.inf)
    o = np.lexsort((key_s, fam))
    Cs, fs, ss = C[o], fam[o], scale[o]

    ncol = 3 if terms is not None else 2
    fig = _u.new_figure((11.5 if terms is not None else 7.6, 6.0))
    gs = fig.add_gridspec(1, ncol, width_ratios=[3.0, 1.4, 2.4][:ncol])
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cs, aspect="auto", interpolation="nearest", cmap=_u.SEQUENTIAL, vmin=0, vmax=1)
    bounds = np.flatnonzero(np.diff(fs)) + 0.5
    for b in bounds:
        ax0.axhline(b, color="w", lw=1.0)
    centres = [(np.flatnonzero(fs == f).min() + np.flatnonzero(fs == f).max()) / 2 for f in range(1, kk + 1)]
    ax0.set_yticks(centres)
    ax0.set_yticklabels([f"{f} (n={int((fs == f).sum())})" for f in range(1, kk + 1)], fontsize=7)
    ax0.set_ylabel("family (genes)")
    ax0.set_title(f"{n} read genes ({desc}), one curve per row", fontsize=8)
    _radius_ticks(ax0, r)
    ax0.set_xlabel("radius (µm)")
    cb = fig.colorbar(im, ax=ax0, shrink=0.6, pad=0.01, location="bottom", aspect=30)
    cb.set_label("Moran's I, scaled to each gene's range")

    ax1 = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("tab10")
    for f in range(1, kk + 1):
        rows = np.flatnonzero(fs == f)
        y = (rows.min() + rows.max()) / 2
        v = ss[rows][np.isfinite(ss[rows])]
        if v.size:
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            ax1.plot([p5, p95], [y, y], color=cmap((f - 1) % 10), lw=1)
            ax1.plot([p25, p75], [y, y], color=cmap((f - 1) % 10), lw=4, solid_capstyle="butt")
            ax1.plot([p50], [y], "o", color="k", ms=4)
    for b in bounds:
        ax1.axhline(b, color="0.85", lw=0.6)
    ax1.set_ylim(n - 0.5, -0.5)
    ax1.set_yticks([])
    ax1.set_title("r*: median, IQR, 5-95%", fontsize=8)
    _radius_axis(ax1, label="r* (µm)")
    fin = ss[np.isfinite(ss)]
    if fin.size:
        ax1.set_xlim(fin.min() / 1.2, fin.max() * 1.2)

    if terms is not None:
        ax2 = fig.add_subplot(gs[2])
        use_fold = "fold" in terms.columns
        ypos, labels, vals, cols = [], [], [], []
        y = 0.0
        heads = []
        for f in range(1, kk + 1):
            tf = terms[terms["family"].astype(int) == f]
            tf = (tf.sort_values("fold", ascending=False) if use_fold else tf.sort_values("q")).head(3)
            heads.append((y, f, len(tf)))
            for _, row in tf.iterrows():
                ypos.append(y)
                t = str(row["term"])
                labels.append(t if len(t) <= 48 else t[:46] + "...")
                vals.append(float(row["fold"]) if use_fold else -np.log10(max(float(row["q"]), 1e-300)))
                cols.append(cmap((f - 1) % 10))
                y += 1
            if tf.empty:
                y += 1
            y += 0.8
        ax2.barh(ypos, vals, color=cols, height=0.7)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(labels, fontsize=7)
        for y0, f, m in heads:
            ax2.annotate(f"{f}" if m else f"{f}  (no terms)", (-0.02, y0 - 0.55),
                         xycoords=("axes fraction", "data"), fontsize=8, ha="right", fontweight="bold")
        ax2.set_ylim(y, -1.2)
        ax2.set_xlabel("fold enrichment" if use_fold else "-log10 q")
        ax2.set_title("terms (top 3 per family)", fontsize=8)
        ax2.yaxis.tick_right()
    fig.suptitle(f"{kk} curve families (k-means, seed {seed}), numbered by median r*", fontsize=10)
    out = _u.finish(fig, fig, show, save)
    if return_labels:
        return out, pd.Series(fam, index=pd.Index(names, name="gene"), name="family")
    return out


# ---------------------------------------------------------------------------------------------
# 9. the cell-type ladder
# ---------------------------------------------------------------------------------------------
def celltype_ladder(table, *, ax=None, show=None, save=None):
    """Cell types placed at the characteristic scale r* of their indicator fields, fine to coarse.

    One row per type, finest at the top, on a log-µm axis; colour gives the amplitude (the top of
    the indicator's Moran curve). Types whose curve never falls to half its range have no r* and
    are listed in a note under the axis.

    Parameters
    ----------
    table : DataFrame
        From :func:`spectrast.organization.celltype_scale`.
    ax, show, save
        The scanpy plotting convention.

    Returns
    -------
    matplotlib.axes.Axes, or None if ``show=True``.

    Examples
    --------
    >>> tab = spectrast.organization.celltype_scale(adata, "cell_type")           # doctest: +SKIP
    >>> spectrast.pl.celltype_ladder(tab)                                        # doctest: +SKIP
    """
    _require(table, ["cell_type", "scale_um", "amplitude"],
             "spectrast.organization.celltype_scale(adata, cell_type_key)")
    t = table.copy()
    x = t["scale_um"].to_numpy(float)
    placed = np.isfinite(x) & (x > 0)
    tp = t[placed].sort_values("scale_um")
    unplaced = t.loc[~placed, "cell_type"].astype(str).tolist()
    n = len(tp)
    fig, ax = _u.new_axes(ax, figsize=(4.4, max(2.4, 0.22 * n + 1.4)))
    if n:
        y = np.arange(n)
        xs = tp["scale_um"].to_numpy(float)
        a = tp["amplitude"].to_numpy(float)
        for yi in y:
            ax.axhline(yi, color="0.9", lw=0.6, zorder=0)
        sc = ax.scatter(xs, y, c=a, cmap=_u.SEQUENTIAL, s=36, edgecolors="k", linewidths=0.4,
                        zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("amplitude (peak Moran's I)")
        ax.set_yticks(y)
        ax.set_yticklabels(tp["cell_type"].astype(str), fontsize=7)
        ax.set_ylim(n - 0.5, -0.5)
        _radius_axis(ax, label="r* of the type's indicator (µm)")
        ax.set_xlim(xs.min() / 1.25, xs.max() * 1.25)
    else:
        ax.set_yticks([])
        ax.set_xlabel("r* of the type's indicator (µm)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if unplaced:
        ax.annotate("no r* (not placed): " + ", ".join(unplaced), (0, -0.16 if n else 0.5),
                    xycoords="axes fraction", fontsize=7, va="top", color="0.3")
    return _u.finish(fig, ax, show, save)

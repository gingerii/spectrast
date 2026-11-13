"""Cell-type ranges, presence and the section report.

Each function draws what a computing step already returned: ``celltype_range`` reads :func:`spectrast.gr.celltype_range`,
``presence`` reads :func:`spectrast.gr.fingerprint_presence` and ``qc`` reads
:func:`spectrast.qc.section_report`. Nothing here recomputes a statistic; ``qc`` rasterises the
cells onto the report's own lattice to show occupancy, which is a count, not a statistic.
"""
from __future__ import annotations

import numpy as np

from . import _utils

_NAN_GREY = "0.88"            # cells / bins with no value


# ---------------------------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------------------------
def _log_colorbar(cb):
    """Plain-number µm ticks on a log colourbar."""
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.yaxis.set_minor_formatter(NullFormatter())


def _bare_map(ax, invert_y=True):
    ax.set_aspect("equal")
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------------------------
# celltype_range
# ---------------------------------------------------------------------------------------------
def _g_columns(table):
    """``(column names, radii_um)`` of the per-annulus ``g_<r>`` columns, in annulus order."""
    cols = [c for c in table.columns if isinstance(c, str) and c.startswith("g_")]
    r = table.attrs.get("radii_um")
    if r is not None and len(r) == len(cols):
        return cols, np.asarray(r, float)
    return cols, np.array([float(c[2:]) for c in cols])


def _peak_sign_and_reason(table, value):
    """Per row: the sign of ``g - 1`` at the peak annulus (+1 enrichment, -1 exclusion, NaN unknown)
    and, where ``value`` is NaN, why: ``"E"`` peak at the window edge / no valid annulus, ``"F"`` the
    peak's ``|g - 1|`` below the floor, ``"W"`` no decay (or crossing) inside the window."""
    from ..gr import G_ZERO, RANGE_FLOOR
    cols, r = _g_columns(table)
    G = table[cols].to_numpy(float) if cols else np.empty((len(table), 0))
    sign = np.full(len(table), np.nan)
    reason = np.full(len(table), "", dtype=object)
    for i, (pk, val) in enumerate(zip(table["r_peak_um"].to_numpy(float), table[value].to_numpy(float))):
        dev = np.nan
        if np.isfinite(pk) and G.shape[1]:
            k = int(np.argmin(np.abs(r - pk)))
            g = G[i, k]
            if np.isfinite(g) and g > G_ZERO:
                dev = g - 1.0
                sign[i] = 1.0 if dev > 0 else -1.0
        if np.isfinite(val):
            continue
        if not np.isfinite(pk):
            reason[i] = "E"
        elif np.isfinite(dev) and abs(dev) < RANGE_FLOOR:
            reason[i] = "F"
        else:
            reason[i] = "W"
    return sign, reason


def celltype_range(table, *, value="r_half_um", ax=None, show=None, save=None):
    """Cell type x cell type heatmap of interaction range, coloured by the sign of the deviation.

    Each cell is one pair from :func:`spectrast.gr.celltype_range`. Hue gives the direction of the
    cross-PCF at its peak annulus — red where the pair is enriched (``g > 1``), blue where it is
    excluded (``g < 1``) — and intensity gives the range in µm on a log scale shared by both hues,
    so a darker cell is a longer range whichever direction it has. Values are printed in the cells
    when there are at most twelve types. Pairs with no range are left blank with a grey code, counted
    beneath the plot: ``F`` the deviation at the peak is below ``gr.RANGE_FLOOR`` (too small to have a
    direction), ``W`` the curve never decays to the level (or, for ``r_cross_um``, never crosses 1)
    inside the window, ``E`` the peak sits at the window edge or the pair has no valid annulus.

    Parameters
    ----------
    table : pandas.DataFrame
        The table :func:`spectrast.gr.celltype_range` returned.
    value : {"r_half_um", "r_peak_um", "r_cross_um"}, optional
        Which length to draw. Default ``"r_half_um"``, the primary range.
    ax : matplotlib.axes.Axes, optional
        Draw into this Axes.
    show : bool, optional
        Call ``plt.show()`` and return ``None``.
    save : str, optional
        Write the figure to this path.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> tab = spectrast.gr.celltype_range(adata, cell_type_key="cell_type")   # doctest: +SKIP
    >>> spectrast.pl.celltype_range(tab)                                       # doctest: +SKIP
    """
    import textwrap
    import matplotlib as mpl
    from matplotlib.colors import LogNorm, ListedColormap
    from ..gr import RANGE_FLOOR, celltype_range_matrix
    need = {"type_a", "type_b", "r_peak_um", value}
    if not hasattr(table, "columns") or not need <= set(table.columns):
        missing = sorted(need - set(getattr(table, "columns", [])))
        raise KeyError(f"not a cell-type range table (missing {missing}): pass the table returned by "
                       f"spectrast.gr.celltype_range(adata, cell_type_key=...)")
    if not len(table):
        raise ValueError("the cell-type range table is empty")
    M = celltype_range_matrix(table, value)
    sign, reason = _peak_sign_and_reason(table, value)
    S = celltype_range_matrix(table.assign(_sign=sign), "_sign").to_numpy(float)
    C = celltype_range_matrix(table.assign(_why=np.arange(len(table), dtype=float)), "_why").to_numpy(float)
    V = M.to_numpy(float)
    levels = list(M.index)
    n = len(levels)

    fin = np.isfinite(V) & np.isfinite(S)
    if fin.any():
        lo, hi = float(np.nanmin(V[fin])), float(np.nanmax(V[fin]))
        hi = hi if hi > lo else lo * 1.5
    else:
        lo, hi = 1.0, 10.0
    norm = LogNorm(vmin=lo, vmax=hi)
    # the light end of each ramp is cut so the shortest range is still distinct from a blank cell
    ramp = np.linspace(0.3, 1.0, 256)
    cmaps = {1.0: ListedColormap(mpl.colormaps["Reds"](ramp)),
             -1.0: ListedColormap(mpl.colormaps["Blues"](ramp))}
    rgba = np.ones((n, n, 4))
    for sgn, cm in cmaps.items():
        sel = fin & (S == sgn)
        rgba[sel] = cm(norm(V[sel]))

    fig, ax = _utils.new_axes(ax, figsize=(1.2 + 0.55 * n + 1.6, 1.4 + 0.5 * n))
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks(range(n), levels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), levels)
    ax.set_xticks(np.arange(-0.5, n), minor=True); ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    counts = {"F": 0, "W": 0, "E": 0}
    for i in range(n):
        for j in range(n):
            if fin[i, j]:
                if n <= 12:
                    lum = 0.2126 * rgba[i, j, 0] + 0.7152 * rgba[i, j, 1] + 0.0722 * rgba[i, j, 2]
                    ax.text(j, i, f"{V[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if lum < 0.5 else "black")
            elif np.isfinite(C[i, j]):
                code = reason[int(C[i, j])]
                if code:
                    ax.text(j, i, code, ha="center", va="center", fontsize=7, color="0.55")
                    if j >= i:
                        counts[code] += 1
    for sgn, label in ((-1.0, "excluded, g < 1"), (1.0, "enriched, g > 1")):   # enriched sits nearest
        if (fin & (S == sgn)).any():
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmaps[sgn])
            cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, fraction=0.05)
            cb.set_label(f"{label}: {value} (µm)")
            _log_colorbar(cb)
    expl = {"F": f"F = |g − 1| at the peak below {RANGE_FLOOR}",
            "W": ("W = never crosses g = 1 inside the window" if value == "r_cross_um"
                  else "W = does not decay to the level inside the window"),
            "E": "E = peak at the window edge or no valid annulus"}
    blank = [f"{expl[k]} ({counts[k]})" for k in ("F", "W", "E") if counts[k]]
    note = ("Blank pairs: " + "; ".join(blank) + ".") if blank else "Every pair has a range."
    ax.set_xlabel(textwrap.fill(note, 70), fontsize=7)
    ax.set_title(f"cell-type range ({value})")
    return _utils.finish(fig, ax, show, save)


# ---------------------------------------------------------------------------------------------
# presence
# ---------------------------------------------------------------------------------------------
_AXIS_LABEL = {
    "scale_peak_moran": "peak multiscale Moran's I",
    "focality": "focality (Gini of local Moran)",
    "texture": "texture (spectral centroid)",
    "orientation": "fraction of tissue oriented",
}


def presence(df, *, axes=None, show=None, save=None):
    """The per-gene evidence behind each tissue-level presence call.

    One panel per axis: the distribution of the per-gene statistic on the tissue against the same
    statistic on the coordinate scramble the test ranked it against (densities, so a multi-scramble
    null is on the same footing). The dashed line is the null's ``1 - alpha`` quantile; the genes
    beyond it are ``n_sig``. Each panel is annotated with ``enrichment`` and ``presence_p`` from the
    table, so the picture and the call can be read together.

    Parameters
    ----------
    df : pandas.DataFrame
        The table :func:`spectrast.gr.fingerprint_presence` returned, with its ``attrs`` intact.
    axes : sequence of str, optional
        Which axes (rows of ``df``) to draw. Default: every row with per-gene values.
    show : bool, optional
        Call ``plt.show()`` and return ``None``.
    save : str, optional
        Write the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> df = spectrast.gr.fingerprint_presence(adata)                  # doctest: +SKIP
    >>> spectrast.pl.presence(df)                                      # doctest: +SKIP
    """
    if not hasattr(df, "columns") or "axis" not in df.columns:
        raise KeyError("not a presence table: pass the table returned by "
                       "spectrast.gr.fingerprint_presence(adata)")
    from matplotlib.ticker import MaxNLocator
    vals = getattr(df, "attrs", {}).get("values")
    if not vals:
        raise ValueError("this presence table carries no per-gene values (df.attrs['values']): it was "
                         "made by an older spectrast, or a pandas operation dropped its attrs. Re-run "
                         "spectrast.gr.fingerprint_presence(adata) and pass its table unchanged.")
    tab = df.set_index("axis")
    if axes is None:
        axes = [a for a in tab.index if a in vals]
    else:
        axes = [axes] if isinstance(axes, str) else list(axes)
        missing = [a for a in axes if a not in vals or a not in tab.index]
        if missing:
            raise KeyError(f"axes {missing} not in this presence table (it has {list(vals)})")
    if not axes:
        raise ValueError("no axis in this table has per-gene values to draw")

    fig = _utils.new_figure((3.3 * len(axes), 2.9))
    axs = fig.subplots(1, len(axes), squeeze=False)[0]
    for ax, name in zip(axs, axes):
        v = vals[name]
        o = np.asarray(v["observed"], float); o = o[np.isfinite(o)]
        nl = np.asarray(v["null"], float); nl = nl[np.isfinite(nl)]
        higher = bool(v.get("higher", True))
        alpha = float(v.get("alpha", 0.05))
        both = np.r_[o, nl]
        if both.size and np.ptp(both) > 0:
            bins = np.linspace(both.min(), both.max(), 31)
        else:
            bins = 10
        if nl.size:
            ax.hist(nl, bins=bins, density=True, color=_utils.MUTED, label="coordinate scramble")
        if o.size:
            ax.hist(o, bins=bins, density=True, histtype="step", linewidth=1.5,
                    color=_utils.HIGHLIGHT, label="tissue")
        if nl.size:
            tau = float(np.quantile(nl, 1 - alpha)) if higher else float(np.quantile(nl, alpha))
            ax.axvline(tau, color="0.3", linestyle="--", linewidth=0.8)
        r = tab.loc[name]
        call = "present" if bool(r["present"]) else ("underpowered" if bool(r["underpowered"]) else "not present")
        ax.set_title(f"{name}: {call}", fontsize=9)
        txt = (f"n_sig {int(r['n_sig'])} of {int(r['n_tested'])}\n"
               f"enrichment {float(r['enrichment']):.2f}x\n"
               f"presence p {float(r['presence_p']):.2g}")
        if not higher:
            txt += "\n(structure = low values)"
        ax.text(0.98 if higher else 0.02, 0.97, txt, transform=ax.transAxes, fontsize=7,
                ha="right" if higher else "left", va="top")
        ax.set_xlabel(_AXIS_LABEL.get(name, name))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel("density")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(loc="center right" if bool(vals[axes[0]].get("higher", True)) else "center left",
                  fontsize=7, frameon=False)
    return _utils.finish(fig, fig, show, save)


# ---------------------------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------------------------
def _occupancy(adata, lattice, basis):
    """Cells per bin on the report's lattice (the same origin and bin the report's null used)."""
    xy = _utils.coords(adata, basis)
    if len(xy) != int(lattice["n_cells"]):
        raise ValueError(f"adata has {len(xy):,} cells but the report was computed on "
                         f"{int(lattice['n_cells']):,}: pass the AnnData the report was made from")
    b = float(lattice["bin_um"])
    xyz = xy - xy.min(0)
    col = (xyz[:, 0] / b).astype(int); row = (xyz[:, 1] / b).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    cnt = np.zeros((H, W))
    np.add.at(cnt, (row, col), 1.0)
    return cnt


def _text_panel(ax, title, lines):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8,
            linespacing=1.5)


def qc(report, *, adata=None, show=None, save=None):
    """One-page diagnostic of a :func:`spectrast.qc.section_report`.

    Panels: the lattice numbers (cells, occupied bins, bin size, cells per bin, the radius window
    against the cell pitch) and the significance summary; the occupancy map (cells per bin on the
    report's lattice) when ``adata`` is given; the saturation curve, the fraction of genes significant
    at each radius, with the ``qc.SATURATION_WARN`` line above which significance has stopped
    discriminating; the design floor when the report has one; and the report's notes, wrapped, at
    the bottom. Like the report, the page reports quantities and issues no verdict.

    Parameters
    ----------
    report : dict
        What :func:`spectrast.qc.section_report` returned.
    adata : AnnData, optional
        The section the report was made from; adds the occupancy map.
    show : bool, optional
        Call ``plt.show()`` and return ``None``.
    save : str, optional
        Write the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> rep = spectrast.qc.section_report(adata)                       # doctest: +SKIP
    >>> spectrast.pl.qc(rep, adata=adata)                              # doctest: +SKIP
    """
    import textwrap
    import matplotlib as mpl
    from matplotlib.colors import LogNorm, Normalize
    from ..qc import SATURATION_WARN
    if not isinstance(report, dict) or "lattice" not in report or "saturation" not in report:
        raise KeyError("not a section report: pass the dict returned by spectrast.qc.section_report(adata)")
    L, S = report["lattice"], report["saturation"]
    if "by_radius" not in S:
        raise KeyError("this report has no per-radius significance (report['saturation']['by_radius']): "
                       "it was made by an older spectrast; re-run spectrast.qc.section_report(adata)")
    prov = report.get("provenance", {}) or {}
    basis = prov.get("basis") or "spatial"
    design = report.get("design")
    cov = report.get("covariates")

    notes = list(L.get("notes", [])) + list(S.get("notes", []))
    if design and not design.get("fdr_reachable", True):
        notes.append("An empty table from this design means no test was performed, not that there is "
                     "no effect.")
    body = "\n".join(textwrap.fill(f"• {n}", 150) for n in notes) if notes else "No notes."
    h_notes = 0.12 + 0.065 * (body.count("\n") + 1)

    if adata is not None:
        mosaic = [["text", "map"], ["sat", "extra" if (design or cov) else "map"], ["notes", "notes"]]
        heights = [1.0, 1.0, h_notes]
    else:
        mosaic = [["text", "sat"]] + ([["extra", "extra"]] if (design or cov) else []) + [["notes", "notes"]]
        heights = [1.0] + ([0.55] if (design or cov) else []) + [h_notes]
    fig = _utils.new_figure((10.0, 3.0 * sum(heights) + 0.4))
    ax = fig.subplot_mosaic(mosaic, height_ratios=heights)

    # lattice numbers
    pitch = float(L.get("pitch_um", np.nan))
    ratio = L["r_min_um"] / pitch if np.isfinite(pitch) and pitch > 0 else np.nan
    lines = [
        f"cells                 {L['n_cells']:,}",
        f"occupied bins         {L['n_bins_occupied']:,}",
        f"bin                   {L['bin_um']:.2f} µm",
        f"cells per bin         {L['cells_per_bin']:.2f}"
        + ("  (above the analytic-null limit)" if L.get("cells_per_bin_high") else ""),
        f"radius window         {L['r_min_um']:.1f}–{L['r_max_um']:.1f} µm, {L['n_radii']} radii",
        f"median cell pitch     {pitch:.1f} µm",
        f"smallest radius       {ratio:.2f} x pitch"
        + ("  (below the pitch)" if L.get("r_min_below_pitch") else ""),
        "",
        f"significant at α={S['alpha']}:  {S['frac_gene_radius_significant']:.1%} of "
        f"{S['n_cells_tested']:,} gene × radius",
        f"                      {S['frac_genes_significant_any_radius']:.1%} of {S['n_genes']:,} genes "
        f"at some radius",
        f"median |I|            {S['median_abs_I_all']:.4f} overall, "
        f"{S['median_abs_I_significant']:.4f} significant",
    ]
    _text_panel(ax["text"], "Lattice and significance", lines)
    ax["text"].texts[-1].set_family("monospace")

    # occupancy map
    if adata is not None:
        cnt = _occupancy(adata, L, basis)
        a = ax["map"]
        img = np.ma.masked_equal(cnt, 0)
        cmap = mpl.colormaps[_utils.SEQUENTIAL].copy()
        cmap.set_bad(_NAN_GREY)
        hi = max(float(cnt.max()), 2.0)
        norm = LogNorm(vmin=1, vmax=hi) if hi > 10 else Normalize(vmin=1, vmax=hi)
        im = a.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        _bare_map(a, invert_y=False)          # row 0 at the top: the orientation of the cell maps
        cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.01)
        cb.set_label("cells per bin")
        if hi > 10:
            _log_colorbar(cb)
        else:
            cb.set_ticks(np.arange(1, int(hi) + 1))
        a.set_title(f"occupancy, {L['bin_um']:.1f} µm bins (grey = empty)", fontsize=10)

    # saturation curve
    a = ax["sat"]
    br = S["by_radius"]
    r = np.asarray(br["radius_um"], float); f = np.asarray(br["frac_significant"], float)
    a.plot(r, f, "o-", color="0.2", markersize=3.5, linewidth=1.2, label="genes significant")
    a.axhline(SATURATION_WARN, color=_utils.HIGHLIGHT, linestyle="--", linewidth=1.0,
              label=f"SATURATION_WARN = {SATURATION_WARN:g}")
    if np.isfinite(pitch):
        a.axvline(pitch, color="0.5", linestyle=":", linewidth=1.0, label="cell pitch")
    _utils.radius_axis(a, np.r_[r, pitch] if np.isfinite(pitch) else r)
    a.set_ylim(0, 1.03)
    a.set_ylabel(f"fraction of genes with p < {S['alpha']}")
    a.set_title("Saturation by radius", loc="left", fontsize=10)
    a.legend(fontsize=7, frameon=False, loc="lower left")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    # design floor and covariate attribution
    if design or cov:
        dl = []
        if design:
            dl += [f"{'units':<24}{design['n_units_A']} vs {design['n_units_B']}"
                   + (" (paired)" if design.get("paired") else ""),
                   f"{'relabellings':<24}{int(design['n_partitions']):,}",
                   f"{'smallest attainable p':<24}{design['p_floor']:.3g}",
                   f"FDR < {design['alpha']} over {int(design['n_features']):,} features "
                   + ("is reachable" if design["fdr_reachable"] else "is NOT reachable")]
        if cov:
            if dl:
                dl.append("")
            dl.append("evidence attributable to a section-wide trend (median |z| free → conditioned):")
            for c in cov:
                dl.append(f"  {c['covariate']}: {c['median_z_free']:.2f} → {c['median_z_conditioned']:.2f}"
                          f"  ({c['median_frac_attributable']:.0%}; 90th pct "
                          f"{c['q90_frac_attributable']:.0%})")
        _text_panel(ax["extra"], "Design floor" if design else "Covariates", dl)
        ax["extra"].texts[-1].set_family("monospace")

    # notes
    _text_panel(ax["notes"], "Notes", [body])
    ax["notes"].texts[-1].set_fontsize(7.5)
    return _utils.finish(fig, fig, show, save)

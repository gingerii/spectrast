"""spectrast.qc — what a section can and cannot support, reported before it is interpreted.

THIS IS A DIAGNOSTIC AND NOT A GATE, and that is a measured position rather than a stylistic one.

The obvious version of this module — sweep counts and density per bin, flag "gradient-dominated"
sections, drop them — was proposed and then **refuted by our own calibration study**. Re-testing the
genes a pooled null gained at ~300 µm under a per-gene null conditioned on ten quantile strata of
total counts cut the *evidence* by about a third and changed
**almost no calls**: 64 → 58, 105 → 92, 200 → 200 across three sections, and the result was
insensitive to how the strata were built. The conclusion recorded there is verbatim:

    "A z that drops from 6.5 to 4.5 under count-strata says a third of a gene's evidence is the
    section's count trend. That is a useful number to report per gene; it is not a gate."

So nothing here returns a pass, a fail, or a recommendation to discard a section. Every block reports
a quantity and what is known about how to read it. If a future caller wants to threshold on one of
these numbers, that is their decision to defend, not one this module makes for them.

**The block that justifies the module** is the saturation one. On sections of 10^5 cells, 95–99 % of
gene × radius cells are significant under a *calibrated* null and 99–100 % of genes are significant at
some radius — a Moran's I of a few thousandths clears the bar. A user thresholding a spatially-variable
gene list on significance is, on a section that size, selecting close to arbitrarily. They should be
told that about **their own section** before they threshold, and that is what this reports.
"""
from __future__ import annotations

import numpy as np

__all__ = ["section_report", "format_report", "SATURATION_WARN"]

#: Above this fraction of significant gene × radius cells, significance has stopped discriminating on
#: the section and the effect size is the quantity to rank on. 0.5 is a floor, not a discovery: the
#: measured rate on 80k–390k-cell sections is 0.95–0.99.
SATURATION_WARN = 0.5


def _lattice_block(attrs, pitch_um):
    """Block 1 — the measuring stick, and Block 2 — where it stops being trustworthy."""
    radii = np.asarray(attrs["radii_um"], float)
    cpb = float(attrs["cells_per_bin"])
    r_min = float(radii.min())
    out = dict(
        n_cells=int(attrs["n_cells"]), n_bins_occupied=int(attrs["n_bins"]),
        bin_um=float(attrs["bin_um"]), cells_per_bin=cpb,
        n_radii=int(len(radii)), r_min_um=r_min, r_max_um=float(radii.max()),
        pitch_um=(float(pitch_um) if pitch_um is not None else np.nan),
    )
    notes = []
    # A radius below the tissue's own spacing does not measure the tissue. This is not hypothetical:
    # it is how the Trem2 capstone's DAM signature ended up with a characteristic length that was a
    # property of the lattice, and it is why the pitch-anchored default starts at 2 bins.
    if pitch_um is not None and np.isfinite(pitch_um) and r_min < float(pitch_um):
        out["r_min_below_pitch"] = True
        notes.append(
            f"the smallest swept radius ({r_min:.1f} µm) is BELOW the median cell spacing "
            f"({pitch_um:.1f} µm). A neighbourhood smaller than the spacing encloses less than one "
            f"cell on average, so any length read at that radius describes the lattice, not the "
            f"tissue. No choice of bin fixes this — only a higher-resolution assay does.")
    else:
        out["r_min_below_pitch"] = False
    # The analytic null permutes BINS; the permutation null permutes cells and re-averages, which
    # adds variance the analytic form cannot see. Measured at sigma_analytic/sigma_perm = 0.92 on a
    # ~2 cells/bin lattice (0.87 at the finest radius), 1.00 at ~1 cell/bin.
    from .gr import NULL_CELLS_PER_BIN_WARN
    out["cells_per_bin_high"] = bool(cpb > NULL_CELLS_PER_BIN_WARN)
    if out["cells_per_bin_high"]:
        notes.append(
            f"{cpb:.1f} cells per bin (> {NULL_CELLS_PER_BIN_WARN}). The analytic null's sigma runs "
            f"8–12 % low in this regime because it permutes bins while the permutation null permutes "
            f"cells and re-averages. Prefer method='permutation' for a marginal call here.")
    out["notes"] = notes
    return out


def _saturation_block(df, alpha):
    """Block 3 — has significance stopped discriminating on this section?"""
    p = df["p"].to_numpy(float)
    sig = p < alpha
    per_gene = df.assign(_s=sig).groupby("gene")._s.any()
    I = np.abs(df["I"].to_numpy(float))
    out = dict(
        alpha=float(alpha),
        n_genes=int(df["gene"].nunique()), n_cells_tested=int(len(df)),
        frac_gene_radius_significant=float(np.mean(sig)),
        frac_genes_significant_any_radius=float(per_gene.mean()),
        median_abs_I_significant=float(np.median(I[sig])) if sig.any() else np.nan,
        median_abs_I_all=float(np.median(I)),
    )
    out["saturated"] = bool(out["frac_gene_radius_significant"] >= SATURATION_WARN)
    # the same significance cells, split by radius: where along the sweep significance saturates
    by_r = df.assign(_s=sig).groupby("r_um", sort=True)._s.mean()
    out["by_radius"] = dict(radius_um=[float(r) for r in by_r.index],
                            frac_significant=[float(v) for v in by_r.to_numpy()])
    out["notes"] = []
    if out["saturated"]:
        out["notes"].append(
            f"{out['frac_gene_radius_significant']:.0%} of gene × radius cells and "
            f"{out['frac_genes_significant_any_radius']:.0%} of genes are significant at "
            f"alpha={alpha}. The median |I| among the significant cells is "
            f"{out['median_abs_I_significant']:.4f}. Significance is not the discriminating quantity "
            f"on this section — a gene list thresholded on it is close to unfiltered. Rank on the "
            f"effect size and use significance as a floor beneath that ranking, not as the ranking.")
    return out


def _covariate_block(adata, covariates, genes, radii, bin_um, layer, basis, n_perm, n_strata, seed):
    """Block 4 — how much of each gene's evidence is a section-wide trend. NOT a filter."""
    from . import gr as _gr
    rows = []
    for cov in covariates:
        base = _gr.moran_null(adata, genes=genes, method="permutation", n_perm=n_perm,
                              radii=radii, bin_um=bin_um, layer=layer, basis=basis, seed=seed,
                              key_added="_qc_n0")
        cond = _gr.moran_null(adata, genes=genes, method="permutation", n_perm=n_perm,
                              covariate=cov, n_strata=n_strata, radii=radii, bin_um=bin_um,
                              layer=layer, basis=basis, seed=seed, key_added="_qc_n1")
        key = ["gene", "r_um"]
        m = base[key + ["z"]].merge(cond[key + ["z"]], on=key, suffixes=("_free", "_cond"))
        keep = np.isfinite(m.z_free) & np.isfinite(m.z_cond) & (m.z_free.abs() > 0)
        m = m[keep]
        if not len(m):
            continue
        # the fraction of |z| that the covariate accounts for, per gene x radius
        frac = 1.0 - (m.z_cond.abs() / m.z_free.abs())
        rows.append(dict(covariate=str(cov), n_cells=int(len(m)),
                         median_z_free=float(m.z_free.abs().median()),
                         median_z_conditioned=float(m.z_cond.abs().median()),
                         median_frac_attributable=float(np.median(frac)),
                         q90_frac_attributable=float(np.quantile(frac, 0.9))))
    return rows


def section_report(adata, *, genes=None, radii=None, bin_um=None, layer=None, basis="spatial",
                   alpha=0.05, covariates=(), covariate_genes=None, n_perm=199, n_strata=10,
                   n_units_A=None, n_units_B=None, paired=False, seed=0):
    """What this section can and cannot support. Returns a dict of blocks; renders with
    :func:`format_report`.

    Parameters
    ----------
    adata : AnnData
        One section.
    genes : sequence, optional
        Features for the saturation block. Default: every gene. The analytic null costs about one
        scalogram for a whole transcriptome, so this is usually affordable as-is.
    radii : sequence of int, optional
        Radii in lattice BINS, not in micrometres; non-integer values are truncated to whole bins.
        ``None`` (default) uses 19 log-spaced radii from 2 bins to 40 × the median cell spacing,
        rounded to whole bins with repeats removed. As in :func:`spectrast.gr.moran_null`.
    bin_um : float, optional
        Lattice bin side, in the units of ``adata.obsm[basis]``. ``None`` (default) uses 0.5 × the
        median cell spacing, coarsened with a warning if the section would need more than four
        million bins.
    layer : str, optional
        Read expression from ``adata.layers[layer]``. ``None`` (default) uses ``adata.X``.
    basis : str, optional
        Key in ``adata.obsm`` holding the ``(n_obs, 2)`` coordinates. Default ``"spatial"``.
    alpha : float, optional
        Significance level of the saturation block (a gene × radius cell counts as significant
        when ``p < alpha``), and the FDR level of the design block. Default ``0.05``.
    covariates : sequence, optional
        Covariates to attribute evidence to: a list whose items are ``obs`` column names or
        per-cell arrays, e.g. ``["total_counts"]``. Wrap a single name in a list; a bare string is
        read one character at a time. Each covariate is smoothed at the largest swept radius and
        cut into ``n_strata`` quantile strata. **Empty by default because this block is
        expensive**: it runs the permutation null twice per covariate, at 2–10 s per gene. Pair
        it with ``covariate_genes``.
    covariate_genes : sequence, optional
        Restrict the covariate block to these features. Strongly recommended when ``covariates`` is
        set; ``genes`` is used when this is ``None``.
    n_perm : int, optional
        Permutations per gene in the covariate block. The saturation block uses the analytic null
        and draws none. Default ``199``.
    n_strata : int, optional
        Number of quantile strata each smoothed covariate is cut into. Default ``10``.
    n_units_A, n_units_B : int, optional
        Replicate counts, if the section is part of a design. Adds the design-floor block, which is
        pure arithmetic and touches no data. Both must be given; one alone adds nothing.
    paired : bool, optional
        Report the floor of a paired (sign-flip) design; then ``n_units_A`` must equal
        ``n_units_B``. Default ``False``. See :func:`spectrast.diff.design_floor`.
    seed : int, optional
        Seed for the covariate block's permutation nulls (gene ``j`` uses ``seed + j``, with ``j``
        its column in ``adata.var``). The analytic null only records it. Default ``0``.

    Returns
    -------
    dict
        ``lattice``, ``saturation``, optionally ``covariates`` and ``design``, plus ``provenance``.
        ``saturation["by_radius"]`` splits the significance fraction by radius: ``radius_um`` and
        ``frac_significant`` (the fraction of genes with ``p < alpha`` at that radius), the curve
        :func:`spectrast.pl.qc` draws.

        * ``lattice``: ``n_cells``, ``n_bins_occupied``, ``bin_um``, ``cells_per_bin``,
          ``n_radii``, ``r_min_um``, ``r_max_um``, ``pitch_um`` (median cell spacing, ``NaN``
          with three cells or fewer), ``r_min_below_pitch``, ``cells_per_bin_high`` (above
          :data:`spectrast.gr.NULL_CELLS_PER_BIN_WARN`) and ``notes`` (list of str).
        * ``saturation``: ``alpha``, ``n_genes``, ``n_cells_tested`` (gene × radius cells),
          ``frac_gene_radius_significant``, ``frac_genes_significant_any_radius``,
          ``median_abs_I_significant`` (``NaN`` when none is significant), ``median_abs_I_all``,
          ``saturated`` (the significant fraction is at least :data:`SATURATION_WARN`),
          ``by_radius`` and ``notes``.
        * ``covariates``: a list with one dict per covariate: ``covariate``, ``n_cells`` (gene ×
          radius cells compared), ``median_z_free``, ``median_z_conditioned``,
          ``median_frac_attributable`` and ``q90_frac_attributable``, where the attributable
          fraction is ``1 - abs(z_conditioned) / abs(z_free)``. A covariate with no comparable
          cell is left out.
        * ``design``: the :func:`spectrast.diff.design_floor` dict plus ``n_units_A``,
          ``n_units_B`` and ``paired``. Its ``n_features`` is the saturation block's ``n_genes``.
        * ``provenance``: ``spectrast_backend``, ``spectrast_null_method``,
          ``spectrast_null_kind``, ``bin_um``, ``radii_um``, ``n_cells``, ``n_bins``,
          ``cells_per_bin``, ``layer`` and ``basis``.

        Radii and lengths are in the units of ``adata.obsm[basis]``. The underlying
        :func:`spectrast.gr.moran_null` calls also write into ``adata``: ``_qc_a0_zmax`` and
        ``_qc_a0_r_zmax_um`` in ``adata.var`` and ``_qc_a0_params`` in ``adata.uns``, and with
        ``covariates`` the same keys under ``_qc_n0`` and ``_qc_n1``.

    Notes
    -----
    Nothing here is a pass/fail. See the module docstring for why: the one gate this module could
    plausibly have offered — drop sections whose signal tracks a technical covariate — was measured
    and does not change calls.
    """
    from . import gr as _gr
    from .gr import _radius_from_xy

    xy = np.asarray(adata.obsm[basis], float)
    pitch = float(_radius_from_xy(xy, factor=1.0)) if len(xy) > 3 else np.nan

    df = _gr.moran_null(adata, genes=genes, method="analytic", radii=radii, bin_um=bin_um,
                        layer=layer, basis=basis, seed=seed, key_added="_qc_a0")
    rep = {
        "lattice": _lattice_block(df.attrs, pitch),
        "saturation": _saturation_block(df, alpha),
        "provenance": {k: df.attrs.get(k) for k in
                       ("spectrast_backend", "spectrast_null_method", "spectrast_null_kind",
                        "bin_um", "radii_um", "n_cells", "n_bins", "cells_per_bin", "layer",
                        "basis")},
    }
    if covariates:
        rep["covariates"] = _covariate_block(
            adata, covariates, covariate_genes if covariate_genes is not None else genes,
            radii, bin_um, layer, basis, n_perm, n_strata, seed)
    if n_units_A is not None and n_units_B is not None:
        from .diff import design_floor
        # design_floor reports the arithmetic; the unit counts are the caller's and are carried
        # alongside so the rendered block can name the design it describes
        rep["design"] = dict(design_floor(int(n_units_A), int(n_units_B), paired=bool(paired),
                                          alpha=alpha,
                                          n_features=int(rep["saturation"]["n_genes"])),
                             n_units_A=int(n_units_A), n_units_B=int(n_units_B),
                             paired=bool(paired))
    return rep


def format_report(rep, width=94):
    """Render :func:`section_report` as text. Reports quantities; issues no verdict.

    Parameters
    ----------
    rep : dict
        The dict returned by :func:`section_report`. ``lattice`` and ``saturation`` are required;
        the ``covariates`` and ``design`` blocks are printed when present.
    width : int, optional
        Width, in characters, of the rules and the centred title. Long lines are not wrapped.
        Default ``94``.

    Returns
    -------
    str
        The report as one multi-line string, ready to ``print``.

    Examples
    --------
    >>> rep = spectrast.qc.section_report(adata)                 # doctest: +SKIP
    >>> print(spectrast.qc.format_report(rep))                   # doctest: +SKIP
    """
    L, S = rep["lattice"], rep["saturation"]
    out = ["=" * width, "spectrast — what this section can support".center(width), "=" * width]
    out.append(f"\nLATTICE  {L['n_cells']:,} cells · {L['n_bins_occupied']:,} occupied bins · "
               f"bin {L['bin_um']:.2f} µm · {L['cells_per_bin']:.2f} cells/bin")
    out.append(f"         {L['n_radii']} radii, {L['r_min_um']:.1f}–{L['r_max_um']:.1f} µm · "
               f"median cell spacing {L['pitch_um']:.1f} µm")
    for n in L["notes"]:
        out.append(f"  [!]    {n}")
    out.append(f"\nSIGNIFICANCE  {S['frac_gene_radius_significant']:.1%} of "
               f"{S['n_cells_tested']:,} gene × radius cells significant at α={S['alpha']}; "
               f"{S['frac_genes_significant_any_radius']:.1%} of {S['n_genes']:,} genes "
               f"at some radius")
    out.append(f"              median |I| overall {S['median_abs_I_all']:.4f}, "
               f"among significant {S['median_abs_I_significant']:.4f}")
    for n in S["notes"]:
        out.append(f"  [!]    {n}")
    if rep.get("covariates"):
        out.append("\nEVIDENCE ATTRIBUTABLE TO A SECTION-WIDE TREND  (a number to report, not a gate)")
        for c in rep["covariates"]:
            out.append(f"         {c['covariate']}: median |z| {c['median_z_free']:.2f} → "
                       f"{c['median_z_conditioned']:.2f} conditioned "
                       f"({c['median_frac_attributable']:.0%} of the evidence; "
                       f"90th pct {c['q90_frac_attributable']:.0%})")
    if rep.get("design"):
        d = rep["design"]
        out.append(f"\nDESIGN   {d['n_units_A']} vs {d['n_units_B']} units → {d['n_partitions']:,} "
                   f"relabellings, smallest attainable p {d['p_floor']:.3g}")
        out.append(f"         FDR<{d['alpha']} over {d['n_features']:,} features "
                   f"{'IS' if d['fdr_reachable'] else 'is NOT'} reachable"
                   + ("" if d["fdr_reachable"] else
                      " — an empty table from this design means no test was performed, "
                      "not that there is no effect"))
    out.append("\n" + "-" * width)
    out.append("No section is passed or failed here. The covariate attribution in particular was "
               "measured\nnot to change calls and must not be used "
               "to discard sections.")
    out.append("-" * width)
    return "\n".join(out)

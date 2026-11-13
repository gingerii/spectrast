"""One reduction, one null, two criteria — the unified organization API.

Supersedes the ad-hoc split in which some curves were reduced per gene and others only
presence-tested, and in which "fingerprint" named both a privileged subset of axes and the
per-gene reduction in general. Every curve reduces the same way here; which reductions are worth
SHOWING and which are worth TESTING becomes a measured property in the returned table rather than a
definition in a docstring.

THE TWO CRITERIA, and why both are needed. A statistic earns a place for one of two different jobs:

``separates_genes``
    Does the reduction tell genes apart WITHIN a section, against that section's own coordinate
    scramble? This is what a display axis needs.
``varies_across_tissues``
    Does the statistic's presence differ BETWEEN sections? This is what a tissue-level descriptor
    needs, and a statistic that is uniform everywhere is useless for it however well it separates.

**These are not computed by the same call, because they are not measurable from the same data.**
``separates_genes`` is a within-section quantity and comes from :func:`presence`. ``varies_across_tissues``
is a between-section quantity and comes from :func:`presence_variation`, which refuses fewer than two
sections rather than returning a column it cannot fill. A single-section call that claimed to report
cross-tissue variation would be reporting nothing.

WHAT THE CRITERIA RETURN IS NOT A RANKING, AND NOT ORTHOGONAL EITHER. Measured over nine banked
sections (2026-10-01) the two criteria are strongly OPPOSED: every statistic that separates genes is
stable across tissues (Moran CV 5.8%, Geary 14.7%, General G 15.5%) and every statistic that fails to
separate is highly variable (LOSH 55.0%, texture 56.3%, focality 55.3%). No statistic occupies the
off-diagonal. A statistic's weakness as a gene-level axis and its strength as a tissue-level
discriminator are, on this evidence, the same fact rather than two independent ones.

EVERY REDUCTION SHIPS ITS OWN DISTRUST FLAG. A reduction whose failure mode is to return a plausible
micrometre value cannot be audited after the fact, and this toolkit was caught twice in development: the
focality landmark returned one for most genes and then failed to reproduce across sections, and LOSH
returns one for 100% of genes while 21% of them sit within ±10% of the radius grid's own centre of
mass on the one section where that coverage was the argument. So ``grid_suspect`` travels with every
scale, and ``scale_supported`` says whether the half-maximum was actually crossed inside the window
rather than extrapolated to reach it.

``grid_suspect`` alone is not enough, and General G is the counter-example that proves it: its curve
is cumulative, so flipping it to the common decaying convention yields a confident half-maximum --
100% "supported", 0% grid-suspect -- for a quantity that has no characteristic scale at all. The
artefact is not centred on the grid's centre of mass; it is placed by where the sweep was cut. Hence
``window_limited``, which asks whether the curve actually flattened inside the window. On the banked
sections it flags General G for essentially every gene and leaves the other five statistics alone.

:func:`fingerprint` is NOT redefined in terms of this module. It returns a gene × axis table and four
figure generators consume that shape; aliasing it here would break them to no purpose. It stays as it
is, and this module is the route for new work.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Every curve this module knows how to reduce. All six share one rasterization and one transform.
ORG_STATS = ("moran", "geary", "general_g", "losh", "focality", "texture")

#: A scale landing this close (fractionally) to the radius grid's centre of mass is flagged.
GRID_SUSPECT_TOL = 0.10

#: ``frac_outside`` must clear this multiple of its own chance level, ``2/(n_perm+1)``, for a
#: statistic to count as separating genes. The multiple is why :func:`presence` refuses a small
#: ``n_perm``: the threshold would exceed 1 and the criterion could never be met.
SEPARATES_MULTIPLE = 3.0

#: A curve whose LAST radius step still carries this fraction of its total range has not flattened
#: inside the window, so its half-maximum is placed by where the sweep was cut rather than by the
#: tissue. Calibrated on three banked sections: General G, which is cumulative and saturates far
#: outside any usable window, sits at 0.197-0.218; every other statistic sits at 0.013-0.061.
#: Monotonicity is NOT the right test here and was rejected -- LOSH is 99.7-100% monotone and Moran
#: 74-96%, so a monotonicity flag condemns two statistics whose scales are sound.
WINDOW_LIMITED_TOL = 0.15


def _orient(C):
    """Flip a curve to a common DECAYING convention, and say which way it ran.

    Data-driven rather than a per-statistic table, because the direction is a property of the
    statistic AND the tissue: Geary rises where Moran falls by construction, but LOSH and texture do
    not have a direction fixed a priori. Returns ``(oriented, sign)`` with ``sign = -1`` where the
    curve was rising.
    """
    C = np.asarray(C, float)
    lo = np.nanmedian(C[0]); hi = np.nanmedian(C[-1])
    sign = 1.0 if lo >= hi else -1.0
    return sign * C, sign


def _reduce_curve(C, radii):
    """The ONE reduction, applied to every statistic identically.

    ``amplitude`` is the peak of the oriented curve and ``scale_um`` the radius at which it falls
    halfway to its own minimum, interpolated in log radius. Both are read off the curve's own range
    rather than a 0-1 rescaling, so the amplitude survives.

    A gene whose curve never crosses its own half-maximum inside the window gets ``scale_um = nan``
    and ``scale_supported = False`` -- NOT an extrapolated value. This is the difference between a
    reduction that reports its own failure and one that returns a plausible number regardless.

    Radii where a curve is undefined (NaN) are skipped: the crossing is interpolated between the
    last defined radius above the half-maximum and the first radius at or below it.

    Parameters
    ----------
    C : array, shape (n_radii, n_genes)
        One statistic's scalogram, in the orientation it was computed.
    radii : array, shape (n_radii,)
        Radii in coordinate units.

    Returns
    -------
    dict of arrays, each shape (n_genes,), plus the scalar ``orientation``.
    """
    C = np.asarray(C, float)
    radii = np.asarray(radii, float)
    if C.shape[0] != radii.size:
        raise ValueError(f"curve has {C.shape[0]} radii, radii has {radii.size}")
    if radii.size < 2:
        raise ValueError(f"a curve needs at least two radii to reduce, got {radii.size}")
    A, sign = _orient(C)
    lr = np.log2(radii)
    n_valid = np.isfinite(A).sum(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # all-NaN genes are reported, not raised
        lo = np.nanmin(A, axis=0); hi = np.nanmax(A, axis=0)
    lvl = lo + 0.5 * (hi - lo)

    scale = np.full(A.shape[1], np.nan)
    for i in range(A.shape[1]):
        y = A[:, i]
        if not np.isfinite(y).any():
            continue
        below = np.where(y <= lvl[i])[0]
        if below.size == 0 or below[0] == 0:
            continue                                       # never crosses inside the window
        j = below[0]
        # interpolate from the last DEFINED radius before the crossing: an undefined radius (a
        # texture band with no Fourier mode) carries no value to interpolate from
        above = np.flatnonzero(np.isfinite(y[:j]))
        if above.size == 0:
            continue                                       # no defined value above the level
        k = above[-1]
        denom = y[k] - y[j]
        f = (y[k] - lvl[i]) / denom if abs(denom) > 1e-12 else 0.0
        scale[i] = 2.0 ** (lr[k] + f * (lr[j] - lr[k]))

    com = 2.0 ** np.mean(lr)                               # the grid's own centre of mass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suspect = np.abs(np.log(scale / com)) < np.log1p(GRID_SUSPECT_TOL)
        # has the curve flattened inside the window, or is it still climbing at the edge?
        rng = hi - lo
        tail = np.abs(A[-1] - A[-2]) / np.where(rng > 1e-12, rng, np.nan)
        window_limited = tail >= WINDOW_LIMITED_TOL

    return dict(scale_um=scale, amplitude=hi, scale_supported=np.isfinite(scale),
                grid_suspect=np.where(np.isfinite(scale), suspect, False),
                window_limited=np.where(np.isfinite(tail), window_limited, True),
                tail_fraction=tail, n_valid_radii=n_valid, orientation=sign, grid_com_um=com)


def presence_variation(tidies, *, stat_col="statistic", value_col="effect_size",
                       section_col="section", cv_threshold=30.0, drop_underpowered=True):
    """Does each statistic's presence VARY BETWEEN sections? The second of the two criteria.

    **Requires at least two sections and refuses fewer.** Cross-tissue variation is not a property a
    single section can exhibit, and a function that returned a column of NaN for it would be
    reporting an answer it does not have.

    Compare across tissues by a panel-independent effect size, NOT by a count of significant genes:
    the presence ``score`` saturates on large panels, so a statistic would look more variable simply
    because the panels differ in size.

    Sections flagged ``underpowered`` are dropped by default rather than averaged in. A section where
    every statistic is at the noise floor -- every block at ``|z| ~ 1`` -- otherwise manufactures apparent cross-tissue variation in every statistic at once,
    which is a statement about that section and not about any statistic.

    Parameters
    ----------
    tidies : DataFrame or sequence of DataFrame
        Per-section presence tables, concatenated or as a list, each with one row per statistic.
        Neither :func:`presence` nor :func:`spectrast.gr.fingerprint_presence` adds a section
        column, so add one before passing them. The column defaults fit no single producer:
        :func:`presence` names its statistics in ``statistic`` but has no ``effect_size`` (pass,
        for example, ``value_col="median_abs_z"``), and :func:`spectrast.gr.fingerprint_presence`
        has ``effect_size`` but names its statistics in ``axis`` (pass ``stat_col="axis"``).
    stat_col : str, optional
        Column naming the statistic. Default ``"statistic"``.
    value_col : str, optional
        Column holding the per-section value compared across sections. Non-finite values are
        ignored. Default ``"effect_size"``.
    section_col : str, optional
        Column naming the section. Default ``"section"``.
    cv_threshold : float, optional
        Coefficient of variation, in per cent, at or above which ``varies_across_tissues`` is
        True. Default ``30.0``.
    drop_underpowered : bool, optional
        If ``True`` (default) and the table has an ``underpowered`` column, drop every section
        that is underpowered on all of its statistics. If dropping would leave fewer than two
        sections, nothing is dropped and a ``RuntimeWarning`` is issued.

    Returns
    -------
    pandas.DataFrame
        One row per statistic, sorted by ``cv_pct``, largest first, with columns ``statistic``
        (always this name, whatever ``stat_col`` is), ``n_sections`` (number of finite values),
        ``cv_pct`` (sample standard deviation over the absolute mean, in per cent; ``inf`` when
        the absolute mean is below 1e-12, ``NaN`` with fewer than two values), ``range``, ``median``,
        ``varies_across_tissues`` and ``n_underpowered`` (sections dropped as underpowered).

    Raises
    ------
    ValueError
        If a required column is missing, or the table holds fewer than two sections.
    """
    import pandas as pd                                     # lazy: keep import-time deps light

    d = pd.concat(list(tidies), ignore_index=True) if not hasattr(tidies, "columns") \
        else tidies.copy()
    for c in (stat_col, value_col, section_col):
        if c not in d.columns:
            raise ValueError(f"missing column {c!r}; got {list(d.columns)}")
    n_all = d[section_col].nunique()
    if n_all < 2:
        raise ValueError(
            f"presence_variation needs at least two sections, got {n_all}. Cross-tissue variation "
            "is a between-section quantity; run presence() on each section and pass the results.")

    n_under = 0
    if drop_underpowered and "underpowered" in d.columns:
        bad = d.groupby(section_col).underpowered.all()     # underpowered on EVERY statistic
        drop = set(bad[bad].index)
        n_under = len(drop)
        if drop and n_all - n_under >= 2:
            d = d[~d[section_col].isin(drop)]
        elif drop:
            warnings.warn(f"{n_under} section(s) are underpowered on every statistic but dropping "
                          "them would leave fewer than two; keeping them and flagging instead.",
                          RuntimeWarning, stacklevel=2)
            n_under = 0

    rows = []
    for s, g in d.groupby(stat_col):
        v = g[value_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            rows.append(dict(statistic=s, n_sections=v.size, cv_pct=np.nan, range=np.nan,
                             median=np.nan, varies_across_tissues=False, n_underpowered=n_under))
            continue
        mu = v.mean()
        cv = 100.0 * v.std(ddof=1) / abs(mu) if abs(mu) > 1e-12 else np.inf
        rows.append(dict(statistic=s, n_sections=v.size, cv_pct=cv, range=float(v.max() - v.min()),
                         median=float(np.median(v)), varies_across_tissues=bool(cv >= cv_threshold),
                         n_underpowered=n_under))
    return pd.DataFrame(rows).sort_values("cv_pct", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ the lattice, shared by all

def _lattice(xy, bin_um):
    """Cell -> bin assignment plus one sparse scatter matrix, built once and reused by every
    statistic and every scramble. The scatter replaces ``np.add.at`` per gene: same arithmetic, a
    cell contributes to exactly one bin, but it is pure host work that otherwise dominates a
    multi-thousand-gene sweep and caps every backend equally."""
    import scipy.sparse as sps

    xyz = np.asarray(xy, float) - np.asarray(xy, float).min(0)
    col = (xyz[:, 0] / bin_um).astype(int)
    row = (xyz[:, 1] / bin_um).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    lin = row * W + col
    scat = sps.csr_matrix((np.ones(lin.size), (lin, np.arange(lin.size))), shape=(H * W, lin.size))
    cnt = np.bincount(lin, minlength=H * W).astype(float).reshape(H, W)
    mask = cnt > 0
    inv = np.maximum(cnt[mask], 1.0)[:, None]
    return dict(row=row, col=col, lin=lin, HW=(H, W), scat=scat, mask=mask, inv=inv)


def _radius_bins(r_um, bin_um, need):
    """Radii in coordinate units -> ``(r_bin, r_um)``: the whole bins they are measured at, a
    repeated bin kept once, and those bins back in coordinate units. Raises ``ValueError`` when
    fewer than ``need`` distinct bins remain."""
    r_um = np.atleast_1d(np.asarray(r_um, float))
    r_bin = np.maximum(np.round(r_um / bin_um).astype(int), 1)
    keep = np.concatenate([[True], np.diff(r_bin) > 0])[: r_bin.size]   # a repeated bin is one radius
    r_bin = r_bin[keep]
    if r_bin.size < need:
        want = "at least two distinct radii" if need == 2 else "at least one radius"
        raise ValueError(
            f"need {want} after rounding to whole bins of {bin_um:g}; the radii "
            f"{[float(v) for v in r_um]} round to {r_bin.size} distinct bin(s): "
            f"{[int(v) for v in r_bin]}. Pass radii further apart or a smaller bin_um.")
    return r_bin, r_bin * bin_um


def _expression(X):
    """The expression matrix as :func:`_stack` reads it: a sparse matrix stays sparse, as CSC so a
    chunk of gene columns is a cheap slice; a dense one is used as is."""
    import scipy.sparse as sps

    return X.tocsc() if sps.issparse(X) else np.asarray(X)


def _stack(lat, Z, cols):
    """Rasterize a chunk of genes onto the lattice: per-bin MEAN, with empty bins left at zero and
    carried by ``mask`` rather than being treated as measured zeros. Only the chunk's columns of
    ``Z`` are densified, so a sparse matrix is never dense as a whole."""
    H, W = lat["HW"]
    blk = Z[:, cols]
    blk = np.asarray(blk.toarray() if hasattr(blk, "toarray") else blk, float)
    g = np.asarray(lat["scat"] @ blk).reshape(H, W, len(cols))
    g[lat["mask"]] /= lat["inv"]
    return g


def _scalograms(Z, lat, r_bin, stats, bin_um, chunk=64):
    """Every requested statistic over every gene, from ONE rasterization and one transform per chunk.

    This is the "one reduction" half of the unification: the statistics are different questions, but
    they share the lattice and the FFT engine, so computing them separately re-rasterizes and
    re-transforms for no gain.

    Texture is :func:`spectrast.spatial.texture_multiscale`: the mean power in the radial-frequency
    band centred on wavelength ``2 r`` for each radius ``r`` in ``r_bin``, so it sits on the same
    radius axis as the five disc statistics.
    """
    from . import spatial as spa

    G = Z.shape[1]
    out = {s: np.zeros((len(r_bin), G)) for s in stats}

    for s0 in range(0, G, chunk):
        cols = list(range(s0, min(s0 + chunk, G)))
        grids = _stack(lat, Z, cols)
        sl = slice(s0, s0 + len(cols))
        if "moran" in stats:
            out["moran"][:, sl] = spa.morans_i_multiscale(grids, lat["mask"], r_bin)
        if "geary" in stats:
            out["geary"][:, sl] = spa.gearys_c_multiscale(grids, lat["mask"], r_bin)
        if "general_g" in stats:
            out["general_g"][:, sl] = spa.general_g_multiscale(grids, lat["mask"], r_bin)
        if "losh" in stats:
            out["losh"][:, sl] = spa.losh_multiscale(grids, lat["mask"], r_bin)
        if "focality" in stats:
            out["focality"][:, sl] = spa.focality_scalogram(grids, lat["mask"], r_bin)
        if "texture" in stats:
            out["texture"][:, sl] = spa.texture_multiscale(grids, lat["mask"], r_bin)
    return out


def organization(adata, *, stats=ORG_STATS, genes=None, radii=None, bin_um=None, layer=None,
                 basis="spatial", chunk=64, return_curves=False):
    """One reduction, applied uniformly to every organization curve. Returns a tidy table.

    Every statistic reduces to the same two numbers -- a characteristic scale and an amplitude --
    and every one of them travels with the flags that say whether to trust it. Which reductions are
    worth SHOWING is then a property of this table, not a definition.

    Parameters
    ----------
    adata : AnnData
    stats : tuple of str
        Any of :data:`ORG_STATS`. ``"texture"`` is the mean spectral power in the radial-frequency
        band centred on wavelength ``2 r`` at each radius ``r``
        (:func:`spectrast.spatial.texture_multiscale`), so it reads the same radius axis as the
        other statistics. A radius whose band holds no Fourier mode of the lattice gives NaN; this
        happens at long wavelengths, where a band is narrower than the lattice's frequency
        resolution or longer than the section.
    genes, radii, bin_um, layer, basis
        As elsewhere in the package. ``radii`` are in COORDINATE UNITS (µm); they are converted to
        whole bins internally, which is the grid the statistics are actually computed on.
        ``genes=None`` uses every gene; ``radii=None`` uses 19 log-spaced radii from 2 to 40 × the
        median cell spacing; ``bin_um=None`` uses 0.5 × the median cell spacing, coarsened with a
        warning above four million bins; ``layer=None`` reads ``adata.X``; ``basis`` defaults to
        ``"spatial"``. Radii that round to the same whole bin are measured once, and at least
        two distinct radii must remain (a curve needs two to be reduced), else ``ValueError``.
    chunk : int, optional
        Number of genes densified, rasterized and transformed together. A sparse expression matrix
        stays sparse (a CSC copy is made if it is in another sparse format) and only one chunk of
        genes is dense at a time: an ``(n_cells, chunk)`` block plus the lattice stack of
        ``chunk`` genes and its transforms, so this working memory grows with ``chunk`` and the
        whole matrix is never dense. Default ``64``.
    return_curves : bool
        Also return the curves the table was reduced from. The table is identical either way.

    Returns
    -------
    DataFrame with one row per (gene, statistic) and columns ``scale_um``, ``amplitude``,
    ``scale_supported``, ``grid_suspect``, ``window_limited``, ``tail_fraction``, ``n_valid_radii``.
    With ``return_curves=True``, the tuple ``(table, curves)`` where ``curves`` is a dict:
    ``radii_um`` (the radii actually measured, whole bins times ``bin_um``), ``genes`` (column
    labels) and one ``(n_radii, n_genes)`` array per statistic, in the orientation it was computed.
    :mod:`spectrast.pl` plots accept this dict wherever they take curves.

    Notes
    -----
    Read the three flags before the two numbers. ``scale_supported`` is False where the
    half-maximum was never crossed inside the window, so the scale is NaN rather than extrapolated.
    Radii where a curve is NaN are skipped when the crossing is located.
    ``grid_suspect`` marks a scale sitting within ±10% of the radius grid's own centre of mass.
    ``window_limited`` marks a curve that had not flattened by the widest radius measured, so its
    half-maximum is placed by where the sweep was cut rather than by the tissue -- the General G
    case, which ``grid_suspect`` does not catch.
    """
    import pandas as pd
    from .gr import _coords, _default_bin_um, _gene_idx, _matrix, _radii_from_xy

    bad = [s for s in stats if s not in ORG_STATS]
    if bad:
        raise ValueError(f"unknown statistics {bad}; choose from {list(ORG_STATS)}")
    stats = tuple(stats)

    xy = _coords(adata, basis)
    X = _matrix(adata, layer=layer)
    names = np.asarray(adata.var_names)
    if genes is not None:
        idx, _ = _gene_idx(adata, genes)            # (positions, names)
        X = X[:, idx]; names = names[idx]
    Z = _expression(X)                          # sparse stays sparse; densified per chunk

    bin_um = float(bin_um) if bin_um is not None else float(_default_bin_um(xy))
    r_um = np.asarray(radii, float) if radii is not None else np.asarray(_radii_from_xy(xy), float)
    r_bin, r_um = _radius_bins(r_um, bin_um, need=2)           # a curve needs two radii to reduce

    lat = _lattice(xy, bin_um)
    cur = _scalograms(Z, lat, r_bin, stats, bin_um, chunk=chunk)

    frames = []
    for s in stats:
        R = _reduce_curve(cur[s], r_um)
        frames.append(pd.DataFrame(dict(
            gene=names, statistic=s, scale_um=R["scale_um"], amplitude=R["amplitude"],
            scale_supported=R["scale_supported"], grid_suspect=R["grid_suspect"],
            window_limited=R["window_limited"], tail_fraction=R["tail_fraction"],
            n_valid_radii=R["n_valid_radii"])))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["bin_um"] = bin_um
    # a list, not an ndarray: pandas compares attrs with == when it truncates a table for display,
    # and an array there makes the table unprintable
    out.attrs["radii_um"] = [float(v) for v in r_um]
    out.attrs["grid_com_um"] = float(2.0 ** np.mean(np.log2(r_um)))
    if return_curves:
        curves = dict(radii_um=np.array(r_um, float), genes=np.asarray(names, dtype=str),
                      **{s: cur[s] for s in stats})
        return out, curves
    return out


def celltype_scale(adata, cell_type_key, *, min_cells=500, types=None, radii=None, bin_um=None,
                   basis="spatial", return_curves=False):
    """Characteristic scale and amplitude of each cell type's 0/1 indicator field.

    Each type with at least ``min_cells`` cells becomes one field, 1 in that type's cells and 0 in
    every other cell, and is measured exactly as :func:`organization` measures a gene with
    ``stats=("moran",)``: the same lattice (each bin holds the fraction of its cells that are of the
    type), the same radii and the same half-max reduction. With the default ``radii`` and ``bin_um``
    the lattice and radii are the ones a plain ``organization(adata)`` call resolves on this section,
    so the types' scales and the genes' scales are commensurate.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]`` and a categorical ``adata.obs[cell_type_key]``.
    cell_type_key : str
        The ``obs`` column holding the labels. Cells with a missing label belong to no type but stay
        in the tissue (they are 0 in every indicator).
    min_cells : int
        Types with fewer cells are skipped and listed in ``attrs["skipped"]``.
    types : sequence of str, optional
        Restrict to these labels (``min_cells`` still applies). Default: every label, most
        abundant first.
    radii, bin_um, basis
        As in :func:`organization`.
    return_curves : bool
        Also return the Moran curves, as :func:`organization` does with ``return_curves=True``.

    Returns
    -------
    DataFrame, one row per type, with columns ``cell_type``, ``n_cells``, ``scale_um``,
    ``amplitude``, ``scale_supported``, ``grid_suspect``, ``window_limited``, ``tail_fraction``,
    ``n_valid_radii``; ``attrs`` carry ``radii_um``, ``bin_um``, ``grid_com_um``,
    ``cell_type_key`` and ``skipped`` (a list of ``(type, n_cells)``). With ``return_curves=True``,
    ``(table, curves)`` where ``curves`` holds ``radii_um``, ``genes`` and ``types`` (both the type
    labels, ``genes`` so that code reading an :func:`organization` curve dict reads this one too)
    and ``moran`` of shape ``(n_radii, n_types)``.

    Examples
    --------
    >>> tab = spectrast.organization.celltype_scale(adata, "cell_type")     # doctest: +SKIP
    >>> spectrast.pl.celltype_ladder(tab)                                    # doctest: +SKIP
    """
    import anndata as ad
    import pandas as pd
    from .gr import _coords

    if cell_type_key not in adata.obs:
        raise KeyError(f"adata.obs[{cell_type_key!r}] not found")
    lab = adata.obs[cell_type_key]
    known = lab.notna().to_numpy()
    lab = lab.astype(str).to_numpy()
    counts = pd.Series(lab[known]).value_counts()
    if types is None:
        cand = list(counts.index)
    else:
        cand = [str(t) for t in ([types] if isinstance(types, str) else types)]
        missing = [t for t in cand if t not in counts.index]
        if missing:
            raise KeyError(f"types not found in adata.obs[{cell_type_key!r}]: {missing}")
    keep = [t for t in cand if counts[t] >= min_cells]
    skipped = [(t, int(counts[t])) for t in cand if counts[t] < min_cells]
    if not keep:
        raise ValueError(f"no type in adata.obs[{cell_type_key!r}] has at least min_cells={min_cells} "
                         f"cells (largest: {counts.iloc[0] if len(counts) else 0}); lower min_cells")

    ind = np.column_stack([(known & (lab == t)).astype(float) for t in keep])
    tmp = ad.AnnData(ind)
    tmp.var_names = keep
    tmp.obs_names = np.asarray(adata.obs_names).astype(str)
    tmp.obsm["spatial"] = np.asarray(_coords(adata, basis), float)[:, :2]
    tab, cur = organization(tmp, stats=("moran",), radii=radii, bin_um=bin_um, basis="spatial",
                            return_curves=True)
    out = tab.drop(columns="statistic").rename(columns={"gene": "cell_type"})
    out.insert(1, "n_cells", [int(counts[t]) for t in keep])
    out.attrs.update(tab.attrs)
    out.attrs["cell_type_key"] = cell_type_key
    out.attrs["skipped"] = skipped
    if return_curves:
        curves = dict(radii_um=cur["radii_um"], genes=cur["genes"],
                      types=cur["genes"].copy(), moran=cur["moran"])
        return out, curves
    return out


def _scramble(lat, rng):
    """Permute the cell -> bin assignment, holding the grid, the mask, the per-bin counts and every
    gene's expression values FIXED. The set of occupied bins is bit-identical to the observed pass,
    so the only thing destroyed is WHICH cell's expression sits where: a strictly spatial null that
    cannot be tissue shape, occupancy or expression marginal, because those are the same arrays."""
    import scipy.sparse as sps

    p = rng.permutation(lat["lin"].size)
    lin = lat["lin"][p]
    H, W = lat["HW"]
    scat = sps.csr_matrix((np.ones(lin.size), (lin, np.arange(lin.size))), shape=(H * W, lin.size))
    cnt = np.bincount(lin, minlength=H * W).astype(float).reshape(H, W)
    mask = cnt > 0
    return dict(lat, scat=scat, mask=mask, inv=np.maximum(cnt[mask], 1.0)[:, None], lin=lin)


def presence(adata, *, stats=ORG_STATS, n_perm=12, genes=None, radii=None, bin_um=None,
             layer=None, basis="spatial", seed=0, chunk=64, min_n=30):
    """Does each statistic carry spatial signal in THIS section? Criterion 1, per statistic.

    **Read the paired columns.** ``frac_outside`` and ``median_abs_z`` compare each gene against its
    OWN null, which controls for expression level exactly, and that is the question this criterion
    asks. ``auc_sep_unpaired_guard`` pools all genes and asks whether an observed value can be told
    from a null value WITHOUT knowing which gene it came from -- a weaker and, for selection,
    misleading question. A statistic whose between-gene spread exceeds its observed-vs-null shift
    reads low on the guard while every individual gene separates cleanly, and between-gene spread is
    the signal clustering consumes, not noise. The guard is kept only to catch a degenerate per-gene
    null sd inflating z. **It is named as a guard because using it as a verdict is a mistake this
    project has already made**: it put focality at 0.33 and texture at 0.32, apparent failures, where
    the paired readout puts them at 0.82 and 0.84 against a chance level of 0.154.

    ``frac_outside`` must be read against its own chance level, ``2/(n_perm+1)``, which is returned:
    with 12 permutations a gene's value at one radius falls outside its null range 15.4% of the
    time by chance alone.

    Returns one row per statistic. ``separates_genes`` is True where ``frac_outside`` exceeds three
    times chance. Cross-tissue variation is NOT returned here -- it is not a property one section can
    exhibit; see :func:`presence_variation`.

    Each null draw permutes which cell sits in which position. The lattice, the set of occupied
    bins, the number of cells per bin and every gene's values stay fixed, so only the arrangement
    of expression is destroyed.

    Parameters
    ----------
    adata : AnnData
        Coordinates in ``adata.obsm[basis]``; expression in ``adata.X`` or
        ``adata.layers[layer]``.
    stats : tuple of str, optional
        Any of :data:`ORG_STATS`. Default: all six. An unknown name raises ``ValueError``.
    n_perm : int, optional
        Number of coordinate scrambles. Each one recomputes every requested statistic for every
        gene, so the cost is about ``n_perm + 1`` calls of :func:`organization`. Must be at least
        6, or ``separates_genes`` could never be True; a smaller value raises ``ValueError``, but
        only after the scrambles have been computed. Default ``12``.
    genes, radii, bin_um, layer, basis, chunk
        As in :func:`organization`. ``radii`` and ``bin_um`` are in coordinate units. One distinct
        radius after rounding to whole bins is enough, except with ``"texture"`` in ``stats``,
        which needs two (each band is bounded by its neighbours); fewer raise ``ValueError``.
    seed : int, optional
        Seed for the scrambles. Default ``0``.
    min_n : int, optional
        A statistic is flagged ``underpowered`` when fewer than ``min_n`` genes have a finite
        curve at every radius, or when ``median_abs_z`` is below 3. Default ``30``.

    Returns
    -------
    pandas.DataFrame
        One row per statistic, with columns:

        * ``statistic``;
        * ``n_tested``: genes whose observed curve is finite at every radius;
        * ``n_perm`` and ``chance_frac_outside``, which is ``2 / (n_perm + 1)``;
        * ``frac_outside``: the fraction of gene × radius values that lie strictly outside the
          range of their own null values (values with a zero or undefined null standard
          deviation are left out);
        * ``median_abs_z``: the median over gene × radius values of the absolute z-score against
          the null mean and standard deviation;
        * ``separates_genes``: ``frac_outside`` above ``SEPARATES_MULTIPLE`` (3) × chance;
        * ``auc_sep_unpaired_guard``: ``abs(2 * AUC - 1)`` for observed against null values,
          pooled over genes and radii;
        * ``underpowered``: see ``min_n``.

        ``attrs`` hold ``bin_um`` and ``radii_um``, the radii actually measured (whole bins times
        ``bin_um``).
    """
    import pandas as pd
    from .gr import _coords, _default_bin_um, _gene_idx, _matrix, _radii_from_xy

    bad = [s for s in stats if s not in ORG_STATS]
    if bad:
        raise ValueError(f"unknown statistics {bad}; choose from {list(ORG_STATS)}")
    stats = tuple(stats)

    xy = _coords(adata, basis)
    X = _matrix(adata, layer=layer)
    names = np.asarray(adata.var_names)
    if genes is not None:
        idx, _ = _gene_idx(adata, genes); X = X[:, idx]; names = names[idx]
    Z = _expression(X)                          # sparse stays sparse; densified per chunk

    bin_um = float(bin_um) if bin_um is not None else float(_default_bin_um(xy))
    r_um = np.asarray(radii, float) if radii is not None else np.asarray(_radii_from_xy(xy), float)
    # texture's bands are bounded by the neighbouring radii, so it needs two; the disc statistics
    # are compared with their nulls radius by radius and need one
    r_bin, r_um = _radius_bins(r_um, bin_um, need=2 if "texture" in stats else 1)

    lat = _lattice(xy, bin_um)
    obs = _scalograms(Z, lat, r_bin, stats, bin_um, chunk=chunk)
    rng = np.random.default_rng(seed)
    nul = {s: np.empty((n_perm,) + obs[s].shape) for s in stats}
    for p in range(n_perm):
        one = _scalograms(Z, _scramble(lat, rng), r_bin, stats, bin_um, chunk=chunk)
        for s in stats:
            nul[s][p] = one[s]

    chance = 2.0 / (n_perm + 1)
    # The criterion is `frac_outside > SEPARATES_MULTIPLE x chance`, and `chance` is 2/(n_perm+1),
    # so with too few permutations the threshold exceeds 1 and the criterion becomes UNREACHABLE
    # however strong the signal -- it would silently report "does not separate" for everything.
    # Refuse rather than return a flag that cannot be True.
    if SEPARATES_MULTIPLE * chance >= 1.0:
        raise ValueError(
            f"n_perm={n_perm} is too small for the separation criterion: chance frac_outside is "
            f"{chance:.3f} and the threshold {SEPARATES_MULTIPLE}x{chance:.3f} = "
            f"{SEPARATES_MULTIPLE * chance:.3f} exceeds 1, so separates_genes could never be True. "
            f"Use n_perm >= {int(np.ceil(2 * SEPARATES_MULTIPLE - 1)) + 1}.")
    rows = []
    for s in stats:
        o, nl = obs[s], nul[s]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = nl.mean(0), nl.std(0, ddof=1)
            good = np.isfinite(o) & np.isfinite(mu) & (sd > 0)
            z = np.where(good, (o - mu) / np.where(sd > 0, sd, 1.0), np.nan)
            outside = np.where(good, (o > nl.max(0)) | (o < nl.min(0)), np.nan)
            # the unpaired guard, pooled over genes and radii
            a = o[np.isfinite(o)].ravel(); b = nl[np.isfinite(nl)].ravel()
            if a.size and b.size:
                rk = pd.Series(np.concatenate([a, b])).rank().to_numpy()
                auc = (rk[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size)
                guard = float(abs(2 * auc - 1))
            else:
                guard = np.nan
            fo = float(np.nanmean(outside))
            zz = float(np.nanmedian(np.abs(z)))
        n_tested = int(np.isfinite(o).all(0).sum())
        rows.append(dict(statistic=s, n_tested=n_tested, n_perm=n_perm,
                         chance_frac_outside=chance, frac_outside=fo, median_abs_z=zz,
                         separates_genes=bool(fo > SEPARATES_MULTIPLE * chance),
                         auc_sep_unpaired_guard=guard,
                         underpowered=bool(n_tested < min_n or zz < 3.0)))
    out = pd.DataFrame(rows)
    out.attrs["bin_um"] = bin_um
    # a list, not an ndarray: pandas compares attrs with == when it truncates a table for display,
    # and an array there makes the table unprintable
    out.attrs["radii_um"] = [float(v) for v in r_um]
    return out

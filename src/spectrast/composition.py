"""Composition-aware nulls — separating *reorganized* from *recomposed*.

Spectrast's statistics measure how a gene's expression is ARRANGED. When two conditions differ in
cell-type COMPOSITION, a gene's scale or focality can shift purely because the mix of cells
expressing it changed, not because its organization changed. The ordinary nulls (coordinate scramble
in :mod:`spectrast.gr`, replicate label-permutation / matched repartition in :mod:`spectrast.diff`)
preserve tissue shape and density but scramble ACROSS cell types, so they cannot tell those apart: a
purely compositional difference can return a small p-value and be read as reorganization.

Two mechanisms, answering different questions. Ship both, and know which you are asking:

**Condition-on** (:func:`stratified_permutation`, :func:`stratified_partition`) — permute only
*within* each cell-type stratum. Composition, per-type abundance and each type's spatial footprint
are preserved exactly; only CROSS-type arrangement is destroyed. A gene significant against this
null carries organization BEYOND what cell-type placement explains. Exact and non-parametric.

**Regress-out** (:func:`residualize`) — regress the organization metric on a local-composition
matrix and test the residual. Removes the LINEAR part of composition and handles CONTINUOUS
composition or cell-state gradients, which condition-on cannot because it needs discrete strata.

=========================================  ==============  ==============
Signal                                     Condition-on    Regress-out
=========================================  ==============  ==============
Within-type reorganization                 captures        captures if orthogonal to composition
Organization carried by cell-type placement missed BY      missed if linearly predictable
                                           DESIGN
Nonlinear composition dependence           captures        missed (linear only)
Continuous gradient, no discrete types     cannot          captures
=========================================  ==============  ==============

.. warning::

   **Composition is not always a nuisance.** If the biology *is* a cell type reorganizing —
   decidualization, atrophy, an expanding niche — then adjusting for composition conditions away the
   mechanism you are trying to report. These nulls answer "organization BEYOND composition". If the
   composition change *is* your finding, do not adjust.

.. warning::

   **Resolution changes the answer, and the tool must not choose for you.** A coarse stratification
   under-corrects (residual composition structure survives inside a stratum); a fine one
   over-corrects and runs into sparse strata. Run at two or more resolutions and inspect
   :func:`resolution_flip_report` — genes whose verdict flips are the ones whose significance is an
   artefact of that choice, not of the data.

.. warning::

   **Circularity.** On a targeted panel, cell-type labels are usually annotated FROM the same genes
   being tested. Conditioning on those labels removes signal by construction for the markers that
   defined them. :func:`check_circularity` warns when a tested gene is a known annotation marker.
"""
from __future__ import annotations

import warnings

import numpy as np

#: Below this many cells a stratum has too little permutation entropy to give a usable null.
DEFAULT_MIN_CELLS = 50

#: Label given to strata merged by the ``"pool"`` small-stratum policy.
POOLED_LABEL = "_pooled_small"


def strata_from_obs(adata, strata_key):
    """Read and validate the stratum labels from ``adata.obs[strata_key]``.

    Fails loudly rather than degrading: a composition-aware null silently falling back to the
    ordinary null would report exactly the false positive it exists to prevent.

    Parameters
    ----------
    adata : AnnData
        Holds the labels in ``adata.obs``.
    strata_key : str
        The ``obs`` column of stratum labels, for example cell types. Any dtype; the labels are
        compared as strings.

    Returns
    -------
    numpy.ndarray of str
        ``(n_obs,)`` label per cell.

    Raises
    ------
    ValueError
        If ``strata_key`` is ``None``, if any cell is unlabelled (a missing value, which reads as
        the string ``"nan"``; a real label spelled ``"nan"`` is refused too), or if there are
        fewer than two distinct labels.
    KeyError
        If ``adata.obs`` has no column ``strata_key``.
    """
    if strata_key is None:
        raise ValueError("no strata_key given")
    if strata_key not in adata.obs:
        raise KeyError(
            f"adata.obs[{strata_key!r}] not found — composition adjustment needs cell-type (or other "
            f"stratum) labels. Available obs columns: {list(adata.obs.columns)[:20]}")
    lab = np.asarray(adata.obs[strata_key].astype(str))
    if np.any(lab == "nan"):
        n = int((lab == "nan").sum())
        raise ValueError(
            f"adata.obs[{strata_key!r}] has {n} unlabelled cell(s). Every cell must belong to a "
            f"stratum, or the null is only partly conditioned — label them (an 'unassigned' category "
            f"is fine and honest) or subset the data first.")
    if len(np.unique(lab)) < 2:
        raise ValueError(
            f"adata.obs[{strata_key!r}] has a single level, so a within-stratum permutation is the "
            f"ordinary permutation and conditions on nothing. Use a stratification with >= 2 levels.")
    return lab


def apply_min_cells(labels, min_cells=DEFAULT_MIN_CELLS, policy="pool"):
    """Guardrail 1 — handle strata too small to permute.

    A stratum of a handful of cells has almost no permutation entropy: its contribution to the null
    is nearly the observed value, so p-values floor and the test looks more significant than it is.
    Rare-type columns are also the unstable ones in the regress-out mode.

    ``policy="pool"`` (default) merges every under-sized stratum into one ``_pooled_small`` group —
    cells are kept, but note that arrangement IS scrambled across the pooled types, so those types
    are under-conditioned. ``policy="drop"`` excludes them (returns a keep-mask instead), which
    conditions correctly but changes the tissue. ``policy="error"`` refuses.

    Returns ``(labels, keep_mask, report)``.

    Parameters
    ----------
    labels : array-like, shape (n_cells,)
        Stratum label per cell, for example from :func:`strata_from_obs`. Compared as strings.
    min_cells : int, optional
        A stratum with fewer cells than this is "small". Default :data:`DEFAULT_MIN_CELLS` (50).
    policy : {"pool", "drop", "error"}, optional
        What to do with small strata, as described above. Default ``"pool"``. Both ``"pool"``
        and ``"drop"`` warn when they act. Another value raises ``ValueError``, but only when a
        small stratum exists.

    Returns
    -------
    labels : numpy.ndarray of str
        ``(n_cells,)``. Under ``"pool"`` the small strata are relabelled ``POOLED_LABEL``
        (``"_pooled_small"``); otherwise the input labels, as strings.
    keep_mask : numpy.ndarray of bool
        ``(n_cells,)``. ``False`` for cells of dropped strata under ``"drop"``; all ``True``
        otherwise.
    report : dict
        ``n_strata`` (distinct labels among kept cells, after pooling), ``small`` (each small
        stratum mapped to its cell count), ``policy``, ``min_cells`` and ``n_dropped``.

    Raises
    ------
    ValueError
        Under ``policy="error"`` when any stratum is small.
    """
    labels = np.asarray(labels, dtype=object).astype(str)
    vals, counts = np.unique(labels, return_counts=True)
    small = {v: int(c) for v, c in zip(vals, counts) if c < min_cells}
    keep = np.ones(len(labels), bool)
    # widen before assigning POOLED_LABEL: a fixed-width unicode array sized to the existing labels
    # TRUNCATES a longer one silently (a "<U4" array turns "_pooled_small" into "_poo"), which would
    # both lose the marker and risk colliding with a real label.
    width = max(int(labels.dtype.itemsize // 4), len(POOLED_LABEL))
    out = labels.astype(f"<U{width}")
    if small:
        if policy == "error":
            raise ValueError(f"strata below min_cells={min_cells}: {small}. Raise min_cells, coarsen "
                             f"the stratification, or pass policy='pool'/'drop'.")
        if policy == "pool":
            out[np.isin(labels, list(small))] = POOLED_LABEL
            n_pooled = int(np.isin(labels, list(small)).sum())
            warnings.warn(
                f"composition: {len(small)} stratum/strata below min_cells={min_cells} "
                f"({small}) merged into {POOLED_LABEL!r} ({n_pooled} cells). Arrangement is "
                f"scrambled ACROSS those types, so they are under-conditioned; pass "
                f"policy='drop' to exclude them instead, or coarsen the stratification.")
        elif policy == "drop":
            keep = ~np.isin(labels, list(small))
            warnings.warn(
                f"composition: dropped {len(small)} stratum/strata below min_cells={min_cells} "
                f"({small}); {int((~keep).sum())} cells excluded from the test.")
        else:
            raise ValueError(f"policy must be 'pool', 'drop' or 'error', got {policy!r}")
        # pooling can still leave the pooled group too small to permute
        if policy == "pool":
            n_pool = int((out == POOLED_LABEL).sum())
            if n_pool < min_cells:
                warnings.warn(f"composition: even the pooled group has only {n_pool} cells "
                              f"(< {min_cells}); its contribution to the null is near-degenerate.")
    report = dict(n_strata=int(len(np.unique(out[keep]))), small=small, policy=policy,
                  min_cells=int(min_cells), n_dropped=int((~keep).sum()))
    return out, keep, report


def stratified_permutation(labels, rng):
    """A permutation that stays inside each stratum.

    ``perm[i]`` is drawn from the cells sharing ``i``'s label, so applying it to coordinates moves a
    cell's expression to another position *of its own type*. Cell-type composition, per-type
    abundance and each type's spatial footprint are therefore preserved exactly, and only the
    arrangement of expression *within* the types is destroyed.

    Parameters
    ----------
    labels : array-like of shape (n_obs,)
        Stratum label per cell.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        Permutation of ``arange(n_obs)``.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrast import composition
    >>> lab = np.array(["a", "a", "b", "b", "b"])
    >>> p = composition.stratified_permutation(lab, np.random.default_rng(0))
    >>> bool((lab[p] == lab).all())            # every cell keeps its own stratum
    True
    """
    labels = np.asarray(labels)
    perm = np.arange(len(labels))
    for v in np.unique(labels):
        idx = np.flatnonzero(labels == v)
        perm[idx] = rng.permutation(idx)
    return perm


def stratified_partition(labels, in_A, rng):
    """Reassign two-arm membership within each stratum, preserving each arm's composition.

    The composition-aware analogue of the matched (N2) repartition null: instead of drawing arm A
    from the pooled cells, draw it stratum by stratum, keeping the number of arm-A cells of each type
    exactly as observed. A difference that is purely compositional then cannot survive, because both
    arms have the observed composition in every permutation.

    Parameters
    ----------
    labels : array-like of shape (n_cells,)
    in_A : array-like of bool, shape (n_cells,)
        Observed arm-A membership; only its per-stratum COUNT is preserved.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray of bool
        New arm-A membership.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrast import composition
    >>> lab = np.array(["a"] * 4 + ["b"] * 4)
    >>> obs = np.array([True] * 2 + [False] * 2 + [True] * 3 + [False])
    >>> new = composition.stratified_partition(lab, obs, np.random.default_rng(0))
    >>> [int((new & (lab == v)).sum()) for v in ("a", "b")]     # per-stratum counts preserved
    [2, 3]
    """
    labels = np.asarray(labels); in_A = np.asarray(in_A, bool)
    out = np.zeros(len(labels), bool)
    for v in np.unique(labels):
        idx = np.flatnonzero(labels == v)
        k = int(in_A[idx].sum())
        if k:
            out[rng.choice(idx, k, replace=False)] = True
    return out


def residualize(values, composition, add_intercept=True):
    """Guardrail-free core of the regress-out mode: least-squares residuals on a composition matrix.

    Removes the part of ``values`` linearly predictable from ``composition`` (per-unit cell-type
    fractions, or any continuous covariate). Handles continuous gradients that the discrete
    within-stratum null cannot, at the cost of assuming the composition dependence is linear.

    Parameters
    ----------
    values : array-like, shape (n_units,) or (n_units, n_features)
    composition : array-like, shape (n_units, n_types)
    add_intercept : bool

    Returns
    -------
    numpy.ndarray
        Residuals, same shape as ``values``.

    Examples
    --------
    >>> import numpy as np
    >>> from spectrast import composition as C
    >>> comp = np.linspace(0, 1, 20)[:, None]
    >>> y = 3.0 + 2.0 * comp[:, 0]                      # purely compositional
    >>> bool(np.allclose(C.residualize(y, comp), 0, atol=1e-9))
    True
    """
    y = np.asarray(values, float)
    one_d = y.ndim == 1
    Y = y[:, None] if one_d else y
    Xc = np.asarray(composition, float)
    Xc = Xc[:, None] if Xc.ndim == 1 else Xc
    if len(Xc) != len(Y):
        raise ValueError(f"composition has {len(Xc)} rows but values has {len(Y)}")
    if add_intercept:
        Xc = np.column_stack([np.ones(len(Xc)), Xc])
    # lstsq (not a normal-equation solve): composition columns sum to 1 and are near-collinear with
    # an intercept, so the design is rank-deficient by construction and must be handled, not fought.
    beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    res = Y - Xc @ beta
    return res[:, 0] if one_d else res


def check_residual_design(C, arm, min_residual_df=2, separation_warn=0.9):
    """Guardrails for the regress-out mode — the two ways it silently destroys the contrast.

    **Saturation.** Regressing on ``k`` composition columns costs ``k + 1`` degrees of freedom. A
    cohort of 4 samples and 5 cell types has none left: the residuals are ~0 by construction and
    every gene turns non-significant, which reads as "no reorganization" rather than "no test".

    **Separation.** If composition nearly predicts *which arm a sample is in* — the usual case when
    the conditions differ compositionally, i.e. exactly when you reach for this — then the arm
    indicator lies in the column space of the covariates and residualizing removes the contrast
    itself, not a nuisance. This is the quantitative form of "composition is not always a nuisance";
    at ``R^2 -> 1`` the adjusted test cannot answer anything.

    Parameters
    ----------
    C : array-like, shape (n_units, n_types)
    arm : array-like of bool, shape (n_units,)
        Arm membership per unit.
    min_residual_df : int
        Minimum ``n_units - n_types - 1``. Below this the mode refuses.
    separation_warn : float
        Warn when the arm indicator's R^2 on composition exceeds this.

    Returns
    -------
    dict
        Design diagnostics, stamped into the result's provenance.
    """
    C = np.asarray(C, float); arm = np.asarray(arm).astype(float)
    n, k = C.shape
    df = n - k - 1
    if df < min_residual_df:
        raise ValueError(
            f"regress-out needs more units than composition columns: {n} unit(s), {k} covariate(s) "
            f"leaves {df} residual d.f. (< {min_residual_df}). The residuals would be ~0 by "
            f"construction and every gene would turn non-significant — which is not a null result, "
            f"it is no test. Collapse the cell types into fewer groups, or use the condition-on "
            f"null (strata_key=) instead, which does not spend degrees of freedom.")
    ss_tot = float(((arm - arm.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0 else 1.0 - float((residualize(arm, C) ** 2).sum()) / ss_tot
    if r2 > separation_warn:
        warnings.warn(
            f"composition: the arm label is {r2:.0%} predictable from composition alone. "
            f"Residualizing therefore removes most of the contrast you are testing, not a nuisance "
            f"— a null result here is close to uninformative. If the composition change IS the "
            f"finding, report it rather than adjusting it away.")
    return dict(n_units=int(n), n_covariates=int(k), residual_df=int(df),
                arm_r2_on_composition=float(r2))


def composition_matrix(adata, strata_key, unit_key):
    """Per-unit cell-type fractions — the covariate matrix for :func:`residualize`.

    Parameters
    ----------
    adata : AnnData
        Cells of every unit, with both label columns in ``adata.obs``.
    strata_key : str
        The ``obs`` column of cell-type (stratum) labels, read and checked by
        :func:`strata_from_obs`, so every cell must be labelled and there must be at least two
        types.
    unit_key : str
        The ``obs`` column naming each cell's unit (for example a section or donor). Labels are
        compared as strings.

    Returns
    -------
    tuple(numpy.ndarray, list, list)
        ``(fractions (n_units, n_types), unit_labels, type_labels)``, units in first-appearance
        order so the rows line up with per-unit statistics built the same way. Each row holds the
        fraction of that unit's cells in each type and sums to 1. ``type_labels`` are sorted;
        both label lists hold strings.

    Raises
    ------
    KeyError
        If either column is missing.
    ValueError
        From :func:`strata_from_obs`, on unlabelled cells or a single type.
    """
    import pandas as pd
    lab = strata_from_obs(adata, strata_key)
    if unit_key not in adata.obs:
        raise KeyError(f"adata.obs[{unit_key!r}] not found")
    unit = np.asarray(adata.obs[unit_key].astype(str))
    units = list(pd.unique(unit)); types = list(np.unique(lab))
    M = np.zeros((len(units), len(types)))
    for i, u in enumerate(units):
        sel = lab[unit == u]
        for j, t in enumerate(types):
            M[i, j] = np.mean(sel == t) if len(sel) else np.nan
    return M, units, types


def check_circularity(tested, marker_map, strata_key=None):
    """Guardrail 3 — warn when tested genes are the markers that defined the strata.

    On a targeted panel the cell-type labels are usually annotated from the same genes being tested.
    Conditioning on those labels then removes the signal for those genes *by construction*, so a
    null result for a marker means nothing. This cannot be detected automatically — it needs the
    annotation's marker list — so pass one when you have it.

    Parameters
    ----------
    tested : sequence of str
        Genes being tested.
    marker_map : dict or sequence of str
        ``{cell_type: [markers]}`` or a flat marker list used to annotate the strata.
    strata_key : str, optional
        Only used in the message.

    Returns
    -------
    list of str
        The offending genes (also warned about).
    """
    if not marker_map:
        return []
    flat = set()
    if isinstance(marker_map, dict):
        for v in marker_map.values():
            flat.update([v] if isinstance(v, str) else v)
    else:
        flat.update(marker_map)
    hits = sorted({g for g in tested if g in flat})
    if hits:
        warnings.warn(
            f"composition: {len(hits)} tested gene(s) are annotation markers for "
            f"{strata_key or 'the strata'} ({hits[:10]}{'...' if len(hits) > 10 else ''}). "
            f"Conditioning on labels derived from these genes removes their signal BY CONSTRUCTION "
            f"— a null result for them is uninformative, not evidence of no reorganization.")
    return hits


def resolution_flip_report(results, key="gene", verdict="significant"):
    """Guardrail 2 — which genes change verdict between stratification resolutions.

    The coarse-vs-fine problem is not solved by choosing a resolution; it is made visible. Run the
    same test at two or more ``strata_key`` resolutions, pass the results here, and inspect the genes
    whose call flips: their significance depends on a modelling choice rather than on the data.

    Parameters
    ----------
    results : dict of {str: pandas.DataFrame}
        One result table per resolution, each with ``key`` and a boolean ``verdict`` column.
    key, verdict : str

    Returns
    -------
    pandas.DataFrame
        One row per feature: the per-resolution verdicts, ``n_significant``, and ``flips`` (True when
        the verdict is not unanimous). Sorted flips-first.

    Examples
    --------
    >>> import pandas as pd
    >>> from spectrast import composition
    >>> a = pd.DataFrame({"gene": ["g0", "g1"], "significant": [True, True]})
    >>> b = pd.DataFrame({"gene": ["g0", "g1"], "significant": [True, False]})
    >>> composition.resolution_flip_report({"coarse": a, "fine": b}).flips.tolist()
    [True, False]
    """
    import pandas as pd
    if len(results) < 2:
        raise ValueError("resolution_flip_report needs >= 2 resolutions — its whole purpose is to "
                         "expose dependence on that choice")
    cols = {}
    for name, df in results.items():
        if key not in df or verdict not in df:
            raise KeyError(f"result {name!r} lacks {key!r} or {verdict!r}")
        cols[name] = df.set_index(key)[verdict].astype(bool)
    out = pd.DataFrame(cols)
    out["n_significant"] = out.sum(1)
    out["flips"] = (out["n_significant"] > 0) & (out["n_significant"] < len(results))
    return (out.sort_values(["flips", "n_significant"], ascending=[False, False])
               .reset_index().rename(columns={"index": key}))

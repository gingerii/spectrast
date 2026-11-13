"""spectrast.diff — DIFFERENTIAL scale-resolved spatial statistics across paired/grouped tissue states.

The inferential layer over the descriptive spectral engine (T1-T4). Given two conditions it tests, per gene
(or per gene/LR/cell-type PAIR) and per scale, whether SPATIAL ORGANIZATION changed — with a null that separates
biological reorganization from the technical differences (density, depth, registration) that otherwise fake a
change.

ONE AXIS PER RUN by default. The organization axis is declared by the caller:
  moran     (default) autocorrelation strength vs scale       — the shipped scalogram
  geary     local-dissimilarity / boundary sharpness vs scale — alternative autocorrelation (NEVER pooled w/ moran)
  focality  diffuse -> punctate (Gini of the local-Moran map) vs scale
  coloc     scale-resolved SIGNED co-expression of a (A, B) pair (cross-spectral) — makes the sign-flip inferential
  cross     distance-resolved CO-ORGANIZATION of a pair (mark cross-correlation, geometry divided out) — asks
            whether the RANGE over which two marks are coupled changed, not whether they are co-expressed
The N1/N2 null machinery (permute -> mean-diff -> standardize by null SD per scale -> max-statistic over scale ->
BH-FDR) is GENERIC over that per-object curve: it does not care which statistic produced it. `differential(...)`
is the single driver; `scalogram_delta(...)` is the back-compat `axis="moran"` wrapper (byte-identical output).
The only multi-axis entry point is `differential_omnibus(...)`, which uses ONE combined statistic, ONE FDR pass,
and a post-hoc (uncorrected) decomposition — so the family size is G genes, never G x S.

Null hierarchy (state which was used):
  N1 "replicate" — within-condition sample-pair spread is the null for the between-condition difference
                   (permute sample->condition labels). Absorbs ALL technical nuisances; primary when replicates
                   exist. Use unit_key to aggregate to the independent experimental unit (patient/animal).
  N2 "matched"   — one co-located sample per condition (e.g. a split, or two registered sections): pool the
                   cells and repartition n_perm times. Optional depth (binomial thinning to a common median) and
                   density (subsample to matched cell count) matching remove the two nuisances most likely to
                   fake a scale change.
All heavy work reuses spectrast.spatial primitives; reduced_rank is never used (a differential must not be confounded
by a denoiser).
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

PAIR_AXES = {"coloc", "cross"}                                          # per-pair axes (the rest are per-gene)
PER_GENE_AXES = {"moran", "geary", "focality"}


def bh_fdr(p):
    """Benjamini–Hochberg false-discovery-rate correction.

    Parameters
    ----------
    p : array-like
        Raw p-values. ``NaN`` is treated as ``1.0``.

    Returns
    -------
    numpy.ndarray
        BH-adjusted q-values in ``[0, 1]``, same length as ``p``.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> q = spectrast.diff.bh_fdr([0.001, 0.02, 0.5, 0.9])
    >>> bool(np.all((q >= 0) & (q <= 1)))
    True
    """
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def _grid(xy, bin_size):
    xy = np.asarray(xy, float); xy = xy - xy.min(0)
    col = (xy[:, 0] / bin_size).astype(int); row = (xy[:, 1] / bin_size).astype(int)
    H, W = int(row.max()) + 1, int(col.max()) + 1
    return H, W, row * W + col


def _thin(counts, target_median, seed=0):
    """Binomial-thin integer counts so per-cell total matches ``target_median`` (depth match). Seeded so the
    stochastic thinning honours the caller's ``seed`` (was a hardcoded rng(0), making it seed-independent)."""
    tot = np.asarray(counts.sum(1)).ravel(); med = np.median(tot[tot > 0])
    if target_median >= med: return counts
    keep = np.clip(target_median / np.maximum(tot, 1), 0, 1)
    rng = np.random.default_rng(seed); c = counts.tocoo()
    kept = rng.binomial(c.data.astype(int), keep[c.row])
    import scipy.sparse as sp
    return sp.csr_matrix((kept, (c.row, c.col)), shape=counts.shape)


# ---------------------------------------------------------------------------
# Shared binning + per-axis curve builders (the statistic plug-ins).
# Every builder has the SAME contract: f(xy, Xg, radii, bin_size, H, W, binidx) -> (nR, ncol),
# so the null driver never branches on the axis except to pick the gate and the id column.
# ---------------------------------------------------------------------------

# A contrast whose smallest attainable p exceeds this cannot produce a conventionally significant
# result at all, so the design is reported rather than left for the reader to infer from an empty table.
_P_FLOOR_WARN = 0.05


def _perm_support(kind, n_a, n_b=0):
    """How many DISTINCT relabelings the permutation null actually admits.

    ``"unpaired"`` -> C(n_a + n_b, n_a) partitions of the units into two arms.
    ``"signflip"`` -> 2 ** n_a sign vectors over the paired units.
    """
    from math import comb
    return 2 ** n_a if kind == "signflip" else comb(n_a + n_b, n_a)


def _perm_floor_num(kind, n_a, n_b=0):
    """Numerator of the exact permutation floor: ``_perm_floor_num / support``.

    The enumerated null holds every relabelling BUT the observed one, so the floor is set by how
    many relabellings are guaranteed at least as extreme as the observed one. The statistic is
    two-sided, so the observed labelling's COMPLEMENT qualifies — **when the complement is itself
    a member of the relabelling set**:

    - ``"signflip"``: negating every sign is always a valid sign vector -> 2.
    - ``"unpaired"`` with ``n_a == n_b``: swapping the arms maps an ``n_a``-subset to another
      ``n_a``-subset, so it is in the set -> 2.
    - ``"unpaired"`` with ``n_a != n_b``: the complement of an ``n_a``-subset has size ``n_b``,
      which is NOT one of the ``C(n_a + n_b, n_a)`` labellings of this design. Only the observed
      labelling itself is guaranteed extreme -> 1.

    MEASURED, not argued (2026-09-28). Smallest p actually returned by an enumerated replicate
    null on a planted effect: 4v4 = 0.028571 = 2/70 and 3v3 = 0.1 = 2/20, but 3v6 = 0.011905 =
    1/84 and 5v6 = 0.0021645 = 1/462 — exactly half the floor the library used to document for
    them. Reporting 2/support for an unbalanced design overstates the floor twofold and can
    declare `fdr_reachable=False` for a design that is in fact capable, which is the harmful
    direction: it tells a user their study cannot work when it can.
    """
    if kind == "signflip":
        return 2
    return 2 if int(n_a) == int(n_b) else 1


def _clamp_perms(n_perm, support, what, *, floor_num=2):
    """Cap ``n_perm`` at the null's own support, so p can never fall below what the design supports.

    The permutation p is ``(1 + #{null >= obs}) / (1 + n_perm)``, so its floor is ``1/(1+n_perm)``
    — a quantity fixed by the CALLER. The exact permutation p, however, cannot go below
    ``floor_num/support``, because the observed labelling is itself one of the relabelings. Draw more
    permutations than the design admits and the sampler simply repeats partitions, yet the
    denominator keeps growing, so a p BELOW the exact floor becomes reportable and the test turns
    anti-conservative.

    Measured on a 4-vs-5 section split (126 partitions) with n_perm=200: a same-condition contrast
    reported p = 1/201 = 0.00497, under the exact floor of 1/126 = 0.0079, and produced a false
    positive that vanished once n_perm was capped.

    This is a no-op for any well-powered design — 9 vs 10 units already admits 92,378 partitions —
    so it cannot move a result that had the replicates to support it.

    **Once clamped, the null is ENUMERATED rather than sampled** (2026-08-19). Drawing ``support-1``
    random permutations from ``support`` possibilities repeats some and misses roughly a third, which
    bites at exactly the sizes where the clamp binds. Enumerating every relabelling EXCEPT the observed
    one, each once, makes ``(1 + #{null >= obs}) / (n_perm + 1)`` the exact permutation p
    ``#{all >= obs} / support``. Because the statistic is two-sided the observed labelling's complement
    is also as extreme WHENEVER that complement is itself a relabelling of this design, so the exact
    floor is ``floor_num/support`` with ``floor_num`` from :func:`_perm_floor_num` — 2 for a sign-flip
    or a BALANCED unpaired design, 1 for an unbalanced one, where no complement exists in the label
    set. Callers pass it; the default of 2 is the balanced convention.

    What this fixes: on a 4-vs-4 design (support 70, exact floor 0.029) the sampler returned a smallest
    p of 0.057, so nothing could reach 0.05 and a strongly injected effect read as zero detection — the
    design looked incapable when only the sampling was. Enumeration also makes the floor deterministic
    and seed-independent, which the sampled version was not.

    Only runs when ``support - 1 <= n_perm``, so its cost is bounded by the ``n_perm`` the caller
    already asked for; above that the sampler is untouched.
    """
    # A DESIGN THAT CANNOT REACH THE THRESHOLD MUST SAY SO. The cap below keeps p honest, but honesty
    # about p is not the same as being able to detect anything: with support S the smallest attainable
    # p is 2/S, so at 3-vs-3 units (S=20, floor 0.1) NOTHING can clear 0.05 however large the effect.
    # An empty result table then means "no test was performed", not "no effect" -- and it looks exactly
    # like a clean negative, which is how it has been misread twice during development (the k-vs-k
    # replicate calibration, and again on the landmark differential). Warned unconditionally, before
    # the capping question, because it applies whether or not n_perm exceeded the support.
    #
    # floor_num/S (2026-09-25, corrected 2026-09-28): the enumerated null holds every relabelling BUT
    # the observed one, so the floor is set by how many relabellings are guaranteed as extreme -- the
    # observed one, plus its complement WHEN the complement is a member of the set. That is 2/S for a
    # sign-flip or a balanced unpaired design and 1/S for an unbalanced one; `_perm_floor_num` owns
    # the rule and `design_floor` reports the same number. The original 1/S comparison stayed silent
    # on exactly the designs a user is most likely to run: at 3-vs-3, 1/20 = 0.05 is not > 0.05, so no
    # warning fired although no gene could reach 0.05. Whenever this fires the support is under 40, so
    # `n_perm` is necessarily above support-1 and the null IS enumerated -- i.e. the floor is
    # attainable, not a bound.
    p_min = float(floor_num) / support
    if p_min > _P_FLOOR_WARN:
        warnings.warn(
            f"the {what} null admits only {support} distinct relabelings, so the smallest attainable "
            f"p is {floor_num}/{support} = {p_min:.3g} (the observed labelling is excluded from the "
            + ("enumeration and its complement is always as extreme, so the floor is 2/support"
               if floor_num == 2 else
               "enumeration; the arms are unequal, so the observed labelling's complement is not "
               "itself a relabelling of this design and the floor is 1/support")
            + f" -- the convention diff.design_floor reports). NOTHING in this contrast can reach "
            f"p < {_P_FLOOR_WARN:g}, "
            f"and after FDR correction the reachable threshold is coarser still. An empty result here "
            f"means NO TEST WAS PERFORMED, not that there is no effect. The only fix is more units.",
            stacklevel=3)
    if support - 1 >= n_perm:
        return n_perm
    capped = max(1, support - 1)
    warnings.warn(
        f"n_perm={n_perm} exceeds the {what} null's support ({support} distinct relabelings); "
        f"capping to {capped}. Beyond the support the sampler only repeats partitions while the "
        f"p-value denominator keeps growing, which lets p fall below the exact floor of "
        f"{floor_num}/{support} = {p_min:.4g} and makes the test anti-conservative. The real fix is "
        f"more units — this design's smallest attainable p is {p_min:.4g}.", stacklevel=3)
    return capped


def design_floor(n_A, n_B, *, paired=False, alpha=0.05, n_features=1):
    """What the smallest attainable p is for this design, before any data is touched.

    A permutation test cannot report a p below the support of its own relabelling set. The library
    enumerates every relabelling except the observed one, so the floor is a count over
    ``n_partitions``: the observed labelling always counts, and its complement counts too whenever
    that complement is itself a relabelling — i.e. ``2 / n_partitions`` for a paired sign-flip or a BALANCED unpaired
    design, but ``1 / n_partitions`` when ``n_A != n_B``.
    Across ``n_features`` features with BH control, the most conservative rung is ``alpha /
    n_features``: a design whose floor sits above it cannot make a feature FDR-significant on its
    own. Features that share the floor p can still pass together, because BH adjusts ``k`` tied
    p-values to ``p_floor * n_features / k``; that needs at least ``n_features * p_floor / alpha``
    of them. Reported rather than left for the reader to infer from an empty table.

    Parameters
    ----------
    n_A, n_B : int
        Units per arm (sections, or donors when sections are grouped into units). With
        ``paired=True`` both are the number of pairs and must be equal.
    paired : bool, optional
        ``True`` for the paired sign-flip null (``2 ** n_A`` sign vectors); ``False`` for the
        unpaired label permutation (``C(n_A + n_B, n_A)`` partitions). Default ``False``.
    alpha : float, optional
        FDR level that ``fdr_reachable`` is judged against. Default ``0.05``.
    n_features : int, optional
        Number of features in the BH family. Values below 1 are treated as 1. Default ``1``.

    Returns
    -------
    dict
        ``kind`` (``"signflip"`` or ``"unpaired"``), ``n_partitions`` (number of distinct
        relabellings), ``p_floor`` (smallest attainable p), ``alpha``, ``n_features`` and
        ``fdr_reachable`` (``True`` when ``p_floor <= alpha / n_features``, so one feature at the
        floor passes BH by itself). :func:`differential` stores the same dict, plus ``n_units_A``,
        ``n_units_B`` and ``paired``, in ``df.attrs["spectrast_design"]``.

    Raises
    ------
    ValueError
        If ``paired=True`` and ``n_A != n_B``.

    Examples
    --------
    >>> import spectrast
    >>> spectrast.diff.design_floor(4, 4)["p_floor"]          # 2 / C(8, 4)
    0.02857142857142857
    """
    kind = "signflip" if paired else "unpaired"
    if paired and int(n_A) != int(n_B):
        raise ValueError(
            f"paired=True needs one unit per pair in each arm, got n_A={n_A} and n_B={n_B}. The "
            f"sign-flip null is indexed by the PAIRS, so an unequal pair of counts does not name a "
            f"design. This used to be accepted with n_B silently ignored, which reported the floor "
            f"of a design the caller had not described.")
    n_part = _perm_support(kind, int(n_A), 0 if paired else int(n_B))
    floor = _perm_floor_num(kind, int(n_A), 0 if paired else int(n_B)) / n_part
    return dict(kind=kind, n_partitions=int(n_part), p_floor=float(floor), alpha=float(alpha),
                n_features=int(n_features),
                fdr_reachable=bool(floor <= float(alpha) / max(int(n_features), 1)))


def _mark_label(spec):
    """A short, stable display name for a mark spec."""
    if isinstance(spec, str):
        return spec
    if callable(spec):
        return getattr(spec, "mark_name", None) or getattr(spec, "__name__", "field")
    kind = spec[0]
    if kind == "obs":
        return str(spec[1])
    if kind == "mean":
        members = list(spec[1])
        return f"{spec[2]}" if len(spec) > 2 else f"mean[{len(members)}]"
    if kind == "fingerprint":
        ax = spec[2] if len(spec) > 2 else "getis"
        return f"{spec[1]}:{ax}"
    return str(spec)


def _mark_key(spec):
    """A hashable identity for de-duplicating specs. Lists are normalized to tuples."""
    if isinstance(spec, str) or callable(spec):
        return spec
    return tuple(tuple(x) if isinstance(x, list) else x for x in spec)


def _resolve_mark(ad, spec, layer=None):
    """One mark spec -> a per-cell float vector on ``ad``.

    Every bivariate mark in the design that is PRECOMPUTABLE reduces to a per-cell field, so it can
    enter the differential through the same binning as expression. Supported forms:

    ``"GENE"``                        expression of one gene (the original behaviour)
    ``("obs", key)``                  a numeric ``obs`` column (covariate, program score, state)
    ``("mean", [genes])``             mean expression over a gene set (regulon / program)
    ``("fingerprint", gene[, axis])`` the gene's clipped organization field (``getis`` or
                                      ``local_moran``) — a mark that is itself a spatial statistic
    callable ``f(ad) -> (n_obs,)``    an escape hatch for anything precomputed

    Marks that need indicator + density-field handling (#2 cross-PCF, #3 gene x cell type) are
    deliberately NOT here; they are the second pass.
    """
    import scipy.sparse as sp
    n = ad.n_obs
    if callable(spec):
        v = np.asarray(spec(ad), float).ravel()
    elif isinstance(spec, str):
        j = list(ad.var_names).index(spec)
        X = ad.X if layer is None else ad.layers[layer]
        col = X[:, j]
        v = np.asarray(col.todense() if sp.issparse(col) else col, float).ravel()
    else:
        kind = spec[0]
        if kind == "obs":
            v = np.asarray(pd.to_numeric(ad.obs[spec[1]], errors="coerce"), float).ravel()
        elif kind == "mean":
            genes = [g for g in spec[1] if g in set(ad.var_names)]
            if not genes:
                raise KeyError(f"mark {_mark_label(spec)!r}: none of its genes are in this sample")
            idx = [list(ad.var_names).index(g) for g in genes]
            X = ad.X if layer is None else ad.layers[layer]
            sub = X[:, idx]
            v = (np.asarray(sub.mean(axis=1)).ravel() if sp.issparse(sub)
                 else np.asarray(sub, float).mean(1))
        elif kind == "fingerprint":
            from .spatial import getis_ord_genes, local_morans_i_genes
            from .gr import default_radius
            gene, ax = spec[1], (spec[2] if len(spec) > 2 else "getis")
            fn = getis_ord_genes if ax == "getis" else local_morans_i_genes
            j = list(ad.var_names).index(gene)
            r = float(spec[3]) if len(spec) > 3 else default_radius(ad)
            X = ad.X if layer is None else ad.layers[layer]
            v = np.clip(np.asarray(fn(np.asarray(ad.obsm["spatial"], float), X, r, [j])).ravel(),
                        0.0, None)
        else:
            raise ValueError(f"unknown mark spec {spec!r}")
    if v.shape[0] != n:
        raise ValueError(f"mark {_mark_label(spec)!r} returned {v.shape[0]} values for {n} cells")
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def _prep_marks(ad, specs, depth_to=None, seed=0, layer=None):
    """(xy, Xg) for a list of mark specs.

    When EVERY spec is a gene name this delegates to :func:`_prep` untouched, so the gene-gene path
    — including depth thinning — is bit-identical to the pre-mark implementation. That is the gate,
    and it holds by construction rather than by comparison.
    """
    import scipy.sparse as sp
    if layer is None and all(isinstance(s, str) for s in specs):
        return _prep(ad, [list(ad.var_names).index(g) for g in specs], depth_to, seed)
    # `_prep` reads ``ad.X`` unconditionally, so delegating to it while a layer was requested would
    # silently return the WRONG matrix. With a layer set, every spec goes through the resolver.
    if depth_to is not None:
        raise ValueError("depth_match is a matched-null nuisance control for EXPRESSION columns and "
                         "has no meaning for derived marks (a program score has no counts to thin); "
                         "pass depth_match=False when any mark is not a gene name.")
    xy = np.asarray(ad.obsm["spatial"], float)
    cols = [_resolve_mark(ad, s, layer) for s in specs]
    return xy, sp.csr_matrix(np.column_stack(cols).astype(np.float32))


def _mean_stat(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """FIRST-ORDER level: the mask-mean of each mark's binned field, broadcast over radii.

    The first-order companion to the scale-resolved axes. It carries no scale information — the
    value is identical at every radius by construction — so it is reported at ``radii=(1,)`` and its
    "scale of maximum change" is meaningless and should be ignored. It exists so that "did the mark
    simply change level?" is tested with the SAME null, the same max-statistic and the same FDR pass
    as "did its organization change", which is what makes the two comparable.
    """
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    m = (fld * mask[..., None]).sum((0, 1)) / max(mask.sum(), 1)                     # (G,)
    return np.repeat(m[None, :], len(radii), axis=0)                                 # (nR, G)


def _bin_fields(xy, Xg, bin_size, H=None, W=None, binidx=None):
    """Bin per-cell values onto a grid -> raw per-bin MEAN field (H, W, G) + mask. Shared by every axis so A and B
    (and every gene/pair member) are registered on identical bins."""
    import scipy.sparse as sp
    if H is None:
        H, W, binidx = _grid(xy, bin_size)
    nb = H * W
    cnt = np.bincount(binidx, minlength=nb).astype(float); mask = (cnt > 0).reshape(H, W)
    M = sp.csr_matrix((np.ones(len(binidx), np.float32), (binidx, np.arange(len(binidx)))), shape=(nb, len(binidx)))
    sums = np.asarray((M @ (Xg.tocsr() if sp.issparse(Xg) else sp.csr_matrix(Xg))).todense())
    fld = np.zeros_like(sums); nz = cnt > 0; fld[nz] = sums[nz] / cnt[nz, None]
    return fld.reshape(H, W, Xg.shape[1]), mask, H, W, binidx


def _zstack(fld, mask):
    """Demean-in-mask + unit-variance per gene (the Moran/Geary convention)."""
    zbar = (fld * mask[..., None]).sum((0, 1)) / mask.sum()
    stack = (fld - zbar) * mask[..., None]
    return stack / (stack.std((0, 1)) + 1e-9)


def _scalogram(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """Per-gene Moran scalogram I_g(r). Xg: (n_cells, G) sparse/dense. Reference per-gene builder."""
    from .spatial import morans_i_multiscale
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    return morans_i_multiscale(_zstack(fld, mask), mask, radii, kind="disc")            # (nR, G)


def _geary_scalogram(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """Per-gene 1 - Geary's C at each radius (complement so the curve DECAYS with scale like Moran: local-
    dissimilarity sensitive; a boundary/transition axis). Never pooled with moran. Uses the shared-FFT
    multiscale primitive (one forward transform amortized over all radii)."""
    from .spatial import gearys_c_multiscale
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    return 1.0 - gearys_c_multiscale(_zstack(fld, mask), mask, radii, kind="disc")           # (nR, G)


def _focality_scalogram(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """Per-gene focality (Gini of the local-Moran map) at each radius: the diffuse -> punctate axis. Uses the
    shared-FFT multiscale primitive (transforms z + mask once, reduces to the Gini curve on the fly)."""
    from .spatial import focality_scalogram
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    return focality_scalogram(_zstack(fld, mask), mask, radii, kind="disc")                   # (nR, G)


def _generalg_scalogram(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """Per-gene Getis-Ord General G at each radius: do HIGH values cluster, rather than like-with-like.

    THE RAW BINNED FIELD IS PASSED, NOT ``_zstack``. General G is a ratio of products of the values
    themselves and is only interpretable for a non-negative variable; centring it would make the
    denominator a sum of signed products that can pass through zero, and `spatial.general_g_multiscale`
    warns about exactly that. Moran, Geary and focality all take the z-stack because they are defined
    on deviations — this axis is not, and the difference is easy to lose when adding it beside them.
    """
    from .spatial import general_g_multiscale
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    return general_g_multiscale(fld * mask[..., None], mask, radii, kind="disc")               # (nR, G)


def _losh_scalogram(xy, Xg, radii, bin_size, H=None, W=None, binidx=None):
    """Per-gene LOSH concentration at each radius: how UNEVENLY local variance is distributed.

    The fourth ESDA question — variance heterogeneity rather than level or autocorrelation. Also fed
    the raw field, since LOSH is built from residuals about a LOCAL mean and normalising globally
    first would flatten the very heterogeneity it measures.

    Reduced with the Gini of the H map (`spatial.losh_multiscale`'s default), which is a declared
    COMPOSITION of two standard tools rather than a canonical statistic — LOSH, unlike Moran, Geary and
    General G, has no established global form. Say so wherever this axis is reported.
    """
    from .spatial import losh_multiscale
    fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
    return losh_multiscale(fld * mask[..., None], mask, radii, kind="disc")                    # (nR, G)


def _counts(binidx, H, W):
    """Cells per bin — the density field the mark cross-correlation needs alongside the mean field."""
    return np.bincount(binidx, minlength=H * W).astype(float).reshape(H, W)


def _cross_edges(radii_um, bin_size):
    """Annulus edges whose centres are the requested radii, FIXED across samples.

    The pair curves have to be comparable between samples and permutations that bin to different
    grid sizes, so the annuli are pinned to the caller's radii rather than derived from each grid.
    Edges are geometric midpoints because the radius grid usually is geometric (1, 2, 4, 8, 16 bins),
    and the innermost edge is floored at one bin — on a lattice the smallest non-zero displacement
    IS one bin, so nothing below that is measurable.
    """
    r = np.asarray(radii_um, float)
    if r.ndim != 1 or len(r) < 2 or np.any(np.diff(r) <= 0):
        raise ValueError("axis='cross' needs at least 2 strictly increasing radii")
    if r[0] < bin_size:
        warnings.warn(f"cross: radius {r[0]:.0f}µm is below one bin ({bin_size:.0f}µm) and is not "
                      f"measurable on this lattice; its annulus is floored at one bin. Use radii "
                      f">= 1 bin, or a finer bin_size.", stacklevel=3)
    mid = np.sqrt(r[:-1] * r[1:])
    return np.concatenate([[max(r[0] ** 2 / mid[0], bin_size)], mid, [r[-1] ** 2 / mid[-1]]])


def _cross_curve(fld, pairs_idx, mask, radii_um, bin_size, counts):
    """Per-pair mark cross-correlation ``k_mm(r) - 1``: distance-resolved co-organization of a pair.

    The bivariate sibling of ``_coloc_curve``. Where coloc reports SIGNED co-expression per
    WAVELENGTH from the cross-spectrum, this reports co-organization per SEPARATION with tissue
    geometry divided out — the mean product of two marks conditional on a cell pair existing at that
    distance (Stoyan's mark correlation). So a change on this axis is a change in the DISTANCE
    STRUCTURE of a pair's coupling, not in how much the two are co-expressed overall.

    Returned as ``k - 1`` rather than ``k`` so that zero means "no co-organization". That is not
    cosmetic: the driver's structure gate is ``|curve|`` against ``structured_min``, and on the raw
    ``k`` scale every pair sits near 1 and would clear any sensible gate.
    """
    from .spatial import mark_cross_correlation_batch
    used = sorted({i for pr in pairs_idx for i in pr})
    lo = min(float(np.nanmin(fld[..., i])) for i in used)
    if lo < 0:
        raise ValueError(
            f"axis='cross' needs non-negative expression (found {lo:.3g}): the statistic is a ratio "
            "to the product of the marginal means and has no interpretation for a sign-changing "
            "mark. Pass raw or log-normalized counts, not scaled/centred values.")
    marks = [np.nan_to_num(fld[..., i]) for i in range(fld.shape[2])]
    out = mark_cross_correlation_batch(marks, mask, pairs_idx, densities=[counts] * len(marks),
                                       normalize="density", bin_size=bin_size,
                                       annuli=_cross_edges(radii_um, bin_size))
    return np.nan_to_num(out["g"].T - 1.0)                              # (nR, nPairs)


def _coloc_curve(fld, pairs_idx, mask, radii_um, bin_size, counts=None):
    """Per-pair SIGNED scale-resolved co-expression. For each (i, j) member pair, the cross-spectral coexpr(k) in
    [-1, 1] is resampled onto the FIXED wavelength axis ``radii_um`` (so curves align across samples/permutations
    that have different grid sizes -> different native CSD bins). Positive = co-localize at that scale, negative =
    anti-localize; the sign can FLIP across scale (the T3 result), which a scalar bivariate Moran averages away.

    NOTE the cross-spectral density resolves wavelengths only down to ~2*bin_size (its Nyquist band); any target
    radius below that is BELOW the measurable band and np.interp CLAMPS it to the finest CSD bin — so several
    sub-Nyquist radii collapse to the same value and are not independently measured. Choose coloc radii >= ~2 bins
    (e.g. for bin_size=30µm use radii>=2 -> >=60µm). We warn once when radii fall below the band."""
    from .spatial import cross_spectral_density
    wl_target = np.asarray(radii_um, float)
    out = np.zeros((len(wl_target), len(pairs_idx)))
    warned = False
    for pi, (i, j) in enumerate(pairs_idx):
        csd = cross_spectral_density(fld[..., i], fld[..., j], mask, n_bins=24, bin_size=bin_size)
        wl = np.asarray(csd["wavelength_um"], float); ce = np.asarray(csd["coexpr"], float)
        order = np.argsort(wl); wl_min = wl.min()
        if not warned and (wl_target < wl_min - 1e-6).any():
            import warnings
            warnings.warn(f"coloc: {(wl_target < wl_min).sum()} target radii below the CSD band (< {wl_min:.0f}µm "
                          f"= ~2*bin_size) are clamped to the finest bin and not independently measured; "
                          f"use coloc radii >= ~2 bins.", stacklevel=3); warned = True
        out[:, pi] = np.interp(wl_target, wl[order], ce[order])         # np.interp clamps outside the CSD band
    return out


STAT = {"moran": _scalogram, "geary": _geary_scalogram, "focality": _focality_scalogram,
        "general_g": _generalg_scalogram, "losh": _losh_scalogram, "mean": _mean_stat}
#: Axes defined on the values themselves rather than on deviations from the mean, so they take the raw
#: binned field instead of the z-stack and are only interpretable for a non-negative variable.
UNCENTRED_AXES = {"general_g", "losh"}

#: Axes that are not admissible as a per-gene differential test, mapped to why. Refused by
#: :func:`differential` rather than deleted, because each underlying statistic is still valid for
#: description. Two different grounds live here, and the message says which:
#:   * ``general_g`` is IN :data:`STAT` — it computes a curve per radius, but the CONTRAST is
#:     confounded by a global shift in expression level.
#:   * ``texture`` is NOT in :data:`STAT` — it has no per-radius curve to contrast in the first
#:     place, so the refusal is structural rather than statistical.
#: Membership here is checked BEFORE the ``STAT`` lookup, so both give the reason instead of
#: "unknown axis".
REFUSED_DIFF_AXES = {
    "general_g":
        "General G is uncentred — it is a ratio of products of the raw values, with no variance "
        "normalization — so a change in overall expression level moves it for every gene by roughly "
        "the same amount. Measured on a paired cohort (1501 genes): 372 of "
        "372 structured genes reached raw p<0.05, 371 of them in the SAME direction, with T_max "
        "confined to 3.59-3.79 — one global shift reported 372 times, not 372 findings. Its own "
        "negative control could not reach p<0.25 (min p 0.2409 over 377 genes), which is what "
        "identifies the effect as global rather than spatial. Use axis='moran' or 'losh' for "
        "organization, and a differential-expression test for abundance.",
    "texture":
        "Texture has no per-radius curve, so there is nothing for this test to contrast. Every axis "
        "in STAT returns one value per RADIUS and the statistic is the studentised maximum over "
        "them; texture is a radial POWER SPECTRUM (frequency bins of |FFT|^2, gr._texture_bins) "
        "reduced to a single centroid per gene, so making it differentiable means first CHOOSING a "
        "per-radius reduction — a band-limited centroid, say. That would be a second estimator "
        "wearing the same name: gr.texture would report a whole-field centroid while this reported a "
        "band-limited one, and one column would hold two statistics. The reduction would also have "
        "to be justified for an axis measured at -0.000 (CI -0.020 to +0.020) added programme "
        "recovery over scale_um + focality. Use axis='moran', 'focality' or 'losh'.",
}
FIRST_ORDER_AXES = {"mean"}                       # no scale content; report at radii=(1,)
PAIR_STAT = {"coloc": _coloc_curve, "cross": _cross_curve}       # same contract, extra counts arg


#: Re-exported from :mod:`spectrast.gr` so there is ONE definition. Set from a nine-section
#: parameter sweep:
#: pitch-anchoring beat fixed-micrometres and cells-per-bin on all five criteria, and 0.5x pitch is
#: uniformly better than 1.0x. Costs 4x the grid area.
#:
#: CORRECTED 2026-08-19 — this note used to say "at 0.5x every one of the nine sections passes (worst
#: 0.095)". Those distances were measured against a k=0.25 reference that had never been checked, and
#: that reference moves 0.106 octaves on ``xenium_breast_cancer`` when halved again. Against a
#: converged k=0.125 reference, 0.5x EXCEEDS the 0.1-octave tolerance on both Xenium 5K sections
#: (0.101 and 0.163), and ``dcis_idc`` (0.099) and ``stereoseq_olf`` (0.093) sit just inside it.
#: Closing that residual would need ~0.125x pitch, 16x the grid area, so it is a limitation to state
#: rather than a bin to refine. The
#: DECISION stands; the claim of convergence everywhere does not. Focality is unaffected
#: (0.001-0.008 octaves on all nine).
from .gr import K_PITCH                                   # noqa: E402  (one home, see gr)

#: Default radius window. The LOW end is fixed in BINS, not in pitch multiples, because that is where
#: the constraint actually lives: a disc of radius 1 bin is a plus-shape, not a disc. Measured on the
#: kernel this code actually convolves with -- ``spatial.neighbor_kernel``, which EXCLUDES the centre
#: bin because Moran's I weights neighbours and never self -- radius 1 is 4 bins against pi*r^2 = 3.14
#: (+27.3%) and radius 2 is 12 bins against 12.57 (-4.5%). Counting the centre as well, the same two
#: are 5 bins (+59.2%) and 13 bins (+3.5%); this comment quoted that second convention until
#: 2026-10-01 without saying so, so anyone checking it against neighbor_kernel found a mismatch.
#: Either way radius 1 is unusable and radius 2 is round enough. The bank's "2-40x pitch" span assumes its own
#: bin = 1x pitch, so its low end is also 2 bins — expressing it in pitch would move the floor when
#: K_PITCH changes and silently drop the shortest resolvable scale. That is not hypothetical: with the
#: low end written as 2x pitch the Trem2 capstone lost Spi1, because the window then started at 61 µm
#: and six of the nine hits have their argmax at 30.
#: The HIGH end follows the bank at 40x pitch. Resolution (``RADII_N``) and the high end are the P2/P3
#: parameters and have NOT been swept — consistency defaults, not validated optima.
RADII_BINS_LO, RADII_PITCH_HI, RADII_N = 2.0, 40.0, 19


def _median_pitch(adatas, basis="spatial"):
    """Median nearest-neighbour cell spacing pooled over sections -> one shared bin for both arms.

    ONE grid for both arms is not a convenience, it is a requirement: rasterizing arm A and arm B on
    different lattices would make the difference partly a difference of measuring sticks. So the pitch
    is the median of the per-section medians over A and B together, never each section's own.
    """
    from . import gr as _gr
    vals = []
    for ad in adatas:
        xy = ad.obsm.get(basis) if hasattr(ad.obsm, "get") else None
        if xy is None:
            continue
        vals.append(float(_gr._radius_from_xy(np.asarray(xy, float), factor=1.0)))
    if not vals:
        raise ValueError(f"cannot derive a bin size: no section has obsm[{basis!r}]. "
                         f"Pass bin_size explicitly.")
    return float(np.median(vals))


def _pitch_anchored_grid(adatas_A, adatas_B, radii, bin_size, basis="spatial"):
    """Resolve ``(radii_in_bins, bin_size)`` when either is left to the pitch-anchored default."""
    if bin_size is not None and radii is not None:
        return list(radii), float(bin_size)
    pitch = _median_pitch(list(adatas_A) + list(adatas_B), basis)
    b = float(bin_size) if bin_size is not None else K_PITCH * pitch
    if radii is not None:
        return list(radii), b
    lo, hi = RADII_BINS_LO, RADII_PITCH_HI * pitch / b
    r = np.unique(np.maximum(np.round(np.geomspace(lo, hi, RADII_N)).astype(int), 1))
    return r.tolist(), b


def _char_scale(scalogram, radii_um, how=None, *, sign=None):
    """Per-gene characteristic organization scale. Same readout ``gr.scale_spectrum`` uses, by
    construction — 'characteristic scale' has to mean one thing in the fingerprint and the differential
    or a Δscale is a difference of two different quantities. ``scalogram`` is (n_radii, G) -> (G,).

    **The default is the half-maximum as of 2026-08-18**; ``how="centroid"`` reproduces pre-flip
    numbers. See :mod:`spectrast.scale`.

    THE OBJECTION THIS FUNCTION USED TO RECORD, AND WHY IT DOES NOT APPLY. This docstring previously
    said a half-max crossing "saturates at the radius-grid ceiling for strongly-autocorrelated fields
    (row-standardized Moran) — collapsing the scale-recovery Δscale". That was true of
    ``spatial.correlation_length``, which crosses half of ``I`` AT THE SMALLEST RADIUS and assigns the
    largest radius to any curve that never gets there: a high floor defeats the threshold and every
    such gene is pinned to the ceiling. The reduction used here crosses half of the curve's OWN RANGE,
    which a floor cannot defeat, and returns NaN rather than the ceiling where there is no crossing.
    Different estimator, different failure mode — and it was re-measured on the scale-recovery contrast
    before the default was flipped rather than reasoned about.
    """
    from . import scale as _scale
    return _scale.reduce_scale(scalogram, radii_um, how or _scale.DEFAULT_REDUCTION, sign=sign)


def _arm_composition(A, B, composition):
    """Per-sample composition covariates for the regress-out mode -> ``(n_samples, n_types), types``.

    ``composition`` is either an ``obs`` column of cell-type labels (fractions are computed per
    sample, over the union of levels so every sample gets the same columns) or an explicit array
    already ordered as ``A + B``.
    """
    samples = list(A) + list(B)
    if isinstance(composition, str):
        from . import composition as _comp
        labs = [_comp.strata_from_obs(ad, composition) for ad in samples]
        types = sorted({v for l in labs for v in np.unique(l)})
        C = np.array([[np.mean(l == t) for t in types] for l in labs], float)
        return C, types
    C = np.asarray(composition, float)
    C = C[:, None] if C.ndim == 1 else C
    if len(C) != len(samples):
        raise ValueError(f"composition has {len(C)} rows but there are {len(samples)} samples "
                         f"({len(A)} in arm A + {len(B)} in arm B, in that order)")
    return C, [f"c{i}" for i in range(C.shape[1])]


def _fft_backend():
    from . import _fft
    return _fft.get_backend()


def _arm_backends(adatas):
    """Backends stamped on a list of samples by an earlier ``gr.*`` run (``uns[...]["backend"]``)."""
    seen = set()
    for ad in adatas:
        uns = getattr(ad, "uns", {}) or {}
        for k, v in uns.items():
            if isinstance(v, dict) and "backend" in v and str(k).startswith("spectrast"):
                seen.add(str(v["backend"]))
    return seen


def _check_backend_consistency(A, B):
    """Warn when the two arms of a contrast were not computed on the same compute backend.

    MPS/CUDA are float32-only, so every statistic carries a small (~1e-3) device-dependent offset.
    That offset is CONSTANT within a backend and cancels in a same-backend difference — but a
    contrast whose arms were computed on different backends keeps it, and it enters the effect
    exactly where the real signal does. This is a silent wrong answer, so it is worth a warning
    even though the magnitude is small.

    Returns the backend this call is running on.
    """
    from . import _fft
    active = _fft.get_backend()
    ba, bb = _arm_backends(A), _arm_backends(B)
    if ba and bb and ba != bb:
        warnings.warn(
            f"differential(): the two arms carry different compute-backend stamps (A={sorted(ba)}, "
            f"B={sorted(bb)}). MPS/CUDA are float32-only, so a per-feature device offset that would "
            f"cancel within one backend does NOT cancel across two — recompute both arms on the same "
            f"backend (spectrast.set_backend(...)) before comparing.", stacklevel=3)
    return active


def _prep(ad, genes_idx, depth_to=None, seed=0):
    xy = np.asarray(ad.obsm["spatial"], float)
    if depth_to is not None and "counts" in getattr(ad, "layers", {}):
        import scipy.sparse as sp
        C = ad.layers["counts"]; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
        Ct = _thin(C, depth_to, seed); X = Ct[:, genes_idx].astype(np.float32)
        X.data = np.log1p(X.data)                                          # match the log1p-normalized convention
    else:
        X = ad.X[:, genes_idx]
        import scipy.sparse as sp
        X = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)).astype(np.float32)
    return xy, X


# ---------------------------------------------------------------------------
# The generic driver.
# ---------------------------------------------------------------------------

def _erl_pvalues(z_obs, z_null):
    """Extreme-rank-length p per feature (Myllymäki, Mrkvička, Grabarnik, Seijo & Hahn 2017, JRSS-B).

    ``z_obs`` (nR, G), ``z_null`` (n_perm, nR, G): the studentised difference and its permutations.

    The max-statistic summarises a curve by its single worst point, so two curves whose worst points
    tie are indistinguishable to it. Extreme rank length instead ranks the WHOLE ordered vector of
    pointwise deviations and compares curves lexicographically on it: the second-worst radius breaks
    the tie at the worst, the third breaks that, and so on. The gain is real and specific — a
    difference that is modest but present at EVERY radius outranks a sharp spike at one, which is the
    case a maximum cannot see.

    Measured on the Trem2 amyloid contrast: identical to max-T on 9 of 9 hits and adds P2ry12, whose
    |z| is 3.07/2.95/2.92/2.89/2.97 against a 0.05 threshold of 2.14 — flat and everywhere extreme.

    Ties count as "at least as extreme", which is the conservative direction and keeps the p valid.
    """
    from scipy.stats import rankdata
    n_perm, nR, G = z_null.shape
    A = np.abs(np.concatenate([z_obs[None], z_null], axis=0))              # (s, nR, G), row 0 = obs
    # rank 1 = MOST extreme at that radius; 'min' so tied curves share the better rank
    R = rankdata(-A, method="min", axis=0)                                 # (s, nR, G)
    S = np.sort(R, axis=1)                                                 # most extreme first
    d = S[1:] - S[0][None]                                                 # (n_perm, nR, G)
    nz = d != 0
    first = np.argmax(nz, axis=1)                                          # (n_perm, G)
    pi, gi = np.meshgrid(np.arange(n_perm), np.arange(G), indexing="ij")
    at_least = (~nz.any(1)) | (d[pi, first, gi] < 0)                       # tie, or lexicographically worse
    return (1.0 + at_least.sum(0)) / (n_perm + 1.0)


def _null_band(null_d, sd, radii_um, band_arms, meanA, meanB, reduction, alpha, sign):
    """Effect sizes the SAME estimators return on the null contrasts — a matched noise floor.

    WHY THIS EXISTS. A volcano's x-axis needs a scale, and the natural one is "how big an effect does a
    contrast with no signal produce". Running a separate control contrast answers that only if the
    control shares the design: effect-estimate noise goes as sqrt(1/nA + 1/nB), so a 4-vs-5 control
    against a 9-vs-10 test returns a band **1.46x too wide** and hides real effects inside "noise".
    Measured on the Trem2 amyloid contrast 2026-08-21 — 3 of 10 hits outside the mismatched band
    against 6 of 10 outside a design-corrected one. This band has no such mismatch by construction: it
    is the same units, the same grid, and the same permutations that produced the p-value.

    ARM CURVES ARE RECONSTRUCTED, NOT RECOMPUTED. Where the null relabels whole units, the arm TOTAL
    is conserved, so with ``S = nA*mA + nB*mB`` fixed and ``d`` the null difference,
    ``mA = (S + nB*d)/(nA+nB)`` and ``mB = (S - nA*d)/(nA+nB)`` exactly. No extra convolutions. This
    fails for cell-level nulls, where a Moran curve is not a mean over cells — hence ``band_arms``
    being None there, and delta_scale getting no band rather than a wrong one.

    Returns ``{"dI_at_max": (lo, hi), "delta_scale_um": (lo, hi) | None, ...}``, pooled over genes and
    permutations exactly as a cross-contrast band would be.
    """
    q = [alpha / 2, 1 - alpha / 2]
    z_null = null_d / sd[None]                                                  # (n_perm, nR, G)
    amax_n = np.abs(z_null).argmax(1)                                           # (n_perm, G)
    dI_null = np.take_along_axis(null_d, amax_n[:, None, :], axis=1)[:, 0, :]   # (n_perm, G)
    out = {"dI_at_max": tuple(float(x) for x in np.nanquantile(dI_null, q)),
           "delta_scale_um": None, "alpha": float(alpha), "n_perm": int(null_d.shape[0]),
           "matched": band_arms is not None, "pooled_over": "genes and permutations"}
    if band_arms is None:
        return out
    nA, nB = band_arms
    S = nA * meanA + nB * meanB                                                 # conserved under relabelling
    ds = np.empty((null_d.shape[0], null_d.shape[2]), float)
    for p in range(null_d.shape[0]):                                            # (nR, G) at a time: (n_perm, nR, G) would
        d = null_d[p]                                                           # be ~0.4 GB on a 1501-gene call
        mA = (S + nB * d) / (nA + nB)
        mB = (S - nA * d) / (nA + nB)
        # ONE orientation sign, the observed one, for every permutation. The convention is a property
        # of the axis, not of the labelling; letting each permutation pick its own would reduce the
        # null on a different convention from the observation and the comparison would be meaningless.
        ds[p] = _char_scale(mA, radii_um, reduction, sign=sign) - \
            _char_scale(mB, radii_um, reduction, sign=sign)
    if np.isfinite(ds).any():
        out["delta_scale_um"] = tuple(float(x) for x in np.nanquantile(ds, q))
    return out


def _degeneracy(pval, dI, structured, alpha=0.05, frac=0.95):
    """Why this axis is not behaving as a per-gene test, or "" if it is.

    ONE GLOBAL EFFECT REPORTED ONCE PER GENE looks exactly like a very powerful test in the output
    table, and nothing else in the result distinguishes them. `general_g` on a paired
    cohort returned raw p<0.05 for 372 of 372 structured genes with 371 in the same direction; the
    per-gene numbers were all individually plausible. Only the JOINT pattern gives it away, so it is
    checked here rather than left to whoever reads the table. Warning, not an error: a strong global
    biological effect can legitimately look like this, and the caller is better placed to judge.
    """
    m = np.asarray(structured, bool)
    if m.sum() < 20:
        return ""
    p, d = np.asarray(pval)[m], np.asarray(dI)[m]
    hit_frac = float(np.mean(p < alpha))
    finite = np.isfinite(d)
    if not finite.any():
        return ""
    same_dir = max(float(np.mean(d[finite] > 0)), float(np.mean(d[finite] < 0)))
    if hit_frac >= frac and same_dir >= frac:
        return (f"{100 * hit_frac:.0f}% of structured features reach raw p<{alpha} and "
                f"{100 * same_dir:.0f}% move the same way")
    return ""


def differential(adatas_A, adatas_B, *, axis="moran", features=None, pairs=None,
                 radii=None, bin_size=None, null="auto", n_perm=200,
                 depth_match=False, density_match=True, structured_min=0.05, seed=0,
                 units_A=None, units_B=None, paired="auto", return_grid=False,
                 strata_key=None, min_cells_per_stratum=None, small_strata="pool",
                 composition=None, min_residual_df=2, layer=None,
                 ordering="max", envelope_alpha=0.05, reduction=None, return_curves=False,
                 _return_internals=False, _field_cache=None):
    """Test whether spatial organization differs between two conditions, on one axis.

    The package's main inferential test. For each feature it builds a per-condition organization
    curve over scale, takes the between-condition difference, and calibrates it against a permutation
    null — reporting a signed effect, the radius at which the studentised difference is largest, and
    a BH-FDR-controlled p-value.
    Because it tests *organization* (not mean expression), it flags genes that spatially reorganize
    without any change in bulk level — invisible to ordinary differential expression.

    The null is chosen from the design: **≥2 samples per condition** → a replicate label-permutation
    null (N1); **exactly one sample per condition on shared coordinates** → a matched pool-and-
    repartition null (N2). Everything downstream of the per-sample curve (permute → mean-difference →
    max-statistic → FDR) is axis-agnostic.

    Parameters
    ----------
    adatas_A, adatas_B : list of AnnData
        Samples for condition A and B (each with ``obsm["spatial"]``). ``≥2`` each selects the
        replicate null; exactly one each (shared coordinates) selects the matched null.
    axis : {"moran", "geary", "focality", "losh", "mean"}, optional
        Organization axis, per gene (pass ``features``). ``"losh"`` is computed
        on the raw binned values, not on the centred field. ``"mean"`` is the mean binned
        expression level, the same at every radius, so it has no scale content. ``"general_g"``
        and ``"texture"`` are refused with a ``ValueError`` that gives the reason (see
        ``REFUSED_DIFF_AXES``). Default ``"moran"``.
    features : list of str, optional
        Genes to test (required for per-gene axes).
    radii : sequence of int, optional
        Neighbourhood radii in grid bins. Default: ``RADII_N`` = 19 log-spaced radii running from
        ``RADII_BINS_LO`` = 2 bins to ``RADII_PITCH_HI`` = 40 × the pooled median cell pitch, matching
        the curve bank. Two bins is a geometry floor, not a convention — a one-bin disc is a five-bin
        cross whose area is off by 27% against the circle it stands for (4 bins against pi*r^2 =
        3.14, measured on ``spatial.neighbor_kernel``, which excludes the centre bin; counting the
        centre it is 5 bins and 59%).
    bin_size : float, optional
        Grid bin size in coordinate units. Default: ``K_PITCH`` × the median cell pitch pooled over
        BOTH arms, so the two conditions share one lattice and one measuring stick. An absolute
        micrometre default cannot do that across platforms whose pitch differs several-fold.
    null : {"auto", "replicate", "matched"}, optional
        Null model. ``"auto"`` picks by sample count. Default ``"auto"``.
    n_perm : int, optional
        Permutations for the null. Default ``200``.
    depth_match, density_match : bool, optional
        Nuisance controls for the matched (N2) null only (no-ops under the replicate null).
        ``depth_match`` also requires a ``"counts"`` layer. Defaults ``False`` / ``True``.
    structured_min : float, optional
        Minimum peak organization for a feature to be tested (else reported unstructured). Default ``0.05``.
        A feature whose tested curve is undefined (NaN) at some radius in any unit is also reported
        unstructured, with ``p = 1`` and a warning: it has no test as given (changed in 0.13.1; before,
        its p fell to the permutation floor).
    units_A, units_B : list, optional
        Experimental-unit labels (e.g. patient) parallel to the sample lists. When given, per-sample
        curves are averaged within a unit before the null and the permutation shuffles units, so
        multiple sections from one donor count as a single replicate rather than pseudo-replicates.
    paired : {"auto", True, False}, optional
        Whether the arms are PAIRED (the same units in both, e.g. a donor at baseline vs on-treatment).
        ``"auto"`` (default) detects a paired design when ``units_A`` and ``units_B`` share labels and then
        uses a within-unit sign-flip null (matching each unit's A/B curve); an unpaired label-permutation
        would double-count shared units and shuffle them across arms — the wrong null. ``True``/``False``
        force paired / unpaired. Requires ``≥2`` shared units when paired.
    seed : int, optional
        Random seed for the permutation null (and the matched-null depth thinning). Default ``0``.
    return_grid : bool, optional
        If ``True`` also return the internal binning grid. Default ``False``.
    return_curves : bool, optional
        If ``True`` also return a dict of the curves the test was computed from, for plotting
        (:mod:`spectrast.pl`: ``diff_curves``, ``unit_curves``, ``diff_heatmap``,
        ``scale_shift``, ``neighborhood_maps``). Default ``False``. The result table is identical either way. Keys:

        * ``radii_um`` (n_radii,), ``radii_bins`` (list of int), ``bin_size`` (coordinate units);
        * ``ids`` (feature names, the table's ``gene`` column), ``id_col``, ``axis``;
        * ``unit_curves_A`` / ``unit_curves_B``: (n_units, n_radii, n_features), one tested curve
          per unit (per section when no units are given). Under the paired null only the units
          present in both arms, which are the ones tested. ``None`` under the matched null, which
          has no units. In ``composition=`` mode these are the residual curves that were tested;
        * ``units_A`` / ``units_B``: a label per unit (``units_A``/``units_B`` labels when given,
          else ``"A0", "A1", ...`` by position in ``adatas_A``); ``None`` under the matched null;
        * ``unit_sections_A`` / ``unit_sections_B``: for each unit, the positions in ``adatas_A`` /
          ``adatas_B`` of the sections averaged into it; ``None`` under the matched null;
        * ``section_labels_A`` / ``section_labels_B``: a name per input section (``"A0", ...``;
          :func:`differential_cohort` fills in the ``sample_key`` values);
        * ``meanA``, ``meanB`` (n_radii, n_features): the arms' mean curves;
        * ``diff`` (n_radii, n_features): the tested difference, ``meanA - meanB`` (under the
          paired null the mean of within-unit differences, equal to it up to rounding);
        * ``sd`` (n_radii, n_features): the null standard deviation per radius that studentises
          ``diff``;
        * ``scale_meanA_um``, ``scale_meanB_um`` (n_features,): the characteristic scale of each
          arm's MEAN curve, reduced on one shared orientation sign; their difference is the table's
          ``delta_scale_um`` exactly. ``NaN`` where a curve has no half-max crossing, and throughout
          in ``composition=`` mode. These differ from the table's ``scale_A_um`` /
          ``scale_B_um``, which are the median over units of each unit's own scale;
        * ``T_alpha`` (n_features,) and ``band = T_alpha * sd``: the simultaneous envelope;
          ``exceed = abs(diff) > band`` (n_radii, n_features), whose column sums are the table's
          ``n_radii_exceeded``; ``envelope_alpha``;
        * ``labels``: ``(label_A, label_B)`` when known (:func:`differential_cohort` sets them from
          its condition values), else ``None``;
        * ``null`` (the table's ``null_used``) and ``n_perm`` (the permutations actually drawn).
    strata_key : str, optional
        **Condition-on** composition adjustment (``matched`` null only): draw each permutation's arm
        membership *within* cell-type strata, so every permutation carries the observed per-arm
        composition. Not supported under the replicate null — N1 permutes sample labels, so there is
        no cell-level permutation to stratify; use ``composition=`` there instead. See
        :mod:`spectrast.composition`.
    min_cells_per_stratum, small_strata
        Guardrail for strata too small to permute (default 50, ``"pool"``).
    composition : str or array-like, optional
        **Regress-out** composition adjustment (``replicate`` null): an ``obs`` column of cell-type
        labels (per-sample fractions are computed for you) or an explicit ``(n_samples, n_types)``
        array ordered ``A`` then ``B``. Each unit's organization curve is regressed on its
        composition and the permutation null runs on the residuals, which removes the LINEAR part of
        composition and handles continuous gradients that discrete strata cannot.

        Two failure modes are guarded: too few units for the number of covariates (refused — the
        residuals would be ~0 and every gene would read non-significant, which is not a null result),
        and composition nearly predicting the arm label (warned — then residualizing removes the
        contrast itself, not a nuisance). ``delta_scale_um`` is ``NaN`` in this mode: a residual
        curve has no characteristic length in µm.
    min_residual_df : int, optional
        Minimum ``n_units - n_types - 1`` for the regress-out mode. Default ``2``.
    layer : str, optional
        Read expression from ``adata.layers[layer]`` in every section instead of ``adata.X``.
        ``None`` (default) uses ``adata.X``. Cannot be combined with ``depth_match=True`` under the matched null
        when the sections carry a ``"counts"`` layer (depth matching thins that layer itself);
        that combination raises ``ValueError``.
    ordering : {"max", "erl"}, optional
        How each feature's p-value ranks its whole studentised difference curve against the null.
        ``"max"`` (default) uses the largest absolute studentised difference over radii,
        ``T_max``. ``"erl"`` uses the extreme rank length (Myllymäki et al. 2017, JRSS-B): it
        compares the sorted per-radius ranks of the observed curve with those of each null curve,
        so a difference that is moderate at every radius can outrank a single sharp peak. Only
        ``p`` and ``fdr`` change; ``T_max``, ``r_at_max_um``, ``dI_at_max`` and the envelope
        columns are the same under both. Recorded in ``df.attrs["spectrast_ordering"]``. Any
        other value raises ``ValueError``.
    envelope_alpha : float, optional
        Level of the simultaneous envelope. Per feature, ``T_alpha`` is the ``1 - envelope_alpha``
        quantile of the null ``T_max`` values, and the band at radius ``r`` is
        ``T_alpha * sd(r)``, with ``sd`` the null standard deviation at that radius. The columns
        ``n_radii_exceeded``, ``radii_exceeded_um``, ``r_diverge_lo_um`` and ``r_diverge_hi_um``
        read where the observed difference leaves this band, and
        ``df.attrs["spectrast_null_band"]`` reports the central ``1 - envelope_alpha`` interval of
        the null effect sizes. It does not change ``p`` or ``fdr``. With ``ordering="max"`` a
        feature with ``p`` below ``envelope_alpha`` leaves its band at one radius at least.
        Default ``0.05``.
    reduction : {"half_max", "centroid"}, optional
        How each arm's organization curve is reduced to a characteristic length before differencing.
        ``None`` uses the package default, which is ``"half_max"`` as of 2026-08-18 and was
        ``"centroid"`` before it — so **this changes every published ``delta_scale_um``**. Pass
        ``"centroid"`` to reproduce pre-flip numbers. Both arms are always reduced on one shared
        orientation sign. See :mod:`spectrast.scale`.

    Returns
    -------
    pandas.DataFrame
        One row per feature with the stable column contract: ``gene``, ``axis``,
        ``delta_scale_um``, ``dI_at_max``, ``T_max``, ``r_at_max_um`` (the ARGMAX of the studentised
        difference — **not** a characteristic scale of the change: the null's spread falls with
        radius, which pushes this argmax toward coarse radii; use ``radii_exceeded_um`` for extent), ``p``, ``fdr``, ``null_used``,
        ``structured``, ``peak_signal``.

        ``delta_scale_um`` is a DESCRIPTIVE shift, not the tested statistic — on a real
        two-condition contrast it is nearly uncorrelated with ``T_max``, while ``|dI_at_max|``
        correlates strongly with it. Read
        ``delta_scale_resolvable`` beside it: the shift is reported as resolvable only when it
        exceeds the mean of the two arms' BETWEEN-UNIT scale spreads. ``r_diverge_lo_um`` /
        ``r_diverge_hi_um`` give the radius interval over which the curves leave the envelope —
        the honest answer to "at what scale do they differ" — and ``dI_raw_peak_um`` is the raw
        difference's own peak beside the studentised ``r_at_max_um``.

        All radii and scales are in coordinate units (µm for micrometre coordinates). The columns,
        in table order:

        * ``p``: the permutation p-value under ``ordering``; ``fdr``: Benjamini–Hochberg over all
          rows. A feature below ``structured_min`` gets ``p = 1`` and ``T_max = 0``.
        * ``n_radii_exceeded``: how many radii leave the envelope (see ``envelope_alpha``);
          ``radii_exceeded_um``: those radii as one string joined by ``"|"`` (empty when none).
        * ``null_used``: ``"replicate"``, ``"replicate-paired"`` or ``"matched"``.
        * ``structured``: whether the feature passed ``structured_min`` and had a finite tested
          curve; ``peak_signal``: the gate value it was judged on, the largest value over radii
          and both arms of the arm's Moran curve (averaged over units under the replicate
          null).
        * ``r_diverge_lo_um``, ``r_diverge_hi_um``: smallest and largest radius outside the
          envelope, ``NaN`` when none.
        * ``dI_raw_peak_um``, ``dI_raw_at_peak``: radius and signed value of the largest absolute
          unstudentised difference.
        * ``scale_A_um``, ``scale_B_um``: per arm, the geometric median over units of each unit's
          own characteristic scale; ``scale_A_sd_log2``, ``scale_B_sd_log2``: the between-unit
          standard deviation of log2 scale (``NaN`` with fewer than two units). All four are
          ``NaN`` under the matched null and in ``composition=`` mode.
        * ``n_units_A``, ``n_units_B``: units tested per arm (shared units only under the paired
          null); ``NaN`` under the matched null.
        * ``delta_scale_resolvable``: nullable boolean, ``True`` when ``abs(log2(A / B))`` of the
          two mean-curve scales exceeds the mean of ``scale_A_sd_log2`` and ``scale_B_sd_log2``;
          ``<NA>`` where the shift or either spread is undefined.

        ``df.attrs`` holds ``spectrast_grid`` (``bin_size``, ``radii`` in bins, ``radii_um``),
        ``spectrast_reduction``, ``spectrast_ordering``, ``spectrast_envelope_alpha``,
        ``spectrast_design`` (the :func:`design_floor` dict plus ``n_units_A``, ``n_units_B`` and
        ``paired``; its numeric fields are ``None`` under the matched null),
        ``spectrast_null_band`` (null quantiles of ``dI_at_max`` and ``delta_scale_um``; ``None``
        in ``composition=`` mode), ``spectrast_degenerate`` (a description of a
        one-global-effect pattern, or ``""``), ``spectrast_backend`` and
        ``spectrast_composition``.

        With ``return_grid=True`` the return is ``(df, grid)``; with ``return_curves=True`` it is
        ``(df, curves)``; with both, ``(df, grid, curves)``. ``grid`` is a DataFrame with one row
        per feature and radius: the ``gene`` column, ``scale_um`` (the radius), ``dI``
        (the signed difference there), ``p`` (a per-radius two-sided permutation p, not corrected
        across radii; 1 where ``dI`` is undefined) and ``fdr`` (BH over all feature × radius rows).

    See Also
    --------
    differential_omnibus : screen several axes at once with one FDR pass.
    groups_from_obs : build the A/B sample lists from an ``obs`` condition key.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> def toy():
    ...     a = ad.AnnData(np.abs(rng.standard_normal((300, 6))).astype("float32"))
    ...     a.obsm["spatial"] = rng.random((300, 2)) * 100
    ...     a.var_names = [f"g{i}" for i in range(6)]
    ...     return a
    >>> A, B = [toy(), toy()], [toy(), toy()]
    >>> res = spectrast.diff.differential(A, B, axis="moran", features=["g0", "g1"], n_perm=20)
    >>> list(res.columns)[:3]
    ['gene', 'axis', 'delta_scale_um']
    """
    import pandas as pd, scipy.sparse as sp
    radii, bin_size = _pitch_anchored_grid(adatas_A, adatas_B, radii, bin_size)
    rng = np.random.default_rng(seed); radii = list(radii); radii_um = np.array(radii) * bin_size
    A = list(adatas_A); B = list(adatas_B); is_pair = axis in PAIR_AXES
    backend_at_start = _check_backend_consistency(A, B)
    from . import composition as _comp
    comp_prov = dict(conditioned=False)
    if strata_key is not None and composition is not None:
        raise ValueError(
            "pass strata_key= OR composition=, not both: condition-on and regress-out answer "
            "different questions (see spectrast.composition) and stacking them adjusts twice for "
            "the same thing, which removes signal without a defined interpretation.")
    if is_pair:
        assert pairs is not None, "pair axis needs pairs=[(A,B), ...]"
        pairs = [tuple(p) for p in pairs]
        if all(isinstance(m, str) for pr in pairs for m in pr):
            members = sorted({m for pr in pairs for m in pr})          # unchanged: the gene-gene path
        else:
            # specs are not orderable across types (str vs tuple vs callable), so de-duplicate by
            # first appearance. Per-pair curves are unaffected by member ORDER — each mark is its own
            # column and the pair indices are remapped consistently.
            members, seen = [], set()
            for pr in pairs:
                for m in pr:
                    k = _mark_key(m)
                    if k not in seen:
                        seen.add(k); members.append(m)
        midx = {_mark_key(g): k for k, g in enumerate(members)}
        pairs_idx = [(midx[_mark_key(a)], midx[_mark_key(b)]) for a, b in pairs]
        extract = members
        ids = [f"{_mark_label(a)}~{_mark_label(b)}" for a, b in pairs]
        id_col = "pair"
    else:
        # refusal FIRST: a refused axis need not be in STAT (texture has no per-radius curve at all),
        # and "unknown axis" would send the caller looking for a typo instead of reading the reason.
        if axis in REFUSED_DIFF_AXES:
            raise ValueError(f"axis={axis!r} is not admissible as a differential test. "
                             f"{REFUSED_DIFF_AXES[axis]}")
        assert axis in STAT, f"unknown axis {axis!r}"
        assert features is not None, "per-gene axis needs features=[gene, ...]"
        extract = list(features)
        ids, id_col = [_mark_label(f) for f in extract], "gene"
    G = len(ids)
    if (units_A is None) != (units_B is None):
        # One list alone used to skip unit grouping without a word, so repeated sections from one donor counted
        # as independent replicates (fixed in 0.14.1). Refuse rather than guess what the other arm is.
        raise ValueError(
            "units_A and units_B must be given together (or neither): with only one, sections cannot be "
            "grouped consistently and repeated sections would count as independent replicates. If the other "
            "arm has one section per unit, pass its section labels, e.g. units_B=[f'B{i}' for i in range(len(B))].")
    null_requested = null                                              # before resolution: was this a
    if null == "auto":                                                 #   silent fallback or an explicit choice?
        null = "replicate" if (len(A) >= 2 and len(B) >= 2) else "matched"
    if null_requested == "auto" and null == "matched" and (units_A is not None or units_B is not None):
        warnings.warn(
            "differential(): units were supplied but the null resolved to 'matched', which "
            "repartitions CELLS and therefore treats cells as replicates — the pseudoreplication "
            "failure mode. Pass null='replicate' to permute unit labels (and see diff.design_floor "
            "for whether this design can support it).", stacklevel=3)

    # Matched-null default, set HERE (immediately after the null kind is resolved, before any
    # branch runs) so the replicate branches below OVERWRITE it with their real unit counts rather
    # than this default clobbering theirs if it were set later. No units exist under the matched
    # (cell-repartition) null, so every design-floor field is None.
    _design_attrs = dict(kind="matched", n_partitions=None, p_floor=None, alpha=None,
                         n_features=None, fdr_reachable=None,
                         n_units_A=None, n_units_B=None, paired=False)

    if depth_match and null == "matched":                             # M10: thinning is a matched-null nuisance
        meds = [np.median(t[t > 0]) for ad in A + B                    #   control ONLY; under the replicate null
                for t in [np.asarray(ad.layers["counts"].sum(1)).ravel()] if "counts" in getattr(ad, "layers", {})]
        depth_to = min(meds) if meds else None                        #   it must be a no-op (as the docstring says)
    else:
        depth_to = None

    # per-sample curve on the tested axis; per-gene axes also carry a Moran GATE curve (structure detector)
    def _curve(xy, Xg, H=None, W=None, binidx=None):
        if is_pair:
            fld, mask, H, W, binidx = _bin_fields(xy, Xg, bin_size, H, W, binidx)
            return PAIR_STAT[axis](fld, pairs_idx, mask, radii_um, bin_size, _counts(binidx, H, W))
        return STAT[axis](xy, Xg, radii, bin_size, H, W, binidx)

    def _gate(xy, Xg, H=None, W=None, binidx=None):
        # per-gene: Moran scalogram (the universal structure gate); pairs: |coexpr| of the axis curve itself
        if is_pair:
            return np.abs(_curve(xy, Xg, H, W, binidx))
        return _scalogram(xy, Xg, radii, bin_size, H, W, binidx)

    def _curve_gate(ad):
        """(axis curve, structure gate) for ONE sample, sharing a single bin + z-stack (E1). For axis='moran'
        the gate IS the curve (same reduction on the same field) -> computed once. For focality/geary the
        Moran gate reuses the shared z-stack instead of re-binning. Byte-identical to _curve()/_gate().
        The binned (fld, mask, z, counts) are axis-INDEPENDENT, so ``_field_cache`` (keyed per sample) lets an
        omnibus reuse them across its per-axis calls (E2) — the cache never changes a value or the null."""
        key = id(ad)
        if _field_cache is not None and key in _field_cache:
            fld, mask, z, cnt = _field_cache[key]
        else:
            xy, Xg = _prep_marks(ad, extract, depth_to, seed, layer)
            fld, mask, H_, W_, binidx_ = _bin_fields(xy, Xg, bin_size)
            z = None if is_pair else _zstack(fld, mask)
            cnt = _counts(binidx_, H_, W_)      # cheap, and always present so a mixed-axis omnibus
            #                                     cannot pull a cache entry that lacks it
            if _field_cache is not None:
                _field_cache[key] = (fld, mask, z, cnt)
        if is_pair:
            c = PAIR_STAT[axis](fld, pairs_idx, mask, radii_um, bin_size, cnt)
            return c, np.abs(c)
        from .spatial import (morans_i_multiscale, gearys_c_multiscale, focality_scalogram,
                              general_g_multiscale, losh_multiscale)
        if axis in FIRST_ORDER_AXES:
            # first-order: read the LEVEL off the already-binned field. It must not come off the
            # z-stack, which demeans in-mask and would divide the very quantity being tested away.
            m = (fld * mask[..., None]).sum((0, 1)) / max(mask.sum(), 1)
            c = np.repeat(m[None, :], len(radii), axis=0)
            return c, morans_i_multiscale(z, mask, radii, kind="disc")
        if axis == "geary":
            c = 1.0 - gearys_c_multiscale(z, mask, radii, kind="disc")
        elif axis == "focality":
            c = focality_scalogram(z, mask, radii, kind="disc")
        elif axis in UNCENTRED_AXES:
            # General G and LOSH are defined on the VALUES, so they get the raw binned field. Handing
            # them `z` would centre the input: General G's denominator becomes a sum of signed products
            # that can pass through zero, and LOSH's residuals would be taken about a global mean it
            # has already removed. The gate stays Moran, as for every other second-order axis.
            raw = fld * mask[..., None]
            c = (general_g_multiscale(raw, mask, radii, kind="disc") if axis == "general_g"
                 else losh_multiscale(raw, mask, radii, kind="disc"))
        elif axis == "moran":                                          # gate == curve
            c = morans_i_multiscale(z, mask, radii, kind="disc")
            return c, c
        else:
            # NOT a silent fallback. This branch used to be `else: <moran>`, so an axis added to STAT
            # but never wired in here computed MORAN and returned it under the new axis's name --
            # `general_g` and `losh` did exactly that, with identical hits, identical run times and a
            # Jaccard of 1.00 against moran, which is what gave it away. Registering an axis is two
            # edits, and this raise is what makes forgetting the second one loud.
            raise AssertionError(
                f"axis {axis!r} is in STAT but has no branch in _curve_gate; add one rather than "
                "letting it fall through to Moran")
        return c, morans_i_multiscale(z, mask, radii, kind="disc")     # Moran gate on the shared z-stack

    null_used = null
    band_arms = None      # (nA, nB) in UNITS, set only by nulls that relabel whole units
    # per-unit curve stacks for the per-arm scale + between-unit spread columns (assembled below,
    # just before the DataFrame is built). None under the matched null, which has no units at all.
    unit_curves_A = unit_curves_B = None
    # unit labels and the input-section positions behind each unit, parallel to unit_curves_*; carried
    # only into the optional `return_curves` dict, never into the table. None under the matched null.
    unit_names_A = unit_names_B = unit_secs_A = unit_secs_B = None
    if strata_key is not None and null == "replicate":
        # The replicate null permutes SAMPLE LABELS, not cells, so there is no cell-level permutation
        # to draw within a stratum — a within-stratum variant of it is simply not defined. Refuse
        # rather than run the unconditioned null under a conditioned-sounding argument, which is the
        # exact silent-wrong-answer this feature exists to prevent.
        raise ValueError(
            "strata_key is not supported with the replicate (N1) null: N1 permutes sample labels, "
            "not cells, so there is no within-stratum permutation to draw. Options: (a) use the "
            "matched null (null='matched', one sample per arm on a shared coordinate frame), where "
            "the repartition is over cells and is conditioned exactly; or (b) adjust at the curve "
            "level instead with spectrast.composition.residualize on a per-unit composition matrix "
            "(see composition.composition_matrix), which is the right tool for a replicate design.")
    if null == "replicate":
        pa = [_curve_gate(ad) for ad in A]
        pb = [_curve_gate(ad) for ad in B]
        IA = np.stack([c for c, _ in pa]); IB = np.stack([c for c, _ in pb])                 # (nS, nR, G)
        gA = np.stack([g for _, g in pa]); gB = np.stack([g for _, g in pb])                 # per-section gate curves
        do_paired = False; labA = labB = None
        unit_names_A = [f"A{i}" for i in range(len(A))]; unit_secs_A = [[i] for i in range(len(A))]
        unit_names_B = [f"B{i}" for i in range(len(B))]; unit_secs_B = [[i] for i in range(len(B))]
        if units_A is not None and units_B is not None:                    # aggregate sections -> one curve per unit
            uA = np.asarray(units_A); uB = np.asarray(units_B)
            labA = list(pd.unique(uA)); labB = list(pd.unique(uB))
            unit_names_A = [str(v) for v in labA]; unit_secs_A = [np.flatnonzero(uA == v).tolist() for v in labA]
            unit_names_B = [str(v) for v in labB]; unit_secs_B = [np.flatnonzero(uB == v).tolist() for v in labB]
            def _agg(I, u, labs):
                u = np.asarray(u); return np.stack([I[u == v].mean(0) for v in labs])
            IA, IB = _agg(IA, uA, labA), _agg(IB, uB, labB)               # (nUnits, nR, G)
            gA, gB = _agg(gA, uA, labA), _agg(gB, uB, labB)               # gate too: one section-heavy unit must not
            shared = [v for v in labA if v in set(labB)]                  #   dominate the structure gate
            do_paired = bool(paired is True or (paired == "auto" and shared))
        gateA, gateB = gA.mean(0), gB.mean(0)
        # regress-out (mechanism B): remove the part of each unit's organization curve that is
        # linearly predictable from that unit's composition, then run the ordinary label-permutation
        # null on the RESIDUALS. This is the composition adjustment that fits a replicate design,
        # where condition-on has no cell-level permutation to stratify. The structure gate stays on
        # the RAW curves — it asks whether there is any signal to compare, which residualizing would
        # destroy.
        if composition is not None:
            C_samples, ctypes = _arm_composition(A, B, composition)
            CA, CB = C_samples[:len(A)], C_samples[len(A):]
            if units_A is not None and units_B is not None:
                CA, CB = _agg(CA, uA, labA), _agg(CB, uB, labB)      # aggregate covariates like curves
            C_all = np.vstack([CA, CB])
            arm = np.r_[np.ones(len(IA), bool), np.zeros(len(IB), bool)]
            design = _comp.check_residual_design(C_all, arm, min_residual_df=min_residual_df)
            I_all = np.concatenate([IA, IB], 0)
            res = _comp.residualize(I_all.reshape(len(I_all), -1), C_all).reshape(I_all.shape)
            IA, IB = res[:len(IA)], res[len(IA):]
            comp_prov.update(conditioned=True, mode="residual", covariates=list(ctypes), **design)
        # Hoisted for the per-arm scale + between-unit spread columns, assembled just before the
        # DataFrame is built. Taken AFTER any residualisation (so a residual mode's stacks are the
        # residual curves, consistent with the comp_prov.mode == "residual" NaN guard down there) and
        # AFTER per-unit aggregation (one curve per unit, not per section). The paired sub-branch below
        # overrides this with the SHARED-unit-only stacks, since only shared units enter that test.
        unit_curves_A, unit_curves_B = IA, IB
        if do_paired:
            # PAIRED / longitudinal design: the SAME unit (donor) appears in both arms. An unpaired
            # label-permutation would double-count that donor and shuffle donors freely across arms — the wrong
            # null. Match each shared unit's A and B curve and use a within-unit SIGN-FLIP null (the exact
            # exchangeability under H0: the A/B label within a donor is arbitrary).
            if len(shared) < 2:
                raise ValueError("paired differential needs >=2 units shared across arms; got "
                                 f"{len(shared)}. Pass paired=False to force the unpaired null.")
            ia = {v: k for k, v in enumerate(labA)}; ib = {v: k for k, v in enumerate(labB)}
            D = np.stack([IA[ia[v]] - IB[ib[v]] for v in shared])          # (nPairs, nR, G) within-unit deltas
            # the structure gate from the SAME units the test uses: averaged over every unit, a section with no
            # partner could move a gene across structured_min without touching a tested pair (fixed in 0.14.1)
            gateA = np.stack([gA[ia[v]] for v in shared]).mean(0)
            gateB = np.stack([gB[ib[v]] for v in shared]).mean(0)
            obs = D.mean(0)
            sup = _perm_support("signflip", len(shared))
            n_perm = _clamp_perms(n_perm, sup, "paired sign-flip",
                                  floor_num=_perm_floor_num("signflip", len(shared)))
            if sup - 1 <= n_perm:
                # EXACT, same convention as the unpaired branch: every sign vector except the observed
                # all-positive one. Its negation is ALWAYS a valid sign vector and always as
                # extreme, so the floor here is exactly 2/support -- unlike the unpaired branch,
                # a sign-flip design cannot be "unbalanced" in the sense that breaks that argument.
                signs = ((np.arange(1, sup)[:, None] >> np.arange(len(shared))[None, :]) & 1) * 2.0 - 1.0
                n_perm = sup - 1
                null_d = (D[None] * signs[:, :, None, None]).mean(1)
            else:
                null_d = np.empty((n_perm, len(radii), G), float)
                for p in range(n_perm):
                    s = rng.choice([-1.0, 1.0], size=len(shared))[:, None, None]
                    null_d[p] = (D * s).mean(0)
            unit_curves_A = np.stack([IA[ia[v]] for v in shared])         # only the shared units were
            unit_curves_B = np.stack([IB[ib[v]] for v in shared])         # actually tested -- match that
            unit_secs_A = [unit_secs_A[ia[v]] for v in shared]
            unit_secs_B = [unit_secs_B[ib[v]] for v in shared]
            unit_names_A = unit_names_B = [str(v) for v in shared]
            meanA = unit_curves_A.mean(0)
            meanB = unit_curves_B.mean(0)
            band_arms = (len(shared), len(shared))
            null_used = "replicate-paired"
            _design_attrs = dict(**design_floor(len(shared), len(shared), paired=True,
                                                n_features=int(G)),
                                 n_units_A=len(shared), n_units_B=len(shared), paired=True)
        else:
            obs = IA.mean(0) - IB.mean(0)                                                      # (nR, G)
            allI = np.concatenate([IA, IB], 0); nA = len(IA); labels = np.arange(len(allI))
            sup = _perm_support("unpaired", nA, len(IB))
            n_perm = _clamp_perms(n_perm, sup, "replicate label",
                                  floor_num=_perm_floor_num("unpaired", nA, len(IB)))
            null_d = np.empty((n_perm, len(radii), G), float)          # float64: identity perm must equal obs exactly
            if sup - 1 <= n_perm:
                # EXACT: every labelling except the observed one, each exactly once. The clamp's own
                # convention is `n_perm = support - 1`, which makes the reported
                # `(1 + #{null >= obs}) / (n_perm + 1)` the exact permutation p `#{all >= obs} /
                # support`. The observed labelling is therefore excluded here rather than counted
                # twice. Its complement stays in and is always as extreme ONLY WHEN THE COMPLEMENT
                # IS ITSELF ONE OF THESE LABELLINGS: for nA == nB the arm swap maps an nA-subset to
                # another nA-subset, giving the floor 2/support — 0.029 at 4-vs-4, where the sampler
                # was delivering 0.057. For nA != nB the complement has size nB and is not in
                # `sel_all` at all, so the floor is 1/support. `_perm_floor_num` owns that rule and
                # `design_floor` reports it; do not restate it as unconditional (corrected 2026-09-28).
                from itertools import combinations
                obs_sel = tuple(range(nA))
                sel_all = [s for s in combinations(range(len(allI)), nA) if s != obs_sel]
                n_perm = len(sel_all)
                null_d = np.empty((n_perm, len(radii), G), float)
                for p, sel in enumerate(sel_all):
                    m = np.zeros(len(allI), bool); m[list(sel)] = True
                    null_d[p] = allI[m].mean(0) - allI[~m].mean(0)
            else:
                for p in range(n_perm):
                    pm = rng.permutation(labels)
                    null_d[p] = allI[pm[:nA]].mean(0) - allI[pm[nA:]].mean(0)
            meanA, meanB = IA.mean(0), IB.mean(0)
            band_arms = (nA, len(IB))
            _design_attrs = dict(**design_floor(nA, len(IB), n_features=int(G)),
                                 n_units_A=int(nA), n_units_B=int(len(IB)), paired=False)
    else:
        assert len(A) == 1 and len(B) == 1, "matched null needs exactly one sample per condition"
        xyA, XA = _prep_marks(A[0], extract, depth_to, seed, layer)
        xyB, XB = _prep_marks(B[0], extract, depth_to, seed, layer)
        # M9: the matched null overlays both samples on ONE grid and treats their cells as exchangeable — valid
        # only on a SHARED coordinate frame (split / co-registered sections). Warn if the bounding boxes barely
        # overlap, which means two unrelated sections are being pooled and the repartition null is uninterpretable.
        loA, hiA = xyA.min(0), xyA.max(0); loB, hiB = xyB.min(0), xyB.max(0)
        inter = np.maximum(0.0, np.minimum(hiA, hiB) - np.maximum(loA, loB)).prod()
        union = np.prod(np.maximum(hiA, hiB) - np.minimum(loA, loB))
        if union > 0 and inter / union < 0.5:
            # NOTE: no local `import warnings` here — the module already imports it, and a
            # function-scoped import makes the name local to the WHOLE function, so any earlier use
            # raises UnboundLocalError. That is exactly what happened when the delta_scale warning
            # was added above this line.
            warnings.warn(f"differential(null='matched'): A and B bounding boxes overlap only "
                          f"{inter / union:.0%} — the matched null assumes a shared coordinate frame "
                          f"(split/registered sections). Results may be uninterpretable for unrelated sections.")
        # read the stratum labels BEFORE any subsetting, then carry them through the same selection
        labA = labB = None
        if strata_key is not None:
            labA = _comp.strata_from_obs(A[0], strata_key)
            labB = _comp.strata_from_obs(B[0], strata_key)
        if density_match:
            n = min(XA.shape[0], XB.shape[0])
            selA = rng.choice(XA.shape[0], n, replace=False); selB = rng.choice(XB.shape[0], n, replace=False)
            xyA, XA, xyB, XB = xyA[selA], XA[selA], xyB[selB], XB[selB]
            if labA is not None:
                labA, labB = labA[selA], labB[selB]
        # composition-aware repartition: draw arm A stratum by stratum so every permutation carries
        # the OBSERVED per-type composition in both arms. A purely compositional difference then has
        # nowhere left to show up, which is the whole point. Resolved BEFORE the grid is built:
        # small_strata="drop" returns a keep-mask that used to be discarded here, so the "excluded"
        # cells stayed in every curve (fixed in 0.14.1). They now leave both arms first.
        strata_pool = None
        if strata_key is not None:
            mc = _comp.DEFAULT_MIN_CELLS if min_cells_per_stratum is None else int(min_cells_per_stratum)
            pool_lab, keep, comp_report = _comp.apply_min_cells(
                np.concatenate([labA, labB]), mc, small_strata)
            comp_prov.update(comp_report, conditioned=True, null="matched-within-stratum")
            kA, kB = keep[:len(labA)], keep[len(labA):]
            if not keep.all():
                xyA, XA, xyB, XB = xyA[kA], XA[kA], xyB[kB], XB[kB]
            strata_pool = np.concatenate([pool_lab[:len(labA)][kA], pool_lab[len(labA):][kB]])
        xy = np.vstack([xyA, xyB]); X = sp.vstack([XA, XB]).tocsr()
        H, W, binidx = _grid(xy, bin_size); nAcells = XA.shape[0]
        stateA = np.zeros(X.shape[0], bool); stateA[:nAcells] = True
        meanA = _curve(xy[stateA], X[stateA], H, W, binidx[stateA])
        meanB = _curve(xy[~stateA], X[~stateA], H, W, binidx[~stateA])
        gateA = _gate(xy[stateA], X[stateA], H, W, binidx[stateA])
        gateB = _gate(xy[~stateA], X[~stateA], H, W, binidx[~stateA])
        obs = meanA - meanB
        pooled = np.arange(X.shape[0])
        # draw ALL permutations up front (sequential RNG -> identical to the serial loop, worker-count
        # independent), then fan the expensive per-perm _curve pair out across threads (Phase 2).
        if strata_pool is None:
            perms = [rng.permutation(pooled) for _ in range(n_perm)]
        else:                                    # per-stratum arm sizes held at their observed values
            perms = [_comp.stratified_partition(strata_pool, stateA, rng) for _ in range(n_perm)]
        def _perm_diff(pm):
            Ap = pm if pm.dtype == bool else np.zeros(X.shape[0], bool)
            if pm.dtype != bool:
                Ap[pm[:nAcells]] = True
            return _curve(xy[Ap], X[Ap], H, W, binidx[Ap]) - _curve(xy[~Ap], X[~Ap], H, W, binidx[~Ap])
        from ._parallel import parallel_map
        null_d = np.stack(parallel_map(_perm_diff, perms))

    # structure gate + characteristic-scale effect size (axis-appropriate)
    # np.maximum (NaN-PROPAGATING) is deliberate, not np.fmax: a gene with no spatial signal in one arm has an
    # undefined Moran/coexpr there (den==0 -> NaN curve -> NaN obs). Propagating that NaN into peak_signal gates the
    # gene OUT (structured=False -> p=1). Using fmax would let it pass the gate while obs stays NaN, and the max-stat
    # would then read a spurious tiny p (comparisons with NaN are False) -> a FALSE POSITIVE. Organization can only
    # be *compared* where both arms have organization; expression appearance/disappearance is DE's job, not this axis.
    peak_signal = np.maximum(gateA, gateB).max(0)                       # (G,) peak Moran (per-gene) or peak|coexpr|
    # sA/sB (the MEAN-CURVE reduced scale) are only assigned in the branch below; hoisted to None so
    # delta_scale_resolvable, computed at the assembly point far below, can tell whether they exist
    # without an UnboundLocalError on the pair-axis / residual-mode paths.
    sA = sB = None
    if is_pair:
        delta_scale = np.full(G, np.nan)                               # no characteristic length for signed covariance
    elif comp_prov.get("mode") == "residual":
        # a residual curve is a deviation from the composition fit, not an organization profile, so
        # its positive-mass centroid is not a length in um. Reporting the RAW delta next to a
        # composition-adjusted p would invite exactly the misreading this mode exists to prevent.
        delta_scale = np.full(G, np.nan)
    else:
        # ONE orientation sign for both arms, taken from the two pooled. Letting each arm choose its
        # own would reduce them on different conventions and the difference would be meaningless.
        from . import scale as _scale
        sgn = _scale.orient_sign(np.concatenate([meanA, meanB], axis=1))
        sA = _char_scale(meanA, radii_um, reduction, sign=sgn)
        sB = _char_scale(meanB, radii_um, reduction, sign=sgn)
        delta_scale = sA - sB
        # A one-armed NaN is the reduction DECLINING, and it is worth explaining rather than leaving as
        # a hole in the table. It happens when a gene is gated in on one arm's organization while the
        # other arm has none: an unstructured curve has no characteristic length, so there is nothing
        # to difference. The centroid returned a number for those curves anyway — the radius grid's own
        # centre of mass — which is how an absence of structure got reported as a length shift.
        one_armed = np.isnan(delta_scale) & (np.isfinite(sA) | np.isfinite(sB))
        if one_armed.any():
            frac = float(one_armed.mean())
            warnings.warn(
                f"{one_armed.sum()} of {G} features ({frac:.0%}) have a characteristic length in one "
                f"arm but not the other, so delta_scale_um is NaN for them. That is the reduction "
                f"declining, not a failure: an arm whose organization curve never falls to its own "
                f"mid-range has no length to difference. p, FDR and T_max are unaffected — they do not "
                f"use this column. Inspect the per-arm curves before treating these as missing data, "
                f"and use reduction='centroid' only if you need the pre-2026-08-18 numbers.",
                stacklevel=3)

    # max-statistic (standardize dI by null SD per scale; collapse scale) -> one p per gene/pair
    sd = null_d.std(0) + 1e-9; z_obs = obs / sd                                                # (nR, G)
    amax = np.abs(z_obs).argmax(0)
    Tmax = np.abs(z_obs).max(0); r_at = radii_um[amax]
    dI_at_max = obs[amax, np.arange(G)]                              # signed dI at the argmax radius
    Tmax_null = np.abs(null_d / sd[None]).max(1)                                                # (n_perm, G)
    p_max = (1 + (Tmax_null >= (Tmax - 1e-9)[None]).sum(0)) / (n_perm + 1)  # vectorized over genes (E3)
    structured = peak_signal >= structured_min
    # A NaN anywhere in the TESTED curve -- of any unit, or of any null draw -- while the Moran gate is finite
    # leaves T_max NaN. That happens when a unit is too sparse for some radius to have neighbouring bins:
    # Geary's denominator vanishes there and Moran's does not. No null value compares as larger than NaN, so
    # p fell to its floor, 1 / (n_perm + 1), on a row marked structured: a false positive (found 2026-10-13
    # on a compartment re-measurement). Such a feature has no test as given, so it is gated out exactly as a
    # NaN gate curve already is, and the caller is told how to test it.
    untestable = structured & ~np.isfinite(Tmax)
    if untestable.any():
        warnings.warn(
            f"differential(axis={axis!r}): {int(untestable.sum())} of {G} features have a NaN in the tested "
            f"curve of at least one unit or null draw at some radius (absent there, or a unit too sparse "
            f"for that radius to have neighbouring bins). They are gated out (p = 1, structured = False) "
            f"instead of reading a p at the permutation floor. To test them, drop the units in which they "
            f"are undefined.", stacklevel=3)
        structured = structured & ~untestable
    if ordering not in ("max", "erl"):
        raise ValueError(f"ordering must be 'max' or 'erl', got {ordering!r}")
    pval = _erl_pvalues(z_obs, null_d / sd[None]) if ordering == "erl" else p_max
    pval = np.where(structured, pval, 1.0); Tmax = np.where(structured, Tmax, 0.0)

    # SIMULTANEOUS ENVELOPE, from the same null quantile that produced the p. band(r) = T_alpha*sd(r),
    # so a gene significant at envelope_alpha is exactly a gene whose curve leaves its band somewhere:
    # this says WHERE without changing WHETHER, and cannot disagree with the p by construction.
    # It is not decoration. On the Trem2 amyloid contrast every hit exits its band at ALL FIVE radii,
    # so r_at_max_um names one radius of a five-radius interval -- and the width of that interval was
    # not recoverable from the output at all before these columns existed.
    #
    # r_at_max_um IS NOT A CHARACTERISTIC SCALE OF THE CHANGE. Three independent lines say so and the
    # column name invites the opposite reading, so it is written here next to where it is computed:
    #   1. the envelope above -- every hit exceeds at every radius, so the argmax picks one point of a
    #      wide interval;
    #   2. it moves wholesale with the radius grid (the capstone pins six of nine hits at 30 um, its
    #      own smallest radius; a finer grid spreads the same nine over 30-330 um);
    #   3. injected ground truth -- foci of KNOWN radius give
    #      rho(r_injected, r_at_max) = -0.71 and -0.11 on two grids over a 4.5x range of injected
    #      radii. It does not track a scale that was actually put there.
    # Report it as the argmax of the studentised difference. Read the envelope columns for extent.
    T_alpha = np.quantile(Tmax_null, 1.0 - envelope_alpha, axis=0)                              # (G,)
    exceed = np.abs(obs) > (T_alpha[None, :] * sd)                                              # (nR, G)
    n_exceed = exceed.sum(0)
    where_ex = ["|".join(f"{radii_um[r]:g}" for r in np.nonzero(exceed[:, g])[0]) for g in range(G)]
    # matched null band + degeneracy check, both computed from what the test already produced
    sgn_band = None
    if not is_pair and comp_prov.get("mode") != "residual":
        from . import scale as _sc
        sgn_band = _sc.orient_sign(np.concatenate([meanA, meanB], axis=1))
    nb = _null_band(null_d, sd, radii_um, band_arms, meanA, meanB, reduction,
                    envelope_alpha, sgn_band) if sgn_band is not None else None
    degen = _degeneracy(pval, dI_at_max, structured)
    if degen:
        warnings.warn(
            f"differential(axis={axis!r}): {degen}. That is the signature of ONE GLOBAL effect "
            f"reported once per feature rather than per-feature findings — an axis with no variance "
            f"normalization will do this whenever overall level shifts between the arms. Check the "
            f"effect sizes: near-identical T_max across features, and a negative control that cannot "
            f"reach small p, both confirm it. See diff.REFUSED_DIFF_AXES for the worked example.",
            stacklevel=3)

    # WHERE the curves differ, as an interval. `r_at_max_um` is the argmax of the STUDENTISED
    # difference and `sd_null` falls with radius, so that argmax is pushed coarse (Abi3: reported
    # 120 um, raw dI peaks at 30 um). The envelope already knows the extent; these columns say it.
    ex_idx = [np.nonzero(exceed[:, g])[0] for g in range(G)]
    r_lo = np.array([radii_um[i[0]] if len(i) else np.nan for i in ex_idx], float)
    r_hi = np.array([radii_um[i[-1]] if len(i) else np.nan for i in ex_idx], float)
    # the RAW difference's own peak, beside the studentised one. `argmax` returns 0 for an all-NaN
    # column (no comparison there is ever True), which would misreport the grid's smallest radius as
    # a "peak" that does not exist -- mask those genes explicitly rather than trust the index.
    raw_amax = np.abs(obs).argmax(0)
    all_nan_obs = np.isnan(obs).all(0)
    r_raw = np.where(all_nan_obs, np.nan, radii_um[raw_amax])
    dI_raw = obs[raw_amax, np.arange(G)]                # already NaN here for an all-NaN column
    # per-arm scale with its BETWEEN-UNIT spread -- the error bar a within-section resample cannot
    # give (extension #3's finding). NaN wherever the arm has <2 units, the axis has no characteristic
    # length (pair axes), or the curve is a composition residual (no length in um by construction --
    # same reasoning as the existing delta_scale_um NaN in residual mode; a scale +/- spread there
    # would contradict it). See `_arm_scale`'s docstring for how this differs from `delta_scale_um`'s
    # own `sA`/`sB` -- use these two columns for "what scale is this arm at, and how much does that
    # estimate move section to section", not to reconstruct `delta_scale_um`.
    def _arm_scale(stack):
        """(n_units, nR, G) -> (median scale per gene, sd of log2 scale per gene).

        The median is taken in LOG2 SPACE and exponentiated back -- a geometric median over units,
        not an arithmetic one -- because scale is a ratio-like quantity (an octave shift up and down
        should weigh the same). This is a different aggregation from `sA`/`sB` above, which reduce
        the arm's MEAN CURVE rather than aggregating each unit's own reduced scale.
        """
        if stack is None or len(stack) == 0 or is_pair or comp_prov.get("mode") == "residual":
            return np.full(G, np.nan), np.full(G, np.nan)
        per = np.stack([_char_scale(stack[u], radii_um, reduction, sign=sgn_band)
                        for u in range(len(stack))])              # (n_units, G)
        with np.errstate(invalid="ignore", divide="ignore"):
            lg = np.log2(np.where(per > 0, per, np.nan))
        med = 2.0 ** np.nanmedian(lg, axis=0)
        sd = np.nanstd(lg, axis=0, ddof=1) if len(stack) > 1 else np.full(G, np.nan)
        return med, sd
    with warnings.catch_warnings():
        # nanmedian/nanstd warn "All-NaN slice"/"Degrees of freedom <= 0" per-gene when a gene has no
        # characteristic length in ANY unit of an arm -- expected (mirrors delta_scale_um's own
        # one-armed-NaN case above) and already surfaced structurally as NaN in the output, not as a
        # warning a caller needs to see for every such gene.
        warnings.filterwarnings("ignore", message=".*All-NaN slice.*")
        warnings.filterwarnings("ignore", message=".*Degrees of freedom.*")
        scA, sdA = _arm_scale(unit_curves_A)
        scB, sdB = _arm_scale(unit_curves_B)
    # `delta_scale_resolvable` must judge the SAME quantity `delta_scale_um` reports -- the
    # MEAN-CURVE reduced scale `sA`/`sB` computed above (None on the pair-axis / residual-mode
    # paths) -- not the per-unit median `scA`/`scB`: those are a different estimator (see
    # `_arm_scale`'s docstring) and could disagree in magnitude, or even sign, with the column the
    # docstring tells the reader to check the flag against. The comparison spread stays `scA`/`scB`'s
    # between-unit `sdA`/`sdB`, which is the reported per-arm estimate's own spread.
    if sA is None:
        shift = np.full(G, np.nan)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            sA_pos = np.where(sA > 0, sA, np.nan)
            sB_pos = np.where(sB > 0, sB, np.nan)
            shift = np.abs(np.log2(sA_pos) - np.log2(sB_pos))
    valid_scale = np.isfinite(shift) & np.isfinite(sdA) & np.isfinite(sdB)
    resolvable_bool = shift > 0.5 * (sdA + sdB)
    resolvable = pd.array(resolvable_bool, dtype="boolean")
    resolvable[~valid_scale] = pd.NA
    n_uA = len(unit_curves_A) if unit_curves_A is not None else np.nan
    n_uB = len(unit_curves_B) if unit_curves_B is not None else np.nan

    # Pre-existing columns keep their ORIGINAL physical order/position (gene/pair .. peak_signal,
    # unchanged since before this branch) -- new columns are appended after, never spliced into the
    # middle, so "add columns, do not reorder existing ones" holds for column POSITION too, not just
    # for values.
    df = pd.DataFrame({id_col: ids, "axis": axis, "delta_scale_um": delta_scale, "dI_at_max": dI_at_max,
                       "T_max": Tmax, "r_at_max_um": r_at, "p": pval, "fdr": bh_fdr(pval),
                       "n_radii_exceeded": n_exceed, "radii_exceeded_um": where_ex,
                       "null_used": null_used, "structured": structured, "peak_signal": peak_signal,
                       "r_diverge_lo_um": r_lo, "r_diverge_hi_um": r_hi,
                       "dI_raw_peak_um": r_raw, "dI_raw_at_peak": dI_raw,
                       "scale_A_um": scA, "scale_B_um": scB,
                       "scale_A_sd_log2": sdA, "scale_B_sd_log2": sdB,
                       "n_units_A": n_uA, "n_units_B": n_uB,
                       "delta_scale_resolvable": resolvable})
    from . import scale as _scalemod
    df.attrs["spectrast_grid"] = dict(bin_size=float(bin_size), radii=list(radii),
                                      radii_um=[float(x) for x in radii_um])
    df.attrs["spectrast_reduction"] = reduction or _scalemod.DEFAULT_REDUCTION
    df.attrs["spectrast_ordering"] = ordering
    df.attrs["spectrast_envelope_alpha"] = float(envelope_alpha)
    df.attrs["spectrast_design"] = _design_attrs
    df.attrs["spectrast_null_band"] = nb        # matched noise floor; None for pair axes / residual mode
    df.attrs["spectrast_degenerate"] = degen    # "" when the axis behaves like a per-feature test
    # Provenance: record the backend BOTH arms actually ran on. They share one call, so they share a
    # backend unless it was switched mid-flight (e.g. from another thread) — which would put the two
    # arms on different float precisions and is worth failing loudly about, not stamping over.
    backend_at_end = _fft_backend()
    if backend_at_end != backend_at_start:
        warnings.warn(f"differential(): the compute backend changed during this call "
                      f"({backend_at_start} -> {backend_at_end}); the two arms were not computed at "
                      f"the same precision and the contrast is not trustworthy. Rerun without "
                      f"switching backends mid-call.")
    df.attrs["spectrast_backend"] = backend_at_end
    df.attrs["spectrast_composition"] = comp_prov      # provenance: what the null conditioned on
    if _return_internals:
        # null_d is the FULL per-radius null, (n_perm, nR, G). Tmax_null has already collapsed the
        # radius axis, so anything that ranks a whole curve rather than its single worst point --
        # extreme-rank-length ordering, interval-wise tests, global envelopes -- cannot be computed
        # from the collapsed version and needs this. Private hook; adding a key breaks no caller.
        return df, dict(Tmax=Tmax, Tmax_null=Tmax_null, structured=structured, obs=obs, sd=sd, r_at=r_at,
                        dI_at_max=dI_at_max, delta_scale=delta_scale, ids=ids, id_col=id_col,
                        null_d=null_d, radii_um=radii_um, meanA=meanA, meanB=meanB)
    curves = None
    if return_curves:
        # Read-only views of what the test already computed: nothing here feeds back into `df`.
        # The arm-mean-curve scales are the two numbers `delta_scale_um` differences; NaN wherever the
        # axis has no length (pair axes, composition residuals) or a curve has no half-max crossing.
        nan_g = np.full(G, np.nan)
        curves = dict(
            scale_meanA_um=nan_g.copy() if sA is None else np.asarray(sA, float).copy(),
            scale_meanB_um=nan_g.copy() if sB is None else np.asarray(sB, float).copy(),
            radii_um=np.asarray(radii_um, float), radii_bins=[int(r) for r in radii],
            bin_size=float(bin_size), ids=list(ids), id_col=id_col, axis=axis,
            unit_curves_A=unit_curves_A, unit_curves_B=unit_curves_B,
            units_A=unit_names_A, units_B=unit_names_B,
            unit_sections_A=unit_secs_A, unit_sections_B=unit_secs_B,
            section_labels_A=[f"A{i}" for i in range(len(A))],
            section_labels_B=[f"B{i}" for i in range(len(B))],
            meanA=np.asarray(meanA), meanB=np.asarray(meanB), diff=np.asarray(obs), sd=sd,
            band=T_alpha[None, :] * sd, T_alpha=T_alpha, exceed=exceed,
            envelope_alpha=float(envelope_alpha), labels=None, null=null_used, n_perm=int(n_perm))
    if not return_grid:
        return (df, curves) if return_curves else df
    # Count over the finite null draws only; where the observed difference is undefined (a feature
    # absent from an arm) there is nothing to test, so the radius is gated out as p = 1, as the main
    # table gates out such a feature, rather than falling to the floor 1 / (n_perm + 1).
    fin = np.isfinite(null_d)
    p_gr = (1 + (fin & (np.abs(null_d) >= np.abs(obs)[None])).sum(0)) / (fin.sum(0) + 1)
    p_gr = np.where(np.isfinite(obs), p_gr, 1.0)
    grid = pd.DataFrame([dict(**{id_col: ids[g]}, scale_um=float(radii_um[r]), dI=float(obs[r, g]),
                              p=float(p_gr[r, g])) for r in range(len(radii)) for g in range(G)])
    grid["fdr"] = bh_fdr(grid.p.values)
    return (df, grid, curves) if return_curves else (df, grid)


def scalogram_delta(adatas_A, adatas_B, genes, radii=None, bin_size=None,
                    null="auto", n_perm=200, depth_match=False, density_match=True, seed=0,
                    return_grid=False):
    """Moran-scalogram differential (thin wrapper over :func:`differential`, ``axis="moran"``).

    Kept for back-compatibility: returns the original column layout (``max_moran`` in place of
    ``peak_signal``, no ``axis`` column). New code should call :func:`differential` directly.

    Parameters
    ----------
    adatas_A, adatas_B : list of AnnData
        Condition samples; ``≥2`` each → replicate null, one each → matched null.
    genes : list of str
        Genes to test.
    radii, bin_size, null, n_perm, depth_match, density_match, seed, return_grid
        As in :func:`differential`.

    Returns
    -------
    pandas.DataFrame
        Columns ``gene, delta_scale_um, dI_at_max, T_max, r_at_max_um, p, fdr, null_used,
        structured, max_moran`` (plus the grid if ``return_grid=True``).

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> def toy():
    ...     a = ad.AnnData(np.abs(rng.standard_normal((300, 5))).astype("float32"))
    ...     a.obsm["spatial"] = rng.random((300, 2)) * 100
    ...     a.var_names = [f"g{i}" for i in range(5)]
    ...     return a
    >>> df = spectrast.diff.scalogram_delta([toy(), toy()], [toy(), toy()], genes=["g0"], n_perm=20)
    >>> "max_moran" in df.columns
    True
    """
    import pandas as pd
    out = differential(adatas_A, adatas_B, axis="moran", features=list(genes), radii=radii, bin_size=bin_size,
                       null=null, n_perm=n_perm, depth_match=depth_match, density_match=density_match, seed=seed,
                       return_grid=return_grid)
    df, grid = (out if return_grid else (out, None))
    df = df.rename(columns={"peak_signal": "max_moran"}).drop(columns=["axis"])
    # The envelope columns are deliberately NOT forwarded. This wrapper's column set is a frozen
    # back-compat contract (tests/test_diff.py asserts it exactly), and an appended column is still a
    # changed contract. Callers who want `n_radii_exceeded` / `radii_exceeded_um` should call
    # `differential(axis="moran", ...)`, which is what the docstring already tells new code to do.
    df = df[["gene", "delta_scale_um", "dI_at_max", "T_max", "r_at_max_um", "p", "fdr",
             "null_used", "structured", "max_moran"]]
    return (df, grid) if return_grid else df


def differential_omnibus(adatas_A, adatas_B, *, axes=("moran", "focality"), features=None,
                         radii=None, bin_size=None, null="auto", n_perm=200,
                         depth_match=False, density_match=True, structured_min=0.05, seed=0):
    """Screen several organization axes at once with a single, exact FDR pass.

    For the exploratory "tell me what changed" question. Combines the per-gene max-statistics across
    ``axes`` into one omnibus statistic ``T_omni = max_a T_max^(a)``, built from the *same* label
    permutations across axes (so the combined null is exact), then applies one BH-FDR pass — the
    family size is the number of genes, not genes × axes. Genes that pass get a post-hoc
    decomposition naming the ``driver_axis`` (the axis that moved most) with its signed effect and
    scale. The decomposition is descriptive (like ANOVA post-hoc); do not read the per-axis values as
    independently significant.

    Parameters
    ----------
    adatas_A, adatas_B : list of AnnData
        Condition samples (per-gene axes only).
    axes : sequence of {"moran", "geary", "focality"}, optional
        Axes to screen jointly. Default ``("moran", "focality")``.
    features : list of str
        Genes to screen (required).
    radii, bin_size, null, n_perm, depth_match, density_match, structured_min, seed
        As in :func:`differential`.

    Returns
    -------
    screen : pandas.DataFrame
        One row per gene: ``gene, T_omni, p, fdr, driver_axis``.
    decomp : pandas.DataFrame
        Post-hoc rows for screen-passing genes: ``gene, axis, T_max, dI_at_max, r_at_max_um,
        delta_scale_um``.

    See Also
    --------
    differential : test a single axis.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> def toy():
    ...     a = ad.AnnData(np.abs(rng.standard_normal((300, 6))).astype("float32"))
    ...     a.obsm["spatial"] = rng.random((300, 2)) * 100
    ...     a.var_names = [f"g{i}" for i in range(6)]
    ...     return a
    >>> screen, decomp = spectrast.diff.differential_omnibus(
    ...     [toy(), toy()], [toy(), toy()], axes=("moran", "focality"),
    ...     features=["g0", "g1"], n_perm=20)
    >>> "driver_axis" in screen.columns
    True
    """
    import pandas as pd
    for a in axes:                      # refuse loudly; the filter below would silently narrow the screen
        if a in REFUSED_DIFF_AXES:
            raise ValueError(f"axis={a!r} is not admissible as a differential test. "
                             f"{REFUSED_DIFF_AXES[a]}")
    axes = [a for a in axes if a in PER_GENE_AXES]
    assert axes and features is not None, "omnibus is per-gene: pass axes from moran/geary/focality and features"
    ids = list(features)
    # reuse each axis's internals computed on the SAME seed -> the driver's rng.permutation stream is identical
    # across axes, so the per-axis null_d rows share the same label shuffle (an exact shared-permutation null).
    per = {}
    field_cache = {}                          # E2: bin + z-stack each sample ONCE, reuse across the axes
    for a in axes:
        _, intern = differential(adatas_A, adatas_B, axis=a, features=ids, radii=radii, bin_size=bin_size,
                                 null=null, n_perm=n_perm, depth_match=depth_match, density_match=density_match,
                                 structured_min=structured_min, seed=seed, _return_internals=True,
                                 _field_cache=field_cache)
        per[a] = intern
    struct = np.any([per[a]["structured"] for a in axes], 0)                                    # structured on any axis
    T_obs = np.stack([per[a]["Tmax"] for a in axes])                                            # (S, G)
    T_null = np.stack([per[a]["Tmax_null"] for a in axes])                                      # (S, n_perm, G)
    T_omni = T_obs.max(0)                                                                       # (G,)
    T_omni_null = T_null.max(0)                                                                 # (n_perm, G)
    p = (1 + (T_omni_null >= (T_omni - 1e-9)[None]).sum(0)) / (T_null.shape[1] + 1)   # vectorized over genes (E3)
    p = np.where(struct, p, 1.0); T_omni = np.where(struct, T_omni, 0.0)
    driver = np.array(axes)[T_obs.argmax(0)]
    screen = pd.DataFrame(dict(gene=ids, T_omni=T_omni, p=p, fdr=bh_fdr(p), driver_axis=driver))
    hit = screen.fdr < 0.05
    rows = []
    for g in np.where(hit.values)[0]:
        for a in axes:
            it = per[a]
            rows.append(dict(gene=ids[g], axis=a, T_max=float(it["Tmax"][g]), dI_at_max=float(it["dI_at_max"][g]),
                             r_at_max_um=float(it["r_at"][g]), delta_scale_um=float(it["delta_scale"][g])))
    decomp = pd.DataFrame(rows)
    return screen, decomp


def split_sections(adata, sample_key):
    """Split ONE concatenated AnnData into per-section views, in first-appearance order.

    Spatial statistics are per-section — cells from two slides share no coordinate frame — so every
    entry point here wants a list of sections. Cohort data usually lives as a single concatenated
    object instead. This is the bridge, and it returns **views**, not copies: the expression matrix is
    not duplicated, which is what makes a many-section contrast fit in memory.

    Parameters
    ----------
    adata : AnnData
        Concatenated sections with coordinates in ``obsm["spatial"]``.
    sample_key : str
        ``obs`` column identifying the physical section.

    Returns
    -------
    tuple(list of AnnData, list of str)
        ``(views, labels)`` parallel to each other.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> a = ad.AnnData(np.abs(rng.standard_normal((40, 3))).astype("float32"))
    >>> a.obs["section"] = ["s0"] * 20 + ["s1"] * 20
    >>> views, labels = spectrast.diff.split_sections(a, "section")
    >>> labels, [v.n_obs for v in views]
    (['s0', 's1'], [20, 20])
    """
    import pandas as pd
    if sample_key not in adata.obs:
        raise KeyError(f"adata.obs[{sample_key!r}] not found")
    lab = adata.obs[sample_key].astype(str).to_numpy()
    labels = list(pd.unique(lab))
    return [adata[lab == s] for s in labels], labels


def differential_cohort(adata, condition_key, values_A, values_B, *, sample_key,
                        unit_key=None, **kwargs):
    """Run :func:`differential` straight from one concatenated cohort AnnData.

    The same test as :func:`differential`, marshalled for how cohort data actually lives: one object
    with a condition column and a section column, rather than two hand-built lists of section objects.
    Sections are split into **views** (:func:`split_sections`), so nothing is copied.

    When ``unit_key`` is given the unit labels are passed through as ``units_A``/``units_B``, which is
    what makes multiple sections from one donor count as a single replicate instead of
    pseudo-replicates — and, if the same units appear in both arms, selects the paired sign-flip null.
    **Pass it whenever donors contribute more than one section.**

    Parameters
    ----------
    adata : AnnData
        Concatenated cohort.
    condition_key : str
        ``obs`` column holding the condition label.
    values_A, values_B : sequence
        Condition values assigned to arm A and arm B.
    sample_key : str
        ``obs`` column identifying the physical section (keyword-only, required — a contrast whose
        sections are undefined would silently pool slides onto one lattice).
    unit_key : str, optional
        ``obs`` column with the experimental unit (donor/patient).
    **kwargs
        Everything else is forwarded to :func:`differential` (``axis``, ``features``, ``radii``,
        ``bin_size``, ``n_perm``, ``paired``, ``return_grid``, ``return_curves``, …).

    Returns
    -------
    pandas.DataFrame
        As :func:`differential`, including its tuple forms. With ``return_curves=True`` the curves
        dict is labelled from the cohort: ``labels`` is ``("/".join(values_A), "/".join(values_B))``,
        ``section_labels_A``/``_B`` are the ``sample_key`` values, and, when ``unit_key`` is not
        given (so each section is its own unit), so are ``units_A``/``units_B``.

    See Also
    --------
    differential : the underlying test.
    groups_from_obs : the same split for an existing list of sections.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> parts = []
    >>> for cond in ("on", "off"):
    ...     for donor in ("d0", "d1"):
    ...         p = ad.AnnData(np.abs(rng.standard_normal((n, 4))).astype("float32"))
    ...         p.obsm["spatial"] = rng.random((n, 2)) * 100
    ...         p.var_names = [f"g{i}" for i in range(4)]
    ...         p.obs["condition"] = cond; p.obs["donor"] = donor
    ...         p.obs["section"] = f"{cond}_{donor}"
    ...         parts.append(p)
    >>> cohort = ad.concat(parts)
    >>> res = spectrast.diff.differential_cohort(
    ...     cohort, "condition", ["on"], ["off"], sample_key="section", unit_key="donor",
    ...     axis="moran", features=["g0", "g1"], n_perm=20)
    >>> list(res.columns)[:2]
    ['gene', 'axis']
    """
    views, labels = split_sections(adata, sample_key)
    if unit_key is not None:
        A, B, uA, uB = groups_from_obs(views, condition_key, values_A, values_B, unit_key=unit_key)
    else:
        A, B = groups_from_obs(views, condition_key, values_A, values_B)
        uA = uB = None
    if not A or not B:
        got = sorted(set(adata.obs[condition_key].astype(str))) if condition_key in adata.obs else []
        raise ValueError(f"condition split left arm A with {len(A)} and arm B with {len(B)} "
                         f"section(s) out of {len(labels)}; values present in "
                         f"adata.obs[{condition_key!r}] are {got}")
    res = differential(A, B, units_A=uA, units_B=uB, **kwargs)
    if kwargs.get("return_curves") and not kwargs.get("_return_internals"):
        curves = res[-1]
        secA = [str(v.obs[sample_key].iloc[0]) for v in A]
        secB = [str(v.obs[sample_key].iloc[0]) for v in B]
        def _arm(v):
            return v if isinstance(v, str) else "/".join(map(str, v))
        curves["labels"] = (_arm(values_A), _arm(values_B))
        curves["section_labels_A"], curves["section_labels_B"] = secA, secB
        if unit_key is None and curves["units_A"] is not None:
            curves["units_A"] = [secA[s[0]] for s in curves["unit_sections_A"]]
            curves["units_B"] = [secB[s[0]] for s in curves["unit_sections_B"]]
    return res


def groups_from_obs(adatas, condition_key, values_A, values_B, unit_key=None):
    """Split a list of AnnData into (A, B) condition lists by an ``obs`` key.

    Convenience marshalling for :func:`differential`: routes each sample to arm A or B by the value of
    ``adata.obs[condition_key]`` (read from the first cell). Samples whose value is in neither list are
    dropped.

    See :func:`differential_cohort` to go straight from one concatenated cohort object instead.

    Parameters
    ----------
    adatas : list of AnnData
        Samples to split.
    condition_key : str
        Column in ``adata.obs`` holding the condition label.
    values_A, values_B : sequence
        Label values assigned to arm A and arm B respectively.
    unit_key : str, optional
        ``obs`` column with the experimental-unit label (e.g. patient/donor). When given, ALSO returns the
        per-sample unit labels for each arm — pass them to :func:`differential` as ``units_A``/``units_B``
        (which enables the unit-aggregated and, for shared units, paired null). Default ``None``.

    Returns
    -------
    tuple(list, list) or tuple(list, list, list, list)
        ``(A_list, B_list)``; or ``(A_list, B_list, units_A, units_B)`` when ``unit_key`` is given.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> def toy(label):
    ...     a = ad.AnnData(np.abs(rng.standard_normal((50, 4))).astype("float32"))
    ...     a.obs["condition"] = label
    ...     return a
    >>> A, B = spectrast.diff.groups_from_obs(
    ...     [toy("amyloid+"), toy("amyloid-")], "condition", ["amyloid+"], ["amyloid-"])
    >>> len(A), len(B)
    (1, 1)
    """
    A, B, uA, uB = [], [], [], []
    for ad in adatas:
        v = str(ad.obs[condition_key].iloc[0]) if condition_key in ad.obs else None
        u = str(ad.obs[unit_key].iloc[0]) if (unit_key and unit_key in ad.obs) else None
        if v in values_A:
            A.append(ad); uA.append(u)
        elif v in values_B:
            B.append(ad); uB.append(u)
    return (A, B, uA, uB) if unit_key is not None else (A, B)

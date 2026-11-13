"""Shared plumbing for :mod:`spectrast.pl`: axes, finishing (show/save), result lookup, and the few
drawing helpers every plot uses. matplotlib is imported inside each function."""
from __future__ import annotations

import numpy as np

SCALOGRAM_KEY = "spectrast_scalogram"
DIVERGING = "RdBu_r"          # signed quantities (local Moran, studentised differences, log g)
SEQUENTIAL = "viridis"        # magnitudes (focality, z, scale)
ARM_COLORS = ("#4C72B0", "#DD8452")   # arm A, arm B
HIGHLIGHT = "#C44E52"         # called / significant
MUTED = "0.75"                # not called / not read


# ---------------------------------------------------------------------------------------------
# figure plumbing
# ---------------------------------------------------------------------------------------------
def new_axes(ax=None, figsize=(4.2, 3.4)):
    """``(fig, ax)``: the given Axes and its figure, or a new constrained-layout figure."""
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    return fig, ax


def new_figure(figsize):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=figsize, layout="constrained")


def finish(fig, ret, show=None, save=None, dpi=300):
    """The scanpy convention: write ``save`` if given; ``show=True`` shows and returns ``None``;
    otherwise return ``ret`` (an Axes for single-panel plots, the Figure for multi-panel ones)."""
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
        return None
    return ret


# ---------------------------------------------------------------------------------------------
# result lookup
# ---------------------------------------------------------------------------------------------
def scalogram(adata, key=SCALOGRAM_KEY):
    """``(curves (n_radii, n_genes), radii_um (n_radii,), gene names (n_genes,))`` written by
    :func:`spectrast.gr.scale_spectrum` or :func:`spectrast.gr.fingerprint`."""
    if key not in adata.uns:
        raise KeyError(f"adata.uns[{key!r}] not found: run spectrast.gr.scale_spectrum(adata) or "
                       f"spectrast.gr.fingerprint(adata) first")
    st = adata.uns[key]
    S = np.asarray(st["scalogram"], float)
    r = np.asarray(st["radii"], float)
    names = np.asarray(st["genes"], dtype=str) if "genes" in st else np.asarray(adata.var_names, dtype=str)
    if S.shape[1] != len(names):
        raise ValueError(f"adata.uns[{key!r}] holds {S.shape[1]} curves but {len(names)} gene labels; "
                         f"recompute it with this version of spectrast")
    return S, r, names


def index_of(names, wanted):
    """Positions of ``wanted`` in ``names``, with one error naming everything missing."""
    names = list(map(str, names))
    pos = {n: i for i, n in enumerate(names)}
    wanted = [wanted] if isinstance(wanted, str) else list(wanted)
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise KeyError(f"not found: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return np.array([pos[w] for w in wanted], int)


def coords(adata, basis="spatial"):
    if basis not in adata.obsm:
        raise KeyError(f"adata.obsm[{basis!r}] not found")
    return np.asarray(adata.obsm[basis], float)[:, :2]


def feature_vector(adata, feature, layer=None):
    """One per-cell vector: a gene (``var_names``), an ``obs`` column, or ``"<obs_key>==<value>"``
    for a 0/1 indicator of one category (e.g. ``"cell_type==B cell"``)."""
    if "==" in feature:
        key, val = (s.strip() for s in feature.split("==", 1))
        if key not in adata.obs:
            raise KeyError(f"obs column {key!r} not found")
        return (adata.obs[key].astype(str).to_numpy() == val).astype(float)
    if feature in adata.var_names:
        j = int(np.flatnonzero(np.asarray(adata.var_names) == feature)[0])
        X = adata.layers[layer] if layer is not None else adata.X
        col = X[:, j]
        col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
        return np.asarray(col, float).ravel()
    if feature in adata.obs:
        return np.asarray(adata.obs[feature], float)
    raise KeyError(f"{feature!r} is neither a gene, an obs column, nor '<obs_key>==<value>'")


# ---------------------------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------------------------
def radius_axis(ax, radii=None, label="radius (µm)", axis="x"):
    """Log-scaled radius axis with plain-number µm ticks at 1, 2 and 5 per decade, so an axis
    spanning less than a decade still carries labelled ticks."""
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_scale("log")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())
    if radii is not None:
        r = np.asarray(radii, float)
        lim = (r.min() / 1.1, r.max() * 1.1)
        (ax.set_xlim if axis == "x" else ax.set_ylim)(*lim)
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(label)


def robust_limits(values, pct=98.0, symmetric=False):
    """Colour limits clipped at ``pct``: ``(-m, m)`` if symmetric, else the central interval."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        m = float(np.percentile(np.abs(v), pct)) or 1.0
        return -m, m
    lo, hi = np.percentile(v, [100 - pct, pct])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def spatial_scatter(ax, xy, values, *, cmap=SEQUENTIAL, vmin=None, vmax=None, s=None, title=None,
                    invert_y=True, colorbar=True, cbar_label=None, order_by_value=False):
    """Cells as points coloured by ``values``; equal aspect, no frame. ``s`` defaults to a size that
    keeps ~10^5 cells legible."""
    xy = np.asarray(xy, float)
    v = np.asarray(values, float)
    if s is None:
        s = float(np.clip(4e4 / max(len(xy), 1), 0.05, 8.0))
    idx = np.argsort(np.abs(np.nan_to_num(v))) if order_by_value else np.arange(len(v))
    sc = ax.scatter(xy[idx, 0], xy[idx, 1], c=v[idx], s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
        if cbar_label:
            cb.set_label(cbar_label)
    return sc


def range_normalise(curves):
    """Each column rescaled to [0, 1] over its own range; flat or non-finite columns become NaN."""
    C = np.asarray(curves, float)
    lo, hi = np.nanmin(C, 0), np.nanmax(C, 0)
    span = hi - lo
    out = (C - lo) / np.where(span > 0, span, np.nan)
    return out


def label_points(ax, x, y, labels, fontsize=7, max_labels=None):
    """Plain text labels offset slightly from each point."""
    n = len(labels) if max_labels is None else min(len(labels), max_labels)
    for i in range(n):
        ax.annotate(str(labels[i]), (x[i], y[i]), xytext=(3, 2), textcoords="offset points",
                    fontsize=fontsize)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_labels(ax, x, y, labels, fontsize=7, obstacles=()):
    """Name points without the labels overlapping each other, the labelled points, the given
    ``obstacles`` (artists) or the Axes edge.

    Greedy: each label takes the first of a fixed set of offsets around its point whose box is free,
    with a thin leader line when it sits away from the point. Box sizes are estimated from the
    character count, positions from the Axes after layout. Labels are kept out of the layout.
    """
    if len(labels) == 0:
        return []
    fig = ax.figure
    try:
        fig.draw_without_rendering()                                  # settle the layout first
    except Exception:                                                 # pragma: no cover - backend specific
        pass
    px = fig.dpi / 72.0
    P = ax.transData.transform(np.c_[np.asarray(x, float), np.asarray(y, float)])
    h = fontsize * 1.2 * px
    pts = [(p[0] - 3 * px, p[1] - 3 * px, p[0] + 3 * px, p[1] + 3 * px) for p in P]
    blocked = []
    for art in obstacles:
        try:
            e = art.get_window_extent()
            blocked.append((e.x0, e.y0, e.x1, e.y1))
        except Exception:                                             # pragma: no cover
            pass
    bb = ax.bbox
    cands = [(4, 2), (4, -11), (-4, 2), (-4, -11), (4, 13), (-4, 13), (4, -22), (-4, -22),
             (4, 24), (-4, 24), (4, -33), (-4, -33), (4, 35), (-4, 35), (4, -44), (-4, -44),
             (14, -55), (-14, -55), (14, 46), (-14, 46)]
    placed, out = [], []
    for i, lab in enumerate(map(str, labels)):
        w = 0.62 * fontsize * len(lab) * px
        choice = None
        inside_first = None
        for dx, dy in cands:
            x0 = P[i, 0] + dx * px - (w if dx < 0 else 0.0)
            box = (x0, P[i, 1] + dy * px, x0 + w, P[i, 1] + dy * px + h)
            inside = box[0] >= bb.x0 and box[2] <= bb.x1 and box[1] >= bb.y0 and box[3] <= bb.y1
            if not inside:
                continue
            inside_first = inside_first or (dx, dy)
            if not any(_overlaps(box, b) for b in placed + blocked) and \
                    not any(_overlaps(box, b) for j, b in enumerate(pts) if j != i):
                choice = (dx, dy)
                break
        choice = choice or inside_first or cands[0]
        dx, dy = choice
        x0 = P[i, 0] + dx * px - (w if dx < 0 else 0.0)
        placed.append((x0, P[i, 1] + dy * px, x0 + w, P[i, 1] + dy * px + h))
        far = abs(dy) > 10
        ann = ax.annotate(lab, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                          fontsize=fontsize, ha="left" if dx >= 0 else "right", va="bottom",
                          arrowprops=dict(arrowstyle="-", lw=0.4, color="0.45", shrinkA=0,
                                          shrinkB=2) if far else None)
        ann.set_in_layout(False)
        out.append(ann)
    return out

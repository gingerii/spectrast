"""spectrast.pl — plotting, in the scanpy convention.

Every function takes the object its computing step produced (an AnnData that a ``spectrast.gr``
call wrote into, or the table a ``gr`` / ``diff`` / ``qc`` call returned), draws it, and follows
scanpy's three keywords:

* ``ax=`` draw into an existing Axes (single-panel functions only);
* ``show=True`` call ``plt.show()`` and return ``None``;
* ``save="path.png"`` write the figure.

Otherwise single-panel functions return their ``Axes`` and multi-panel ones their ``Figure``.
Nothing here recomputes a statistic: the few maps that need a per-cell field (local Moran at one
radius, a neighbourhood mean) compute that one field and say so in their docstring.

matplotlib is imported inside each function, so ``import spectrast`` stays cheap. Functions are
loaded on first use.
"""
from __future__ import annotations

from importlib import import_module

# name -> submodule. Loaded lazily (module __getattr__), so a missing optional dependency in one
# plot never breaks ``import spectrast`` or the others.
_WHERE = {
    # one section: organization of genes and cell types
    "scale_plane": "_section",
    "curves": "_section",
    "significance_heatmap": "_section",
    "gene_card": "_section",
    "local_map": "_section",
    "local_filmstrip": "_section",
    "curve_embedding": "_section",
    "curve_families": "_section",
    "celltype_ladder": "_section",
    # two conditions / many sections
    "volcano": "_diff",
    "diff_curves": "_diff",
    "unit_curves": "_diff",
    "diff_heatmap": "_diff",
    "diff_fingerprint": "_diff",
    "scale_shift": "_diff",
    "effect_plane": "_diff",
    "neighborhood_maps": "_diff",
    "section_scales": "_diff",
    # cell-type pairs and diagnostics
    "celltype_range": "_diag",
    "presence": "_diag",
    "qc": "_diag",
}

__all__ = sorted(_WHERE)


def __getattr__(name):
    mod = _WHERE.get(name)
    if mod is None:
        raise AttributeError(f"module 'spectrast.pl' has no attribute {name!r}")
    return getattr(import_module(f".{mod}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_WHERE))

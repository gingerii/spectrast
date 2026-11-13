---
sd_hide_title: true
---

# Spectrast

**Spectrast** computes multiscale spatial statistics for single-cell spatial transcriptomics as FFT
convolutions on a lattice. It measures how each gene or cell type is spatially organized at every
length scale. It computes Moran's I, Geary's C, Getis-Ord General G, LOSH, focality and texture as
curves over radius, and tests whether that organization changes between conditions. It works on
Xenium, MERFISH, CosMx and Visium HD data in AnnData and follows the scanpy/squidpy layout: `gr`
computes, `pl` plots.

```{image} _static/spectrast_overview.png
:alt: How Spectrast computes a spatial statistic at every radius
:width: 720px
:align: center
```

- **A.** Each statistic is a normalised quadratic form of a gene's expression *y* under the
  spatial weight matrix Ωᵣ at radius *r*.
- **B.** On irregular cells the weight matrix is a cell × cell graph. On a lattice it becomes a
  convolution kernel kᵣ.
- **C.** Spectrast rasterizes cells onto a lattice (bin ≈ *r*/5). Cells where *r* is small
  relative to the bin are computed exactly on the cell graph (adaptive fusion). Every other
  neighbour sum and count is an FFT convolution.
- **D.** Sweeping *r* turns each gene into a curve, which is then reduced to a characteristic
  scale.

## Install

```bash
pip install "spectrast[full] @ git+https://github.com/gingerii/spectrast.git"
```

The core package needs only numpy, scipy, pandas and psutil. `[full]` adds scanpy, anndata,
matplotlib and scikit-learn. For a GPU, add `cuda` (NVIDIA) or `mps` (Apple silicon) to the extras
and see {func}`spectrast.set_backend`.

## Quick start

```python
import scanpy as sc
import spectrast
from spectrast import gr, pl, diff

# one section; cell coordinates in µm in adata.obsm["spatial"]
adata = sc.read_h5ad("section.h5ad")

# each gene's Moran's I curve over radius -> characteristic scale, amplitude
tab = gr.organization(adata, stats=("moran",))
pl.scale_plane(tab)

# a few genes' curves against a calibrated per-gene, per-radius null
genes = ["CXCL13", "CCL19", "MS4A1"]
gr.fingerprint(adata, genes=genes)         # stores the curves in adata.uns
null = gr.moran_null(adata, genes=genes)
pl.curves(adata, genes, null=null)

# where each cell type sits on the tissue's length scale
pl.celltype_ladder(gr.celltype_scale(adata, "cell_type"))

# two conditions: cohort is one AnnData of many sections, with obs columns
# "condition" and "section_id"; which genes reorganize, and over which radii?
res, curves = diff.differential_cohort(
    cohort, "condition", ["case"], ["control"],
    sample_key="section_id", axis="moran", return_curves=True)
pl.volcano(res)
pl.diff_curves(curves, res.sort_values("p").gene.iloc[0])
```

## Getting started

::::{grid} 1 1 2 2
:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Preprocess two public datasets, then analyse one section and a two-condition cohort end to end.
:::
:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public function, by module.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
api/index
```

This documentation was co-authored by Claude, an AI model made by Anthropic.

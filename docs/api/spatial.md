# Low-level engine: `spatial`

Array/grid functions operating on `(H, W, G)` fields and `(N, 2)` point clouds. The high-level `gr`
API wraps these; use them directly for custom pipelines.

```{eval-rst}
.. currentmodule:: spectrast.spatial
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   morans_i_multiscale
   gearys_c_multiscale
   focality_scalogram
   correlation_length
   variance_decomposition
   morans_scalogram_genes
   morans_global_genes
```

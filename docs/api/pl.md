# Plotting: `pl`

Scanpy-style plots of the results above. Each function takes what its computing step produced (an
AnnData a `gr` call wrote into, or a table a `gr` / `diff` / `qc` call returned) and accepts `ax=`
(single-panel plots), `show=` and `save=`.

## One section

```{eval-rst}
.. currentmodule:: spectrast.pl
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   scale_plane
   curves
   significance_heatmap
   gene_card
   local_map
   local_filmstrip
   curve_embedding
   curve_families
   celltype_ladder
```

## Two conditions and many sections

```{eval-rst}
.. currentmodule:: spectrast.pl
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   volcano
   diff_curves
   unit_curves
   diff_heatmap
   diff_fingerprint
   scale_shift
   effect_plane
   neighborhood_maps
   section_scales
```

## Diagnostics

```{eval-rst}
.. currentmodule:: spectrast.pl
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   presence
   qc
```

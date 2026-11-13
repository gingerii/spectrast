# Statistics: `gr`

AnnData-native statistics, imported as `spectrast.gr`.

Read coordinates from `adata.obsm["spatial"]`, write results into `adata` under `spectrast_*` keys.

## Per-gene organization

```{eval-rst}
.. currentmodule:: spectrast.gr
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   default_radius
   default_radii
   organization
   spatial_autocorr
   fingerprint
   fingerprint_by
   fingerprint_signature
   aggregate_fingerprint
   scale_spectrum
   scale_fit_check
   focality
   focality_landmark
   texture
   moran_null
   fingerprint_presence
   autotune_rank
   celltype_scale
   presence
   presence_variation
```

## Local maps

```{eval-rst}
.. currentmodule:: spectrast.gr
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   local_moran
   getis_ord
   local_geary
   losh
```

# Inference

Per-gene significance for the multiscale curves, and nulls that hold cell-type composition fixed.

## `spectrast.null` — per-gene calibrated nulls

Per-gene, per-radius significance for the multiscale curves: permutation nulls, analytic moments for
Moran, Geary and General G, a fitted surrogate null, and covariate-stratified variants. Significance is
a floor, not a ranking; rank genes by amplitude and gate them by these.

```{eval-rst}
.. currentmodule:: spectrast.null
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   gene_inference
   permutation_null
   moran_permutation_null
   moran_null_batch
   moran_analytic_moments
   moran_analytic_z
   geary_analytic_z
   general_g_binary_moments
   general_g_analytic_z
   fit_surrogate_null
   gene_features
   covariate_strata
```

## `spectrast.composition` — composition-aware nulls

Separate *reorganized* from *recomposed*. When two tissues differ in cell-type composition, a gene's
scale or focality can shift because the mix of cells expressing it changed, not because its
arrangement changed. Pass `strata_key=` to `gr.fingerprint_presence` (or `diff.differential` under
the matched null) to permute **within** each cell-type stratum, and read the warnings in this
module's docstring first — composition is not always a nuisance.

```{eval-rst}
.. currentmodule:: spectrast.composition
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   stratified_permutation
   stratified_partition
   apply_min_cells
   residualize
   check_residual_design
   composition_matrix
   resolution_flip_report
   check_circularity
   strata_from_obs
```

# Curve reductions

From an organization curve to the numbers reported about it: a characteristic scale, a decay fit that checks it, and a shape class.

## `spectrast.scale` — one characteristic length per curve

The single reduction from an organization curve to a characteristic scale in micrometres, shared by
the fingerprint and the differential so that a Δ-scale compares like with like.

```{eval-rst}
.. currentmodule:: spectrast.scale
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   reduce_scale
   half_max
   centroid
   orient_sign
```

## `spectrast.curvefit` — decay fits as a check on the scale

Fits exponential and stretched-exponential decay families to an organization curve. The fit checks
whether a reported scale is a length the tissue has or an artefact of where the radius sweep stopped;
it never replaces the scale.

```{eval-rst}
.. currentmodule:: spectrast.curvefit
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   fit_decay
   fit_summary
   one
   stretched
   pinned
   orient
   bic
```

## `spectrast.curveshape` — shape classes and landmarks

A per-section descriptive coordinate for curves no single model family fits: smoothing, turning
points, a shape class, a landmark radius and a roughness score.

```{eval-rst}
.. currentmodule:: spectrast.curveshape
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   classify
   landmark
   turning_points
   smooth
   roughness
```

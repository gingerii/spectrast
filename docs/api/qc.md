# Diagnostics and input

## `spectrast.qc` — what a section can support

A diagnostic report on what one section can support, read before its results are interpreted: its
lattice, how saturated significance is at its size, optional covariate attribution, and the design's
smallest attainable p. It is deliberately not a gate: it passes or fails nothing.

```{eval-rst}
.. currentmodule:: spectrast.qc
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   section_report
   format_report
```

## `spectrast.io_visium` — Space Ranger input

Read a Space Ranger output directory into an AnnData with spot coordinates in micrometres.

```{eval-rst}
.. currentmodule:: spectrast.io_visium
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   read_spaceranger
   read_sections
```

## `spectrast.simulate` — synthetic expression

Synthetic expression with known low-rank spatial structure on cell coordinates you supply, for
checking an analysis against a known answer.

```{eval-rst}
.. currentmodule:: spectrast.simulate
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   simulate_st
```

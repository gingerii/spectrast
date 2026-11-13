# API reference

`import spectrast`, then reach each module as `spectrast.gr`, `spectrast.pl` and so on. Most
analyses need only `gr`, `pl` and `diff`.

| Page | Contents |
|---|---|
| [`gr`](gr.md) | Statistics on an AnnData: per-gene organization curves and scales, and local maps. Results are written into `adata` under `spectrast_*` keys. |
| [`pl`](pl.md) | Plots of what `gr`, `diff` and `qc` return, in the scanpy convention (`ax=`, `show=`, `save=`). |
| [`diff`](diff.md) | Differential organization between two conditions, with a replicate-calibrated permutation null. |
| [Inference](inference.md) | `null`: per-gene, per-radius nulls. `composition`: nulls that permute within cell types. |
| [Curve reductions](reductions.md) | `scale`, `curvefit`, `curveshape`: from a curve to the numbers reported about it. |
| [Diagnostics and input](qc.md) | `qc`: what a section can support. `io_visium`: Space Ranger input. `simulate`: synthetic expression. |
| [Compute backend](backend.md) | CPU, CUDA or Apple GPU for the transforms, and thread counts. |
| [`spatial`](spatial.md) | The low-level engine: functions on lattices and point clouds that the modules above wrap. |

```{toctree}
:hidden:

gr
pl
diff
inference
reductions
qc
backend
spatial
```

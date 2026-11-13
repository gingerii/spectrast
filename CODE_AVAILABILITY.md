# Code availability

Packages and versions, grouped by environment (and language).

**atlas_FM_svd env (Python 3.11.15):** numpy 2.4.4, scipy 1.17.1, pandas 2.3.3,
scanpy 1.11.5, anndata 0.12.10, scikit-learn 1.8.0, numba 0.65.0, matplotlib 3.10.8,
psutil 7.2.2, statsmodels 0.14.6, adjustText 1.4.0, h5py 3.16.0; data import (remote reads of
public bundles): remotezip 0.12.3, fsspec 2026.3.0, requests 2.33.1; spectrast 0.16.1 (this project). Benchmark baselines:
squidpy 1.8.1 (spatial_neighbors / spatial_autocorr / nhood_enrichment / co_occurrence); the
field-standard reference for the per-cell local tools (esda/PySAL-style Gi*/LISA/CCI and
scikit-gstat-style variogram) is reimplemented in-repo on `sklearn.neighbors.radius_neighbors_graph`
(no extra dependency). Dev/test: pytest. FFT backend: `scipy.fft` (threaded, `workers=-1`) is the
default CPU path; the permutation-null thread pool uses the stdlib `concurrent.futures`.

**GPU FFT backend (optional; xenium_morpho env, Python 3.12):** torch 2.12.0 — the optional CUDA /
Apple-MPS FFT backend (`spectrast._fft`). torch is never a hard dependency; the base package runs on the
threaded-scipy CPU path alone. MPS accuracy validated against the float64 CPU reference (relative-L2
~1e-7 up to 1100²).

**R (4.4.1):** onion 1.5.3, rsvd 1.0.5, irlba 2.3.7, Seurat 5.4.0,
SeuratData 0.2.2.9001, Matrix 1.7.3; QSC 0.7.0 (reference package).

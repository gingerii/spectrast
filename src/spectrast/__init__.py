"""spectrast — FFT-native multiscale spatial statistics for single-cell spatial transcriptomics.

Rasterize cells to a lattice → FFT/convolution neighborhood statistics at every scale.
Heavy deps are lazy-imported inside functions; importing this package is cheap.

The quaternion/hypercomplex origin layer (quaternion, octonion, io, apps) has been moved
to ``spectrast._archive`` and is no longer part of the live API. Import it explicitly from there
if you need the original quaternion tooling.
"""
from . import (config, spatial, gr, diff, orient, memory, composition, bivariate, qc,  # noqa: F401
               curveshape, curvefit, scale, null, io_visium, pl)  # noqa: F401
from ._fft import set_backend, get_backend, set_workers  # pluggable FFT backend (cpu/cuda/mps)
from ._parallel import set_n_jobs, get_n_jobs            # permutation-null thread count

__all__ = ["config", "spatial", "gr", "diff", "orient", "memory", "composition", "bivariate",
           "curveshape", "curvefit", "scale", "null", "io_visium", "pl",
           "set_backend", "get_backend", "set_workers", "set_n_jobs", "get_n_jobs"]
__version__ = "0.16.1"

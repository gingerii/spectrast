"""Thread-pool fan-out for the FFT-bound permutation nulls.

The matched-null differential and the colocalization / CCI nulls recompute a full scalogram (or
convolution) PER permutation — independent, embarrassingly parallel, and dominated by FFT + numpy
reductions that both RELEASE the GIL. So a thread pool (no pickling, shared memory for the large
field stack) parallelizes them; processes are unnecessary unless a benchmark shows GIL contention.

Determinism contract (a correctness requirement, not a nicety):
  * Callers draw ALL permutation indices UP FRONT from the master RNG, in order, then hand the
    expensive per-permutation compute to :func:`parallel_map`. Because the RNG draw is sequential
    and the results are collected in submission order, the output is BIT-IDENTICAL to the serial
    loop AND independent of worker count — a stronger guarantee than seed-per-permutation.
  * Inside the pool the inner FFT drops to a single worker (``_fft.set_workers(1)``) AND the native BLAS /
    OpenMP pools are pinned to 1 thread (``threadpoolctl``, if installed) so N concurrent permutations ×
    all-core FFT/matmul don't oversubscribe the cores; parallelism comes from the permutations.
"""
from __future__ import annotations

import os
import threading

_N_JOBS: int | None = None   # None -> auto (os.cpu_count())
_WORKERS_LOCK = threading.Lock()   # serialize the _fft global worker-count save/restore (nested/concurrent calls)


def set_n_jobs(n: int | None) -> None:
    """Set the permutation-null worker count. None or <=0 -> auto (all cores). 1 -> serial."""
    global _N_JOBS
    _N_JOBS = None if (n is None or n <= 0) else int(n)


def get_n_jobs() -> int:
    return _N_JOBS if _N_JOBS else (os.cpu_count() or 1)


def parallel_map(fn, items):
    """Map ``fn`` over ``items`` across a thread pool, preserving order.

    Serial (identical result) when n_jobs<=1 or there is <=1 item. Otherwise runs on a
    ThreadPoolExecutor with the inner FFT pinned to 1 worker to avoid oversubscription.
    """
    items = list(items)
    n = get_n_jobs()
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import contextlib
    from concurrent.futures import ThreadPoolExecutor
    from . import _fft
    try:
        from threadpoolctl import threadpool_limits           # M7: cap BLAS/OpenMP so n_jobs perms x all-core
        blas_ctx = threadpool_limits(1)                       #   matmul/einsum (reduced-rank nulls) don't thrash
    except Exception:
        blas_ctx = contextlib.nullcontext()
    with _WORKERS_LOCK:                                       # m28: serialize the global save/restore
        prev = _fft.get_workers()
        _fft.set_workers(1)                                   # perms run concurrently; each inner FFT single-threaded
    try:
        with blas_ctx, ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
            return list(ex.map(fn, items))
    finally:
        with _WORKERS_LOCK:
            _fft.set_workers(prev)

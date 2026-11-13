"""Memory-management regression tests (code-review scope 6)."""
import numpy as np
import pytest
from spectrast.memory import gene_chunk, guard_shape, available_bytes


def test_gene_chunk_floor_does_not_exceed_budget():
    """M6: under real memory pressure the chunk must drop BELOW the soft floor (down to 1), not be forced up
    to floor and OOM. A comfortable budget still caps at `cap`."""
    per_col = 700_000 * 8 * 8                                  # N=700k rows, 8 buffers, float64
    tight = gene_chunk(700_000, budget=2 * per_col)           # budget fits ~2 columns, well below floor=64
    assert 1 <= tight <= 2, f"floor forced a chunk above the budget: {tight}"
    assert gene_chunk(1000, budget=10 ** 12) == 128           # ample budget -> capped at cap


def test_guard_shape_raises_over_budget():
    """M4: guard_shape turns an oversized (N x m) OUTPUT allocation into an explicit MemoryError rather than
    a silent OOM, and is silent for a request that fits."""
    guard_shape((1000, 10))                                    # tiny -> fine, no raise
    big_rows = int(available_bytes() / 8 * 2)                 # 2x the live budget in float64 elements
    with pytest.raises(MemoryError):
        guard_shape((big_rows, 1))

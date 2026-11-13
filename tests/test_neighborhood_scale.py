"""The neighbourhood-scale estimators must depend on cell SPACING, not on cell COUNT."""
import numpy as np
import pytest

from spectrast.gr import _radius_from_xy
from spectrast.orient import _cell_spacing


def _poisson_at_density(n, density, seed):
    """n points at a fixed density, so the area grows with n and the spacing is constant."""
    area = n / density
    side = np.sqrt(area)
    return np.random.default_rng(seed).random((n, 2)) * side


@pytest.mark.parametrize("fn,kw", [(_radius_from_xy, dict(factor=1.0)), (_cell_spacing, {})])
def test_estimate_is_invariant_to_cell_count_at_fixed_density(fn, kw):
    """THE regression. Both estimators subsample for speed; if the subsample defines the point set
    rather than just the query points, the answer inflates by ~sqrt(N/sample) and two sections of the
    same density but different size get different neighbourhoods. Sections here span 2k to 80k cells
    at one density — 16x the subsample cap at the top — so the old code inflated by ~4x across them."""
    vals = [fn(_poisson_at_density(n, 1e-3, seed=1), **kw) for n in (2_000, 20_000, 80_000)]
    spread = max(vals) / min(vals)
    assert spread < 1.15, f"estimate varies {spread:.2f}x with cell count alone: {np.round(vals, 2)}"


@pytest.mark.parametrize("fn,kw", [(_radius_from_xy, dict(factor=1.0)), (_cell_spacing, {})])
def test_estimate_tracks_density(fn, kw):
    """It must still respond to the thing it is supposed to measure: 4x the density is half the
    spacing in 2-D."""
    sparse = fn(_poisson_at_density(20_000, 1e-3, seed=2), **kw)
    dense = fn(_poisson_at_density(20_000, 4e-3, seed=2), **kw)
    assert 0.45 < dense / sparse < 0.55, f"expected a ~2x drop, got {dense / sparse:.3f}"


def test_radius_is_order_invariant():
    """The subsample of QUERY points is a coordinate-sorted stride, so a row-shuffled copy of the same
    tissue must give the same radius."""
    xy = _poisson_at_density(30_000, 1e-3, seed=3)
    perm = np.random.default_rng(0).permutation(len(xy))
    assert _radius_from_xy(xy) == pytest.approx(_radius_from_xy(xy[perm]), rel=1e-12)

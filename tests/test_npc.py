"""Sanity tests for the FFT multiscale NPC co-localization prototype (spectrast.spatial.npc_colocalization_multiscale)."""
import numpy as np
import pytest
from spectrast.spatial import npc_colocalization_multiscale


def _planted_niche(seed=0):
    """A niche region where types A and B co-occur, plus a background of A, B and a filler type C."""
    rng = np.random.default_rng(seed)
    # niche blob: A and B interleaved in a 200x200 um patch centred at (500,500)
    n_niche = 800
    nx = rng.normal(500, 60, (n_niche, 2))
    nlab = np.where(rng.random(n_niche) < 0.5, "A", "B")
    # background mostly filler C (A,B globally rare), so A-B co-occurrence in the blob is far above chance
    n_bg = 4000
    bx = rng.uniform(0, 1000, (n_bg, 2))
    blab = rng.choice(["A", "B", "C"], n_bg, p=[0.1, 0.1, 0.8])
    xy = np.vstack([nx, bx]); labels = np.concatenate([nlab, blab])
    in_niche = np.concatenate([np.ones(n_niche, bool), np.zeros(n_bg, bool)])
    return xy, labels, in_niche


def test_npc_detects_planted_niche_and_is_abundance_normalized():
    xy, labels, in_niche = _planted_niche()
    radii = [1, 2, 4, 8]                                       # bins of 20um
    npc = npc_colocalization_multiscale(xy, labels, ["A", "B"], radii, bin_size=20.0,
                                        abundance_normalize=True, n_perm=0)
    assert npc.shape == (len(xy), len(radii))
    # niche cells score far above chance (>1); background A/B cells ~<= niche
    fine = npc[:, 1]                                           # 40um neighbourhood
    assert np.nanmean(fine[in_niche]) > 3.0, "NPC should be strongly enriched in the planted niche"
    assert np.nanmean(fine[in_niche]) > 2 * np.nanmean(fine[~in_niche]), "niche must beat background"


def test_npc_permutation_pvalues_flag_the_niche():
    xy, labels, in_niche = _planted_niche(seed=1)
    npc, p = npc_colocalization_multiscale(xy, labels, ["A", "B"], [2], bin_size=20.0, n_perm=100, seed=2)
    assert p.shape == npc.shape and p.min() >= 1.0 / 101 - 1e-9   # (r+1)/(n+1), never exactly 0
    # niche cells are more often significant than background
    assert (p[in_niche, 0] < 0.05).mean() > (p[~in_niche, 0] < 0.05).mean()


def test_npc_multivariate_requires_two_labels():
    xy, labels, _ = _planted_niche()
    with pytest.raises(ValueError):
        npc_colocalization_multiscale(xy, labels, ["A"], [2], bin_size=20.0)

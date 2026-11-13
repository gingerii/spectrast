"""Freeze the multiscale scalograms' outputs BEFORE a refactor, so the refactor can be held to bit-identity.

Run once, on the code being replaced:
    python tests/data/make_multiscale_reference.py
It writes multiscale_reference.npz (inputs AND outputs) next to itself and records the git commit the
outputs came from. tests/test_multiscale_equivalence.py then requires the current code to reproduce every
array exactly (assert_array_equal), and the fused entry point to equal the per-statistic functions.

The cases are chosen to reach every branch: irregular masks with holes and isolated (neighbourless)
occupied bins, a thin strip, a fully occupied grid, all three kernel kinds, radii from 1 to beyond half the
grid, a constant gene (zero variance -> NaN path) and an all-zero gene (Gini guard).
"""
import os
import subprocess

import numpy as np
from scipy.ndimage import gaussian_filter

from spectrast import spatial as sp

HERE = os.path.dirname(os.path.abspath(__file__))
FUNCS = {"moran": sp.morans_i_multiscale, "geary": sp.gearys_c_multiscale,
         "general_g": sp.general_g_multiscale, "losh": sp.losh_multiscale,
         "focality": sp.focality_scalogram, "texture": sp.texture_multiscale}


def _field(rng, H, W, G, sigma):
    f = np.stack([gaussian_filter(rng.standard_normal((H, W)), sigma) for _ in range(G)], -1)
    f = f - f.min((0, 1)) + 0.1 * rng.random((H, W, G))           # non-negative (General G / LOSH)
    return f


def cases():
    rng = np.random.default_rng(20261012)
    out = []
    # A: irregular two-blob tissue with holes and isolated occupied bins
    H, W = 60, 80
    yy, xx = np.mgrid[:H, :W]
    m = ((yy - 20) ** 2 + (xx - 22) ** 2 < 16 ** 2) | ((yy - 40) ** 2 / 2 + (xx - 58) ** 2 < 15 ** 2)
    m &= rng.random((H, W)) > 0.08                                  # holes
    m[2, 2] = m[57, 5] = m[3, 75] = True                            # isolated, neighbourless at r=1
    g = _field(rng, H, W, 5, 3.0)
    g[..., 3] = 2.5                                                 # constant -> zero variance
    g[..., 4] = 0.0                                                 # all zero
    out.append(("A", g, m, [1, 2, 3, 5, 8, 13], "disc"))
    # B: thin strip + scatter, gaussian kernel
    H, W = 41, 37
    m = np.zeros((H, W), bool); m[10:14, :] = True; m[rng.random((H, W)) > 0.93] = True
    out.append(("B", _field(rng, H, W, 3, 1.5), m, [2, 4, 7], "gaussian"))
    # C: fully occupied, queen kernel
    H, W = 32, 32
    out.append(("C", _field(rng, H, W, 4, 2.0), np.ones((H, W), bool), [1, 2, 4, 8], "queen"))
    # A1: case A's first gene ALONE -- numpy sums a single column pairwise, several row by row, so the
    # single-gene call is its own case (a refactor can be exact at G=5 and still move the last bit at G=1)
    a = out[0]
    out.append(("A1", a[1][..., :1].copy(), a[2], a[3], a[4]))
    # D: twelve genes on a ragged mask -- enough columns that the Gini sort runs on threads
    H, W = 50, 50
    yy, xx = np.mgrid[:H, :W]
    m = ((yy - 25) ** 2 + (xx - 25) ** 2 < 22 ** 2) & (rng.random((H, W)) > 0.15)
    out.append(("D", _field(rng, H, W, 12, 2.5), m, [1, 3, 6, 11], "disc"))
    return out


def amend_0_13_2(arrays):
    """spectrast 0.13.2 (2026-10-13): LOSH reads NaN where a field has no residual variance on the bins
    it is measured on, because H is 0/0 there. The e4753f8 reference holds that code's values for such fields --
    rounding noise for a nonzero constant, 0 for an all-zero field -- so those LOSH entries become NaN. Nothing else
    changes. Idempotent: on code from 0.13.2 on, the generated entries are NaN already."""
    for name in sorted({k.split("/")[0] for k in arrays if "/" in k}):
        g, m = arrays[f"{name}/grid"], np.asarray(arrays[f"{name}/mask"], bool)
        g3 = g if g.ndim == 3 else g[..., None]
        const = np.ptp(g3[m], axis=0) == 0
        L = np.array(arrays[f"{name}/losh"], dtype=float)
        if const.any():
            L[:, const] = np.nan
        arrays[f"{name}/losh"] = L
    arrays["amended"] = np.array("0.13.2: LOSH is NaN on a field with no residual variance")
    return arrays


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/spectrast/spatial.py"],
                           capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(HERE))).stdout
    if dirty.strip():
        raise SystemExit("spatial.py has uncommitted changes: the reference must come from committed code")
    arrays = {"commit": np.array(commit)}
    for name, g, m, radii, kind in cases():
        arrays[f"{name}/grid"], arrays[f"{name}/mask"] = g, m
        arrays[f"{name}/radii"], arrays[f"{name}/kind"] = np.array(radii), np.array(kind)
        for s, fn in FUNCS.items():
            arrays[f"{name}/{s}"] = fn(g, m, radii, kind)
    arrays = amend_0_13_2(arrays)
    np.savez_compressed(os.path.join(HERE, "multiscale_reference.npz"), **arrays)
    print(f"wrote multiscale_reference.npz from {commit}: " + ", ".join(sorted({k.split('/')[0] for k in arrays
                                                                                if '/' in k})))


if __name__ == "__main__":
    main()

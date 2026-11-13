"""Tests for spectrast.diff — differential scale-resolved spatial statistics. Run: pytest tests/test_diff.py -q.

Locks the two properties the framework lives or dies on:
  * NEGATIVE control — splitting one tissue into two halves (matched null) reports ~no significant genes,
  * POSITIVE control — spatially scrambling one gene in state B flags that gene and (mostly) not the others,
  * the reusable API returns the interpretable effect-size columns.
"""
import warnings

import numpy as np
import pytest

import spectrast


def _toy(n=2500, seed=0):
    """A little AnnData with spatially-structured genes (smooth gradients + blobs)."""
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, size=(n, 2))
    # structured fields: a gradient, a sine, a blob — all with clear spatial autocorrelation
    g0 = xy[:, 0] / 300.0
    g1 = np.sin(xy[:, 1] / 40.0)
    g2 = np.exp(-(((xy[:, 0] - 150) ** 2 + (xy[:, 1] - 150) ** 2) / (2 * 60.0 ** 2)))
    g3 = np.cos(xy[:, 0] / 60.0)
    X = np.clip(np.stack([g0, g1, g2, g3], 1) + 0.15 * rng.standard_normal((n, 4)), 0, None).astype("float32")
    A = ad.AnnData(X=X)
    A.var_names = ["g0", "g1", "g2", "g3"]; A.obsm["spatial"] = xy
    return A


def test_negative_control_reports_no_change():
    from spectrast import diff
    A = _toy(seed=1); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    df = diff.scalogram_delta([A[o[:h]].copy()], [A[o[h:]].copy()], ["g0", "g1", "g2", "g3"],
                              radii=[1, 2, 4], bin_size=20, null="matched", n_perm=40, seed=0)
    assert set(["delta_scale_um", "T_max", "r_at_max_um", "p", "fdr", "null_used"]).issubset(df.columns)
    assert (df.fdr < 0.05).sum() == 0                       # a technical replicate must not flag change


def test_positive_control_flags_scrambled_gene():
    from spectrast import diff
    A = _toy(seed=2); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    adA = A[o[:h]].copy(); adB = A[o[h:]].copy()
    Xb = adB.X.copy(); Xb[:, 0] = Xb[rng.permutation(adB.n_obs), 0]      # spatially scramble g0 in state B
    adB.X = Xb
    df = diff.scalogram_delta([adA], [adB], ["g0", "g1", "g2", "g3"], radii=[1, 2, 4], bin_size=20,
                              null="matched", n_perm=100, seed=0).set_index("gene")
    assert df.loc["g0", "fdr"] < 0.05                       # the scrambled gene is detected
    assert (df.drop("g0").fdr < 0.05).sum() <= 1            # specificity: at most one unperturbed false flag


def test_replicate_null_runs_and_calibrates():
    from spectrast import diff
    A = _toy(seed=3); rng = np.random.default_rng(0); q = np.array_split(rng.permutation(A.n_obs), 4)
    df = diff.scalogram_delta([A[q[0]].copy(), A[q[1]].copy()], [A[q[2]].copy(), A[q[3]].copy()],
                              ["g0", "g1", "g2", "g3"], radii=[1, 2, 4], bin_size=20, null="replicate", n_perm=40, seed=0)
    assert df.null_used.iloc[0] == "replicate"
    assert (df.fdr < 0.05).sum() == 0


# --- refactor: scalogram_delta is now a thin axis='moran' wrapper over differential ---------------------------

def test_unit_aggregation_shuffles_units_not_samples():
    """units_A/units_B average per-sample curves within a unit and permute UNITS: pseudo-replicated sections from
    one patient count once. Negative control with 3 units/arm (2 sections each) stays clean; and duplicating a
    section under the same unit does NOT change the result (aggregation absorbs it)."""
    from spectrast import diff
    A = _toy(seed=11); rng = np.random.default_rng(0); q = np.array_split(rng.permutation(A.n_obs), 6)
    secs = [A[qi].copy() for qi in q]                                       # 6 sections, same biology
    # arm A = 3 sections in 2 units (p1 gets 2 sections, p2 gets 1); arm B likewise; same biology -> ~0
    df = diff.differential(secs[:3], secs[3:6], axis="focality",
                           features=["g0", "g1", "g2", "g3"], radii=[1, 2, 4], bin_size=20, null="replicate",
                           n_perm=60, units_A=["p1", "p1", "p2"], units_B=["p3", "p3", "p4"], seed=0)
    assert (df.fdr < 0.05).sum() == 0
    # aggregation identity: adding a duplicate section to an existing unit leaves obs unchanged vs 1 section/unit
    base = diff.differential([secs[0], secs[1]], [secs[2], secs[3]], axis="moran",
                             features=["g0", "g1"], radii=[1, 2], bin_size=20, null="replicate", n_perm=30,
                             units_A=["p1", "p2"], units_B=["p3", "p4"], seed=1)
    dup = diff.differential([secs[0], secs[0], secs[1]], [secs[2], secs[3]], axis="moran",
                            features=["g0", "g1"], radii=[1, 2], bin_size=20, null="replicate", n_perm=30,
                            units_A=["p1", "p1", "p2"], units_B=["p3", "p4"], seed=1)
    assert np.allclose(base.dI_at_max.values, dup.dI_at_max.values)         # unit p1's mean unchanged by exact dup


def test_replicate_null_permutes_units_not_samples():
    """F4: under units_A/units_B the replicate null must operate on the POST-AGGREGATION units, never on
    the raw sections. 3 sections in 2 units per arm -> 4 units, 6 sections.

    Asserted on the null's SUPPORT rather than by spying on the sampler, because the two are not the
    same claim and only the support survives both code paths: at 4 units the support is C(4,2) = 6, at
    6 sections it would be C(6,3) = 20, so the reported support alone separates them. The sampler spy
    this test used to rely on stopped firing when small supports began to be enumerated instead of
    drawn (`_clamp_perms`) — the null was still built over units, but `rng.permutation` was no longer
    called at all, so the probe measured the implementation rather than the property."""
    import unittest.mock as mock
    from spectrast import diff
    A = _toy(seed=11); rng0 = np.random.default_rng(0); q = np.array_split(rng0.permutation(A.n_obs), 6)
    secs = [A[qi].copy() for qi in q]                                       # 6 sections, same biology
    seen = []
    real = np.random.default_rng
    class _SpyRng:
        def __init__(self, g): self._g = g
        def permutation(self, x, *a, **k):
            seen.append(len(np.arange(x) if np.isscalar(x) else np.asarray(x)))
            return self._g.permutation(x, *a, **k)
        def __getattr__(self, n): return getattr(self._g, n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mock.patch("spectrast.diff.np.random.default_rng", lambda s=None: _SpyRng(real(s))):
            diff.differential(secs[:3], secs[3:6], axis="focality",
                              features=["g0", "g1"], radii=[1, 2], bin_size=20, null="replicate",
                              n_perm=20, units_A=["p1", "p1", "p2"], units_B=["p3", "p3", "p4"], seed=0)
    msgs = " ".join(str(w.message) for w in caught)
    assert "only 6 distinct relabelings" in msgs                            # C(4,2): units, post-aggregation
    assert "20 distinct relabelings" not in msgs                            # C(6,3): the 6 raw sections
    assert 6 not in seen                                                    # and never shuffles them directly


def test_small_support_null_is_enumerated_exactly():
    """A clamped null must be ENUMERATED, so its floor is exact and does not depend on the seed.

    Sampling `support-1` permutations out of `support` repeats some and misses others, which leaves the
    smallest attainable p wherever the draw happened to land. On a 4-vs-4 design (support C(8,4) = 70,
    exact floor 2/70 = 0.029) that produced a smallest p of 0.057 — above 0.05, so a strongly perturbed
    contrast returned nothing and the DESIGN looked incapable when only the sampling was. Enumeration
    fixes the floor at 2/support and makes it seed-independent.

    Both properties are asserted because either alone is weak: a seed-independent p could still be
    wrong, and a correct floor on one seed could still be luck."""
    from spectrast import diff
    A = [_toy(seed=100 + i) for i in range(4)]
    B = [_toy(seed=200 + i) for i in range(4)]
    kw = dict(features=["g0", "g1"], radii=[1, 2], bin_size=20, null="replicate", n_perm=500)
    runs = [diff.differential(A, B, **kw, seed=s) for s in (0, 1, 2)]
    for r in runs[1:]:
        assert np.array_equal(r.p.to_numpy(), runs[0].p.to_numpy())          # enumeration ignores the seed
    # the null holds every labelling but the observed one, so p can only be a multiple of 1/support
    support = 70
    assert np.allclose(runs[0].p.to_numpy() * support, np.round(runs[0].p.to_numpy() * support))
    assert runs[0].p.min() >= 2.0 / support - 1e-12                          # complement is always as extreme


def test_scalogram_delta_wraps_differential_identically():
    """Regression: the wrapper's numeric columns equal differential(axis='moran'); back-compat column names kept."""
    from spectrast import diff
    A = _toy(seed=1); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    kw = dict(radii=[1, 2, 4], bin_size=20, null="matched", n_perm=60, seed=0)
    old = diff.scalogram_delta([A[o[:h]].copy()], [A[o[h:]].copy()], ["g0", "g1", "g2", "g3"], **kw)
    new = diff.differential([A[o[:h]].copy()], [A[o[h:]].copy()], axis="moran",
                            features=["g0", "g1", "g2", "g3"], **kw)
    # exact old column set/order (byte-compatible output), and peak_signal <-> max_moran identity
    assert list(old.columns) == ["gene", "delta_scale_um", "dI_at_max", "T_max", "r_at_max_um",
                                 "p", "fdr", "null_used", "structured", "max_moran"]
    for c in ["delta_scale_um", "dI_at_max", "T_max", "r_at_max_um", "p", "fdr"]:
        assert np.allclose(old[c].values, new[c].values, equal_nan=True)
    assert np.allclose(old["max_moran"].values, new["peak_signal"].values)


# --- calibration battery, compact form, for each NEW axis ------------

def test_multiscale_primitives_match_single_radius():
    """The shared-FFT multiscale Geary/focality/Moran must equal a single-radius computation to machine precision
    (rtol=0 — the amortization is a speedup, not a different statistic). A POROUS mask (some cells neighbourless at
    r=1) is used deliberately: it exercises the W1>0.5 degree gate that otherwise lets fft roundoff leak spurious
    LISA into neighbourless cells."""
    from spectrast import spatial as S
    from spectrast.spatial import neighbor_kernel, _fftconv_same
    rng = np.random.default_rng(0)
    H, W, G = 40, 46, 5
    mask = rng.random((H, W)) > 0.3                                         # porous -> some neighbourless cells at r=1
    grid = rng.standard_normal((H, W, G)) * mask[..., None]
    radii = [1, 2, 4, 8]
    tol = dict(atol=1e-9, rtol=0.0, equal_nan=True)                         # rtol=0: exact, not the allclose default 1e-5
    # Geary: multiscale vs looped single-radius
    ms = S.gearys_c_multiscale(grid, mask, radii, kind="disc")
    loop = np.stack([S.gearys_c(grid, mask, radius=r) for r in radii])
    assert np.allclose(ms, loop, **tol)
    # focality: multiscale Gini curve vs looped local_morans_i_batch + positive Gini
    def gini_pos(v):
        v = np.where(np.isfinite(v), v, 0.0); pos = np.sort(np.clip(v, 0.0, None), 0); n = pos.shape[0]; s = pos.sum(0)
        return np.nan_to_num(((2 * np.arange(1, n + 1) - n - 1)[:, None] * pos).sum(0) / np.where(s > 0, n * s, np.nan))
    fs = S.focality_scalogram(grid, mask, radii, kind="disc")
    mb = mask.astype(bool)
    loopf = np.stack([gini_pos(S.local_morans_i_batch(grid, mask, radius=r)[mb]) for r in radii])
    assert np.allclose(fs, loopf, **tol)
    # Moran: morans_i_multiscale (ROW-STANDARDIZED, B1 fix) vs an INDEPENDENT single-radius reference through the
    # other conv path (_fftconv_same -> direct ndimage for small kernels), cross-checking crop/offset/normalization.
    m = mask.astype(float); msum = m.sum()
    zbar = (grid * m[..., None]).sum((0, 1)) / msum; z = (grid - zbar) * m[..., None]; den = (z * z).sum((0, 1))
    ref = []
    for r in radii:
        k = neighbor_kernel(r, "disc"); Wz = _fftconv_same(z, k); Wm = _fftconv_same(m[..., None], k)[..., 0]
        invW = np.where(Wm > 1e-12, 1.0 / np.maximum(Wm, 1e-12), 0.0)[..., None]   # divide by per-cell degree
        ref.append((z * (Wz * invW)).sum((0, 1)) / np.where(den == 0, np.nan, den))
    assert np.allclose(S.morans_i_multiscale(grid, mask, radii, kind="disc"), np.stack(ref), **tol)


@pytest.mark.parametrize("axis", ["geary", "focality"])
def test_new_pergene_axis_battery(axis):
    from spectrast import diff
    genes = ["g0", "g1", "g2", "g3"]
    # NEG: split one tissue -> no axis may invent a change between technical halves
    A = _toy(seed=5); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    neg = diff.differential([A[o[:h]].copy()], [A[o[h:]].copy()], axis=axis, features=genes,
                            radii=[1, 2, 4], bin_size=20, null="matched", n_perm=100, seed=0)
    assert (neg.fdr < 0.05).sum() == 0, f"{axis} negative control not clean"
    # POS: spatially scramble g0 in B -> that gene's field organization changes on every per-gene axis
    adA = A[o[:h]].copy(); adB = A[o[h:]].copy()
    Xb = adB.X.copy(); Xb[:, 0] = Xb[rng.permutation(adB.n_obs), 0]; adB.X = Xb
    pos = diff.differential([adA], [adB], axis=axis, features=genes, radii=[1, 2, 4], bin_size=20,
                            null="matched", n_perm=100, seed=0).set_index("gene")
    assert pos.loc["g0", "fdr"] < 0.05, f"{axis} missed the scrambled gene"
    assert (pos.drop("g0").fdr < 0.05).sum() <= 1, f"{axis} false-flagged unperturbed genes"


def _toy_coloc(n=7000, seed=0):
    """AnnData with a genuinely CO-LOCALIZED pair (gA, gB share a blob) + an unrelated pair (gC gradient, gD sine).
    Larger/cleaner than _toy so the cross-spectral grid has enough radial modes for a reliable coloc test."""
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed); xy = rng.uniform(0, 360, size=(n, 2))
    blob = np.exp(-(((xy[:, 0] - 180) ** 2 + (xy[:, 1] - 180) ** 2) / (2 * 55.0 ** 2)))
    gA = blob + 0.05 * rng.standard_normal(n); gB = blob + 0.05 * rng.standard_normal(n)   # co-localized
    gC = xy[:, 0] / 360.0; gD = np.sin(xy[:, 1] / 40.0)
    X = np.clip(np.stack([gA, gB, gC, gD], 1), 0, None).astype("float32")
    A = ad.AnnData(X=X); A.var_names = ["gA", "gB", "gC", "gD"]; A.obsm["spatial"] = xy
    return A


def test_coloc_axis_battery():
    from spectrast import diff
    pairs = [("gA", "gB"), ("gC", "gD")]; kw = dict(radii=[2, 4, 8, 16], bin_size=15, null="matched", n_perm=100, seed=0)
    # NEG: split -> co-localization is identical in both halves
    A = _toy_coloc(seed=6); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    neg = diff.differential([A[o[:h]].copy()], [A[o[h:]].copy()], axis="coloc", pairs=pairs, **kw)
    assert "pair" in neg.columns and neg.delta_scale_um.isna().all()      # pairs carry no characteristic length
    assert (neg.fdr < 0.05).sum() == 0, "coloc negative control not clean"
    # POS: scramble one member of the co-localized pair in B -> destroys its co-localization; pair 1 untouched
    adA = A[o[:h]].copy(); adB = A[o[h:]].copy()
    Xb = adB.X.copy(); Xb[:, 1] = Xb[rng.permutation(adB.n_obs), 1]; adB.X = Xb   # scramble gB (member of pair 0)
    pos = diff.differential([adA], [adB], axis="coloc", pairs=pairs, **kw).set_index("pair")
    assert pos.loc["gA~gB", "fdr"] < 0.05, "coloc missed the broken pair"


def test_omnibus_single_fdr_and_decomp():
    from spectrast import diff
    genes = ["g0", "g1", "g2", "g3"]
    A = _toy(seed=7); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    # NEG omnibus: split -> combined screen flags nothing (one FDR pass over G genes)
    scr, dec = diff.differential_omnibus([A[o[:h]].copy()], [A[o[h:]].copy()], axes=("moran", "focality"),
                                         features=genes, radii=[1, 2, 4], bin_size=20, null="matched", n_perm=100, seed=0)
    assert set(["gene", "T_omni", "p", "fdr", "driver_axis"]).issubset(scr.columns)
    assert (scr.fdr < 0.05).sum() == 0
    # POS omnibus: scramble g0 in B -> screen flags g0, decomp names a driver axis for it
    adA = A[o[:h]].copy(); adB = A[o[h:]].copy()
    Xb = adB.X.copy(); Xb[:, 0] = Xb[rng.permutation(adB.n_obs), 0]; adB.X = Xb
    scr2, dec2 = diff.differential_omnibus([adA], [adB], axes=("moran", "focality"), features=genes,
                                           radii=[1, 2, 4], bin_size=20, null="matched", n_perm=100, seed=0)
    assert scr2.set_index("gene").loc["g0", "fdr"] < 0.05
    assert "g0" in set(dec2.gene)
    assert "delta_scale_um" in dec2.columns              # decomp carries the documented characteristic-scale column


def test_omnibus_shared_permutation_null_is_exact():
    """The single combined statistic must be built from the SAME permutations across axes: reconstruct the omnibus
    p by maxing the per-axis (same-seed) Tmax_null and assert it equals differential_omnibus's p exactly. This locks
    the 'one shared null, family size = G not G*S' claim — a stray rng draw desyncing an axis would break it."""
    from spectrast import diff
    genes = ["g0", "g1", "g2", "g3"]; axes = ("moran", "focality")
    A = _toy(seed=8); rng = np.random.default_rng(0); o = rng.permutation(A.n_obs); h = A.n_obs // 2
    adA, adB = A[o[:h]].copy(), A[o[h:]].copy()
    kw = dict(radii=[1, 2, 4], bin_size=20, null="matched", n_perm=80, seed=0)
    per = {a: diff.differential([adA], [adB], axis=a, features=genes, _return_internals=True, **kw)[1] for a in axes}
    T_obs = np.stack([per[a]["Tmax"] for a in axes]); T_null = np.stack([per[a]["Tmax_null"] for a in axes])
    struct = np.any([per[a]["structured"] for a in axes], 0)
    T_omni = T_obs.max(0); T_omni_null = T_null.max(0); nperm = T_null.shape[1]
    p_exp = np.array([(1 + int((T_omni_null[:, g] >= T_omni[g] - 1e-9).sum())) / (nperm + 1) for g in range(len(genes))])
    p_exp = np.where(struct, p_exp, 1.0)
    scr, _ = diff.differential_omnibus([adA], [adB], axes=axes, features=genes, **kw)
    assert np.allclose(scr.p.values, p_exp)             # omnibus p == maxed per-axis shared-permutation null


def test_char_scale_centroid_recovers_coarsening_without_ceiling():
    """The differential's characteristic scale is the positive-scalogram centroid (matching gr.scale_spectrum):
    a coarser field (autocorrelation persisting to larger radius) must report a LARGER scale, and — unlike a
    half-max correlation_length crossing — it must NOT saturate at the radius-grid ceiling for a strongly
    autocorrelated (coarsened) field. Locks the 2026-07 fix for the scale-recovery Δscale collapse."""
    from spectrast.diff import _char_scale
    radii_um = np.array([30., 60., 120., 240., 480.])
    fine = np.array([0.50, 0.30, 0.12, 0.04, 0.01])[:, None]        # decays fast -> small scale
    coarse = np.array([0.96, 0.99, 0.98, 0.93, 0.80])[:, None]      # persists -> large scale (half-max would ceiling)
    s_fine = _char_scale(fine, radii_um)[0]; s_coarse = _char_scale(coarse, radii_um)[0]
    assert s_coarse > s_fine + 30, f"coarser field must report larger scale: fine {s_fine:.0f} vs coarse {s_coarse:.0f}"
    assert s_fine < radii_um[-1] and s_coarse < radii_um[-1]        # centroid is interior, not pinned to the edge


def test_paired_null_detected_and_overridable():
    """M8 (review 2026-07): when the two arms SHARE unit labels (a donor at baseline vs on-treatment), the
    replicate null must switch to a within-unit SIGN-FLIP null — an unpaired label shuffle would double-count
    the shared donor and shuffle donors freely across arms. Auto-detected; overridable with paired=False; and
    <2 shared units under paired=True raises rather than silently degenerating."""
    from spectrast import diff
    import anndata as ad
    rng = np.random.default_rng(0)

    def sample():
        n = 400; xy = rng.random((n, 2)) * 200.0
        X = np.abs(np.c_[np.sin(xy[:, 0] / 30.0) + 0.2 * rng.standard_normal(n),
                         rng.standard_normal((n, 2))]).astype("float32")
        a = ad.AnnData(X); a.obsm["spatial"] = xy.astype("float32"); a.var_names = ["g0", "g1", "g2"]
        return a

    donors = ["d0", "d1", "d2", "d3"]
    A = [sample() for _ in donors]; B = [sample() for _ in donors]
    auto = diff.differential(A, B, axis="moran", features=["g0"], units_A=donors, units_B=donors, n_perm=100)
    assert auto["null_used"].iloc[0] == "replicate-paired"        # auto-detected paired design
    unp = diff.differential(A, B, axis="moran", features=["g0"], units_A=donors, units_B=donors,
                            paired=False, n_perm=100)
    assert unp["null_used"].iloc[0] == "replicate"                # forced unpaired
    for r in (auto, unp):
        assert 0.0 < float(r["p"].iloc[0]) <= 1.0
    with pytest.raises(ValueError):                               # <2 shared units under paired -> informative error
        diff.differential(A, B, axis="moran", features=["g0"],
                          units_A=["d0", "d1", "x2", "x3"], units_B=["d0", "y1", "y2", "y3"],
                          paired=True, n_perm=20)


def test_groups_from_obs_returns_units_with_unit_key():
    """m21: groups_from_obs must honour unit_key (return the parallel unit labels), not silently ignore it."""
    from spectrast import diff
    import anndata as ad

    def toy(cond, donor):
        a = ad.AnnData(np.abs(np.random.default_rng(0).standard_normal((30, 3))).astype("float32"))
        a.obs["condition"] = cond; a.obs["donor"] = donor
        return a

    adatas = [toy("on", "d0"), toy("off", "d0"), toy("on", "d1"), toy("off", "d1")]
    A, B = diff.groups_from_obs(adatas, "condition", ["on"], ["off"])              # 2-tuple when no unit_key
    assert len(A) == 2 and len(B) == 2
    A2, B2, uA, uB = diff.groups_from_obs(adatas, "condition", ["on"], ["off"], unit_key="donor")
    assert uA == ["d0", "d1"] and uB == ["d0", "d1"]                               # units parallel to each arm


def test_differential_warns_on_mixed_backend_arms():
    """R4: MPS/CUDA are float32-only, so every statistic carries a small device-dependent offset. That
    offset cancels in a SAME-backend difference but not across a mixed one, where it enters the effect
    exactly where real signal does. A contrast whose arms carry different backend stamps must warn —
    it is otherwise a silent wrong answer."""
    import warnings as _w
    from spectrast import diff
    import anndata as ad

    def toy(backend, seed):
        rng = np.random.default_rng(seed)
        a = ad.AnnData(np.abs(rng.standard_normal((400, 4))).astype("float32"))
        a.obsm["spatial"] = rng.random((400, 2)) * 200
        a.var_names = [f"g{i}" for i in range(4)]
        a.uns["spectrast_moranI_params"] = dict(radius=30.0, backend=backend)
        return a

    A = [toy("mps", 0), toy("mps", 1)]
    B = [toy("cpu", 2), toy("cpu", 3)]
    with pytest.warns(UserWarning, match="different compute-backend stamps"):
        df = diff.differential(A, B, axis="moran", features=["g0", "g1"], n_perm=20)
    assert df.attrs["spectrast_backend"] == spectrast.get_backend()   # provenance on the result

    same = [toy("cpu", 4), toy("cpu", 5)]
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        # n_perm=5, not 20: this 2-vs-2 design admits only C(4,2)=6 relabelings, so a larger n_perm
        # legitimately trips the permutation-support cap and would fail this assertion for a reason
        # that has nothing to do with backends.
        diff.differential(same, B, axis="moran", features=["g0"], n_perm=5)
    # Assert on the BACKEND warning specifically rather than on silence. This 2-vs-2 design also
    # (correctly) trips the design-power warning — 6 relabelings put the smallest attainable p at
    # 0.167 — and blanket `simplefilter("error", UserWarning)` would fail on that unrelated, true
    # statement. What this test is about is that agreeing backends produce no backend complaint.
    assert not any("compute-backend" in str(w.message) for w in rec)


def _cohort(n=1200, seed=0):
    """One concatenated cohort: 2 conditions x 2 donors x 2 sections. g0 is spatially scrambled in
    arm B, so a working contrast must flag it; the rest are shared structure."""
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    parts = []
    for cond in ("on", "off"):
        for donor in ("d0", "d1"):
            for rep in (0, 1):
                xy = (rng.random((n, 2)) * 300).astype("float32")
                cols = []
                for j in range(4):
                    c = rng.random(2) * 300
                    cols.append(np.exp(-((xy - c) ** 2).sum(1) / (2 * 50.0 ** 2))
                                + rng.normal(0, 0.05, n))
                X = np.column_stack(cols).astype("float32")
                if cond == "off":
                    X[:, 0] = X[rng.permutation(n), 0]          # destroy g0's structure in arm B only
                X -= X.min(0)
                p = ad.AnnData(X); p.obsm["spatial"] = xy
                p.var_names = [f"g{j}" for j in range(4)]
                p.obs["condition"] = cond; p.obs["donor"] = donor
                p.obs["section"] = f"{cond}_{donor}_{rep}"
                parts.append(p)
    out = ad.concat(parts, index_unique="-")
    out.obs["section"] = pd.Categorical(out.obs["section"])
    return out


def test_split_sections_returns_views_not_copies():
    """T2 #4: the cohort bridge must not duplicate X — that is the whole memory argument for it."""
    from spectrast import diff
    a = _cohort(n=200)
    views, labels = diff.split_sections(a, "section")
    assert len(views) == len(labels) == 8
    assert sum(v.n_obs for v in views) == a.n_obs
    assert all(v.is_view for v in views)                        # views, so X is not copied
    assert labels == list(dict.fromkeys(a.obs["section"].astype(str)))   # first-appearance order


def test_differential_cohort_matches_the_hand_built_lists():
    """differential_cohort is marshalling, not a different test: it must reproduce, exactly, what you
    get by splitting the cohort by hand and calling differential — including the unit labels that
    stop two sections from one donor counting as two replicates."""
    from spectrast import diff
    a = _cohort()
    res = diff.differential_cohort(a, "condition", ["on"], ["off"], sample_key="section",
                                   unit_key="donor", axis="moran",
                                   features=["g0", "g1", "g2"], n_perm=100, seed=0)
    views, _ = diff.split_sections(a, "section")
    A, B, uA, uB = diff.groups_from_obs(views, "condition", ["on"], ["off"], unit_key="donor")
    ref = diff.differential(A, B, units_A=uA, units_B=uB, axis="moran",
                            features=["g0", "g1", "g2"], n_perm=100, seed=0)
    assert np.array_equal(res.T_max.to_numpy(), ref.T_max.to_numpy())
    assert np.array_equal(res.p.to_numpy(), ref.p.to_numpy())
    assert res.null_used.iloc[0] == "replicate-paired"          # donors shared across arms -> paired
    # the positive control: the scrambled gene is the one that moves
    assert res.set_index("gene").loc["g0", "T_max"] > res.set_index("gene").loc[["g1", "g2"], "T_max"].max()


def test_differential_cohort_errors_are_actionable():
    """A condition value that matches nothing must say what IS present, not fail deep inside the null."""
    from spectrast import diff
    a = _cohort(n=200)
    with pytest.raises(ValueError, match="values present"):
        diff.differential_cohort(a, "condition", ["treated"], ["off"], sample_key="section",
                                 axis="moran", features=["g0"], n_perm=10)
    with pytest.raises(KeyError):
        diff.differential_cohort(a, "condition", ["on"], ["off"], sample_key="no_such_key",
                                 axis="moran", features=["g0"], n_perm=10)


# ---------------------------------------------------------------- permutation support
def test_perm_support_counts_the_distinct_relabelings():
    """C(nA+nB, nA) for the unpaired label null; 2**n for the paired sign-flip."""
    from spectrast import diff
    assert diff._perm_support("unpaired", 4, 5) == 126             # the trem2 NEG design
    assert diff._perm_support("unpaired", 9, 10) == 92378          # its real contrast
    assert diff._perm_support("signflip", 3) == 8
    assert diff._perm_support("signflip", 24) == 16777216


def test_n_perm_is_capped_at_the_null_support_so_p_cannot_beat_the_exact_floor():
    """Drawing more permutations than the design admits makes the test ANTI-CONSERVATIVE.

    p is (1 + #{null >= obs}) / (1 + n_perm), so its floor is set by the caller — but the EXACT
    permutation p cannot fall below 1/support, since the observed labelling is itself one of the
    relabelings. Past the support the sampler only repeats partitions while the denominator keeps
    growing, so a p below the exact floor becomes reportable. Measured on a 4-vs-5 split (126
    partitions, n_perm=200): a same-condition contrast reported p = 1/201 = 0.00497 against an exact
    floor of 1/126, and produced a false positive that vanished once capped.
    """
    from spectrast import diff
    A = [_toy(seed=s) for s in range(3)]
    B = [_toy(seed=s) for s in range(10, 13)]
    support = diff._perm_support("unpaired", 3, 3)                 # C(6,3) = 20
    assert support == 20
    with pytest.warns(UserWarning, match="exceeds the replicate label null's support"):
        r = spectrast.diff.differential(A, B, features=["g0", "g1"], n_perm=500,
                                        radii=(1, 2, 4), bin_size=30.0, seed=0)
    assert r["p"].min() >= 1.0 / support - 1e-12, "p fell below the exact permutation floor"


def test_the_cap_is_a_no_op_for_a_well_powered_design():
    """The safety proof: with support > n_perm the clamp cannot move a single value.

    Compared against the pre-clamp behaviour directly, by disabling the clamp — so this asserts
    byte-identity of the capstone-style designs rather than merely asserting no warning.
    """
    from spectrast import diff
    A = [_toy(seed=s) for s in range(5)]
    B = [_toy(seed=s) for s in range(10, 15)]
    kw = dict(features=["g0", "g1", "g2"], n_perm=50, radii=(1, 2, 4), bin_size=30.0, seed=0)
    assert diff._perm_support("unpaired", 5, 5) == 252 > 50        # ample support
    with warnings.catch_warnings():
        warnings.simplefilter("error")                             # any clamp warning fails the test
        clamped = spectrast.diff.differential(A, B, **kw)
    orig = diff._clamp_perms
    diff._clamp_perms = lambda n, support, what, **_: n            # the pre-clamp code path
    #                                          ^^^ swallows floor_num; the stub must not pin
    #                                          the real signature or a keyword added to it
    #                                          fails this test instead of the code it guards.
    try:
        unclamped = spectrast.diff.differential(A, B, **kw)
    finally:
        diff._clamp_perms = orig
    for col in ("T_max", "p", "fdr", "dI_at_max", "delta_scale_um"):
        assert np.array_equal(np.asarray(clamped[col]), np.asarray(unclamped[col])), \
            f"{col} moved on a well-powered design — the clamp is not a no-op"


def test_paired_signflip_support_is_also_capped():
    """The paired null has 2**n_shared sign vectors; 3 donors admit only 8."""
    from spectrast import diff
    a = [_toy(seed=s) for s in range(3)]
    b = [_toy(seed=s) for s in range(10, 13)]
    u = ["p1", "p2", "p3"]
    with pytest.warns(UserWarning, match="paired sign-flip null's support"):
        r = spectrast.diff.differential(a, b, features=["g0"], n_perm=100, radii=(1, 2, 4),
                                        bin_size=30.0, seed=0, units_A=u, units_B=u, paired=True)
    assert r["p"].min() >= 1.0 / 8 - 1e-12


def test_all_per_gene_axes_run_and_uncentred_ones_stay_non_negative():
    """Every registered per-gene axis must work through `differential`, and the two defined on the
    VALUES rather than on deviations must not be handed a centred field.

    General G is a ratio of products of x, so centring makes its denominator a sum of signed products
    that can pass through zero; `spatial.general_g_multiscale` warns when it sees negative input. That
    warning firing is the failure mode this guards — it would be easy to add the axis beside Moran and
    reflexively feed it `_zstack`, which produces numbers that look fine and mean nothing."""
    import warnings as _w
    from spectrast import diff
    A = [_toy(seed=s) for s in (1, 2, 3)]
    B = [_toy(seed=s) for s in (4, 5, 6)]
    curves = {}
    per_gene = sorted(set(diff.STAT) - diff.PAIR_AXES - {"mean"} - set(diff.REFUSED_DIFF_AXES))
    assert {"moran", "geary", "focality", "losh"} <= set(per_gene)
    assert "general_g" not in per_gene, "general_g is refused as a differential axis; see REFUSED_DIFF_AXES"
    for axis in per_gene:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            d = diff.differential(A, B, axis=axis, features=["g0", "g1"], radii=[1, 2],
                                  bin_size=25, null="replicate", n_perm=50, seed=0)
        assert len(d) == 2 and d.p.notna().all(), axis
        assert not [c for c in caught if "negative" in str(c.message)], f"{axis} was given a centred field"
        curves[axis] = d.set_index("gene").dI_at_max.to_numpy()

    # THE ASSERTION THAT MATTERS: each axis must compute something DIFFERENT. When general_g and losh
    # were first registered they had no branch in `_curve_gate`, so they fell through to Moran and
    # returned Moran's numbers under their own names -- every hit identical, Jaccard 1.00, run times
    # identical to the millisecond. "It ran and produced p-values" cannot catch that; only comparing
    # the axes against each other can.
    import itertools
    for a, b in itertools.combinations(sorted(curves), 2):
        assert not np.allclose(curves[a], curves[b], equal_nan=True), \
            f"axes {a!r} and {b!r} returned identical curves — one is falling through to the other"


def test_refused_differential_axes_raise_with_the_reason():
    """`general_g` computes fine but is not a per-gene differential test, and must say so.

    It is uncentred, so a global change in expression level moves it for every gene alike. On a
    paired cohort it returned raw p<0.05 for 372 of 372 structured genes, 371 in the same direction,
    while its own negative control could not reach p<0.25 — one global effect reported once per gene.
    Refusing beats returning 372 confident answers nobody can interpret. The statistic stays in
    `STAT` (and in `spatial`) because it is still valid for description; only the CONTRAST is refused.
    """
    from spectrast import diff
    A = [_toy(seed=s) for s in (1, 2, 3)]
    B = [_toy(seed=s) for s in (4, 5, 6)]
    assert "general_g" in diff.REFUSED_DIFF_AXES and "general_g" in diff.STAT
    # texture is refused on a DIFFERENT ground: it has no per-radius curve, so it is not in STAT at
    # all. The refusal must therefore be checked BEFORE the STAT lookup, or the caller gets "unknown
    # axis" and goes hunting for a typo instead of reading why the test does not exist.
    assert "texture" in diff.REFUSED_DIFF_AXES and "texture" not in diff.STAT
    for axis, why in diff.REFUSED_DIFF_AXES.items():
        with pytest.raises(ValueError, match="not admissible as a differential test"):
            diff.differential(A, B, axis=axis, features=["g0", "g1"], radii=[1, 2],
                              bin_size=25, null="replicate", n_perm=50, seed=0)
        assert len(why) > 100, f"{axis} refusal must explain WHY, not just refuse"
        # the omnibus filters unknown axes silently, which would narrow the screen without saying so
        with pytest.raises(ValueError, match="not admissible as a differential test"):
            diff.differential_omnibus(A, B, axes=("moran", axis), features=["g0", "g1"],
                                      radii=[1, 2], bin_size=25, null="replicate", n_perm=20, seed=0)


def test_null_band_arm_reconstruction_is_exact():
    """The band reconstructs each permutation's arm curves from the difference. Check the identity.

    Where the null relabels whole units the arm TOTAL is conserved, so `mA = (S + nB*d)/(nA+nB)` is
    exact, not an approximation. Feeding it the OBSERVED difference must return the OBSERVED arm
    curves — if that fails, every null scale in the band is wrong in a way no smoke test would show.
    """
    from spectrast import diff
    A = [_toy(seed=s) for s in (1, 2, 3)]
    B = [_toy(seed=s) for s in (4, 5, 6, 7)]
    df, internals = diff.differential(A, B, axis="moran", features=["g0", "g1"], radii=[1, 2, 4],
                                      bin_size=25, null="replicate", n_perm=40, seed=0,
                                      _return_internals=True)
    mA, mB = internals["meanA"], internals["meanB"]
    nA, nB = len(A), len(B)
    S = nA * mA + nB * mB
    obs = mA - mB
    np.testing.assert_allclose((S + nB * obs) / (nA + nB), mA, rtol=0, atol=1e-12)
    np.testing.assert_allclose((S - nA * obs) / (nA + nB), mB, rtol=0, atol=1e-12)


def test_null_band_is_matched_and_tightens_with_more_units():
    """The band must come from the test's own null, and must respond to the DESIGN.

    Effect-estimate noise goes as sqrt(1/nA + 1/nB), so a bigger design has to give a tighter band.
    A band that ignores the design is exactly the mismatched-control bug this replaces (a 4v5 control
    against a 9v10 test reads 1.46x too wide).
    """
    from spectrast import diff
    kw = dict(axis="moran", features=["g0", "g1", "g2"], radii=[1, 2, 4], bin_size=25,
              null="replicate", n_perm=60, seed=0)
    small = diff.differential([_toy(seed=s) for s in (1, 2, 3)],
                              [_toy(seed=s) for s in (4, 5, 6)], **kw)
    big = diff.differential([_toy(seed=s) for s in range(1, 8)],
                            [_toy(seed=s) for s in range(8, 15)], **kw)
    for d in (small, big):
        nb = d.attrs["spectrast_null_band"]
        assert nb is not None and nb["matched"], "replicate null must yield a matched band"
        assert nb["dI_at_max"][0] < nb["dI_at_max"][1]
    w_small = np.ptp(small.attrs["spectrast_null_band"]["dI_at_max"])
    w_big = np.ptp(big.attrs["spectrast_null_band"]["dI_at_max"])
    assert w_big < w_small, f"band did not tighten with more units: {w_big:.4f} vs {w_small:.4f}"


def test_degenerate_axis_is_flagged_not_silently_reported():
    """An axis reporting ONE GLOBAL effect once per feature must say so.

    This is what general_g did on a paired cohort -- 372/372 features significant, 371 the same direction -- and
    every per-feature number looked plausible. Only the joint pattern gives it away, so `differential`
    checks it rather than leaving it to whoever reads the table.
    """
    from spectrast import diff
    assert diff._degeneracy(np.full(50, 0.001), np.full(50, -0.4), np.ones(50, bool)), \
        "all-significant, all-one-direction must be flagged"
    rng = np.random.default_rng(0)
    ok = diff._degeneracy(rng.uniform(size=200), rng.normal(size=200), np.ones(200, bool))
    assert ok == "", f"a well-behaved axis must not be flagged, got {ok!r}"
    # one condition alone is not enough: a real strong contrast can be all-one-direction
    assert diff._degeneracy(rng.uniform(size=200), np.full(200, -0.4), np.ones(200, bool)) == ""

"""Tests for spectrast.orient — windowed orientation discovery.

Covers: (1) the frequency-resolved polar spectrum recovers a KNOWN stripe orientation and reads isotropic
noise as ~0; (2) rings are defined relative to window Nyquist (scale-comparable); (3) the per-window scramble
null is calibrated (isotropic field -> uniform-ish p, not systematically tiny despite finite-window bias);
(4) the AnnData scan + scale aggregation + region discovery recover a planted laminar sub-region and rank the
carrier gene above a noise gene; (5) region gene ranking. All small + fast (CPU backend)."""
import numpy as np
import anndata as ad
import pytest
import spectrast
from spectrast import orient


def _stripe_field(H, W, period, angle_deg=0.0):
    """Sinusoidal stripes of given period (bins) at a real-space orientation angle_deg."""
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.radians(angle_deg)
    proj = xx * np.sin(th) + yy * np.cos(th)          # vary ACROSS the stripe axis
    return np.sin(2 * np.pi * proj / period)


def test_windowed_polar_recovers_orientation():
    f = _stripe_field(64, 64, period=8, angle_deg=0.0)          # horizontal stripes -> vertical variation
    out = orient.windowed_polar(f, np.ones_like(f, bool))
    assert out["anisotropy"].max() > 0.5                        # strongly oriented
    # real-space elongation axis of horizontal stripes is ~0 deg (mod 180); allow wrap tolerance
    ang = out["structure_orientation"][np.argmax(out["anisotropy"])]
    assert min(ang % 180, 180 - (ang % 180)) < 20


def test_isotropic_noise_low_anisotropy_and_calibrated_null():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((40, 40))
    m = np.ones_like(f, bool)
    tap = orient._taper2d(40, 40)
    geom = orient._polar_geom(40, 40, 36, 4, 0.15, orient.INSCRIBED_KMAX)
    rgeom = orient._radial_geom(40, 40)
    a, _, _ = orient._polar_batch(f, m, tap, geom)
    null = orient._iso_null(f, m, tap, geom, rgeom, n_perm=200, seed_k=0)[0]     # (n_perm, n_rings)
    # per-ring permutation p should NOT be systematically significant for pure noise
    p = (1 + (null >= a[None] - 1e-12).sum(0)) / (200 + 1)
    assert (p > 0.05).mean() >= 0.5                             # calibrated: noise mostly non-significant


def _smooth_isotropic(H, W, sigma, rng):
    """A spatially AUTOCORRELATED but genuinely ISOTROPIC field.

    NOT scipy.ndimage.gaussian_filter, which this file used to call: that filter is SEPARABLE and
    truncated at 4 sigma, so its transfer function is G(kx)G(ky) plus axis-aligned truncation sidelobes
    that pass ~4.5e4 times more high-|k| power along 0/90 deg than along the diagonals. Harmless for the
    magnitude calibration these tests assert, fatal for anything about DIRECTION. Filtering in Fourier
    with a radially symmetric transfer function makes E|F(k)|^2 a function of |k| alone — isotropic by
    construction, which is exactly the null an orientation estimator must report as directionless."""
    ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.fftfreq(W)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    T = np.exp(-0.5 * (2 * np.pi * kr * sigma) ** 2)              # radially symmetric: no preferred axis
    return np.fft.ifft2(T * np.fft.fft2(rng.standard_normal((H, W)))).real


def test_iso_null_calibrated_on_autocorrelated_isotropic_field():
    # THE regression for the white-noise-null bug: on a SMOOTH (autocorrelated) but isotropic field, the
    # per-window anisotropy must not be systematically significant. A value-scramble null (which whitens the
    # field) gave ~56% false-positive here; the isotropic phase-randomized surrogate must restore ~alpha.
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    geom = orient._polar_geom(48, 48, 36, 4, 0.15, orient.INSCRIBED_KMAX)
    rgeom = orient._radial_geom(48, 48)
    tap = orient._taper2d(48, 48)
    m = np.ones((48, 48), bool)
    fp = []
    for _ in range(24):
        f = _smooth_isotropic(48, 48, 1.8, rng)                # autocorrelated (realistic scale), no orientation
        a, _, _ = orient._polar_batch(f, m, tap, geom)
        nullc = orient._iso_null(f, m, tap, geom, rgeom, n_perm=100, seed_k=int(rng.integers(1 << 30)))[0]
        p = norm.sf((a - nullc.mean(0)) / (nullc.std(0) + 1e-9))
        fp.append((p < 0.05).mean())
    # calibrated to ~alpha (was ~0.56 with the whitening value-scramble null). Calibration is tightest for
    # in-band autocorrelation and degrades only for extreme red fields whose power sits below the ring band.
    assert np.mean(fp) < 0.15


def test_screen_boundary_negative_control():
    # a straight tissue-mask edge must NOT manufacture a "coherently oriented" call for a SMOOTH ISOTROPIC
    # gene. (The hard aperture leaks anisotropy aligned to the edge in every straddling window; the surrogate
    # null, carrying the same aperture, must calibrate it out.)
    import anndata as ad
    rng = np.random.default_rng(2)
    xy = rng.random((5000, 2)) * 400.0
    keep = xy[:, 1] < xy[:, 0] + 40.0                          # one long straight diagonal tissue edge
    xy = xy[keep]
    ctr = rng.random((18, 2)) * 400.0                          # isotropic smooth gene = sum of radial gaussians
    val = np.zeros(len(xy))
    for c in ctr:
        val += np.exp(-((xy - c) ** 2).sum(1) / (2 * 35.0 ** 2))
    a = ad.AnnData(np.c_[val, rng.standard_normal(len(xy))].astype("float32"))
    a.obsm["spatial"] = xy.astype("float32"); a.var_names = ["smooth", "noise"]
    df = orient.screen_oriented_genes(a, genes=["smooth", "noise"], window_um=(120.0, 200.0), n_perm=100,
                                      min_cells=10, seed=0)
    frac = df.set_index("gene")["frac_oriented"]
    assert frac["smooth"] < 0.15        # the edge does not fake a coherently-oriented smooth gene


def test_axis_presence_small_n_not_anticonservative():
    # two i.i.d. no-structure samples must be called "present" at ~alpha regardless of panel size. The old
    # binomial-vs-fixed-quantile test gave ~18-21% false-present at n=12-40; the rank test must be ~<=alpha.
    rng = np.random.default_rng(3)
    fp = 0; trials = 200
    for _ in range(trials):
        o = rng.standard_normal(12); nl = rng.standard_normal(12)
        if orient.axis_presence(o, nl, min_n=1)["present"]:    # min_n=1 so the guard doesn't mask calibration
            fp += 1
    assert fp / trials < 0.12          # calibrated at small n (was ~0.18-0.21 with the binomial test)


def test_frequency_ring_tracks_feature_scale():
    # the novel axis: within a window, a FINER stripe (higher spatial frequency) concentrates POWER in a
    # HIGHER ring than a coarser stripe — orientation resolved by feature scale. (Raw per-ring anisotropy is
    # biased high at low-k rings with few Fourier samples — the null calibrates that; the power tells us where
    # the feature actually lives.)
    coarse = orient.windowed_polar(_stripe_field(64, 64, period=10), np.ones((64, 64), bool), n_rings=4)
    fine = orient.windowed_polar(_stripe_field(64, 64, period=4), np.ones((64, 64), bool), n_rings=4)
    assert np.argmax(fine["ring_power"]) > np.argmax(coarse["ring_power"])
    # both are strongly oriented at their power-carrying ring
    assert coarse["anisotropy"][np.argmax(coarse["ring_power"])] > 0.5
    assert fine["anisotropy"][np.argmax(fine["ring_power"])] > 0.5
    # ring feature-scale (physical wavelength) is monotone decreasing in ring index
    assert np.all(np.diff(2 * 1.0 / np.maximum(coarse["ring_kfrac"], 1e-6)) < 0)


def _laminar_adata(seed=0):
    """A tissue with a laminar sub-region (25µm stripes) in the left half and isotropic noise on the right."""
    rng = np.random.default_rng(seed)
    xy = rng.random((6000, 2)) * np.array([400.0, 400.0])
    left = xy[:, 0] < 200
    lam = np.where(left, np.sin(2 * np.pi * xy[:, 1] / 25.0), 0.0)   # horizontal laminae, left half only
    lam = lam + 0.05 * rng.standard_normal(xy.shape[0])
    noise = rng.standard_normal(xy.shape[0])
    a = ad.AnnData(np.c_[lam, noise].astype("float32"))
    a.obsm["spatial"] = xy.astype("float32"); a.var_names = ["lam", "noise"]
    return a, left


def test_scan_and_aggregate_and_segment():
    a, left = _laminar_adata()
    scan = orient.orientation_field(a, summary="pc1", genes=["lam"], window_um=(120.0, 200.0),
                                    n_perm=150, seed=0)
    assert set(scan) == {120.0, 200.0}
    # the calibrated gate recovers a coherent laminar sub-region, unsupervised
    seg = orient.segment_regions(scan[120.0], min_windows=2)
    assert seg["n_regions"] >= 1
    # and the passing windows sit in the laminar (left) half, not the isotropic right
    passing = seg["gate"]; cx = scan[120.0].centers_um[:, :, 0]
    assert (cx[passing] < 200).mean() > 0.7
    # scale aggregation writes per-cell laminarity; laminar (left) half scores higher on average
    agg = orient.aggregate_scales(scan, a, coherence_min=0.5)
    assert a.obs["orient_laminarity"].to_numpy().shape[0] == a.n_obs
    assert agg["laminarity"][left].mean() > agg["laminarity"][~left].mean()


def test_rank_region_genes_ranks_carrier_over_noise():
    a, left = _laminar_adata()
    df = orient.rank_region_genes(a, box=(0, 200, 0, 400), genes=["lam", "noise"], n_perm=150, min_cells=10)
    assert df.iloc[0]["gene"] == "lam"                          # carrier ranks first
    assert df.set_index("gene").loc["lam", "z"] > df.set_index("gene").loc["noise", "z"]


def test_screen_oriented_genes_finds_carrier():
    # the headline descriptive statistic: discover WHICH genes carry oriented structure, unsupervised.
    a, left = _laminar_adata()
    df = orient.screen_oriented_genes(a, genes=["lam", "noise"], window_um=(120.0, 200.0), n_perm=150,
                                      min_cells=10)
    assert df.iloc[0]["gene"] == "lam"                          # oriented carrier tops the ranking
    dd = df.set_index("gene")
    assert dd.loc["lam", "frac_oriented"] > dd.loc["noise", "frac_oriented"]
    assert dd.loc["lam", "n_sig_windows"] >= 1                  # found significant oriented windows
    assert np.isfinite(dd.loc["lam", "scale_window_um"])        # reports the scale it fired at
    assert np.isfinite(dd.loc["lam", "orientation_deg"])        # and the axis


def test_screen_batched_matches_per_gene():
    # the cross-gene-batched screen must be NUMERICALLY IDENTICAL to a per-gene scan (same statistic, batched).
    a, _ = _laminar_adata()
    win = (120.0, 200.0)
    df = orient.screen_oriented_genes(a, genes=["lam", "noise"], window_um=win, n_perm=80, min_cells=10, seed=0)
    for g in ["lam", "noise"]:
        scan = orient.gene_orientation_field(a, g, window_um=win, n_perm=80, seed=0)   # per-gene reference
        best = 0.0; nsig = 0; nval = 0
        for w, res in scan.items():
            passing = (res.fdr < 0.05) & (res.coherence >= 0.5)
            nsig += int((passing.any(2) & res.valid).sum()); nval += int(res.valid.sum())
            gated = np.where(passing & res.valid[..., None], res.laminarity, 0.0)
            best = max(best, float(gated.max()) if gated.size else 0.0)
        row = df.set_index("gene").loc[g]
        assert abs(row["frac_oriented"] - nsig / max(nval, 1)) < 1e-9
        assert abs(row["max_laminarity"] - best) < 1e-6


def test_orientation_streamlines_shape_and_gating():
    a, left = _laminar_adata()
    scan = orient.gene_orientation_field(a, "lam", window_um=(120.0,), n_perm=150, seed=0)
    res = scan[120.0]
    s = orient.orientation_streamlines(res)
    assert s["U"].shape == res.grid and s["V"].shape == res.grid
    assert s["X"].shape[0] == res.grid[1] and s["Y"].shape[0] == res.grid[0]
    # unit vectors where oriented, zero where gated off
    on = s["speed"] > 0
    assert np.allclose(np.hypot(s["U"][on], s["V"][on]), 1.0, atol=1e-6)
    assert np.allclose(s["U"][~on], 0.0) and np.allclose(s["V"][~on], 0.0)


def test_axis_presence_calibrated():
    rng = np.random.default_rng(0)
    obs = np.r_[rng.normal(3, 1, 40), rng.normal(0, 1, 160)]        # 40 structured genes
    null = rng.normal(0, 1, 200)
    r = orient.axis_presence(obs, null)
    assert r["present"] and r["n_sig"] > r["n_expected"] and r["enrichment"] > 2 and r["n_tested"] == 200
    assert not orient.axis_presence(rng.normal(0, 1, 200), rng.normal(0, 1, 200))["present"]   # noise -> absent
    lo = np.r_[rng.normal(-3, 1, 40), rng.normal(0, 1, 160)]        # structure = LOW statistic (texture centroid)
    assert orient.axis_presence(lo, rng.normal(0, 1, 200), higher=False)["present"]


def test_orientation_presence_runs():
    a, _ = _laminar_adata()
    r = orient.orientation_presence(a, genes=["lam", "noise"], window_um=(120.0, 200.0), n_perm=60, min_cells=10)
    assert {"presence_p", "enrichment", "score", "effect_size", "n_tested"} <= set(r)
    assert r["effect_size"] >= 0.0 and 0 <= r["score"] <= 1


def test_backend_setting_smoke():
    # module must run under whatever backend resolves (cpu here); just exercise the path.
    assert spectrast.set_backend("cpu") == "cpu"
    f = _stripe_field(32, 32, 8)
    out = orient.windowed_polar(f, np.ones_like(f, bool))
    assert np.isfinite(out["anisotropy"]).all()


# ---------------------------------------------------------------------------------------------
# The measurement band must stay inside the inscribed disc of the frequency grid.
#
# A window's FFT samples frequencies on a SQUARE: each axis reaches Nyquist (1/2 cycles per bin) while
# the corner reaches sqrt(2)/2. kmax_frac is a fraction of that CORNER, so above 1/sqrt(2) the outermost
# ring asks for frequencies that exist on the diagonals and do not exist on the axes. The wedges centred
# on 0 and 90 deg are truncated, keep only the low-frequency part of their ring where power is higher,
# and tip the reported orientation onto the pixel axes. Measured on isotropic controls, the former
# default 0.8 put the outer ring's angle within 11.25 deg of a lattice axis 34-40% of the time against
# the 25% expected.
# ---------------------------------------------------------------------------------------------
def _c4(theta_deg):
    """Four-fold order parameter: +1 all on the lattice axes, -1 all on the diagonals, 0 directionless."""
    return float(np.cos(4 * np.radians(np.asarray(theta_deg, float))).mean())


def test_polar_geom_clamps_beyond_the_inscribed_disc():
    with pytest.warns(RuntimeWarning, match="inscribed disc"):
        g = orient._polar_geom(40, 40, 36, 4, 0.15, 0.9)
    assert g["ring_kfrac"].max() <= orient.INSCRIBED_KMAX + 1e-9, "the band was not clamped"


def test_rectangular_window_clamps_below_one_over_sqrt2():
    """1/sqrt(2) is exact only for a SQUARE window; a rectangular one inscribes less of its own corner,
    and windowed_polar / rank_region_genes accept an arbitrary box."""
    g = orient._polar_geom(41, 40, 36, 4, 0.15, orient.INSCRIBED_KMAX)   # default, must not warn
    assert g["ring_kfrac"].max() < orient.INSCRIBED_KMAX, "rectangular window was not clamped tighter"


def test_wedges_are_complete_at_the_default_cut():
    """The mechanism itself: past the inscribed disc the outer ring's cardinal wedges are starved, and
    that is what tips the resultant. Two things this pins down.

    First, clamping makes an over-wide request IDENTICAL to the default rather than merely closer to it.
    Second, the ratio at the default is 0.60 rather than 1.0, and that residual is discretization, not
    truncation: for EVEN N, np.fft.fftfreq yields -1/2 but not +1/2, so the outermost cardinal pixels
    exist on one side of the grid only. It costs a little precision in those wedges and, per
    test_orientation_is_directionless_on_an_isotropic_field, no measurable direction."""
    def card_diag(kmax):
        g = orient._polar_geom(40, 40, 36, 4, 0.15, kmax)
        c = g["counts"].reshape(4, 36)[-1]
        return c[np.r_[0, 35, 17, 18]].mean() / c[np.r_[8, 9, 26, 27]].mean()
    at_default = card_diag(orient.INSCRIBED_KMAX)
    with pytest.warns(RuntimeWarning):
        assert card_diag(0.9) == at_default, "an over-wide request must clamp to the default geometry"
    # the pre-fix geometry, built directly, starved the cardinal wedges more than twice as hard
    ky = np.fft.fftshift(np.fft.fftfreq(40)); kx = ky
    KX, KY = np.meshgrid(kx, ky); kf = np.hypot(KX, KY) / np.hypot(KX, KY).max()
    ang = np.degrees(np.arctan2(KY, KX)) % 180.0
    band = (kf >= 0.6375) & (kf <= 0.8)                 # the old outer ring
    card = band & (np.minimum(ang % 90, 90 - ang % 90) < 2.5)
    diag = band & (np.abs(ang - 45) < 2.5)
    old_ratio = card.sum() / diag.sum()
    assert old_ratio < 0.35, "the old outer ring's cardinal wedges were not truncated"
    assert at_default > 1.7 * old_ratio, "the fix did not materially even out the wedges"


def test_orientation_is_directionless_on_an_isotropic_field():
    """THE regression for the grid artifact. On fields whose expected power depends only on |k|, the
    reported orientation must have no preferred direction — and did not, at the old cut."""
    rng = np.random.default_rng(4)
    tap = orient._taper2d(40, 40)
    m = np.ones((40, 40), bool)
    F = np.stack([_smooth_isotropic(40, 40, 0.6, rng) for _ in range(300)])
    def outer_c4(kmax):
        g = orient._polar_geom(40, 40, 36, 4, 0.15, kmax)
        _, struct, _ = orient._polar_batch(F, m, tap, g)
        return _c4(struct[:, -1])
    assert abs(outer_c4(orient.INSCRIBED_KMAX)) < 0.12, "the default cut still prefers a direction"


def test_no_public_default_exceeds_the_inscribed_disc():
    """Guards against the fix being undone by a new entry point or a copied signature."""
    import inspect
    bad = {}
    for name in ("windowed_polar", "orientation_field", "screen_oriented_genes", "rank_region_genes"):
        d = inspect.signature(getattr(orient, name)).parameters["kmax_frac"].default
        if d > orient.INSCRIBED_KMAX + 1e-9:
            bad[name] = d
    assert not bad, f"kmax_frac defaults exceed the inscribed disc: {bad}"


# ---------------------------------------------------------------------------
# Collapsed polar reduction + device residency (perf work, must not move numbers)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("win,nt,nr", [(40, 36, 4), (16, 18, 3), (13, 72, 6), (7, 36, 4)])
def test_collapsed_reduction_equals_the_legacy_wedge_path(win, nt, nr):
    """``geom["Wc"]`` folds the (H*W, n_rings*n_theta) one-hot binning, the 1/counts division and the
    cos2/sin2 wedge weights into ONE (H*W, 3*n_rings) operator. That is exact algebra, not an
    approximation, so the two paths must agree to round-off at every geometry — including the small
    windows where most wedges are empty and the collapse could plausibly differ."""
    rng = np.random.default_rng(win)
    g = orient._polar_geom(win, win, nt, nr, 0.15, orient.INSCRIBED_KMAX)
    tap = orient._taper2d(win, win)
    F = rng.normal(size=(200, win, win))
    M = rng.random((200, win, win)) > 0.3
    aL, oL, pL = orient._polar_batch(F, M, tap, g, impl="legacy")
    aF, oF, pF = orient._polar_batch(F, M, tap, g)
    assert np.abs(aF - aL).max() < 1e-10
    assert np.abs(oF - oL).max() < 1e-8                      # degrees
    assert np.abs(pF - pL).max() <= 1e-10 * max(np.abs(pL).max(), 1e-30)


def test_collapsed_operator_does_not_depend_on_n_theta_in_width():
    """The whole point of the collapse: the reduction's cost stops scaling with n_theta. Guard the
    property that would silently regress if someone reinstated the per-wedge intermediate."""
    for nt in (18, 36, 72):
        g = orient._polar_geom(24, 24, nt, 4, 0.15, orient.INSCRIBED_KMAX)
        assert g["Wc"].shape == (24 * 24, 3 * 4)
        assert g["S"].shape[1] == nt * 4                     # legacy path still reachable


def test_device_backend_agrees_with_host_within_float32_tolerance():
    """MPS/CUDA is float32-only. Anisotropy is a RATIO whose denominator is a sum of strictly positive
    powers, so unlike the Moran path (~1e-3, cancellation-amplified) it holds float32 well — validated
    on real expression at 5.2e-07 relative-L2. Tolerance here is deliberately far tighter than the
    Moran path's would be; loosening it means the conditioning argument stopped holding."""
    from spectrast import _fft
    prev = _fft.get_backend()
    try:
        if _fft.set_backend("mps") not in ("mps", "cuda"):
            pytest.skip("no GPU backend available")
        rng = np.random.default_rng(0)
        g = orient._polar_geom(40, 40, 36, 4, 0.15, orient.INSCRIBED_KMAX)
        tap = orient._taper2d(40, 40)
        F = rng.normal(size=(500, 40, 40)); M = rng.random((500, 40, 40)) > 0.3
        aD, oD, _ = orient._polar_batch(F, M, tap, g)
        _fft.set_backend("cpu")
        aH, oH, _ = orient._polar_batch(F, M, tap, g)
        assert np.abs(aD - aH).max() < 1e-4
        i = np.arange(len(aH)); k = aH.argmax(-1)
        assert np.abs(((oD[i, k] - oH[i, k]) + 90) % 180 - 90).max() < 0.05      # degrees
    finally:
        _fft.set_backend(prev)


def test_mask_repeat_matches_an_explicit_repeat_across_chunk_boundaries():
    """``mask_repeat`` broadcasts one mask row over the genes sharing a tile instead of materializing a
    repeated copy the size of the field stack. The device path chunks the batch, so the chunk size is
    snapped to a multiple of the repeat — get that arithmetic wrong and masks silently misalign with
    fields part-way through a batch, which would look like a plausible result rather than a crash."""
    from spectrast import _fft
    from spectrast.orient import _polar_reduce_device
    rng = np.random.default_rng(2)
    win, ntile, rep = 16, 53, 7                          # 371 windows, deliberately not chunk-aligned
    g = orient._polar_geom(win, win, 18, 3, 0.15, orient.INSCRIBED_KMAX)
    tap = orient._taper2d(win, win)
    F = rng.normal(size=(ntile * rep, win, win))
    Mt = rng.random((ntile, win, win)) > 0.3
    exp = orient._polar_batch(F, np.repeat(Mt, rep, axis=0), tap, g, impl="legacy")
    prev = _fft.get_backend()
    try:
        if _fft.set_backend("mps") not in ("mps", "cuda"):
            pytest.skip("no GPU backend available")
        for chunk in (100, 64, 371, 5000):               # force ragged + exact + oversized boundaries
            a, b, tot0 = _polar_reduce_device(F, Mt[None] if False else Mt, tap, g,
                                              chunk=chunk, mask_repeat=rep)
            ani = np.sqrt(a ** 2 + b ** 2) / (tot0 + 1e-12)
            assert np.abs(ani - exp[0]).max() < 1e-4, f"chunk={chunk}"
    finally:
        _fft.set_backend(prev)


# ---------------------------------------------------------------------------
# Direct polar sampling (step 2b) — removes the lattice's number-theoretic angle bias
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [16, 24, 40])
@pytest.mark.parametrize("true_deg", [0.0, 30.0, 75.0, 120.0])
def test_direct_sampling_recovers_a_planted_angle_and_agrees_with_the_lattice(B, true_deg):
    """Direct sampling must measure the same physical thing as the lattice path, not merely something
    well-behaved. Stripes built as sin(2*pi*(x sin th + y cos th)/period) have their wavevector at
    90-th, so the STRUCTURE runs at (-th) mod 180 — pinning the convention as well as the accuracy."""
    yy, xx = np.mgrid[0:B, 0:B]
    th = np.radians(true_deg)
    f = np.sin(2 * np.pi * (xx * np.sin(th) + yy * np.cos(th)) / (B / 4.0))[None]
    M = np.ones((1, B, B), bool)
    tap = orient._taper2d(B, B)
    gd = orient._polar_geom_direct(B, B, 36, 4, 0.15, orient.INSCRIBED_KMAX)
    gl = orient._polar_geom(B, B, 36, 4, 0.15, orient.INSCRIBED_KMAX)
    ad, od, _ = orient._polar_batch(f, M, tap, gd)
    al, ol, _ = orient._polar_batch(f, M, tap, gl)
    got, gotl = od[0, ad[0].argmax()], ol[0, al[0].argmax()]
    exp = (-true_deg) % 180.0
    assert abs((got - exp + 90) % 180 - 90) < 6.0
    assert abs((got - gotl + 90) % 180 - 90) < 6.0


@pytest.mark.parametrize("B", [16, 24])
def test_direct_sampling_is_directionless_on_noise_where_the_lattice_is_not(B):
    """The regression this whole change exists to prevent. On a structureless field the lattice path
    piles orientations onto the diagonals — C4 z about -5 at B=16 and -4 at B=24 — because which wedges
    a B-point lattice can populate is a property of B, not the tissue. Five of the eleven datasets can
    only afford B=16-24, so without direct sampling they would be measuring the lattice. The bias does
    NOT decay smoothly with B (it is worse at 19 than at 14 or 25), so this is asserted at the specific
    B values the feasibility table chose, not at a 'safe' one."""
    rng = np.random.default_rng(5)
    N = 4000
    F = rng.normal(size=(N, B, B)); M = np.ones((N, B, B), bool)
    tap = orient._taper2d(B, B)
    g = orient._polar_geom_direct(B, B, 36, 4, 0.15, orient.INSCRIBED_KMAX)
    a, o, _ = orient._polar_batch(F, M, tap, g)
    t = np.radians(o[np.arange(N), a.argmax(-1)])
    z = np.cos(4 * t).mean() / (1 / np.sqrt(2 * N))          # four-fold order parameter, s.e. 1/sqrt(2N)
    assert abs(z) < 3.0, f"direct sampling shows lattice alignment at B={B}: C4 z={z:.1f}"


def test_direct_sampling_holds_float32_on_a_gpu_backend():
    from spectrast import _fft
    prev = _fft.get_backend()
    try:
        if _fft.set_backend("mps") not in ("mps", "cuda"):
            pytest.skip("no GPU backend available")
        rng = np.random.default_rng(0)
        g = orient._polar_geom_direct(40, 40, 36, 4, 0.15, orient.INSCRIBED_KMAX)
        tap = orient._taper2d(40, 40)
        F = rng.normal(size=(800, 40, 40)); M = rng.random((800, 40, 40)) > 0.3
        a32, _o32, _ = orient._polar_batch(F, M, tap, g)
        _fft.set_backend("cpu")
        a64, _o64, _ = orient._polar_batch(F, M, tap, g)
        assert np.linalg.norm(a32 - a64) / np.linalg.norm(a64) < 1e-5
    finally:
        _fft.set_backend(prev)


# ---------------------------------------------------------------------------
# The SHIPPED DEFAULTS. Until 2026-08-12 nothing pinned these: every test built its geometry by
# calling _polar_geom/_polar_geom_direct explicitly, so the library could have been reverted to the
# sliding-B lattice path and the whole suite would still have passed. These lock the two decisions
# that changed what the library computes.
# ---------------------------------------------------------------------------
def test_default_sampling_is_direct_not_lattice():
    """Direct sampling is what makes a small window usable: the lattice reads a four-fold anisotropy
    of z = -4.8 on white noise at B=16, where direct reads +0.4. A silent revert here would reinstate
    a grid-fixed direction in every result the library produces."""
    assert orient._geom_for(16, 36, 4, 0.15, orient.INSCRIBED_KMAX)["sampling"] == "direct"
    assert "sampling" not in orient._geom_for(16, 36, 4, 0.15, orient.INSCRIBED_KMAX, "lattice")
    with pytest.raises(ValueError, match="direct.*lattice"):
        orient._geom_for(16, 36, 4, 0.15, orient.INSCRIBED_KMAX, "fft")


def test_default_bin_rule_holds_the_window_at_a_constant_number_of_bins():
    """The window must be the SAME number of bins at every point of the sweep. Under the old floored
    rule the pitch floor binds at the fine end and the window shrinks in bins instead — 40 down to 6
    across one sweep — so the measurement geometry changes along the very axis being swept, and the
    null floor rides 3.2x with it."""
    a, _ = _laminar_adata()
    wins = (120.0, 200.0, 340.0)
    scan = orient.orientation_field(a, summary="pc1", genes=["lam"], window_um=wins, n_perm=10)
    bins = [round(w / scan[w].bin_size) for w in wins]
    assert bins == [orient.DEFAULT_BINS_PER_WINDOW] * 3, f"bins per window drifted: {bins}"


def test_floored_bin_rule_still_reaches_the_pre_2026_08_12_behaviour():
    """The old rule stays reachable — it is what every result computed before 2026-08-12 used, so
    reproducing one needs it. Asserted by its DEFECT: the bins per window must vary across the sweep."""
    a, _ = _laminar_adata()
    wins = (120.0, 200.0, 340.0)
    scan = orient.orientation_field(a, summary="pc1", genes=["lam"], window_um=wins, n_perm=10,
                                    bins_per_window=40, bin_rule="floored", min_bin=12.0)
    bins = [round(w / scan[w].bin_size) for w in wins]
    assert len(set(bins)) > 1, f"floored rule should let the window slide, got {bins}"


def test_default_windows_low_end_is_set_by_constant_bin_feasibility():
    """The sweep's fine end is where the raster pitch would fall below the cell spacing, not a fixed
    fraction of the tissue. Range is what carries scale (7.28x sweep -> peak-recovery slope 0.67, the
    same sweep cut to 3.07x -> 0.35), so it must adapt to what the section can actually support."""
    a, _ = _laminar_adata()
    w = orient.default_windows(a, bins_per_window=16)
    assert len(w) == 7
    pitch = orient._cell_spacing(np.asarray(a.obsm["spatial"], float))
    assert w[0] == pytest.approx(16 * pitch, rel=1e-9)
    # and the legacy extent-fraction form stays available
    wl = orient.default_windows(a, lo_frac=1 / 18, hi_frac=1 / 4, n=5)
    assert len(wl) == 5 and wl[-1] / wl[0] == pytest.approx(4.5, rel=1e-6)


def test_default_windows_warns_when_the_sweep_is_too_short_to_place_a_scale():
    """A sweep below ~4.7x holds any given wavelength in-band at EVERY window, so the curve's peak
    location is noise. Measured across eleven sections, realised range predicts how reproducibly the
    curve's shape survives a split-half (Spearman +0.83) — this is the load-bearing warning."""
    rng = np.random.default_rng(0)
    xy = rng.random((2000, 2)) * 300.0                        # coarse spacing vs extent -> short sweep
    a = ad.AnnData(rng.standard_normal((2000, 2)).astype("float32"))
    a.obsm["spatial"] = xy.astype("float32"); a.var_names = ["g0", "g1"]
    with pytest.warns(RuntimeWarning, match="4.7x|below the"):
        orient.default_windows(a, bins_per_window=16)

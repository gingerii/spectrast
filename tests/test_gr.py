"""gr-layer regression tests (code-review scope 5): reproducibility + non-mutation + key-collision contracts."""
import warnings

import numpy as np
import anndata as ad
import pytest

import spectrast
from spectrast import gr
from spectrast import spatial


def _adata(n=400, g=12, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(np.abs(rng.standard_normal((n, g))).astype("float32"))
    a.obsm["spatial"] = (rng.random((n, 2)) * 200).astype("float32")
    return a


def test_default_radius_order_invariant():
    """m8: a row-reordered copy of the same tissue must give the SAME default radius (was an RNG choice over
    positional indices, so a sort/reorder of identical data silently changed the default radius)."""
    a = _adata(n=8000)                                       # > sample=5000 -> exercises the subsample path
    r0 = gr.default_radius(a)
    perm = np.random.default_rng(1).permutation(a.n_obs)
    r1 = gr.default_radius(a[perm].copy())
    assert np.isclose(r0, r1, rtol=1e-12), f"default_radius order-dependent: {r0} vs {r1}"


def test_autotune_rank_copy_does_not_mutate():
    """M12: autotune_rank(copy=True) must not touch the caller's adata.uns (it was the only gr function that
    wrote uns with no copy= escape)."""
    a = _adata()
    keys0 = set(a.uns.keys())
    out = gr.autotune_rank(a, copy=True)
    assert isinstance(out, ad.AnnData) and "spectrast_autotune_rank" in out.uns
    assert set(a.uns.keys()) == keys0, "autotune_rank(copy=True) mutated the original uns"


def test_scale_spectrum_key_added_no_collision():
    """m6: two scale_spectrum calls with different key_added must keep BOTH scalograms (the scalogram key was
    hardcoded, so the second silently clobbered the first)."""
    a = _adata()
    gr.scale_spectrum(a, key_added="scaleA")
    gr.scale_spectrum(a, key_added="scaleB")
    assert "scaleA_scalogram" in a.uns and "scaleB_scalogram" in a.uns
    assert "scaleA" in a.var and "scaleB" in a.var
    gr.scale_spectrum(a)                                     # default key_added keeps the documented location
    assert "spectrast_scalogram" in a.uns


def test_gr_moran_reproducible_and_copy_nonmutating():
    """Reproducibility: identical input -> bit-identical output; copy=True leaves the original var untouched."""
    a = _adata()
    v1 = gr.spatial_autocorr(a.copy(), mode="moran")
    v2 = gr.spatial_autocorr(a.copy(), mode="moran")
    assert np.array_equal(v1, v2)
    ncols = a.var.shape[1]
    _ = gr.spatial_autocorr(a, mode="moran", copy=True)
    assert a.var.shape[1] == ncols, "copy=True mutated the original var"


# --------------------------------------------------------------------------- texture + fingerprint presence
def _structured_adata(n=3000, n_struct=40, n_noise=20, seed=0):
    """Structured genes = smooth Gaussian bumps (autocorrelated, focal, low-texture); rest = noise."""
    rng = np.random.default_rng(seed)
    xy = (rng.random((n, 2)) * 300).astype("float32")
    cols = []
    for _ in range(n_struct):
        c = rng.random(2) * 300; sig = rng.uniform(30, 60)
        cols.append(np.exp(-((xy - c) ** 2).sum(1) / (2 * sig ** 2)) + rng.normal(0, 0.05, n))
    for _ in range(n_noise):
        cols.append(rng.normal(0, 1, n))
    X = np.column_stack(cols).astype("float32"); X -= X.min(0)
    a = ad.AnnData(X); a.obsm["spatial"] = xy
    a.var_names = [f"s{i}" for i in range(n_struct)] + [f"n{i}" for i in range(n_noise)]
    return a


def test_texture_writer_orders_smooth_below_fine():
    """gr.texture: the spectral-centroid statistic must put a smooth (low-freq) gene BELOW a fine
    (high-freq) one, and write one finite value per gene. Uses a dense lattice so the binned grid
    faithfully samples the field (sparse random points swamp the centroid with shot noise)."""
    g = 120; lin = np.linspace(0, 300, g); xx, yy = np.meshgrid(lin, lin)
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype("float32")
    smooth = np.sin(2 * np.pi * xx.ravel() / 250.0)          # ~1 cycle across tissue
    fine = np.sin(2 * np.pi * xx.ravel() / 12.0)             # ~25 cycles
    X = np.column_stack([smooth, fine]).astype("float32"); X -= X.min(0)
    a = ad.AnnData(X); a.obsm["spatial"] = xy; a.var_names = ["smooth", "fine"]
    val = gr.texture(a, bin_um=300 / g)
    assert np.isfinite(val).all() and a.var["spectrast_texture"].shape == (2,)
    assert a.var["spectrast_texture"]["smooth"] < a.var["spectrast_texture"]["fine"]


def test_fingerprint_presence_calibrated():
    """gr.fingerprint_presence: a structured tissue reads PRESENT on scale and focality; a coordinate
    scramble of it (same panel/density, structure destroyed) reads ABSENT, and the panel-independent
    scale effect_size collapses — the matched-power calibration the shipped presence test must hold."""
    a = _structured_adata()
    AX = ("scale_peak_moran", "focality", "texture")
    obs = gr.fingerprint_presence(a, axes=AX, n_perm=1).set_index("axis")
    assert bool(obs.loc["scale_peak_moran", "present"]) and bool(obs.loc["focality", "present"])
    rng = np.random.default_rng(1)
    scr = a.copy(); scr.obsm["spatial"] = a.obsm["spatial"][rng.permutation(a.n_obs)]
    nul = gr.fingerprint_presence(scr, axes=AX, n_perm=1).set_index("axis")
    assert not bool(nul.loc["scale_peak_moran", "present"]) and not bool(nul.loc["focality", "present"])
    # matched-power drop
    assert nul.loc["scale_peak_moran", "effect_size"] < obs.loc["scale_peak_moran", "effect_size"]


def test_fingerprint_presence_orientation_combined():
    """orientation=True appends a fourth axis row from orient.orientation_presence, so one call returns
    all four fingerprint axes with a common schema."""
    a = _structured_adata(n=1500, n_struct=20, n_noise=10)
    df = gr.fingerprint_presence(a, axes=("scale_peak_moran",), orientation=True, orient_n_perm=20,
                                 window_um=(120.0, 200.0), min_cells=10)
    assert list(df["axis"]) == ["scale_peak_moran", "orientation"]
    assert {"presence_p", "effect_size", "n_tested", "present"} <= set(df.columns)


def test_presence_scale_alias_retired_but_still_works():
    """The bare axis name 'scale' was ambiguous — it meant the dimensionless peak-Moran statistic here
    while gr.scale_spectrum used it for the um centroid. It must still resolve (so old scripts run) but
    warn, and report under the unambiguous name."""
    a = _structured_adata(n=1200, n_struct=10, n_noise=6)
    with pytest.warns(DeprecationWarning, match="ambiguous"):
        df = gr.fingerprint_presence(a, axes=("scale",), n_perm=1)
    assert list(df["axis"]) == ["scale_peak_moran"]
    with pytest.raises(ValueError, match="unknown axes"):
        gr.fingerprint_presence(a, axes=("char_scale",), n_perm=1)


def test_texture_batched_matches_legacy():
    """R2 regression: the batched real-input rfft2 through the _fft backend must reproduce the original
    per-gene complex np.fft.fft2 loop it replaced. Conjugate symmetry makes the half-grid sum with
    Hermitian multiplicity weights EXACTLY the full-grid sum, so the only difference allowed is
    float64 round-off between pocketfft's real and complex transforms."""
    a = _structured_adata(n=6000, n_struct=14, n_noise=8, seed=3)
    xy = np.asarray(a.obsm["spatial"], float)
    X = np.asarray(a.X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for bin_um in (300 / 128.0, 300 / 77.0):                    # even and odd grid widths
        old = gr._isotropic_fingerprint(xy, Z, bin_um, axes=("texture",), texture_impl="legacy")["texture"]
        new = gr._isotropic_fingerprint(xy, Z, bin_um, axes=("texture",), texture_impl="batched")["texture"]
        assert np.allclose(new, old, rtol=1e-10, atol=0)
        assert np.array_equal(np.argsort(old), np.argsort(new))   # the presence test ranks these


def test_fingerprint_default_is_two_axes_and_texture_stays_available():
    """The default fingerprint vector is (scale_um, focality). Texture is opt-in, not deleted.

    Texture adds -0.000 (CI -0.020 to +0.020) to co-expression programme recovery over these two,
    so it is not a default coordinate. It must still
    COMPUTE — the presence test reports it and `gr.texture` is public — so this pins both halves at
    once: the default narrowed, the axis did not disappear.
    """
    a = _structured_adata(n=1200, n_struct=8, n_noise=6, seed=3)
    assert gr._FINGERPRINT_DEFAULT == ("scale_um", "focality")
    assert list(gr.fingerprint(a.copy()).columns) == ["scale_um", "focality"]
    opt = gr.fingerprint(a.copy(), axes=("scale_um", "focality", "texture"))
    # scale_um is legitimately NaN for the unstructured genes (no half-max crossing), so check the
    # axis that was made opt-in rather than the whole frame
    assert list(opt.columns) == ["scale_um", "focality", "texture"]
    assert np.isfinite(opt[["focality", "texture"]].to_numpy()).all()
    # still computable and still a PRESENCE axis -- the two roles were separated deliberately
    assert "texture" in gr._ISO_AXES and "texture" in gr._PRESENCE_AXES
    assert np.isfinite(gr.texture(a.copy())).all()


def test_fingerprint_scale_um_matches_scale_spectrum():
    """R1: gr.fingerprint()'s 'scale_um' IS gr.scale_spectrum's characteristic scale — same quantity,
    same code path, same number. This equality is the whole point of the rename: a user must never get
    two different 'scale' values from the public API."""
    a = _structured_adata(n=2000, n_struct=10, n_noise=6, seed=5)
    ref = gr.scale_spectrum(a.copy())
    fp = gr.fingerprint(a, axes=("scale_um", "focality", "texture"))
    # equal_nan: the half-max reduction returns NaN for a curve that never crosses its own mid-range,
    # which the centroid never did. The equality being asserted is still exact — NaN must line up too.
    assert np.array_equal(fp["scale_um"].to_numpy(), np.asarray(ref), equal_nan=True)
    assert np.array_equal(a.var["spectrast_char_scale"].to_numpy(), np.asarray(ref), equal_nan=True)
    assert list(fp.index) == list(a.var_names)
    # the grid axes agree with their standalone writers, and texture is the same code
    assert np.allclose(fp["texture"].to_numpy(), gr.texture(a.copy()), rtol=1e-12)
    # scale_um (um) and scale_peak_moran (dimensionless) are different quantities, not a rescale
    fp2 = gr.fingerprint(a, axes=("scale_um", "scale_peak_moran"))
    assert fp2["scale_um"].max() > 1.0 and fp2["scale_peak_moran"].max() <= 1.0
    assert a.uns["spectrast_fingerprint_params"]["backend"] == spectrast.get_backend()


def test_fingerprint_signature_vector_only():
    """R3: fingerprint_signature takes a PRE-SCORED field (obs column, obsm matrix, or array) and
    fingerprints it on the same axes as a gene. SpectraST does not score gene sets."""
    a = _structured_adata(n=2000, n_struct=10, n_noise=6, seed=7)
    a.obs["module"] = np.asarray(a.X)[:, :3].mean(1)
    fp = gr.fingerprint_signature(a, "module")
    assert list(fp.index) == ["module"] and list(fp.columns) == ["scale_um", "focality"]
    assert np.isfinite(fp.to_numpy()).all()
    # a bare array and an explicit name work identically
    fp2 = gr.fingerprint_signature(a, np.asarray(a.obs["module"]), names=["module"])
    assert np.allclose(fp2.to_numpy(), fp.to_numpy())
    # an obsm matrix scores every column (the decoupler shape)
    a.obsm["acts"] = np.column_stack([a.obs["module"], np.asarray(a.X)[:, 3]])
    assert len(gr.fingerprint_signature(a, "acts")) == 2
    with pytest.raises(KeyError, match="score the signature first|neither"):
        gr.fingerprint_signature(a, "not_a_column")
    a.obs["holey"] = np.where(np.arange(a.n_obs) < 5, np.nan, 1.0)
    with pytest.raises(ValueError, match="NaN"):
        gr.fingerprint_signature(a, "holey")


def _sectioned_adata(n_per=900, n_sec=3, seed=0):
    """Sections with genuinely different spatial structure AND different cell spacing, so a
    per-section grid and a shared grid give measurably different answers."""
    import pandas as pd
    parts = []
    for s in range(n_sec):
        rng = np.random.default_rng(seed + s)
        extent = 200.0 * (1 + 0.6 * s)                       # sections differ in size -> different defaults
        xy = (rng.random((n_per, 2)) * extent).astype("float32")
        cols = []
        for j in range(6):
            c = rng.random(2) * extent; sig = extent / (6.0 + 2 * j)
            cols.append(np.exp(-((xy - c) ** 2).sum(1) / (2 * sig ** 2)) + rng.normal(0, 0.05, n_per))
        X = np.column_stack(cols).astype("float32"); X -= X.min(0)
        a = ad.AnnData(X); a.obsm["spatial"] = xy
        a.var_names = [f"g{j}" for j in range(6)]
        a.obs["section"] = f"s{s}"
        a.obs["donor"] = "d0" if s < 2 else "d1"
        a.obs["compartment"] = pd.Categorical(
            rng.choice(["stroma", "epithelium"], n_per, p=[0.6, 0.4]))
        parts.append(a)
    out = ad.concat(parts, index_unique="-")
    return out


def test_fingerprint_by_shared_grid_makes_sections_comparable():
    """T2 #3: fingerprint_by returns a tidy (feature x section) table. The default shared grid is the
    point of it — per-section radii/bin adapt to each section's own spacing, which would give every
    section a different measuring stick and make between-section differences partly a grid artefact."""
    a = _sectioned_adata()
    tidy = gr.fingerprint_by(a, "section")
    assert list(tidy.columns) == ["gene", "section", "n_obs", "scale_um", "focality"]
    assert len(tidy) == 3 * a.n_vars and sorted(tidy.section.unique()) == ["s0", "s1", "s2"]
    assert np.isfinite(tidy[["scale_um", "focality"]].to_numpy()).all()
    assert tidy.attrs["shared_grid"] is True and tidy.attrs["bin_um"] is not None

    # shared vs per-section really do differ here (else the default would be a no-op claim)
    per = gr.fingerprint_by(a, "section", shared_grid=False)
    assert not np.allclose(tidy.scale_um.to_numpy(), per.scale_um.to_numpy())
    assert per.attrs["shared_grid"] is False

    # one section fingerprinted alone must match its row in the shared-grid table exactly
    solo = gr.fingerprint(a[a.obs.section == "s1"].copy(),
                          radii=tidy.attrs["radii"], bin_um=tidy.attrs["bin_um"])
    row = tidy[tidy.section == "s1"].set_index("gene").loc[solo.index]
    assert np.allclose(row.scale_um.to_numpy(), solo.scale_um.to_numpy())


def test_fingerprint_by_skips_and_reports_tiny_sections():
    """A section too small to mean anything must be skipped and REPORTED, not silently fingerprinted."""
    a = _sectioned_adata(n_per=300, n_sec=2)
    sec = a.obs["section"].to_numpy().copy()
    sec[300:595] = "s0"                                  # leaves s1 with a handful of cells
    a.obs["section"] = sec
    with pytest.warns(UserWarning, match="skipped"):
        tidy = gr.fingerprint_by(a, "section", min_cells=100)
    assert "s1" in tidy.attrs["skipped"] and "s1" not in set(tidy.section)
    with pytest.raises(ValueError, match="min_cells"):
        gr.fingerprint_by(a, "section", min_cells=10_000)


def test_within_restricts_to_the_compartments_own_lattice():
    """T2 #7: within= computes the fingerprint from one compartment's cells only. The other cells are
    ABSENT, not zero — so the answer differs from the whole-tissue one, which is the intent."""
    a = _sectioned_adata(n_sec=1)
    whole = gr.fingerprint(a.copy())
    stroma = gr.fingerprint(a.copy(), within=("compartment", "stroma"))
    assert list(stroma.index) == list(whole.index)
    assert not np.allclose(stroma.scale_um.to_numpy(), whole.scale_um.to_numpy())

    # equivalent spellings agree, and match hand-subsetting
    mask = (a.obs.compartment.astype(str) == "stroma").to_numpy()
    d = gr.fingerprint(a.copy(), within={"compartment": "stroma"})
    m = gr.fingerprint(a.copy(), within=mask)
    hand = gr.fingerprint(a[mask].copy())
    for other in (d, m, hand):
        assert np.allclose(stroma.to_numpy(), other.to_numpy())

    a.uns.pop("spectrast_fingerprint_params", None)
    gr.fingerprint(a, within=("compartment", "stroma"))
    assert a.uns["spectrast_fingerprint_params"]["n_obs"] == int(mask.sum())   # provenance records it

    with pytest.raises(ValueError, match="selected 0 cells"):
        gr.fingerprint(a.copy(), within=("compartment", "no_such_type"))
    with pytest.raises(KeyError):
        gr.fingerprint(a.copy(), within=("no_such_key", "x"))


def test_aggregate_fingerprint_pools_sections_within_a_unit():
    """T2 #8: pool section fingerprints to one row per donor. Each section keeps its own spatial
    field; only the resulting numbers are averaged — never the lattices."""
    a = _sectioned_adata()
    tidy = gr.fingerprint_by(a, "section")
    agg = gr.aggregate_fingerprint(tidy, "section", unit={"s0": "d0", "s1": "d0", "s2": "d1"})
    assert set(agg.unit) == {"d0", "d1"} and len(agg) == 2 * a.n_vars
    d0 = agg[agg.unit == "d0"].set_index("gene")
    ref = tidy[tidy.section.isin(["s0", "s1"])].groupby("gene").scale_um.mean()
    assert np.allclose(d0.scale_um.to_numpy(), ref.loc[d0.index].to_numpy())
    assert (d0.n_sections == 2).all()
    assert d0.n_obs.iloc[0] == tidy[tidy.section.isin(["s0", "s1"])].groupby("gene").n_obs.sum().iloc[0]

    # a unit column already on the table works too, and median is available
    tidy2 = tidy.merge(a.obs[["section", "donor"]].drop_duplicates(), on="section")
    med = gr.aggregate_fingerprint(tidy2, "section", unit="donor", reducer="median")
    assert set(med.unit) == {"d0", "d1"}
    assert len(gr.aggregate_fingerprint(tidy2, "section", unit="donor", min_sections=2)) == a.n_vars
    with pytest.raises(KeyError, match="no unit given"):
        gr.aggregate_fingerprint(tidy, "section", unit={"s0": "d0"})


def test_focality_batched_matches_legacy():
    """Batching focality's convolution over genes must reproduce the per-gene fftconvolve loop it
    replaced. Both run on the float64 HOST path (the single-radius neighbour sum is paired with a
    float64 degree field, so it must not move to a float32 backend), so the only allowed difference
    is FFT round-off."""
    a = _structured_adata(n=6000, n_struct=14, n_noise=8, seed=4)
    xy = np.asarray(a.obsm["spatial"], float)
    X = np.asarray(a.X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for bin_um in (300 / 128.0, 300 / 77.0):
        old = gr._isotropic_fingerprint(xy, Z, bin_um, axes=("focality",),
                                        focality_impl="legacy")["focality"]
        new = gr._isotropic_fingerprint(xy, Z, bin_um, axes=("focality",),
                                        focality_impl="batched", focality_chunk=5)["focality"]
        assert np.allclose(new, old, rtol=1e-9, atol=0)
        assert np.array_equal(np.argsort(old), np.argsort(new))


def test_isotropic_fingerprint_is_unit_independent():
    """The magnitude axes must not depend on whether a section is stored in um or in mm.

    The multiscale sweep used to be built as a physical length and rounded,
    ``np.round(np.geomspace(2 * bin_um, 16 * bin_um, 12))``. That silently assumes a coordinate
    unit of about a micrometre: on a section stored in millimetres the bin is ~0.015, np.round
    sends every radius to zero, and the twelve-radius Moran curve collapses onto a single one-bin
    radius — so ``scale_peak_moran`` stopped being a multiscale peak at all. It cost the paper's
    merfish_sagittal column its scale and focality scores.

    Rescaling the coordinates rescales the bin with them, so the lattice, the mask and every
    radius-in-bins are identical and the fingerprint must be too.
    """
    a = _structured_adata(n=6000, n_struct=14, n_noise=8, seed=7)
    xy = np.asarray(a.obsm["spatial"], float)
    X = np.asarray(a.X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    axes = ("scale_peak_moran", "focality", "texture")

    def score(coords):
        bin_um = max(np.ptp(coords, 0).min() / 60.0, 1e-9)      # the driver's own bin rule
        return gr._isotropic_fingerprint(coords, Z, bin_um, axes=axes)

    um = score(xy)
    # A power-of-two rescale is exact in binary, so every cell's bin index is bit-identical and any
    # difference left is the algorithm reading the unit. Moran's I and the Gini are dimensionless
    # and must not move at all. Texture is a spectral centroid in CYCLES PER COORDINATE UNIT, so it
    # must move by exactly the rescale — a value, not a bug, but it is why raw texture is only
    # comparable within a dataset and why the presence test ranks it rather than thresholding it.
    exact = score(xy / 1024.0)
    for ax in ("scale_peak_moran", "focality"):
        assert np.array_equal(um[ax], exact[ax]), f"{ax} depends on the coordinate unit"
    assert np.allclose(exact["texture"], um["texture"] * 1024.0, rtol=1e-9, atol=0)

    # The realistic conversion, um -> mm, is not exact: 1/1000 is a repeating binary fraction, so a
    # few cells fall across a bin boundary. That jitter is the only difference allowed, and it must
    # not reorder the genes — the presence test ranks these against a null.
    mm = score(xy / 1000.0)
    for ax, unit in (("scale_peak_moran", 1.0), ("focality", 1.0), ("texture", 1000.0)):
        v = um[ax] * unit
        assert np.allclose(v, mm[ax], rtol=5e-2, atol=1e-3 * unit), f"{ax} moved more than binning jitter"
        assert np.corrcoef(np.argsort(np.argsort(v)),
                           np.argsort(np.argsort(mm[ax])))[0, 1] > 0.99, f"{ax} rank order moved"


def test_isotropic_fingerprint_sweep_spans_scales():
    """The sweep must actually be multiscale — 2 to 16 bins — whatever the coordinate unit.

    Guards the failure directly: with the old length-rounded construction a millimetre section
    produced twelve copies of a one-bin radius, which no equality test between two mm runs would
    have caught. A constant Moran curve means the peak is a single-radius statistic.
    """
    a = _structured_adata(n=5000, n_struct=10, n_noise=4, seed=8)
    X = np.asarray(a.X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for scale in (1.0, 1e-3):                                    # um and mm
        xy = np.asarray(a.obsm["spatial"], float) * scale
        bin_um = max(np.ptp(xy, 0).min() / 60.0, 1e-9)
        xyz = xy - xy.min(0)
        col = (xyz[:, 0] / bin_um).astype(int)
        row = (xyz[:, 1] / bin_um).astype(int)
        H, W = int(row.max()) + 1, int(col.max()) + 1
        lin = row * W + col
        cnt = np.bincount(lin, minlength=H * W).astype(float).reshape(H, W)
        mk = cnt > 0
        g = np.bincount(lin, weights=Z[:, 0], minlength=H * W).reshape(H, W)
        g[mk] /= np.maximum(cnt[mk], 1)
        rb = [max(int(round(b)), 1) for b in np.geomspace(2.0, 16.0, 12)]
        assert len(set(rb)) >= 8, f"sweep collapsed to {sorted(set(rb))} at scale {scale}"
        curve = spatial.morans_i_multiscale(g[..., None], mk, rb, kind="disc")[:, 0]
        assert curve.std() > 1e-6, "a flat multiscale curve means the sweep is not multiscale"


def test_focality_landmark_ships_a_scoped_third_coordinate():
    """The landmark is a per-section descriptive coordinate: it must classify before it reduces, decline
    to produce a number for a monotone curve, and declare its coverage. It must ALSO record that it is
    not a differential axis — measured on 19 replicate sections its between-condition shift is 0.223
    octaves against a 0.663 within-condition scatter, so differencing it between arms cannot resolve it.
    """
    import warnings as _w
    import anndata as ad
    from spectrast import curveshape
    rng = np.random.default_rng(0)
    N = 700
    xy = rng.random((N, 2)) * 400
    X = np.abs(rng.standard_normal((N, 6))).astype("float32")
    X[:, 0] += 6 * np.exp(-(((xy[:, 0] - 200) ** 2 + (xy[:, 1] - 200) ** 2) / 1500))   # focal blob
    a = ad.AnnData(X)
    a.obsm["spatial"] = xy
    a.var_names = [f"g{i}" for i in range(6)]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out = spectrast.gr.focality_landmark(a, radii=np.geomspace(12, 160, 16))

    assert set(out.columns) == {"shape_class", "landmark_um", "landmark_kind", "prominence",
                                "n_extrema", "monotone_rho", "roughness"}
    assert set(out.shape_class) <= set(curveshape.CLASSES)
    # a monotone curve gets NO landmark — that is the reduction declining, not failing
    mono = out.shape_class.str.startswith("monotone")
    assert np.isnan(out.landmark_um[mono]).all()
    assert (out.landmark_kind[mono] == "none").all()
    assert np.isfinite(out.landmark_um[~mono & (out.shape_class != "flat")]).all()

    u = a.uns["spectrast_focality_landmark"]
    assert u["differential_axis"] is False              # the scope is recorded, not just documented
    assert 0.0 <= u["coverage"] <= 1.0
    assert abs(sum(u["class_fractions"].values()) - 1.0) < 1e-9
    assert all(f"spectrast_focality_{c}" in a.var for c in out.columns)


def test_focality_landmark_warns_when_coverage_collapses():
    """Coverage is the number that decides whether the coordinate means anything: where few curves turn
    over, the ones that do are the extreme-scale genes and the third coordinate degenerates onto the
    first. Below the threshold it must say so rather than return a small number."""
    import warnings as _w
    import anndata as ad
    from spectrast import gr
    rng = np.random.default_rng(3)
    a = ad.AnnData(np.abs(rng.standard_normal((500, 5))).astype("float32"))
    a.obsm["spatial"] = rng.random((500, 2)) * 300
    a.var_names = [f"g{i}" for i in range(5)]
    old = gr.LANDMARK_MIN_COVERAGE
    try:
        gr.LANDMARK_MIN_COVERAGE = 1.01                       # force the branch deterministically
        with pytest.warns(UserWarning, match="focality landmark"):
            gr.focality_landmark(a, radii=np.geomspace(15, 150, 12))
    finally:
        gr.LANDMARK_MIN_COVERAGE = old


def test_curvefit_recovers_a_planted_decay_and_flags_the_window():
    """The fit must recover xi where the curve HAS one, and — the reason the module exists — must say
    so when xi lands outside the swept window, because there the reported half-max is being set by
    where the sweep stopped rather than by the tissue."""
    from spectrast import curvefit as cf
    r = np.geomspace(10, 400, 16)
    inside = 0.9 * np.exp(-r / 80.0) + 0.05                   # xi = 80, comfortably inside
    beyond = 0.9 * np.exp(-r / 40000.0) + 0.05                # essentially flat: xi far past r[-1]
    fit = cf.fit_decay(np.vstack([inside, beyond]), r)
    assert abs(fit["xi_um"][0] - 80.0) / 80.0 < 0.05
    assert fit["scale_supported"][0] and not fit["xi_outside_window"][0]
    assert fit["xi_outside_window"][1] and not fit["scale_supported"][1]


def test_curvefit_beta_identifies_a_heavy_tail_and_pins_when_it_cannot():
    """beta < 1 is the estimator finding the model-free scale rests on, so it must be recovered on a
    curve that has it. And a beta sitting on its bound is a parameter the data do not identify — it
    must be flagged, never reported as a shape."""
    from spectrast import curvefit as cf
    r = np.geomspace(10, 400, 16)
    heavy = 0.9 * np.exp(-((r / 80.0) ** 0.5)) + 0.05
    fit = cf.fit_decay(np.vstack([heavy]), r)
    assert not fit["beta_pinned"][0]
    assert abs(fit["beta"][0] - 0.5) < 0.1
    assert fit["heavier_than_exponential"][0]
    assert fit["d_bic"][0] > cf.D_BIC                          # stretched beats the single decay
    # a curve fitted against a collapsed bound box can only pin, and must say so
    tight = cf.fit_decay(np.vstack([heavy]), r, beta_bounds=(0.9, 1.1))
    assert tight["beta_pinned"][0]
    assert not tight["heavier_than_exponential"][0]            # pinned is never a tail finding


def test_curvefit_refuses_four_parameters_on_too_few_radii():
    """Four parameters cannot be identified from five points. The stretched family must decline rather
    than return a number, and the three-parameter fit must be refused outright below four radii."""
    from spectrast import curvefit as cf
    r5 = np.geomspace(30, 480, 5)
    y = (0.9 * np.exp(-r5 / 80.0) + 0.05)[None, :]
    with pytest.warns(UserWarning, match="too few"):
        fit = cf.fit_decay(y, r5)
    assert np.isnan(fit["beta"][0]) and np.isnan(fit["d_bic"][0])
    assert np.isfinite(fit["xi_um"][0])                        # the trust flag still works
    with pytest.raises(ValueError, match="three-parameter"):
        cf.fit_decay(y[:, :3], r5[:3])


def test_curvefit_does_not_ship_the_retracted_two_component_family():
    """The two-component fit wins the BIC contest and was retracted as biology — its components track
    the measurement window. It stays in the notebook that documents the retraction; shipping it would
    hand users a fit that looks like a second length scale and is not one."""
    from spectrast import curvefit as cf
    assert not hasattr(cf, "two")
    assert not any("two" == n or "two_component" in n for n in dir(cf))


def test_scale_fit_check_qualifies_the_scale_without_replacing_it():
    """It is a CHECK, not a fourth coordinate: it must reuse the scalogram scale_spectrum already
    computed (so the pair costs one sweep), write flags beside the scale, and never touch the scale."""
    a = _adata(n=500, g=10, seed=5)
    a.var_names = [f"g{i}" for i in range(10)]
    rad = np.geomspace(10, 160, 12)
    scale = gr.scale_spectrum(a, radii=rad).copy()
    out = gr.scale_fit_check(a, radii=rad)
    assert a.uns["spectrast_scale_fit_summary"]["scalogram_from_cache"] is True
    assert a.uns["spectrast_scale_fit_summary"]["differential_axis"] is False
    assert np.allclose(a.var["spectrast_char_scale"].to_numpy(), scale, equal_nan=True)  # untouched
    assert {"scale_supported", "beta_pinned", "xi_um"} <= set(out.columns)
    assert "spectrast_scale_fit_scale_supported" in a.var
    # a mismatched cache must be ignored rather than silently fitted on the wrong radii
    b = _adata(n=500, g=10, seed=5)
    b.var_names = a.var_names
    gr.scale_spectrum(b, radii=rad)
    gr.scale_fit_check(b, radii=np.geomspace(10, 160, 11))
    assert b.uns["spectrast_scale_fit_summary"]["scalogram_from_cache"] is False


def test_default_sweep_is_nineteen_radii_and_identifies_beta():
    """0.14.0: the default sweep is 19 radii (4.4 per octave over [2, 40] x the NN spacing). At the old six,
    every default scale_fit_check ran below WEAK_RADII_STRETCHED and warned that beta was weakly identified;
    the radius-count sweep (study_grid_p2p3) puts six in the untested gap just above the cliff where beta is
    pinned at a bound for every gene. A default call must now clear that threshold and stay silent about it."""
    from spectrast import curvefit as cf
    a = _adata(n=500, g=8, seed=3)
    a.var_names = [f"g{i}" for i in range(8)]
    rad = gr.default_radii(a)
    assert rad.shape == (19,) and np.all(np.diff(rad) > 0)
    base = gr._radius_from_xy(a.obsm["spatial"], factor=1.0)
    assert np.isclose(rad[0], 2.0 * base) and np.isclose(rad[-1], 40.0 * base)       # span unchanged
    assert np.isclose(len(rad) / np.log2(rad[-1] / rad[0]), 19 / np.log2(20))       # 4.40 per octave
    assert len(rad) >= cf.WEAK_RADII_STRETCHED
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gr.scale_fit_check(a)
    assert not [x for x in w if "weakly identified" in str(x.message)], [str(x.message) for x in w]


def test_scale_fit_check_never_reuses_a_scalogram_computed_on_other_settings():
    """Regression (0.14.1): the cache matched on radii and gene count only, so a different layer
    (or basis, bin_size, reduced_rank, rank) silently fitted the OLD curves. The cache now carries the settings
    it was computed with, and a fit on other settings recomputes and matches a cache-free run exactly."""
    a = _adata(n=500, g=6, seed=7)
    a.var_names = [f"g{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    a.layers["other"] = np.ascontiguousarray(a.X[rng.permutation(a.n_obs)])      # same genes, other fields
    rad = np.geomspace(10, 160, 12)
    gr.scale_spectrum(a, radii=rad)                                           # caches the X curves
    out = gr.scale_fit_check(a, radii=rad, layer="other")
    assert a.uns["spectrast_scale_fit_summary"]["scalogram_from_cache"] is False
    fresh = a.copy(); del fresh.uns["spectrast_scalogram"]
    ref = gr.scale_fit_check(fresh, radii=rad, layer="other")
    assert np.allclose(out["xi_um"].to_numpy(), ref["xi_um"].to_numpy(), equal_nan=True)
    for kw in (dict(bin_size=4.0), dict(basis="spatial_copy"), dict(reduced_rank=True, rank=3)):
        a.obsm["spatial_copy"] = np.asarray(a.obsm["spatial"]).copy()
        gr.scale_spectrum(a, radii=rad)
        gr.scale_fit_check(a, radii=rad, **kw)
        assert a.uns["spectrast_scale_fit_summary"]["scalogram_from_cache"] is False, kw
    gr.scale_spectrum(a, radii=rad)                                           # same settings still reuse it
    gr.scale_fit_check(a, radii=rad)
    assert a.uns["spectrast_scale_fit_summary"]["scalogram_from_cache"] is True


def test_scale_fit_check_warns_when_the_window_sets_the_scale():
    """Where most fitted lengths fall outside the swept window, the scales on that section are an
    artefact of where the sweep stopped. That has to be a warning, not a column nobody reads."""
    import anndata as ad
    rng = np.random.default_rng(11)
    a = ad.AnnData(np.abs(rng.standard_normal((500, 6))).astype("float32"))
    a.obsm["spatial"] = rng.random((500, 2)) * 300
    a.var_names = [f"g{i}" for i in range(6)]
    old = gr.FIT_MIN_SUPPORTED
    try:
        gr.FIT_MIN_SUPPORTED = 1.01                            # force the branch deterministically
        with pytest.warns(UserWarning, match="swept window"):
            gr.scale_fit_check(a, radii=np.geomspace(15, 150, 12))
    finally:
        gr.FIT_MIN_SUPPORTED = old


def test_half_max_is_not_the_ceiling_saturating_correlation_length():
    """The reduction the differential REJECTED was spatial.correlation_length — half of I at the
    SMALLEST radius, with any curve that never gets there pinned to the largest radius. A high floor
    defeats that threshold and every such gene saturates at the ceiling, which collapses a Δscale.
    The shipped half-max crosses half of the curve's OWN range, which a floor cannot defeat. This is
    the distinction the default flip rests on, so it is pinned here."""
    from spectrast import scale as sc_
    from spectrast import spatial as sp_
    r = np.geomspace(30, 480, 5)
    # a decaying curve on a HIGH floor: I never falls to half of I(r_min), but it does cross its own
    # mid-range partway through the sweep
    y = 0.90 + 0.10 * np.exp(-r / 90.0)
    S = y[:, None]
    clen = sp_.correlation_length(S, r)[0]
    hm = sc_.half_max(S, r)[0]
    assert np.isclose(clen, r[-1])                     # saturated at the ceiling, as documented
    assert np.isfinite(hm) and hm < r[-1] / 2          # the range crossing is unbothered by the floor
    # and the centroid, which the differential used instead, sits near the grid's own centre of mass
    assert sc_.centroid(S, r)[0] > hm


def test_half_max_returns_nan_rather_than_inventing_a_length():
    """A curve with no crossing has no half-max. NaN is the answer; the centroid's habit of returning
    a plausible micrometre value for every curve including the unstructured ones is exactly how it
    ends up reporting the radius grid."""
    from spectrast import scale as sc_
    r = np.geomspace(30, 480, 6)
    flat = np.ones_like(r)                             # never crosses
    decay = np.exp(-r / 100.0)
    out = sc_.half_max(np.column_stack([flat, decay]), r)
    assert np.isnan(out[0]) and np.isfinite(out[1])
    assert np.isfinite(sc_.centroid(np.column_stack([flat, decay]), r)).all()   # it never declines


def test_scale_reduction_orientation_is_population_wide_and_shareable():
    """Orientation must be decided once for a population, never per curve — a per-curve flip would make
    the sign of the statistic a free parameter. And two arms of a differential must be reducible on ONE
    shared sign, or the difference compares two conventions."""
    from spectrast import scale as sc_
    r = np.geomspace(30, 480, 6)
    rising = np.stack([1 - np.exp(-r / 100.0), 1 - np.exp(-r / 200.0)], axis=1)
    assert sc_.orient_sign(rising) == -1.0
    assert np.isfinite(sc_.half_max(rising, r)).all()          # handled by the population flip
    # forcing the wrong sign must NOT silently succeed with a plausible number
    assert np.isnan(sc_.half_max(rising, r, sign=+1.0)).all()
    with pytest.raises(ValueError, match="reduction must be"):
        sc_.reduce_scale(rising, r, "nope")


def test_scale_spectrum_default_is_half_max_and_centroid_is_still_reachable():
    """The flip must be real (the default moved) and reversible (published numbers reproducible)."""
    from spectrast import scale as sc_
    a = _structured_adata(n=1500, n_struct=8, n_noise=4, seed=7)
    rad = np.geomspace(10, 160, 12)
    default = gr.scale_spectrum(a.copy(), radii=rad)
    hm = gr.scale_spectrum(a.copy(), radii=rad, reduction="half_max")
    ct = gr.scale_spectrum(a.copy(), radii=rad, reduction="centroid")
    assert np.array_equal(default, hm, equal_nan=True)
    assert sc_.DEFAULT_REDUCTION == "half_max"
    fin = np.isfinite(hm) & np.isfinite(ct)
    assert fin.any() and not np.allclose(hm[fin], ct[fin])     # they are genuinely different numbers
    b = a.copy(); gr.scale_spectrum(b, radii=rad)
    assert b.uns["spectrast_scalogram"]["reduction"] == "half_max"   # provenance, so old files are identifiable


def test_differential_reduction_moves_only_the_effect_size_not_the_test():
    """delta_scale_um is a reported effect size, not the tested statistic. Flipping the reduction must
    leave p, FDR, T_max and the hit list bit-identical — measured on real cohorts, and pinned here so a
    future change to the reduction cannot quietly move the paper's hit lists."""
    from spectrast import diff as diff_
    rng = np.random.default_rng(3)
    def arm(shift, seed):
        r2 = np.random.default_rng(seed)
        n = 600
        xy = r2.random((n, 2)) * 300
        sig = np.exp(-((xy[:, 0] - 150) ** 2 + (xy[:, 1] - 150) ** 2) / (2 * (40 + shift) ** 2))
        X = np.column_stack([sig + 0.1 * r2.standard_normal(n) for _ in range(6)])
        b = ad.AnnData(np.abs(X).astype("float32"))
        b.obsm["spatial"] = xy.astype("float32")
        b.var_names = [f"g{i}" for i in range(6)]
        return b
    A = [arm(0, s) for s in (1, 2, 3)]
    B = [arm(35, s) for s in (4, 5, 6)]
    kw = dict(axis="moran", features=[f"g{i}" for i in range(6)], radii=(1, 2, 4, 8),
              bin_size=25.0, null="replicate", n_perm=60, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = diff_.differential(A, B, reduction="centroid", **kw).set_index("gene")
        h = diff_.differential(A, B, reduction="half_max", **kw).set_index("gene")
    for col in ("p", "fdr", "T_max", "r_at_max_um", "dI_at_max", "peak_signal"):
        assert np.allclose(c[col].to_numpy(), h[col].to_numpy(), equal_nan=True), col
    assert h.attrs["spectrast_reduction"] == "half_max"
    assert c.attrs["spectrast_reduction"] == "centroid"


def test_result_tables_print_when_the_display_truncates_them():
    """pandas compares DataFrame.attrs with == when it truncates a wide table for display (repr and
    Jupyter's HTML both do), so an ndarray stored in attrs made `organization` and `fingerprint_by`
    tables unprintable ("truth value of an array is ambiguous"). Their attrs now hold plain lists."""
    import anndata as ad
    import pandas as pd
    from spectrast import organization as org

    def sec(seed, n=800):
        r = np.random.default_rng(seed)
        a = ad.AnnData(np.abs(r.standard_normal((n, 5))).astype("float32"))
        a.obsm["spatial"] = r.random((n, 2)) * 300
        a.var_names = [f"g{i}" for i in range(5)]
        return a

    tables = {
        "organization": org.organization(sec(0), stats=("moran", "geary")),
        "fingerprint_by": spectrast.gr.fingerprint_by(
            ad.concat([sec(1), sec(2)], label="s", keys=["s0", "s1"]), "s", min_cells=10),
    }
    with pd.option_context("display.width", 60, "display.max_columns", 4):
        for name, t in tables.items():
            assert not any(isinstance(v, np.ndarray) for v in t.attrs.values()), name
            repr(t)
            t._repr_html_()

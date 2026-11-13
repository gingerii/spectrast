"""spectrast.orient — a descriptive statistic for GENE spatial orientation: which genes are directionally
organized, at what scale, and where.

``spatial.polar_power_spectrum`` reads a SINGLE global (anisotropy, orientation) from a whole cropped field,
so directional structure is only visible if the user already cropped to a region they knew about. This module
inverts that: it slides a tapered window across the tissue, runs the polar machinery per window, and stacks
the results into a per-gene ORIENTATION FIELD with a calibrated significance — DISCOVERING, without prior
knowledge, the genes whose expression carries directional structure, the scale and location at which it does,
and the axis along which it runs (for streamline plots). The primary output is descriptive and per-gene
(:func:`screen_oriented_genes` → :func:`gene_orientation_field` → :func:`orientation_streamlines`); it does
NOT classify tissue into regions. It is general to any organized tissue — laminar cortex, fiber tracts,
crypt/gland axes, tumour fronts — not just brain. A pooled-field mode (:func:`orientation_field` on a summary
+ :func:`segment_regions`) additionally supports unsupervised sub-region discovery.

Two multiscale axes, both FFT-native:
  * **window size** = the scale of the STRUCTURE sought (fine laminae vs coarse tracts). We SWEEP it (a
    log-spaced set of sizes, the window-axis analogue of the Moran scalogram) so the tool is robust to
    tissues whose organization scale is unknown or varies — no fixed-window assumption. ``aggregate_scales``
    then picks, per location, the window scale of maximal calibrated laminarity (a characteristic
    organization scale), the way the scalogram reads a characteristic length.
  * **frequency ring within a window** = the scale of the FEATURE (the novel axis): orientation as a function
    of feature scale AT ONE LOCATION, separating superimposed structures at different scales in the same
    place. The annulus is defined RELATIVE to each window's Nyquist (fractions of it), never in absolute k,
    so rings stay comparable across window sizes (a window cannot resolve structure coarser than itself).

The discriminating statistic is NOT anisotropy alone (a small window is anisotropic by finite-sample chance):
genuine laminar structure is SPATIALLY COHERENT — neighboring windows agree on orientation. We report
``laminarity = anisotropy x coherence`` and gate on both, with a per-window isotropic phase-randomized
surrogate null (BH-FDR across windows) — surrogates preserve each field's radial power spectrum (its
autocorrelation) but randomize orientation, and pass through the same mask+taper aperture, so the null removes
BOTH the small-sample upward bias of windowed anisotropy AND the false anisotropy a hard tissue-mask edge
manufactures (a value-scramble null cannot: it whitens the field and mis-calibrates autocorrelated genes).

Compute: every window's FFT — observed AND the whole permutation null — is batched through the pluggable
``_fft`` backend (threaded CPU / CUDA / MPS) in single transforms, so the sweep×null cost stays affordable.

Prior art: windowed structure tensors / orientation fields are old (materials grain mapping, collagen-fiber
orientation, fingerprint ridges) — that machinery is NOT claimed as novel. The contribution is the
unsupervised, scale-resolved discovery of per-gene directional sub-structure in spatial transcriptomics and
the frequency-resolved-orientation readout. Descriptive by design (v1): no condition-differential wiring.

Heavy deps lazy-imported. AnnData-native at the public layer; ``import spectrast`` then ``spectrast.orient``.
"""
from __future__ import annotations
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# The outer edge of the measurement band (gotcha 0: a square grid is not a circle)
# ---------------------------------------------------------------------------
#: Largest ``kmax_frac`` for which every angular wedge is complete, = 1/sqrt(2).
#:
#: The FFT of a window samples frequencies on a SQUARE: each axis reaches the Nyquist frequency 1/2
#: cycles per bin, so the corner reaches sqrt(2)/2 and the grid extends 41% further along the diagonals
#: than along the axes. ``kmax_frac`` is a fraction of that corner radius, so any value above
#: (1/2)/(sqrt(2)/2) = 1/sqrt(2) asks for frequencies that exist on the diagonals and DO NOT EXIST on the
#: axes. The wedges centred on 0 and 90 deg are then truncated — at a 40x40 window with n_theta=36 and the
#: former default 0.8, they held 4.5 pixels against the diagonal wedges' 16.0 — and since they keep only
#: the low-frequency part of their ring, where power is higher for any field whose spectrum falls with
#: frequency, they read high and tip the resultant onto the pixel axes.
#:
#: Measured on isotropic controls: at 0.8 the outer ring reported a
#: near-cardinal angle 34-40% of the time against the 25% expected by chance (four-fold order parameter
#: C4 = +0.10 to +0.39, z up to 14, at EVERY combination of n_theta in {18,36,72} and n_rings in {3,4,6});
#: at 1/sqrt(2) every ring is directionless (|C4| <= 0.04, max |z| = 3.7 over ~500 tests). The cut is
#: geometric, not tuned. It costs nothing real: it discards wavelengths between 1.77 and 2.00 bins, and
#: two bins IS the sampling limit.
INSCRIBED_KMAX = 2 ** -0.5


# ---------------------------------------------------------------------------
# Window taper (gotcha 1: a hard window edge leaks a spectral cross that FAKES 0/90deg anisotropy)
# ---------------------------------------------------------------------------
def _taper2d(h, w, kind="hann"):
    """Separable 2-D taper (Hann/Tukey) of shape (h, w). Multiplying a window by this before the FFT
    suppresses the edge-discontinuity spectral leakage that would otherwise manufacture anisotropy."""
    def _1d(n):
        if n < 2:
            return np.ones(n)
        i = np.arange(n)
        if kind == "hann":
            return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))
        if kind == "tukey":
            a = 0.5; w1 = np.ones(n); e = int(a * (n - 1) / 2.0)
            if e > 0:
                ramp = 0.5 * (1 + np.cos(np.pi * (np.arange(e) / e - 1)))
                w1[:e] = ramp; w1[-e:] = ramp[::-1]
            return w1
        raise ValueError(kind)
    return np.outer(_1d(h), _1d(w)).astype(np.float64)


# ---------------------------------------------------------------------------
# Polar-spectrum geometry (precomputed ONCE per window size) + batched backend core
# ---------------------------------------------------------------------------
def _polar_geom(H, W, n_theta, n_rings, kmin_frac, kmax_frac):
    """Angular×radial binning geometry for an (H, W) window — independent of the field, so it is built once
    per window size and reused across every window and every permutation. Returns a one-hot assignment matrix
    ``S`` (H*W, n_rings*n_theta), per-bin counts, doubled-angle cos/sin per wedge, and ring centre fractions.

    ``kmax_frac`` is a fraction of the LARGEST radius the FFT grid carries, which is the CORNER of the
    k-square — a factor sqrt(2) further out than the Nyquist frequency on the axes. Ask for more than
    :data:`INSCRIBED_KMAX` and the outermost ring runs off the grid in the 0 and 90 deg directions, so the
    wedges centred there are truncated and, on any field whose power varies with frequency, read high
    (see :data:`INSCRIBED_KMAX`). That is a directional bias with no tissue in it, so the caller is warned.
    """
    ky = np.fft.fftshift(np.fft.fftfreq(H)); kx = np.fft.fftshift(np.fft.fftfreq(W))
    KX, KY = np.meshgrid(kx, ky); Kr = np.sqrt(KX ** 2 + KY ** 2)
    # CLAMP, don't just warn. INSCRIBED_KMAX = 1/sqrt(2) is exact for a SQUARE window of either parity,
    # but a rectangular one inscribes a smaller fraction of its own corner (0.698 at 41x40, 0.695 at
    # 40x31), and windowed_polar / rank_region_genes accept an arbitrary box. Clamping here means no
    # caller can measure outside the complete-wedge disc, and because it happens before the ring edges
    # are cut, ``ring_kfrac`` reports the band actually used.
    inscribed = float(min(np.abs(kx).max(), np.abs(ky).max()) / (Kr.max() + 1e-12))
    # Warn only when the CALLER asked for more than any window could give. A rectangular window inscribes
    # slightly less than INSCRIBED_KMAX, and clamping the default down to it is the library doing its job,
    # not a mistake worth a warning on every call.
    if kmax_frac > INSCRIBED_KMAX + 1e-9:
        warnings.warn(
            f"kmax_frac={kmax_frac:g} exceeds the inscribed disc of this {H}x{W} window's frequency "
            f"grid ({inscribed:.4f}) and has been clamped to it. Beyond that radius the wedges centred "
            f"on 0 and 90 deg are truncated, which biases the reported orientation toward the pixel "
            f"axes (see spectrast.orient.INSCRIBED_KMAX).", RuntimeWarning, stacklevel=2)
    kmax_frac = min(float(kmax_frac), inscribed)
    kfrac = (Kr / (Kr.max() + 1e-12)).ravel()
    ang = (np.degrees(np.arctan2(KY, KX)) % 180.0).ravel()             # axial fold to [0,180)
    redges = np.linspace(kmin_frac, kmax_frac, n_rings + 1)
    tedges = np.linspace(0, 180, n_theta + 1)
    ring = np.digitize(kfrac, redges) - 1                              # in [0,n_rings) inside band, else out
    theta = np.clip(np.digitize(ang, tedges) - 1, 0, n_theta - 1)
    inband = (ring >= 0) & (ring < n_rings)
    binid = np.where(inband, ring * n_theta + theta, -1)
    nbins = n_rings * n_theta
    S = np.zeros((H * W, nbins), np.float64)
    pix = np.where(inband)[0]
    S[pix, binid[pix]] = 1.0
    counts = S.sum(0)                                                  # (nbins,) pixels per (ring,theta)
    tctr = np.radians(0.5 * (tedges[:-1] + tedges[1:]))
    cnt = np.maximum(counts, 1.0)
    # EXACT collapse of the wedge pipeline. anisotropy/orientation/ring power need only THREE numbers
    # per ring -- sum_theta(Pth), sum_theta(Pth cos2), sum_theta(Pth sin2) -- and Pth = (P @ S)/counts
    # is linear in P, so the per-wedge weights fold back through S into one (H*W, 3*n_rings) operator.
    # Identical arithmetic to the S-then-reduce form (agrees to 8e-16), but the reduction shrinks from
    # n_rings*n_theta columns to 3*n_rings and its cost stops depending on n_theta at all. ``S`` is kept
    # so the legacy path stays reachable for the equivalence test.
    Sn = (S / cnt).reshape(H * W, n_rings, n_theta)
    Wc = np.concatenate([(Sn * np.cos(2 * tctr)).sum(-1),
                         (Sn * np.sin(2 * tctr)).sum(-1),
                         Sn.sum(-1)], axis=1)                          # (H*W, 3*n_rings)
    return dict(S=S, counts=cnt, cos2=np.cos(2 * tctr), sin2=np.sin(2 * tctr), Wc=Wc,
                n_theta=n_theta, n_rings=n_rings, ring_kfrac=0.5 * (redges[:-1] + redges[1:]),
                kr_max=float(Kr.max()), HW=(H, W))


def _polar_geom_direct(H, W, n_theta, n_rings, kmin_frac, kmax_frac, n_sub=3):
    """Sample the spectrum DIRECTLY on a polar grid instead of binning the Cartesian k-lattice.

    Why this exists. :func:`_polar_geom` assigns each lattice point to a (ring, wedge) bin, and WHICH
    wedges a bin can reach is a property of the lattice, not the tissue: at a B-point window the
    representable radii are sqrt(i^2+j^2)/B, a sparse set, so different rings can only be populated
    from different directions. A ring reachable from two diagonal wedges reports anisotropy near 1 at
    45 deg, and the max-over-rings aggregation preferentially selects exactly that ring. Measured on
    white noise, the four-fold order parameter C4 reads z = -30 at B=6 and is STILL z = -12 at B=19,
    and it does not decay smoothly with B — it jitters, because it depends on which radii a B-point
    lattice happens to represent. That is a number-theoretic artefact, so no "minimum window" rule can
    be certified without testing every B individually.

    Sampling directly removes it at the source: every (ring, wedge) is evaluated at exactly the
    frequency asked for, so angular occupancy is uniform BY DEFINITION. Same white-noise test, direct
    sampling: |z| <= 4.6 at B=6 and <= 1.8 for every B >= 10.

    What it costs and does not cost. This is a dense complex GEMM against a precomputed operator
    rather than an FFT, so it is O(H*W*n_points) — measured CHEAPER than the lattice path on CPU
    (101 ms vs 153 ms at B=40, batch 8000), because one large BLAS call runs nearer peak than
    thousands of small transforms. It does NOT fix the noise floor: a B x B window carries only ~B^2
    independent spectral values however they are sampled, so the floor still falls with B and only a
    CONSTANT-B sweep removes that ramp from the curve.

    Each ring is integrated over ``n_sub`` sub-radii so a ring is an ANNULUS rather than a circle,
    matching what the lattice version averages over. Returns the complex sampling operator plus the
    same collapsed (n_points, 3*n_rings) reduction the lattice path uses, so both share one downstream.
    """
    if kmax_frac > INSCRIBED_KMAX + 1e-9:
        warnings.warn(f"kmax_frac={kmax_frac:g} exceeds the inscribed disc and has been clamped to "
                      f"{INSCRIBED_KMAX:.4f}.", RuntimeWarning, stacklevel=2)
    kmax_frac = min(float(kmax_frac), INSCRIBED_KMAX)
    kr_max = float(np.hypot(np.abs(np.fft.fftfreq(H)).max(), np.abs(np.fft.fftfreq(W)).max()))
    redges = np.linspace(kmin_frac, kmax_frac, n_rings + 1) * kr_max
    tctr = np.radians(np.linspace(0.0, 180.0, n_theta, endpoint=False) + 90.0 / n_theta)
    rs = np.concatenate([np.linspace(redges[i], redges[i + 1], n_sub + 2)[1:-1] for i in range(n_rings)])
    KX = np.outer(rs, np.cos(tctr)).ravel(); KY = np.outer(rs, np.sin(tctr)).ravel()
    yy, xx = np.mgrid[0:H, 0:W]
    E = np.exp(-2j * np.pi * (KX[None, :] * xx.ravel()[:, None] + KY[None, :] * yy.ravel()[:, None]))
    npts = n_rings * n_sub * n_theta
    # Collapse (n_points -> 3*n_rings) exactly as the lattice path does: average the sub-radii within a
    # ring, then take the cos2/sin2/total reduction over wedges. One matrix, so both samplings share
    # the identical downstream arithmetic and any difference between them is the SAMPLING, nothing else.
    sel = np.zeros((npts, n_rings)); idx = np.arange(npts)
    sel[idx, (idx // n_theta) // n_sub] = 1.0 / n_sub          # ring membership, averaging sub-radii
    ang = np.tile(tctr, n_rings * n_sub)
    Wc = np.concatenate([sel * np.cos(2 * ang)[:, None], sel * np.sin(2 * ang)[:, None], sel], axis=1)
    return dict(sampling="direct", E=E.astype(np.complex128), Wc=Wc, n_theta=n_theta, n_rings=n_rings,
                n_sub=n_sub, ring_kfrac=0.5 * (redges[:-1] + redges[1:]) / kr_max,
                kr_max=kr_max, HW=(H, W))


#: Bins across a window. 16, not the 40 this shipped with until 2026-08-12 — see ``_geom_for``. Small B is
#: only usable with direct sampling, so the two defaults changed together and should not be split.
DEFAULT_BINS_PER_WINDOW = 16


def _geom_for(win, n_theta, n_rings, kmin_frac, kmax_frac, sampling="direct", n_sub=1):
    """Polar geometry for a window, by sampling scheme. ``"direct"`` is the default as of 2026-08-12.

    The lattice path bins an FFT's square frequency grid into polar wedges. At the window sizes this
    sweep actually uses, that grid is too sparse to fill them: at 16 bins across, most wedges catch no
    lattice point at all, and ``_polar_geom`` clamps an empty wedge's count to 1 so it contributes zero
    power rather than being excluded. The surviving wedges are the ones the lattice happens to favour,
    which is a direction fixed by the GRID, not by the tissue — on white noise through a real mask it
    reads a four-fold anisotropy of z = -4.8 at B=16, where direct sampling reads +0.4.

    Direct sampling evaluates the DFT at chosen polar frequencies instead (one complex GEMM against
    precomputed exponentials). Same transform, different points; the FFT's factorization only buys
    speed on its own regular grid, and measured on CPU the GEMM is in fact CHEAPER at these sizes
    (101 ms vs 153 ms at B=40, batch 8000) because one large BLAS call beats thousands of small FFTs.

    ``"lattice"`` remains reachable, and is what the equivalence tests compare against. It is also what
    every result computed before 2026-08-12 used, so reproducing an old number needs it explicitly.
    """
    if sampling == "direct":
        return _polar_geom_direct(win, win, n_theta, n_rings, kmin_frac, kmax_frac, n_sub=n_sub)
    if sampling == "lattice":
        return _polar_geom(win, win, n_theta, n_rings, kmin_frac, kmax_frac)
    raise ValueError(f"sampling must be 'direct' or 'lattice', got {sampling!r}")


def _radial_geom(H, W, nb=None):
    """Radial-frequency binning for an (H, W) window (unshifted FFT layout), reused across genes/perms. Used
    to estimate a field's ISOTROPIC (angularly-averaged) power spectrum for the phase-randomized surrogate
    null. Returns per-pixel radial-bin id, per-bin pixel count, and n_bins."""
    ky = np.fft.fftfreq(H); kx = np.fft.fftfreq(W)
    KX, KY = np.meshgrid(kx, ky); Kr = np.sqrt(KX ** 2 + KY ** 2).ravel()
    if nb is None:
        nb = max(H, W)
    kbin = np.clip((Kr / (Kr.max() + 1e-12) * (nb - 1)).round().astype(int), 0, nb - 1)
    cnt = np.bincount(kbin, minlength=nb).astype(float).clip(min=1.0)
    return dict(kbin=kbin, cnt=cnt, nb=int(nb))


def _polar_batch(fields, mask, tap, geom, impl="auto", mask_repeat=1):
    """Frequency-resolved polar spectrum for a BATCH of same-shape windows through the pluggable FFT backend.

    ``fields`` (B, H, W) or (H, W). ``mask`` is (H, W) shared across the batch OR (B, H, W) per item — the
    latter lets heterogeneous windows (each with its own occupancy mask) go through ONE batched FFT, which
    maximizes GPU/threaded-backend utilization and minimizes device round-trips. Returns per-ring anisotropy
    (B, n_rings), structure_orientation deg (B, n_rings) and mean ring power (B, n_rings).

    ``impl`` selects the reduction. ``"auto"`` uses the collapsed ``geom["Wc"]`` operator and, on a GPU
    backend, keeps the WHOLE chain (demean, taper, FFT, squared magnitude, reduction) device-resident so
    the window stack crosses the bus once per chunk instead of once per transform — the shipped path sent
    a numpy array into ``_fft.fft2``, which round-trips per call and made MPS *slower* than CPU on small
    windows. ``"legacy"`` reaches the original bin-then-divide-then-reduce path; it exists so the
    equivalence test has an independent implementation to compare against. Both agree to ~1e-16.
    """
    from . import _fft as _F
    F = np.asarray(fields, float); single = F.ndim == 2
    if single:
        F = F[None]
    B, H, W = F.shape
    M = np.asarray(mask, bool)
    if M.ndim == 2:
        M = M[None]                                                              # broadcast shared mask
    if geom.get("sampling") == "direct":
        a, b, tot0 = _polar_reduce_direct(F, M, tap, geom, mask_repeat=mask_repeat)
    elif impl == "auto" and _F.get_backend() in ("cuda", "mps"):
        a, b, tot0 = _polar_reduce_device(F, M, tap, geom, mask_repeat=mask_repeat)
    else:
        if mask_repeat > 1 and M.shape[0] > 1:
            M = np.repeat(M, mask_repeat, axis=0)      # host path: materialize, correctness over speed
        msum = M.sum((1, 2)).clip(min=1)                                         # (B,) or (1,)
        mu = (F * M).sum((1, 2)) / msum
        Fz = (F - mu[:, None, None]) * M * tap
        P = np.abs(np.fft.fftshift(_F.fft2(Fz, axes=(1, 2)), axes=(1, 2))) ** 2   # (B,H,W), backend FFT
        if impl == "legacy":
            Pbins = P.reshape(B, H * W) @ geom["S"]                               # (B, nbins)
            Pth = (Pbins / geom["counts"]).reshape(B, geom["n_rings"], geom["n_theta"])
            a = (Pth * geom["cos2"]).sum(-1); b = (Pth * geom["sin2"]).sum(-1)    # (B,n_rings)
            tot0 = Pth.sum(-1)
        else:
            nr = geom["n_rings"]
            R = P.reshape(B, H * W) @ geom["Wc"]                                  # (B, 3*n_rings)
            a, b, tot0 = R[:, :nr], R[:, nr:2 * nr], R[:, 2 * nr:]
    aniso = np.sqrt(a ** 2 + b ** 2) / (tot0 + 1e-12)
    porient = (0.5 * np.degrees(np.arctan2(b, a))) % 180.0
    struct = (porient + 90.0) % 180.0
    ringpow = tot0 / geom["n_theta"]                                             # == old Pth.mean(-1)
    if single:
        return aniso[0], struct[0], ringpow[0]
    return aniso, struct, ringpow


def _polar_reduce_direct(F, M, tap, geom, mask_repeat=1, chunk=4096):
    """Direct polar sampling: |demeaned, masked, tapered field @ E|^2, then the shared collapse.

    Runs on the active backend. On CPU this is float64 (E is complex128) and is the reference; on a
    GPU backend it drops to complex64, which was validated on real expression rather than synthetic —
    the toolkit skill records a synthetic parity benchmark under-reporting float32 error by ~3 orders.
    """
    from . import _fft as _F
    B, H, W = F.shape
    nr = geom["n_rings"]
    r = int(mask_repeat)
    Mfull = M if (r == 1 or M.shape[0] == 1) else np.repeat(M, r, axis=0)
    if _F.get_backend() in ("cuda", "mps"):
        import torch
        dev = _F._TORCH_DEVICE
        Et = torch.from_numpy(np.ascontiguousarray(geom["E"], np.complex64)).to(dev)
        Wt = torch.from_numpy(np.ascontiguousarray(geom["Wc"], np.float32)).to(dev)
        tt = torch.from_numpy(np.ascontiguousarray(tap, np.float32)).to(dev)
        out = []
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            Ft = torch.from_numpy(np.ascontiguousarray(F[s:e], np.float32)).to(dev)
            Mb = Mfull if Mfull.shape[0] == 1 else Mfull[s:e]
            Mt = torch.from_numpy(np.ascontiguousarray(Mb, np.float32)).to(dev)
            mu = (Ft * Mt).sum((-2, -1)) / Mt.sum((-2, -1)).clamp(min=1)
            Fz = ((Ft - mu.reshape(-1, 1, 1)) * Mt * tt).reshape(e - s, H * W).to(torch.complex64)
            out.append((((Fz @ Et).abs() ** 2) @ Wt).detach().to("cpu").numpy())
        R = np.concatenate(out, 0).astype(np.float64)
    else:
        msum = np.maximum(Mfull.sum((1, 2)), 1)
        mu = (F * Mfull).sum((1, 2)) / msum
        Fz = ((F - mu[:, None, None]) * Mfull * tap).reshape(B, H * W)
        R = (np.abs(Fz @ geom["E"]) ** 2) @ geom["Wc"]
    return R[:, :nr], R[:, nr:2 * nr], R[:, 2 * nr:]


_DEV_CACHE = {}          # id(geom) -> (Wc tensor, taper tensor, device) — constants, re-sent per call otherwise


def _polar_reduce_device(F, M, tap, geom, chunk=4096, mask_repeat=1):
    """Device-resident demean+taper+FFT+reduce for a batch of windows. Returns the three per-ring sums
    (cos2-weighted, sin2-weighted, total) as host float64, so every caller sees the same downstream
    arithmetic regardless of backend.

    Validated on REAL expression (10,000 windows, 85% zeros — the dynamic range that made an earlier
    synthetic parity benchmark under-report float32 error by ~3 orders): relative-L2 5.2e-07 vs the host
    float64 path, worst-case relative error 6.7e-05 in the near-cancelling anisotropy<0.02 stratum, and
    the C4 lattice-alignment statistic identical to three decimals. Unlike the Moran path (~1e-3, see the
    spectrast-toolkit skill) anisotropy is a RATIO whose denominator is a sum of strictly positive powers,
    so the resultant's cancellation is normalized by the quantity that produced it and float32 holds up.
    """
    import torch
    from . import _fft as _F
    dev = _F._TORCH_DEVICE
    B, H, W = F.shape
    nr = geom["n_rings"]
    # Profiling says this path is TRANSFER-bound, not compute-bound: the FFT is ~1.4% of wall time while
    # host->device copies are ~40%. So cache the two per-geometry constants instead of re-sending them on
    # every call, and never materialize a repeated mask — broadcasting one mask row across the genes that
    # share a tile removes a copy equal in size to the field stack itself.
    ck = (id(geom), str(dev))
    if _DEV_CACHE.get("key") != ck:
        _DEV_CACHE.clear()
        _DEV_CACHE["key"] = ck
        _DEV_CACHE["Wc"] = torch.from_numpy(np.ascontiguousarray(geom["Wc"], np.float32)).to(dev)
        _DEV_CACHE["tap"] = torch.from_numpy(np.ascontiguousarray(tap, np.float32)).to(dev)
    Wt, tt = _DEV_CACHE["Wc"], _DEV_CACHE["tap"]
    r = int(mask_repeat)
    chunk = max(chunk - chunk % r, r)                    # keep chunk boundaries on mask rows
    out = []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        Ft = torch.from_numpy(np.ascontiguousarray(F[s:e], np.float32)).to(dev)
        Mb = M if M.shape[0] == 1 else M[s // r:-(-e // r)]
        Mt = torch.from_numpy(np.ascontiguousarray(Mb, np.float32)).to(dev)
        if r > 1 and Mt.shape[0] > 1:                    # (nt,H,W) -> (nt,1,H,W) broadcast over r genes
            Ft = Ft.reshape(-1, r, H, W); Mt = Mt.unsqueeze(1)
        msum = Mt.sum((-2, -1)).clamp(min=1)
        mu = (Ft * Mt).sum((-2, -1)) / msum
        Fz = ((Ft - mu.unsqueeze(-1).unsqueeze(-1)) * Mt * tt).reshape(-1, H, W)
        P = torch.fft.fftshift(torch.fft.fft2(Fz, dim=(1, 2)).abs() ** 2, dim=(1, 2))
        out.append((P.reshape(P.shape[0], H * W) @ Wt).detach().to("cpu").numpy())
    R = np.concatenate(out, 0).astype(np.float64)
    return R[:, :nr], R[:, nr:2 * nr], R[:, 2 * nr:]


def windowed_polar(field, mask, *, n_theta=36, n_rings=4, kmin_frac=0.05, kmax_frac=INSCRIBED_KMAX,
                   taper="hann"):
    """Frequency-resolved polar power spectrum of a single window (convenience wrapper over the batched core).

    Bins spectral power into ``n_rings`` radial rings × ``n_theta`` angular wedges and returns the axial
    anisotropy + orientation PER RING (rather than collapsing the annulus to one number). Rings span
    ``[kmin_frac, kmax_frac]`` of THIS window's largest radial frequency (the corner of its FFT grid,
    about 0.707 cycles per bin) — relative, so a ring means the same fraction-of-resolvable-scale in every
    window regardless of size. The field is mean-centred over ``mask``, multiplied by ``mask`` and the
    taper, and binned on the FFT lattice (the scans in this module default to direct polar sampling
    instead; see :func:`orientation_field`).

    Angles are in degrees in ``[0, 180)``, measured from the +x (column) axis toward the +y (row) axis.

    Parameters
    ----------
    field : numpy.ndarray
        ``(H, W)`` field, one value per bin.
    mask : numpy.ndarray
        Boolean ``(H, W)`` occupancy; bins outside it are set to zero after centring.
    n_theta : int, optional
        Angular wedges over ``[0, 180)`` degrees. Default ``36`` (5 degrees each).
    n_rings : int, optional
        Radial rings, equally spaced in frequency between ``kmin_frac`` and ``kmax_frac``. Default ``4``.
    kmin_frac : float, optional
        Inner edge of the innermost ring, as a fraction of the largest radial frequency. Default ``0.05``.
    kmax_frac : float, optional
        Outer edge of the outermost ring, same units. Default :data:`INSCRIBED_KMAX` (``1/sqrt(2)``).
        Larger values are clamped to the disc inscribed in the frequency grid, with a ``RuntimeWarning``.
    taper : {"hann", "tukey"}, optional
        Separable window taper applied before the FFT (``"tukey"`` uses a 0.5 taper fraction).
        Default ``"hann"``.

    Returns
    -------
    dict
        Per-ring arrays of length ``n_rings``: ``anisotropy`` 0->1, ``structure_orientation`` deg
        (real-space elongation axis = power-peak axis + 90), ``ring_power`` (mean power per wedge in the
        ring), ``ring_kfrac`` (ring centre as a fraction of the largest radial frequency). The feature
        wavelength of a ring is ``1 / (ring_kfrac * k_corner)`` bins, with ``k_corner`` about 0.707.

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> yy, xx = np.mgrid[0:64, 0:64]
    >>> stripes = np.sin(2 * np.pi * yy / 8.0)                         # horizontal laminae
    >>> out = spectrast.orient.windowed_polar(stripes, np.ones_like(stripes, bool))
    >>> bool(out["anisotropy"].max() > 0.5)                            # strongly oriented
    True
    """
    H, W = np.asarray(field).shape
    geom = _polar_geom(H, W, n_theta, n_rings, kmin_frac, kmax_frac)
    tap = _taper2d(H, W, taper)
    a, s, rp = _polar_batch(field, mask, tap, geom)
    return dict(anisotropy=a, structure_orientation=s, ring_power=rp, ring_kfrac=geom["ring_kfrac"])


def _iso_amp(F, M, tap, rgeom):
    """Per-item ISOTROPIC amplitude spectrum sqrt(angularly-averaged power). ``F`` (B, H, W), ``M`` (B, H, W)
    bool. Estimates each field's radial power profile (its autocorrelation, orientation-agnostic) so the
    surrogate reproduces the field's SPATIAL SCALE but with no preferred direction. The estimate uses the SAME
    mask+taper as the observed anisotropy measurement (:func:`_polar_batch`), so the surrogate's in-band power
    matches what the statistic actually sees — without the taper the low-frequency spectral leakage differs and
    the null under-shoots the observed anisotropy. Returns amp (B, H, W)."""
    B, H, W = F.shape
    ms = np.clip(M.sum((1, 2)), 1, None)
    mu = (F * M).sum((1, 2)) / ms
    g = (F - mu[:, None, None]) * M * tap                               # match the observed mask+taper pipeline
    P = np.abs(np.fft.fft2(g, axes=(1, 2))) ** 2                        # (B, H, W) unshifted power
    kbin = rgeom["kbin"]; cnt = rgeom["cnt"]; nb = rgeom["nb"]
    idx = (np.arange(B)[:, None] * nb + kbin[None, :]).ravel()         # per-item radial-bin flat index
    prad = np.bincount(idx, weights=P.reshape(B * H * W), minlength=B * nb).reshape(B, nb) / cnt
    return np.sqrt(np.maximum(prad[:, kbin], 0.0)).reshape(B, H, W)


def _iso_null(F, M, tap, geom, rgeom, *, n_perm, seed_k, max_batch=2048):
    """Per-ring anisotropy null for genes sharing ONE window, via ISOTROPIC phase-randomized surrogates.

    The failure of a value-scramble null is that shuffling the masked values WHITENS the field (destroys its
    autocorrelation), so an autocorrelated real gene — which has far fewer effective independent Fourier modes
    per ring than white noise — looks anisotropic by chance far above that white null (measured ~56% per-window
    false-positive on a smooth isotropic field). This null instead preserves each field's RADIAL power spectrum
    (its autocorrelation) and randomizes only the PHASES, giving fields with the same spatial scale but NO
    preferred orientation. Each surrogate is then pushed through the SAME mask+taper aperture as the observed
    field, so it also reproduces the anisotropy a hard tissue-mask edge manufactures — calibrating out the
    boundary artifact that defeated the coherence gate.

    ``F`` (B, win, win) genes sharing window mask ``M`` (win, win). Phases are seeded from ``seed_k`` and SHARED
    across the B genes (per-window seeding), so the batched cross-gene screen is numerically identical to a
    per-gene scan. Returns (B, n_perm, n_rings) null anisotropy.
    """
    F = np.asarray(F, float)
    if F.ndim == 2:
        F = F[None]
    B, H, W = F.shape
    M2 = np.broadcast_to(np.asarray(M, bool), (B, H, W))
    amp = _iso_amp(F, M2, tap, rgeom)                                         # (B, H, W) per-gene radial spectrum
    rng = np.random.default_rng(seed_k)
    # isotropic GRF synthesis: filter white noise by the radial amplitude. Keep the FULL complex white FFT
    # (random amplitudes AND phases, Hermitian by construction -> real ifft2), NOT just its phase — the random
    # per-mode amplitude is what reproduces the finite-window ANGULAR power fluctuations of a real isotropic
    # field. (Keeping only the phase forces |F|^2 = amp^2, a perfectly isotropic spectrum with zero angular
    # fluctuation, which collapses the null anisotropy to ~0 and makes everything falsely significant.)
    # Anisotropy is scale-invariant, so the white-noise normalization is irrelevant.
    Wf = np.fft.fft2(rng.standard_normal((n_perm, H, W)), axes=(1, 2))          # (n_perm, H, W) complex Gaussian
    Mw = np.asarray(M, bool)
    out = np.empty((B, n_perm, geom["n_rings"]))
    per = max(1, max_batch // max(n_perm, 1))                                  # genes per surrogate chunk (bound mem)
    for s in range(0, B, per):
        e = min(s + per, B)
        sur = np.fft.ifft2(amp[s:e, None] * Wf[None], axes=(2, 3)).real         # (e-s, n_perm, H, W) isotropic GRF
        a, _, _ = _polar_batch(sur.reshape((e - s) * n_perm, H, W), Mw, tap, geom)
        out[s:e] = a.reshape(e - s, n_perm, geom["n_rings"])
    return out


# ---------------------------------------------------------------------------
# Window-centre grid orientation coherence + result container
# ---------------------------------------------------------------------------
class OrientationResult:
    """Per-window-scale orientation scan. Arrays are (Wr, Wc, n_rings) over the window-centre grid.

    Attributes: ``window_um``, ``bin_size`` (raster pitch), ``grid`` (Wr, Wc), ``valid`` (Wr, Wc bool),
    ``centers_um`` (Wr, Wc, 2), ``ring_um`` (feature wavelength ≈ 1.41*bin/ring_kfrac, since ``ring_kfrac``
    is a fraction of the grid's corner frequency, about 0.707 cycles per bin), ``ring_kfrac``,
    ``anisotropy``, ``orientation`` (structure, deg), ``coherence``, ``laminarity`` (= aniso*coherence),
    ``p``, ``fdr``.
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"OrientationResult(window_um={self.window_um:.0f}, grid={self.grid}, "
                f"valid={int(self.valid.sum())}/{self.valid.size}, n_rings={len(self.ring_um)})")

    def ring_laminarity(self, ring):
        """(Wr, Wc) laminarity map for one ring, NaN where the window is invalid."""
        lam = self.laminarity[:, :, ring].copy(); lam[~self.valid] = np.nan
        return lam


def _neighbor_coherence(orient_deg, valid):
    """Axial orientation coherence per window: circular concentration (resultant length) of the doubled
    structure-orientation across the 8 window-neighbours. High = neighbours agree (real laminar structure
    bends smoothly); low = random neighbour orientation (finite-sample noise)."""
    from scipy.ndimage import uniform_filter
    Wr, Wc = orient_deg.shape
    if Wr < 2 or Wc < 2:                                    # a degenerate 1xN / Nx1 grid cannot assess coherence
        return np.zeros_like(orient_deg)
    th2 = np.radians(2 * orient_deg); v = valid.astype(float)
    C = np.cos(th2) * v; S = np.sin(th2) * v
    box = lambda A: uniform_filter(A, size=3, mode="constant") * 9  # 3x3 sum via mean*9
    n = box(v) - v                                          # neighbour count (exclude self)
    cs = box(C) - C * v; ss = box(S) - S * v               # neighbour vector sum (exclude self)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.sqrt(cs ** 2 + ss ** 2) / n
    coh[~np.isfinite(coh)] = 0.0
    return coh * valid


def _coherence_batch(O, valid):
    """Neighbour orientation coherence for a BATCH of genes at once. ``O`` (G, Wr, Wc, n_rings), ``valid``
    (Wr, Wc). Identical to :func:`_neighbor_coherence` per (gene, ring) but the 3x3 neighbour sums are one
    ND ``uniform_filter`` over the whole stack instead of a Python loop over genes — the win on
    whole-transcriptome screens."""
    from scipy.ndimage import uniform_filter
    v = valid.astype(float)[None, :, :, None]
    th2 = np.radians(2 * O); C = np.cos(th2) * v; S = np.sin(th2) * v
    box = lambda A: uniform_filter(A, size=(1, 3, 3, 1), mode="constant") * 9      # 3x3 sum over the window grid
    n = uniform_filter(valid.astype(float), size=3, mode="constant")[None, :, :, None] * 9 - v
    cs = box(C) - C * v; ss = box(S) - S * v
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.sqrt(cs ** 2 + ss ** 2) / n
    coh[~np.isfinite(coh)] = 0.0
    return coh * valid.astype(float)[None, :, :, None]


# ---------------------------------------------------------------------------
# Rasterize a scalar field to scan — a SINGLE GENE (the descriptive per-gene mode) or a pooled summary.
# ---------------------------------------------------------------------------
def _build_field(adata, summary, genes, bin_size, basis, layer):
    from .gr import _coords, _matrix, _gene_idx
    from .adaptive import grid_index
    import scipy.sparse as sp
    xy = _coords(adata, basis)
    row, col, H, W, pitch = grid_index(xy, bin_size)
    if summary == "density":
        g = np.zeros((H, W)); np.add.at(g, (row, col), 1.0); mask = g > 0
        return g, mask, pitch, xy.min(0)
    if isinstance(summary, str) and summary in adata.var_names:            # a single gene's expression field
        idx, _ = _gene_idx(adata, summary)
        X = _matrix(adata, layer)
        val = (X[:, idx].toarray().ravel() if sp.issparse(X) else np.asarray(X)[:, idx].ravel()).astype(float)
    else:
        idx, _ = _gene_idx(adata, genes)
        X = _matrix(adata, layer)
        Xg = (X[:, idx].toarray() if sp.issparse(X) else np.asarray(X)[:, idx]).astype(float)
        if summary == "pc1":
            u, s, _ = np.linalg.svd(Xg - Xg.mean(0), full_matrices=False); val = u[:, 0] * s[0]
        elif summary == "mean":
            val = (Xg - Xg.mean(0)).mean(1)
        else:
            raise ValueError(f"summary must be a gene name, 'density', 'pc1', or 'mean', got {summary!r}")
    grid = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(grid, (row, col), val); np.add.at(cnt, (row, col), 1.0)
    mask = cnt > 0; grid[mask] /= cnt[mask]
    return grid, mask, pitch, xy.min(0)


def default_windows(adata, *, n=7, basis="spatial", bins_per_window=DEFAULT_BINS_PER_WINDOW,
                    hi_frac=1 / 4, lo_frac=None, min_bin=None):
    """Log-spaced window-size sweep (coordinate units) — the window-axis analogue of ``gr.default_radii``.

    RANGE, not resolution, is what makes this sweep carry scale. Measured directly: on one dataset at
    one fixed geometry, a 7.28x sweep recovers a planted wavelength with log-log slope 0.67, and the
    same sweep truncated to 3.07x gives 0.35. Independently, across eleven sections, the realised range
    predicts how reproducibly the curve's SHAPE survives splitting the section in half (Spearman +0.83,
    p = 0.002) — the widest sweeps reach 0.7, the narrowest 0.0.

    There is a floor on how short a useful sweep can be. For a given wavelength the span of windows
    that hold it inside the measurement band is about 4.7x, so a sweep shorter than that sits wholly
    in-band and its peak location is decided by noise rather than by the tissue. That figure is derived
    from the band geometry, not tuned.

    So the low end is set by feasibility rather than by a fraction of the tissue. At constant bins per
    window the raster pitch is ``window / bins_per_window``, which cannot go below the cell spacing
    without leaving most bins empty — hence ``lo = bins_per_window x spacing``. Before 2026-08-12 this
    swept ``1/18 .. 1/4`` of the extent, a fixed 4.5x on every section regardless of what the section
    could support.

    Pass ``lo_frac`` to force the old extent-fraction behaviour.

    Parameters
    ----------
    adata : AnnData
        Cell coordinates in ``adata.obsm[basis]``.
    n : int, optional
        Number of window sizes. Default ``7``.
    basis : str, optional
        Key in ``adata.obsm`` holding the coordinates. Default ``"spatial"``.
    bins_per_window : int, optional
        Bins across a window. The smallest window is ``bins_per_window`` times the cell spacing.
        Default ``16``.
    hi_frac : float, optional
        Largest window as a fraction of the shorter side of the coordinate bounding box. Default ``1/4``.
    lo_frac : float, optional
        If given, the smallest window is this fraction of the shorter side instead of the
        spacing-based floor. Default ``None``.
    min_bin : float, optional
        Cell spacing to use, in coordinate units. ``None`` (default) measures it as the median
        nearest-neighbour distance. Ignored when ``lo_frac`` is given.

    Returns
    -------
    tuple of float
        ``n`` window sizes in coordinate units, log-spaced from the smallest to the largest.

    Warns
    -----
    RuntimeWarning
        When the smallest feasible window is not below the largest (the sweep then falls back to
        ``hi / 4.5 .. hi``), and when the sweep spans less than 4.7x.
    """
    from .gr import _coords
    xy = _coords(adata, basis); span = float(min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    hi = span * hi_frac
    if lo_frac is not None:
        lo = span * lo_frac
    else:
        lo = bins_per_window * (float(min_bin) if min_bin is not None else _cell_spacing(xy))
    if hi <= lo:
        warnings.warn(f"tissue extent supports no window sweep at bins_per_window={bins_per_window}: "
                      f"the finest feasible window ({lo:.0f}) is already past {hi_frac:.2f} of the "
                      f"extent ({hi:.0f}). Falling back to a 4.5x sweep below the coarse end; the "
                      f"orientation curve will not resolve scale.", RuntimeWarning, stacklevel=2)
        lo = hi / 4.5
    if hi / lo < 4.7:
        warnings.warn(f"window sweep spans only {hi / lo:.2f}x, below the ~4.7x needed to place a "
                      f"structure's scale: any peak in the curve is inside the band at every window, "
                      f"so its location is noise. Laminarity MAGNITUDE is still meaningful; its scale "
                      f"is not.", RuntimeWarning, stacklevel=2)
    return tuple(np.geomspace(lo, hi, n))


def _cell_spacing(xy, sample=5000, seed=0):
    """Median nearest-neighbour distance (coordinate units) — the physical resolution floor for the raster
    pitch: a bin much finer than this is mostly empty, so windows fail the mask-fraction gate.

    The tree is BUILT ON EVERY CELL and only QUERIED from a subsample. Building it on the subsample
    instead — which this did until 2026-08-07, matching the same defect in ``gr._radius_from_xy`` —
    measures the spacing of a sparser point set and inflates it by about sqrt(N / sample), so the
    raster pitch scaled with cell COUNT rather than with cell spacing and every large section was
    scanned on a coarser grid than intended (4x on an 81,452-cell MERFISH section).
    """
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, float)
    tree = cKDTree(xy)                                       # ALL cells define the density
    q = xy
    if len(xy) > sample:
        q = xy[np.random.default_rng(seed).choice(len(xy), sample, replace=False)]
    d, _ = tree.query(q, k=2)
    return float(np.median(d[:, 1]))


# ---------------------------------------------------------------------------
# Scan one window size across the field (all FFTs batched through the backend)
# ---------------------------------------------------------------------------
def _scan_one(grid, mask, bin_um, origin, *, window_um, overlap, n_theta, n_rings, kmin_frac, kmax_frac,
              taper, mask_frac_min, min_bins, n_perm, seed, pnull="z", sampling="direct"):
    from .diff import bh_fdr
    H, W = grid.shape
    win = max(int(round(window_um / bin_um)), 6)
    step = max(int(round(win * (1 - overlap))), 1)
    r0s = list(range(0, max(H - win, 0) + 1, step)) or [0]
    c0s = list(range(0, max(W - win, 0) + 1, step)) or [0]
    Wr, Wc = len(r0s), len(c0s)
    geom = _geom_for(win, n_theta, n_rings, kmin_frac, kmax_frac, sampling)
    tap = _taper2d(win, win, taper)
    A = np.zeros((Wr, Wc, n_rings)); O = np.zeros((Wr, Wc, n_rings)); Pv = np.ones((Wr, Wc, n_rings))
    valid = np.zeros((Wr, Wc), bool); centers = np.zeros((Wr, Wc, 2))
    ox, oy = origin
    # eligible windows (each has its OWN occupancy mask). We batch ALL windows' observed polar into ONE FFT
    # call, and the whole permutation null (every window × every perm) through the backend in memory-bounded
    # chunks — heterogeneous masks are carried per-item so the GPU/threaded backend runs a handful of big
    # transforms instead of thousands of tiny ones.
    widx = []; fws = []; mws = []
    for i, r0 in enumerate(r0s):
        for j, c0 in enumerate(c0s):
            centers[i, j] = [ox + (c0 + win / 2) * bin_um, oy + (r0 + win / 2) * bin_um]
            fw = grid[r0:r0 + win, c0:c0 + win]; mw = mask[r0:r0 + win, c0:c0 + win]
            if mw.shape != (win, win) or mw.mean() < mask_frac_min or mw.sum() < min_bins:
                continue                                    # gotchas 2+3: mask-fraction + abundance floors
            widx.append((i, j)); fws.append(fw); mws.append(mw); valid[i, j] = True
    if widx:
        Fw = np.stack(fws); Mw = np.stack(mws)                                   # (nW,win,win)
        aobs, sobs, _ = _polar_batch(Fw, Mw, tap, geom)                         # observed, one batched FFT
        rgeom = _radial_geom(win, win)
        nullc = np.empty((len(widx), n_perm, n_rings))                          # (nW, n_perm, n_rings)
        for k in range(len(widx)):                                             # per-window iso-surrogate null,
            nullc[k] = _iso_null(Fw[k], Mw[k], tap, geom, rgeom,               #   seeded seed+1+k (matches the
                                 n_perm=n_perm, seed_k=seed + 1 + k)[0]        #   batched screen -> identity holds)
        # calibrate against the per-window isotropic-surrogate null. Windowed anisotropy is biased UPWARD in
        # small samples (and by the mask aperture), so we subtract the null's own mean; "z" then normalizes by
        # its spread -> a graded p that resolves below the 1/(n_perm+1) permutation floor (needed because BH
        # runs over hundreds of overlapping windows). "perm" keeps the raw permutation-rank p (floored).
        if pnull == "perm":
            pv = (1 + (nullc >= aobs[:, None, :] - 1e-12).sum(1)) / (n_perm + 1)
        else:
            from scipy.stats import norm
            mu = nullc.mean(1); sd = nullc.std(1) + 1e-9
            pv = norm.sf((aobs - mu) / sd)                                       # one-sided parametric-null p
        for k, (i, j) in enumerate(widx):
            A[i, j] = aobs[k]; O[i, j] = sobs[k]; Pv[i, j] = pv[k]
    fdr = np.ones((Wr, Wc, n_rings))
    for ri in range(n_rings):
        pv = Pv[:, :, ri][valid]
        if pv.size:
            fdr[:, :, ri][valid] = bh_fdr(pv)
    coh = np.stack([_neighbor_coherence(O[:, :, ri], valid) for ri in range(n_rings)], -1)
    lam = A * coh
    ring_um = bin_um / np.maximum(geom["ring_kfrac"] * geom["kr_max"], 1e-6)   # feature wavelength (µm)
    return OrientationResult(window_um=float(window_um), bin_size=float(bin_um), grid=(Wr, Wc), valid=valid,
                             centers_um=centers, ring_um=ring_um, ring_kfrac=geom["ring_kfrac"],
                             anisotropy=A, orientation=O, coherence=coh, laminarity=lam, p=Pv, fdr=fdr)


def _scan_chunk(grids, mask, bin_um, origin, *, window_um, overlap, n_theta, n_rings, kmin_frac, kmax_frac,
                taper, mask_frac_min, min_bins, n_perm, seed, pnull="z", sampling="direct"):
    """Batched scan of MANY genes that share one cell grid, mask and window tiling (``grids`` = (H, W, G)).
    Numerically identical to running :func:`_scan_one` per gene — the observed polar and the scramble null are
    batched ACROSS GENES per window (one big backend FFT instead of one per gene), which amortizes the
    per-call launch overhead that dominates whole-transcriptome screens. Returns per-gene arrays
    (G, Wr, Wc, n_rings) + the shared valid/centres/ring_um."""
    from .diff import bh_fdr
    from scipy.stats import norm
    H, W, G = grids.shape
    win = max(int(round(window_um / bin_um)), 6); step = max(int(round(win * (1 - overlap))), 1)
    r0s = list(range(0, max(H - win, 0) + 1, step)) or [0]; c0s = list(range(0, max(W - win, 0) + 1, step)) or [0]
    Wr, Wc = len(r0s), len(c0s)
    geom = _geom_for(win, n_theta, n_rings, kmin_frac, kmax_frac, sampling); tap = _taper2d(win, win, taper)
    rgeom = _radial_geom(win, win)
    A = np.zeros((G, Wr, Wc, n_rings)); O = np.zeros((G, Wr, Wc, n_rings)); Pv = np.ones((G, Wr, Wc, n_rings))
    valid = np.zeros((Wr, Wc), bool); centers = np.zeros((Wr, Wc, 2)); ox, oy = origin
    widx = []
    for i, r0 in enumerate(r0s):
        for j, c0 in enumerate(c0s):
            centers[i, j] = [ox + (c0 + win / 2) * bin_um, oy + (r0 + win / 2) * bin_um]
            mw = mask[r0:r0 + win, c0:c0 + win]
            if mw.shape != (win, win) or mw.mean() < mask_frac_min or mw.sum() < min_bins:
                continue
            widx.append((i, j)); valid[i, j] = True
    for k, (i, j) in enumerate(widx):
        r0, c0 = r0s[i], c0s[j]; mw = mask[r0:r0 + win, c0:c0 + win]
        fw = np.ascontiguousarray(grids[r0:r0 + win, c0:c0 + win, :].transpose(2, 0, 1))   # (G, win, win)
        aobs, sobs, _ = _polar_batch(fw, mw, tap, geom)                          # observed, batched over genes
        A[:, i, j] = aobs; O[:, i, j] = sobs
        # isotropic phase-randomized surrogate null, per-window seed shared across genes (matches _scan_one)
        nulla = _iso_null(fw, mw, tap, geom, rgeom, n_perm=n_perm, seed_k=seed + 1 + k)     # (G, n_perm, n_rings)
        if pnull == "perm":
            Pv[:, i, j] = (1 + (nulla >= aobs[:, None, :] - 1e-12).sum(1)) / (n_perm + 1)
        else:
            Pv[:, i, j] = norm.sf((aobs - nulla.mean(1)) / (nulla.std(1) + 1e-9))
    fdr = np.ones((G, Wr, Wc, n_rings))
    for g in range(G):                                                            # BH-FDR per (gene, ring); cheap
        for ri in range(n_rings):
            pv = Pv[g, :, :, ri][valid]
            if pv.size:
                fdr[g, :, :, ri][valid] = bh_fdr(pv)
    coh = _coherence_batch(O, valid)                                              # coherence batched over genes
    return dict(valid=valid, centers=centers,
                ring_um=bin_um / np.maximum(geom["ring_kfrac"] * geom["kr_max"], 1e-6),
                A=A, O=O, lam=A * coh, fdr=fdr, coh=coh)


def orientation_field(adata, *, summary="pc1", genes=None, bin_size=None,
                      bins_per_window=DEFAULT_BINS_PER_WINDOW, min_bin=None,
                      window_um=None, overlap=0.5, n_theta=36, n_rings=4, kmin_frac=0.15,
                      kmax_frac=INSCRIBED_KMAX,
                      taper="hann", mask_frac_min=0.5, min_bins=None, n_perm=200, pnull="z", basis="spatial",
                      layer=None, seed=0, sampling="direct", bin_rule="constant"):
    """Discover directional (laminar/axial) sub-structure across a tissue, unsupervised, over a window sweep.

    Rasterizes a scalar SUMMARY field (top expression PC, transcript density, or mean), then sweeps tapered
    windows over a log-spaced set of SIZES and returns a calibrated orientation field + laminarity map per
    window scale. Sweeping the window size (rather than fixing it) makes the tool robust to tissues whose
    organization scale is unknown or varies; :func:`aggregate_scales` collapses the sweep to a scale-robust
    per-cell readout.

    Parameters
    ----------
    adata : AnnData
        With ``obsm[basis]`` spatial coordinates.
    summary : str, optional
        Scalar field to scan. A **gene name** in ``var_names`` scans that gene's expression (the descriptive
        per-gene mode; see :func:`gene_orientation_field`). Otherwise a pooled summary: ``"pc1"`` = top PC of
        ``genes`` (dominant expression axis), ``"density"`` = per-bin cell count, ``"mean"`` = mean centred
        expression. Default ``"pc1"``. Expression summaries are averaged over the cells in each bin.
    genes : list of str, optional
        Genes for the ``pc1``/``mean`` summary (default: all). Ignored for a gene name or ``"density"``.
    bin_size : float, optional
        Fixed raster pitch (coordinate units) shared across the sweep. If ``None`` (default) each window size
        is rastered at ``window / bins_per_window`` (``bin_rule="constant"``) or
        ``max(window / bins_per_window, min_bin)`` (``bin_rule="floored"``). With a fixed pitch a window
        spans ``round(window / bin_size)`` bins, at least 6.
    bins_per_window : int, optional
        Bins across a window — held CONSTANT over the sweep, so every window has identical frequency
        geometry and only its physical size changes. Default 16 (was 40 before 2026-08-12).
    min_bin : float, optional
        Raster-pitch floor (coordinate units), used only by ``bin_rule="floored"``. Default
        ``2.5 × median nearest-neighbour spacing``.
    bin_rule : {"constant", "floored"}, optional
        How the raster pitch follows the window. ``"constant"`` (default) uses ``window / bins_per_window``
        throughout, so the window is always ``bins_per_window`` bins across.

        ``"floored"`` is what shipped before 2026-08-12: ``max(window / bins_per_window, min_bin)``. Its
        intent was to stop the raster outrunning the cells, but the effect is that once the floor binds
        the bin can no longer shrink, so the WINDOW shrinks in bins instead — sliding from 40 down to 6
        across a single sweep. The measurement geometry then changes from point to point along the very
        axis the sweep is meant to isolate, and the null floor rides with it: measured on real sections it
        falls 3.2x from the fine end to the coarse end. That ramp is shared by every gene, so per-gene
        z-scoring cannot remove it, and any curve SHAPE read off such a sweep is partly the artefact.
        Kept for reproducing pre-2026-08-12 results, not recommended otherwise.
    sampling : {"direct", "lattice"}, optional
        Polar-spectrum sampling; see :func:`_geom_for`. ``"direct"`` by default. ``"lattice"`` is the
        pre-2026-08-12 path and manufactures direction at small ``bins_per_window``.
    window_um : sequence of float, optional
        The window-size SWEEP (structure scales), in coordinate units. Default :func:`default_windows`
        (7 log-spaced sizes).
    overlap : float, optional
        Window overlap fraction (neighbour coherence needs overlap). Default 0.5. Windows step by
        ``round(window_bins * (1 - overlap))`` bins, at least 1.
    n_rings, n_theta, kmin_frac, kmax_frac, taper
        Frequency-resolved polar-spectrum settings (see :func:`windowed_polar`). Defaults ``4``, ``36``,
        ``0.15``, :data:`INSCRIBED_KMAX` and ``"hann"``; note ``kmin_frac`` defaults higher here than in
        :func:`windowed_polar`.
    mask_frac_min : float, optional
        Minimum within-window occupied fraction. It keeps windows that fall mostly outside the tissue from reporting edge structure. Default 0.5. Windows below it are invalid.
    min_bins : int, optional
        Minimum occupied bins per window, a floor on how many bins the gene occupies. Default ``(window/bin)**2 * 0.15``.
    n_perm : int, optional
        Isotropic surrogate fields drawn per window for the null: each keeps the window's radial power
        spectrum, randomizes its phases, and goes through the same mask and taper. Default 200.
    pnull : {"z", "perm"}, optional
        How the per-window anisotropy is calibrated against the surrogate null. ``"z"`` (default) subtracts the
        null mean (removing the small-sample upward bias) and normalizes by its spread → a graded p that
        resolves below the ``1/(n_perm+1)`` permutation floor, which matters because BH-FDR runs over hundreds
        of overlapping windows. ``"perm"`` uses the raw permutation-rank p (floored, more conservative).
    basis : str, optional
        Key in ``adata.obsm`` holding the coordinates. Default ``"spatial"``.
    layer : str, optional
        Expression from ``adata.layers[layer]``; ``None`` (default) uses ``adata.X``.
    seed : int, optional
        Seed for the surrogate null. The ``k``-th valid window of each window size uses ``seed + 1 + k``.
        Default ``0``.

    Returns
    -------
    dict of float to OrientationResult
        One calibrated scan per window size in the sweep, keyed by window size (coordinate units). Each
        :class:`OrientationResult` holds ``(Wr, Wc, n_rings)`` arrays over the window-centre grid:
        ``anisotropy``, ``orientation`` (degrees in ``[0, 180)``, from the +x axis toward +y),
        ``coherence``, ``laminarity`` (= anisotropy × coherence), ``p`` and ``fdr`` (BH over valid
        windows, per ring); plus ``valid`` ``(Wr, Wc)``, ``centers_um`` ``(Wr, Wc, 2)`` (x, y),
        ``ring_um`` (feature wavelength per ring, coordinate units), ``ring_kfrac``, ``window_um``,
        ``bin_size`` (raster pitch used) and ``grid`` ``(Wr, Wc)``. Invalid windows hold anisotropy 0,
        ``p = 1`` and ``fdr = 1``.

    See Also
    --------
    aggregate_scales : collapse the sweep to a scale-robust per-cell laminarity + characteristic scale.
    segment_regions : threshold + connected-component a laminarity map into sub-regions.
    rank_region_genes : rank genes by directional structure within a discovered region.

    Examples
    --------
    >>> import numpy as np, anndata as ad, spectrast
    >>> rng = np.random.default_rng(0)
    >>> xy = rng.random((800, 2)) * 400
    >>> stripes = np.sin(2 * np.pi * xy[:, 1] / 40.0)                 # horizontal laminae
    >>> a = ad.AnnData(np.c_[stripes, rng.standard_normal(800)].astype("float32"))
    >>> a.obsm["spatial"] = xy; a.var_names = ["lam", "noise"]
    >>> scan = spectrast.orient.orientation_field(
    ...     a, summary="pc1", genes=["lam"], window_um=(120.0, 200.0), n_perm=50)
    >>> len(scan)
    2
    """
    from .gr import _coords
    if window_um is None:
        window_um = default_windows(adata, basis=basis, bins_per_window=bins_per_window)
    window_um = tuple(float(w) for w in window_um)
    if min_bin is None:
        min_bin = 2.5 * _cell_spacing(_coords(adata, basis))            # only used by bin_rule="floored"
    out = {}
    for w in window_um:
        # CONSTANT bins per window: the bin grows with the window and the frequency geometry never moves,
        # so a change along the sweep is a change in the tissue rather than in the measurement. See the
        # bin_rule docstring for what the old floored rule did instead.
        bw = (float(bin_size) if bin_size is not None else
              (w / bins_per_window if bin_rule == "constant" else max(w / bins_per_window, min_bin)))
        grid, mask, bin_um, origin = _build_field(adata, summary, genes, bw, basis, layer)
        mb = min_bins if min_bins is not None else int((w / bin_um) ** 2 * 0.15)
        out[w] = _scan_one(grid, mask, bin_um, origin, window_um=w, overlap=overlap, n_theta=n_theta,
                           n_rings=n_rings, kmin_frac=kmin_frac, kmax_frac=kmax_frac, taper=taper,
                           mask_frac_min=mask_frac_min, min_bins=mb, n_perm=n_perm, pnull=pnull, seed=seed,
                           sampling=sampling)
    return out


def gene_orientation_field(adata, gene, **kw):
    """Orientation scan for ONE gene's expression field (the descriptive per-gene mode): where, at what
    scale, and along which axis does this gene's expression carry directional structure. Thin wrapper over
    :func:`orientation_field` with ``summary=gene`` — returns ``{window_um: OrientationResult}`` whose
    orientation field feeds :func:`orientation_streamlines` for plotting. See :func:`orientation_field`
    for parameters.

    Parameters
    ----------
    adata : AnnData
        With ``obsm[basis]`` spatial coordinates.
    gene : str
        A name in ``adata.var_names``. Its expression is averaged per bin and scanned.
    **kw
        Forwarded to :func:`orientation_field` (``window_um``, ``bin_size``, ``n_perm``, ``layer``,
        ``basis``, ``seed``, ...). ``summary`` is set to ``gene`` and cannot be passed; ``genes`` is
        ignored.

    Returns
    -------
    dict of float to OrientationResult
        As :func:`orientation_field`: one scan per window size, keyed by window size (coordinate units).
    """
    return orientation_field(adata, summary=gene, **kw)


def screen_oriented_genes(adata, genes=None, *, window_um=None, bin_size=None,
                          bins_per_window=DEFAULT_BINS_PER_WINDOW, min_bin=None,
                          n_perm=100, fdr_max=0.05, coherence_min=0.5, min_cells=50, n_theta=36, n_rings=4,
                          kmin_frac=0.15, kmax_frac=INSCRIBED_KMAX, taper="hann", mask_frac_min=0.5, min_bins=None,
                          overlap=0.5, pnull="z", basis="spatial", layer=None, seed=0, gene_chunk=256,
                          sampling="direct", bin_rule="constant"):
    """DISCOVER which genes carry oriented spatial structure — at some scale, at some location.

    The headline descriptive statistic. For each gene it scans the tissue with the windowed orientation
    engine (window-size sweep × frequency rings, calibrated isotropic surrogate null, coherence gate) and
    summarizes whether — and where and at what scale — the gene's expression is directionally organized.
    Purely descriptive: it flags oriented genes, it does not classify tissue regions. General to any
    organized tissue (laminar cortex, fiber tracts, crypt/gland axes, tumour fronts), not just brain.

    A window counts as oriented for a gene when at least one ring has BH-FDR ``< fdr_max`` and
    neighbour coherence ``>= coherence_min``.

    Parameters
    ----------
    adata : AnnData
        With ``obsm[basis]`` spatial coordinates.
    genes : list of str, optional
        Genes to screen (default: all ``var_names``). Either way, genes detected (non-zero) in fewer than
        ``min_cells`` cells are dropped.
    window_um : sequence of float, optional
        Window-size sweep in coordinate units (default :func:`default_windows`). Fewer/coarser sizes =
        faster screen.
    bin_size : float, optional
        Fixed raster pitch shared across the sweep, as in :func:`orientation_field`. ``None`` (default)
        follows ``bin_rule``.
    n_perm : int, optional
        Isotropic surrogate fields per window, as in :func:`orientation_field`. Default ``100``.
    fdr_max : float, optional
        BH-FDR threshold for a window and ring to count as oriented. Default ``0.05``.
    coherence_min : float, optional
        Minimum neighbour orientation coherence (0 to 1) for a window and ring to count. Default ``0.5``.
    min_cells : int, optional
        Detection floor: a gene must be non-zero in at least this many cells. Default ``50``.
    basis, layer, seed
        As in :func:`orientation_field`.
    bins_per_window, min_bin, bin_rule, sampling
        Lattice and sampling settings (as in :func:`orientation_field`). Defaults changed 2026-08-12:
        constant bins per window at 16, direct polar sampling.
    n_theta, n_rings, kmin_frac, kmax_frac, taper, mask_frac_min, min_bins, overlap, pnull
        Scan settings (as in :func:`orientation_field`).
    gene_chunk : int, optional
        Genes rasterized + scanned together (the batching that makes whole-transcriptome screens fast).
        Default ``256``. It changes memory use, not the result.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``frac_oriented``, then ``max_laminarity`` (both descending): one row per gene with columns
        ``gene``, ``max_laminarity`` (largest gated laminarity over every window size, window and ring; 0 if
        none passes), ``frac_oriented`` (fraction of valid windows significantly and coherently oriented,
        pooled over window sizes), ``n_sig_windows``, and the peak signal's ``scale_window_um`` (window
        size), ``feature_um`` (ring wavelength), ``x``, ``y`` (window centre) and ``orientation_deg``
        (real-space axis, degrees from +x toward +y). Lengths are in coordinate units; the peak columns are
        NaN when no window passes. Genes with no oriented structure sort to the bottom
        (``frac_oriented`` ≈ 0).

    Notes
    -----
    All genes in a dataset share the same cell grid, occupancy mask and window tiling, so the observed polar
    and the surrogate null are batched ACROSS GENES (see :func:`_scan_chunk`) — one large backend FFT per
    window instead of one per gene. The result is numerically identical to a per-gene scan.

    See Also
    --------
    gene_orientation_field : the full per-window scan for one discovered gene (feeds streamlines).
    orientation_streamlines : orientation field of a scan result for plotting.
    """
    import pandas as pd, scipy.sparse as sp
    from .gr import _coords, _matrix, _gene_idx
    from .adaptive import grid_index
    idx, names = _gene_idx(adata, genes)
    X = _matrix(adata, layer)
    det = np.asarray((X[:, idx] != 0).sum(0)).ravel() if sp.issparse(X) else (np.asarray(X)[:, idx] != 0).sum(0)
    keep = np.where(det >= min_cells)[0]
    kidx = np.asarray(idx)[keep]; knames = [names[i] for i in keep]; G = len(knames)
    xy = _coords(adata, basis)
    if window_um is None:
        window_um = default_windows(adata, basis=basis, bins_per_window=bins_per_window)
    window_um = tuple(float(w) for w in window_um)
    if min_bin is None:
        min_bin = 2.5 * _cell_spacing(xy)                                # only used by bin_rule="floored"
    best = np.zeros(G); n_sig = np.zeros(G, int); n_val = np.zeros(G, int)
    r_scale = np.full(G, np.nan); r_feat = np.full(G, np.nan); r_x = np.full(G, np.nan)
    r_y = np.full(G, np.nan); r_or = np.full(G, np.nan)
    for w in window_um:
        bw = (float(bin_size) if bin_size is not None else
              (w / bins_per_window if bin_rule == "constant" else max(w / bins_per_window, min_bin)))
        row, col, H, W, pitch = grid_index(xy, bw); origin = xy.min(0)
        flat = row * W + col; cnt = np.bincount(flat, minlength=H * W).astype(float); mask = (cnt > 0).reshape(H, W)
        mb = min_bins if min_bins is not None else int((w / pitch) ** 2 * 0.15)
        for gs in range(0, G, gene_chunk):
            cols = kidx[gs:gs + gene_chunk]
            Xg = (X[:, cols].toarray() if sp.issparse(X) else np.asarray(X)[:, cols]).astype(float)
            grids = np.zeros((H * W, len(cols))); np.add.at(grids, flat, Xg)     # rasterize all chunk genes at once
            nz = cnt > 0; grids[nz] /= cnt[nz][:, None]; grids = grids.reshape(H, W, len(cols))
            res = _scan_chunk(grids, mask, pitch, origin, window_um=w, overlap=overlap, n_theta=n_theta,
                              n_rings=n_rings, kmin_frac=kmin_frac, kmax_frac=kmax_frac, taper=taper,
                              mask_frac_min=mask_frac_min, min_bins=mb, n_perm=n_perm, seed=seed, pnull=pnull,
                              sampling=sampling)
            valid = res["valid"]; nval_w = int(valid.sum())
            passing = (res["fdr"] < fdr_max) & (res["coh"] >= coherence_min)      # (Gc, Wr, Wc, nr)
            nsig_g = (passing.any(-1) & valid).reshape(len(cols), -1).sum(1)
            for lg in range(len(cols)):
                gg = gs + lg
                n_sig[gg] += int(nsig_g[lg]); n_val[gg] += nval_w
                gated = np.where(passing[lg] & valid[..., None], res["lam"][lg], 0.0)
                if gated.size and gated.max() > best[gg]:
                    best[gg] = float(gated.max()); i, j, r = np.unravel_index(gated.argmax(), gated.shape)
                    r_scale[gg] = w; r_feat[gg] = res["ring_um"][r]
                    r_x[gg] = res["centers"][i, j, 0]; r_y[gg] = res["centers"][i, j, 1]
                    r_or[gg] = res["O"][lg, i, j, r]
    df = pd.DataFrame(dict(gene=knames, max_laminarity=best, frac_oriented=n_sig / np.maximum(n_val, 1),
                           n_sig_windows=n_sig, scale_window_um=r_scale, feature_um=r_feat, x=r_x, y=r_y,
                           orientation_deg=r_or))
    if len(df):
        df = df.sort_values(["frac_oriented", "max_laminarity"], ascending=False).reset_index(drop=True)
    return df


def axis_presence(observed, null, *, alpha=0.05, higher=True, n_tested=None, min_n=30):
    """Tissue-level PRESENCE test for one fingerprint axis, from its per-gene statistic and a scramble null.

    Answers "does this tissue exhibit this organization class at all?" by comparing the observed per-gene axis
    statistic against the axis's OWN spatial-scramble null. The presence p is a **rank-based one-sided shift
    test** (Mann-Whitney): is the observed per-gene statistic stochastically larger than the scramble null?
    This replaces an earlier binomial-vs-fixed-quantile test that treated each gene as an independent
    Bernoulli exceedance of a NOISY finite-sample threshold — that test was anti-conservative for panels below
    ~300 genes (empirically ~18-21% false-present at n=12-40 despite a nominal 5%). The rank test is calibrated
    at any panel size and is robust to inter-gene correlation. General to any axis: scale (peak Moran),
    focality (local-Moran Gini), orientation (fraction-of-tissue oriented), etc. For an axis where MORE
    structure means a LOWER statistic (e.g. a spectral-centroid 'texture'), pass ``higher=False``.

    ``score`` (fraction of genes above the null's ``(1-alpha)`` quantile) and ``enrichment`` (vs the
    ``alpha·n_tested`` expected) remain as descriptive effect sizes, but ``presence_p`` no longer derives from
    them. Compare axes ACROSS tissues by a panel-independent effect size (a high percentile of the per-gene
    statistic), NOT by ``score`` — the count saturates on large panels. Below ``min_n`` tested genes the call
    is flagged ``underpowered`` and ``present`` is forced False.

    Parameters
    ----------
    observed, null : array-like, shape (n_genes,)
        Per-gene axis statistic on the real data and on a spatial scramble of it (same panel, density, mask).
    alpha : float
        Per-gene false-positive rate used for the descriptive ``score``/``enrichment``. Default 0.05.
    higher : bool
        True if structure = HIGH statistic (scale/focality/orientation); False for a texture centroid.
    n_tested : int, optional
        Genes tested (the power figure). Default: number of finite ``observed`` values.
    min_n : int
        Minimum tested genes for a ``present`` call; below it the result is ``underpowered``. Default 30.

    Returns
    -------
    dict: ``n_tested, n_sig, n_expected, enrichment, presence_p`` (Mann-Whitney one-sided), ``score``
    (= n_sig/n_tested), ``underpowered``, ``present`` (``presence_p < 0.05`` and ``enrichment >= 2`` and not
    underpowered).

    Examples
    --------
    >>> import numpy as np, spectrast
    >>> rng = np.random.default_rng(0)
    >>> obs = np.r_[rng.normal(3, 1, 40), rng.normal(0, 1, 160)]   # 40 structured genes
    >>> null = rng.normal(0, 1, 200)
    >>> r = spectrast.orient.axis_presence(obs, null)
    >>> r["present"], r["n_sig"] > r["n_expected"]
    (True, True)
    """
    from scipy.stats import mannwhitneyu
    o = np.asarray(observed, float); nl = np.asarray(null, float)
    o = o[np.isfinite(o)]; nl = nl[np.isfinite(nl)]
    if not higher:
        o, nl = -o, -nl
    n = int(o.size) if n_tested is None else int(n_tested)
    tau = float(np.nanquantile(nl, 1 - alpha)) if nl.size else np.inf
    n_sig = int(np.sum(o > tau)); exp = alpha * max(n, 0)
    enr = n_sig / max(exp, 1e-9)
    if o.size >= 3 and nl.size >= 3 and (np.ptp(o) > 0 or np.ptp(nl) > 0):
        p = float(mannwhitneyu(o, nl, alternative="greater").pvalue)      # calibrated at any n; tie-robust
    else:
        p = 1.0
    underpowered = n < min_n
    return dict(n_tested=n, n_sig=n_sig, n_expected=round(exp, 2), enrichment=round(enr, 3),
                presence_p=p, score=n_sig / max(n, 1), underpowered=bool(underpowered),
                present=bool(p < 0.05 and enr >= 2 and not underpowered))


def orientation_presence(adata, *, window_um=None, n_perm=100, alpha=0.05, effect_pct=95,
                         basis="spatial", layer=None, seed=0, return_screens=False, **screen_kw):
    """Tissue-level ORIENTATION presence + panel-independent effect size, in one call.

    Screens the observed data and a COORDINATE-SCRAMBLE of it (shuffling cell positions breaks spatial
    structure at matched panel and density), then applies :func:`axis_presence` to the per-gene fraction of
    tissue each gene is coherently oriented over. Also returns the ``effect_pct`` percentile of that fraction —
    the panel-INDEPENDENT quantity to compare orientation across datasets of different panel size.

    Returns the :func:`axis_presence` dict plus ``effect_size`` (percentile of frac_oriented, e.g. p95). With
    ``return_screens=True`` also returns the observed and null screen DataFrames. ``screen_kw`` passes through
    to :func:`screen_oriented_genes` (genes, min_cells, gates, ...).

    The scramble permutes the rows of ``adata.obsm[basis]`` on a copy of ``adata``, so memory use is about
    twice that of ``adata``.

    Parameters
    ----------
    adata : AnnData
        With ``obsm[basis]`` spatial coordinates.
    window_um : sequence of float, optional
        Window-size sweep in coordinate units, used for both screens. ``None`` (default) uses
        :func:`default_windows`.
    n_perm : int, optional
        Isotropic surrogate fields per window in each screen. Default ``100``.
    alpha : float, optional
        Per-gene false-positive rate for the descriptive ``score`` and ``enrichment`` of
        :func:`axis_presence`. Default ``0.05``. The ``present`` call uses ``presence_p < 0.05``
        whatever ``alpha`` is.
    effect_pct : float, optional
        Percentile (0 to 100) of the observed per-gene ``frac_oriented`` reported as ``effect_size``.
        Default ``95``.
    basis, layer
        As in :func:`orientation_field`.
    seed : int, optional
        Seeds the coordinate scramble and both screens' surrogate nulls. Default ``0``.
    return_screens : bool, optional
        Also return the two screen tables. Default ``False``.
    **screen_kw
        Forwarded to :func:`screen_oriented_genes` for both screens. ``window_um``, ``n_perm``,
        ``basis``, ``layer`` and ``seed`` are set here and cannot be passed again.

    Returns
    -------
    dict
        The :func:`axis_presence` keys (``n_tested``, ``n_sig``, ``n_expected``, ``enrichment``,
        ``presence_p``, ``score``, ``underpowered``, ``present``) computed on per-gene
        ``frac_oriented``, observed against scrambled (a gene missing from the scrambled screen counts as
        0); plus ``effect_size`` (float). With ``return_screens=True`` also ``screen`` and
        ``null_screen``, the :func:`screen_oriented_genes` tables for the observed and scrambled data.
        Fewer than 30 screened genes gives ``underpowered=True`` and ``present=False``.
    """
    from .gr import _coords
    xy = _coords(adata, basis)
    scr = adata.copy(); scr.obsm[basis] = xy[np.random.default_rng(seed).permutation(len(xy))]
    so = screen_oriented_genes(adata, window_um=window_um, n_perm=n_perm, basis=basis, layer=layer, seed=seed, **screen_kw)
    sn = screen_oriented_genes(scr, window_um=window_um, n_perm=n_perm, basis=basis, layer=layer, seed=seed, **screen_kw)
    obs = so.set_index("gene")["frac_oriented"]
    null = sn.set_index("gene")["frac_oriented"].reindex(obs.index).fillna(0.0)
    out = axis_presence(obs.to_numpy(), null.to_numpy(), alpha=alpha, higher=True)
    out["effect_size"] = float(np.nanpercentile(obs.to_numpy(), effect_pct))
    if return_screens:
        out["screen"], out["null_screen"] = so, sn
    return out


def orientation_streamlines(result, *, ring="max", fdr_max=0.05, coherence_min=0.5):
    """Orientation field of a scan result as an evenly-gridded vector field for ``matplotlib.streamplot``.

    Returns ``dict(X, Y, U, V, speed)``: ``X`` (Wc,) and ``Y`` (Wr,) are the evenly-spaced window-centre
    coordinates; ``U, V`` (Wr, Wc) are the unit orientation vectors (0 where a window is not significantly +
    coherently oriented, so no streamline is drawn there); ``speed`` (Wr, Wc) is the laminarity, for line
    width/colour. Axial orientation is mapped to the upper half-plane so streamlines flow consistently.

    Parameters
    ----------
    result : OrientationResult
        One window size of a scan, e.g. ``scan[w]`` from :func:`gene_orientation_field` or
        :func:`orientation_field`.
    ring : int or "max", optional
        Frequency ring to draw. ``"max"`` (default) takes, per window, the ring with the largest passing
        laminarity. An integer selects that ring index (0 = lowest frequency).
    fdr_max : float, optional
        A window is drawn only if its BH-FDR is below this. Default ``0.05``.
    coherence_min : float, optional
        A window is drawn only if its neighbour coherence is at least this. Default ``0.5``.

    Returns
    -------
    dict
        ``X`` ``(Wc,)`` and ``Y`` ``(Wr,)`` window-centre coordinates (coordinate units); ``U``, ``V``
        ``(Wr, Wc)`` unit vectors of the structure orientation, 0 where a window does not pass or is
        invalid; ``speed`` ``(Wr, Wc)`` the laminarity of the drawn ring, 0 where nothing is drawn.

    Examples
    --------
    Plotting::

        s = spectrast.orient.orientation_streamlines(gene_scan[window])
        plt.streamplot(s["X"], s["Y"], s["U"], s["V"], color=s["speed"], density=1.2)
    """
    if ring == "max":
        passing = (result.fdr < fdr_max) & (result.coherence >= coherence_min)
        gated = np.where(passing & result.valid[..., None], result.laminarity, 0.0)
        rbest = gated.argmax(2)
        ii, jj = np.indices(result.grid)
        ang = result.orientation[ii, jj, rbest]; lam = gated.max(2)
    else:
        passing = (result.fdr[:, :, ring] < fdr_max) & (result.coherence[:, :, ring] >= coherence_min) & result.valid
        ang = result.orientation[:, :, ring]; lam = np.where(passing, result.laminarity[:, :, ring], 0.0)
    on = lam > 0
    th = np.radians(ang % 180.0)                                # axial -> upper half-plane (sin >= 0)
    U = np.where(on, np.cos(th), 0.0); V = np.where(on, np.sin(th), 0.0)
    X = result.centers_um[0, :, 0]; Y = result.centers_um[:, 0, 1]
    return dict(X=X, Y=Y, U=U, V=V, speed=lam)


# ---------------------------------------------------------------------------
# Scale aggregation (the window-axis "characteristic scale") -> per-cell, AnnData-native
# ---------------------------------------------------------------------------
def aggregate_scales(scan, adata=None, *, fdr_max=0.05, coherence_min=0.5, basis="spatial", key="orient"):
    """Collapse the window-size sweep to a SCALE-ROBUST readout: per location, take the window scale of
    maximal calibrated laminarity. Answers "the organization scale may differ across tissue" directly.

    For each window size, laminarity is gated (FDR < ``fdr_max`` AND coherence >= ``coherence_min``, best
    ring per window). Each cell is assigned its nearest valid window centre at every scale; across scales the
    cell keeps the maximal laminarity, with the winning window size (characteristic organization scale) and
    that window's orientation.

    Parameters
    ----------
    scan : dict of float to OrientationResult
        Output of :func:`orientation_field` or :func:`gene_orientation_field`.
    adata : AnnData
        The AnnData the scan was built from; required (``None`` raises ``ValueError``). Results are
        written to ``obs[f"{key}_laminarity"]``, ``obs[f"{key}_scale_um"]``, ``obs[f"{key}_orientation"]``
        and returned.
    fdr_max : float, optional
        A window and ring count only if their BH-FDR is below this. Default ``0.05``.
    coherence_min : float, optional
        A window and ring count only if their neighbour coherence is at least this. Default ``0.5``.
    basis : str, optional
        Key in ``adata.obsm`` holding the cell coordinates; use the one the scan was built on. Default
        ``"spatial"``.
    key : str, optional
        Prefix of the three ``adata.obs`` columns written. Default ``"orient"``.

    Returns
    -------
    dict
        Per-cell arrays of length ``n_obs``: ``laminarity``, ``scale_um`` (winning window size, coordinate
        units), ``orientation`` (deg), ``fdr`` (of the winning window/ring). A cell is matched to the nearest
        valid window centre at each scale, with no distance limit. Cells with no passing window keep
        ``laminarity = 0``, ``scale_um`` and ``orientation`` NaN, and ``fdr = 1``. ``fdr`` is not written to
        ``adata.obs``.
    """
    from scipy.spatial import cKDTree
    from .gr import _coords
    if adata is None:
        raise ValueError("pass adata to map the aggregated sweep onto cells")
    xy = _coords(adata, basis); N = xy.shape[0]
    best_lam = np.zeros(N); best_scale = np.full(N, np.nan); best_orient = np.full(N, np.nan)
    best_fdr = np.ones(N)
    for w, res in scan.items():
        if not res.valid.any():
            continue
        vg = res.valid
        gated = np.where((res.fdr < fdr_max) & (res.coherence >= coherence_min), res.laminarity, 0.0)
        ring_best = gated.argmax(2)                                     # best ring index per window
        ii, jj = np.indices(res.grid)
        lam2d = gated.max(2)                                            # best-ring laminarity per window
        or2d = res.orientation[ii, jj, ring_best]
        fdr2d = res.fdr[ii, jj, ring_best]
        centers = res.centers_um[vg]; lamv = lam2d[vg]; orv = or2d[vg]; fdrv = fdr2d[vg]
        tree = cKDTree(centers)
        _, nn = tree.query(xy)
        take = lamv[nn] > best_lam
        best_lam[take] = lamv[nn][take]; best_scale[take] = float(w)
        best_orient[take] = orv[nn][take]; best_fdr[take] = fdrv[nn][take]
    res = dict(laminarity=best_lam, scale_um=best_scale, orientation=best_orient, fdr=best_fdr)
    if adata is not None:
        adata.obs[f"{key}_laminarity"] = best_lam
        adata.obs[f"{key}_scale_um"] = best_scale
        adata.obs[f"{key}_orientation"] = best_orient
    return res


# ---------------------------------------------------------------------------
# Two-level output: (1) region discovery, (2) per-region gene ranking
# ---------------------------------------------------------------------------
def segment_regions(result, *, ring="max", fdr_max=0.05, coherence_min=0.5, min_windows=3):
    """Unsupervised sub-region discovery from a laminarity map.

    A window is 'laminar' if its anisotropy clears the null (BH-FDR < ``fdr_max``) AND its orientation is
    spatially coherent (``coherence >= coherence_min``) — gating on BOTH. Coherent laminar windows are
    grouped into connected components (8-connectivity) of ``>= min_windows``.

    Parameters
    ----------
    result : OrientationResult
    ring : int or "max"
        Frequency ring to segment on; ``"max"`` uses, per window, the ring of strongest passing laminarity.
    fdr_max, coherence_min, min_windows : gates.

    Returns
    -------
    dict: ``labels`` (Wr, Wc int; 0 = background), ``n_regions``, ``gate`` (Wr, Wc bool).
    """
    from scipy import ndimage
    if ring == "max":
        passing = (result.fdr < fdr_max) & (result.coherence >= coherence_min)
        gate = (passing & (result.laminarity > 0)).any(2) & result.valid
    else:
        gate = ((result.fdr[:, :, ring] < fdr_max) & (result.coherence[:, :, ring] >= coherence_min)
                & result.valid)
    labels, n = ndimage.label(gate, structure=np.ones((3, 3)))
    out = np.zeros_like(labels); keep = 0
    for lab in range(1, n + 1):
        if (labels == lab).sum() >= min_windows:
            keep += 1; out[labels == lab] = keep
    return dict(labels=out, n_regions=keep, gate=gate)


def region_box_um(result, seg, region):
    """Bounding box (xmin, xmax, ymin, ymax) in coordinate units for a discovered region label.

    The box spans the region's window centres, padded by half a window on each side.

    Parameters
    ----------
    result : OrientationResult
        The scan that was segmented.
    seg : dict
        Output of :func:`segment_regions` on ``result``.
    region : int
        Region label in ``seg["labels"]`` (1 to ``seg["n_regions"]``; 0 is background). A label not
        present raises ``ValueError``.

    Returns
    -------
    tuple of float
        ``(xmin, xmax, ymin, ymax)`` in coordinate units, ready to pass as ``box`` to
        :func:`rank_region_genes`.
    """
    sel = seg["labels"] == region
    if not sel.any():
        raise ValueError(f"region {region} not found")
    cx = result.centers_um[:, :, 0][sel]; cy = result.centers_um[:, :, 1][sel]
    half = result.window_um / 2.0
    return float(cx.min() - half), float(cx.max() + half), float(cy.min() - half), float(cy.max() + half)


def rank_region_genes(adata, box, *, genes=None, bin_size=None, n_theta=36, n_rings=4, kmin_frac=0.15,
                      kmax_frac=INSCRIBED_KMAX, taper="hann", n_perm=200, basis="spatial", layer=None, seed=0,
                      min_cells=50):
    """Rank genes by directional structure WITHIN a region (answers 'which genes carry the structure, here').

    Crops ``adata`` to the box, rasterizes each gene, and scores it by max-over-rings anisotropy vs the
    isotropic surrogate null (z, permutation p, BH-FDR), reporting the peak ring's feature scale and
    orientation. Genes below ``min_cells`` detections in the region are dropped (abundance floor). Each
    gene's null is one batched FFT through the backend.

    The whole crop is one window, binned on the FFT lattice (as in :func:`windowed_polar`). The ring is
    chosen on the observed anisotropy, and ``z`` and ``p`` compare it with the null at that ring only.

    Parameters
    ----------
    adata : AnnData
        With ``obsm[basis]`` spatial coordinates.
    box : tuple of float
        ``(xmin, xmax, ymin, ymax)`` in coordinate units, bounds included, e.g. from
        :func:`region_box_um`.
    genes : list of str, optional
        Genes to rank. ``None`` (default) uses all ``var_names``.
    bin_size : float, optional
        Raster pitch in coordinate units. ``None`` (default) uses the longer box side / 24.
    n_theta, n_rings, kmin_frac, kmax_frac, taper
        As in :func:`windowed_polar`. ``kmin_frac`` defaults to ``0.15`` here.
    n_perm : int, optional
        Isotropic surrogate fields per gene. Default ``200``.
    basis, layer
        As in :func:`orientation_field`.
    seed : int, optional
        Seed for the surrogate null; gene ``i`` of ``genes`` uses ``seed + i``. Default ``0``.
    min_cells : int, optional
        Genes non-zero in fewer than this many cells inside the box are dropped. Default ``50``.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``z`` desc, one row per kept gene: ``gene``, ``anisotropy`` (largest over rings), ``z``
        and ``p`` (against the null at that ring; ``p`` is ``(1 + #null >= observed) / (n_perm + 1)``),
        ``fdr`` (BH over the returned genes), ``ring_um`` (feature wavelength of that ring, coordinate units),
        ``orientation_deg`` (structure orientation, degrees from +x toward +y) and ``n_detect`` (non-zero
        cells in the box). Empty if no gene passes ``min_cells``.
    """
    import pandas as pd, scipy.sparse as sp
    from .gr import _coords, _matrix, _gene_idx
    from .adaptive import grid_index
    from .diff import bh_fdr
    xy = _coords(adata, basis); x0, x1, y0, y1 = box
    inb = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    sub = adata[inb]; xs = _coords(sub, basis)
    if bin_size is None:
        bin_size = max((x1 - x0), (y1 - y0)) / 24.0
    row, col, H, W, pitch = grid_index(xs, bin_size)
    idx, names = _gene_idx(sub, genes)
    X = _matrix(sub, layer)
    Xg = X[:, idx].toarray() if sp.issparse(X) else np.asarray(X)[:, idx]
    geom = _polar_geom(H, W, n_theta, n_rings, kmin_frac, kmax_frac); tap = _taper2d(H, W, taper)
    rgeom = _radial_geom(H, W)
    rows = []
    for gi, name in enumerate(names):
        v = np.asarray(Xg[:, gi], float); nd = int((v != 0).sum())
        if nd < min_cells:
            continue
        grid = np.zeros((H, W)); cnt = np.zeros((H, W))
        np.add.at(grid, (row, col), v); np.add.at(cnt, (row, col), 1.0)
        mask = cnt > 0; grid[mask] /= cnt[mask]
        a, s, _ = _polar_batch(grid, mask, tap, geom)
        null = _iso_null(grid, mask, tap, geom, rgeom, n_perm=n_perm, seed_k=seed + gi)[0]   # (n_perm, n_rings)
        ri = int(np.argmax(a)); an = null[:, ri]
        rows.append(dict(gene=name, anisotropy=float(a[ri]),
                         z=float((a[ri] - an.mean()) / (an.std() + 1e-12)),
                         p=float((1 + (an >= a[ri] - 1e-12).sum()) / (n_perm + 1)),
                         ring_um=float(pitch / max(geom["ring_kfrac"][ri] * geom["kr_max"], 1e-6)),
                         orientation_deg=float(s[ri]), n_detect=nd))
    df = pd.DataFrame(rows)
    if len(df):
        df["fdr"] = bh_fdr(df["p"].to_numpy())
        df = df.sort_values("z", ascending=False).reset_index(drop=True)
    return df

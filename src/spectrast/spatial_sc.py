"""Single-cell (point-cloud) spatial-statistics core -- the numeric implementations behind spectrast.spatial's
``*_genes`` / ``*_sc`` entry points, for Xenium / MERFISH cell coordinates.

This is where the per-gene statistics live: Moran's I (global / local / scalogram), Geary's C (global /
local), Getis-Ord Gi*, focality, the bivariate matrix, correlation length, and the variogram. Each takes a
``reduced_rank`` FLAG that selects one of two computation paths for the SAME statistic:

  reduced_rank=False  (default)  compute each gene DIRECTLY on its own field -- full rank.
  reduced_rank=True              compute via a rank-k SVD reconstruction of the gene.

The reduced-rank path exploits that every quadratic/bilinear local statistic is a pointwise function of
NEIGHBOURHOOD REDUCTIONS of a gene field: a gene's rank-k reconstruction ``x_g = E~ a`` (a = loadings V[g,:],
E~ = centred embeddings) makes the statistic reconstruct EXACTLY from a couple of small k x k matrices computed
once -- no fitted model -- giving (a) a speed-up on the gene axis (k neighbourhood convolutions instead of G)
and (b) DENOISING (the rank-k gene borrows strength across genes, suppressing per-gene noise/sparsity). It is a
MODE, not the module's purpose -- most tools default to full rank. Numerically verified
identity (rank-k stat == direct stat on the rank-k gene).

Neighbourhood convolutions reuse the FFT engine (via :mod:`spectrast.adaptive`), composing with the adaptive
rasterization + fusion for single-cell data. Re-exported through ``spectrast.spatial``. Heavy deps (scipy, sklearn)
are imported lazily.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------- SVD embeddings
def svd_embeddings(X, rank, center=True, seed=0):
    """Top-``rank`` SVD embeddings of the (N, G) expression matrix.

    Returns dict with ``Ec`` (N,k centred embeddings E~ = U Σ − mean), ``V`` (G,k loadings),
    ``sigma`` (k,), ``mean`` (G,), ``rank``.
    """
    from sklearn.utils.extmath import randomized_svd  # lazy
    import scipy.sparse as sp  # lazy
    if sp.issparse(X):
        # memory-safe path for wide sparse matrices (e.g. whole-transcriptome ~18k genes): do NOT
        # densify. Randomised range-finder SVD of the CENTRED matrix A = X - 1·mean^T, with the mean
        # applied implicitly through mat-mats so it is never materialised. Only N×(k+p) and G×(k+p)
        # intermediates — matches the dense centred SVD but stays fast + low-memory.
        Xc = X.tocsr(); Nn, G = Xc.shape
        rank = int(min(rank, min(Nn - 1, G)))   # cell-centring drops 1 dim along N only → cap = min(N-1, G)
        mean = np.asarray(Xc.mean(0)).ravel().astype(np.float32) if center else np.zeros(G, np.float32)
        ones = np.ones((Nn, 1), np.float32)

        def amat(B):    # A @ B = X@B - 1·(mean·B)
            return np.asarray(Xc @ B) - ones @ (mean @ B)[None, :]

        def atmat(B):   # A^T @ B = X^T@B - mean·(1·B)
            return np.asarray(Xc.T @ B) - np.outer(mean, B.sum(0))

        rng = np.random.default_rng(seed); l = min(rank + 10, G)
        Q = amat(rng.standard_normal((G, l)).astype(np.float32))
        for _ in range(4):                        # power iterations for accuracy
            Q, _ = np.linalg.qr(Q)
            Q, _ = np.linalg.qr(atmat(Q))
            Q = amat(Q)
        Q, _ = np.linalg.qr(Q)                    # (N, l) orthonormal range
        Bm = atmat(Q).T                           # (l, G) = Q^T A
        Ub, s, Vt = np.linalg.svd(Bm, full_matrices=False)
        U = Q @ Ub[:, :rank]; s = s[:rank]; Vt = Vt[:rank]
        E = U * s
        return dict(Ec=E - E.mean(0), V=Vt.T, sigma=s, mean=mean, rank=rank)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    rank = int(min(rank, min(X.shape[0] - 1, X.shape[1])))   # cap = min(N-1, G); centring drops 1 dim on N
    mean = X.mean(0) if center else np.zeros(X.shape[1])
    U, s, Vt = randomized_svd(X - mean, n_components=rank, random_state=seed)
    E = U * s
    return dict(Ec=E - E.mean(0), V=Vt.T, sigma=s, mean=mean, rank=rank)


# ----------------------------------------------------------------- neighbourhood reductions
def _neighbor_sum_count(xy, fields, radius_um, bin_size):
    """(neighbour SUM per field (N,K), neighbour COUNT (N,)) via the FFT engine, self excluded."""
    from .adaptive import _fft_neighbor_reduce, autotune_grid  # lazy
    xy = np.asarray(xy, float)
    F = np.asarray(fields, float)
    if F.ndim == 1:
        F = F[:, None]
    if bin_size is None:
        bin_size = autotune_grid(xy, radius_um)["bin_um"]   # field-INDEPENDENT probe: the bin is a
        # deterministic function of (xy, radius) so the direct and reduced-rank paths (and every gene)
        # share ONE rasterization — otherwise the reduced-rank "exact identity" breaks under autotune.
    nsum, ncnt = _fft_neighbor_reduce(xy, F, radius_um, bin_size)
    return nsum, ncnt, float(bin_size)


def _standardize_cols(M):
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    mu = M.mean(0); sd = M.std(0) + 1e-9
    return (M - mu) / sd


def _embed_neighbor_mean(xy, Ec, radius_um, bin_size, fuse=False, tol=0.1, tree=None):
    """Cell-weighted neighbour MEAN of the k embedding columns (N, k). ``fuse=True`` composes the
    THIRD axis: the FFT reduction is corrected on the rasterization-sensitive minority of cells with
    the exact graph (flagged by the bin-b vs bin-b/2 proxy on the embeddings) — so the reconstructed
    per-gene statistics inherit single-cell exactness. Returns (Mbar (N,k), bin_size, flagged_frac)."""
    from .adaptive import _fft_neighbor_reduce, _exact_neighbor_reduce, autotune_grid  # lazy
    xy = np.asarray(xy, float); Ec = np.asarray(Ec, float)
    if bin_size is None:
        bin_size = autotune_grid(xy, radius_um)["bin_um"]   # field-independent (see _neighbor_sum_count)
    nsum, ncnt = _fft_neighbor_reduce(xy, Ec, radius_um, bin_size)
    Mbar = nsum / np.maximum(ncnt, 1)[:, None]
    flagged = 0
    if fuse:
        n2, c2 = _fft_neighbor_reduce(xy, Ec, radius_um, bin_size / 2.0)
        Mbar2 = n2 / np.maximum(c2, 1)[:, None]
        sd = Mbar.std(0) + 1e-12                          # per-embedding-column scale
        proxy = (np.abs(Mbar - Mbar2) / sd).mean(1)       # mean relative bin-b vs bin-b/2 disagreement
        fidx = np.where(proxy > tol)[0]; flagged = fidx.size
        if fidx.size:
            nx, cx = _exact_neighbor_reduce(xy, Ec, radius_um, sub_idx=fidx, tree=tree)
            Mbar[fidx] = nx / np.maximum(cx, 1)[:, None]
    return Mbar, float(bin_size), flagged / max(len(xy), 1)


def _dense_gene_subset(X, genes):
    """Column-slice the requested ``genes`` while X is still SPARSE, then densify only that subset ->
    (N, len(genes)) dense. Avoids materialising the full (N × G) matrix to read a few columns (the Atera
    densification bug). Guards the subset densify against the live RAM budget. Dense X is already in RAM,
    so it is only fancy-indexed (no guard, no whole-panel copy)."""
    import scipy.sparse as sp  # lazy
    if sp.issparse(X):
        from .memory import guard_densify  # lazy
        Xg = X[:, genes]
        guard_densify(Xg)
        return np.asarray(Xg.todense(), float)
    Xd = np.asarray(X, float)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    return Xd[:, genes]


def _col_block_dense(X, s, n):
    """Densify a CONTIGUOUS column block ``X[:, s:s+n]`` (sparse -> dense subset, dense -> view/copy).
    For whole-transcriptome functions that stream 256-gene blocks so the full matrix is never dense."""
    import scipy.sparse as sp  # lazy
    blk = X[:, s:s + n]
    return np.asarray(blk.todense(), float) if sp.issparse(blk) else np.asarray(blk, float)


# ----------------------------------------------------------------- Moran's I (mean-neighbour form)
def morans_global(xy, X, radius_um, *, reduced_rank=False, rank=30, bin_size=None, svd=None,
                  fuse=False, tol=0.1, tree=None):
    """Global Moran's I (mean-neighbour form) per gene: ``I_g = mean_n z_g[n]·meanN(z_g)[n]``.

    ``X`` : (N, G) expression (or (N,) single gene). ``reduced_rank=False`` computes each gene
    directly; ``reduced_rank=True`` runs an SVD (rank ``rank``) and reconstructs every gene's I
    exactly from the k×k embedding matrices. ``fuse=True`` (with ``reduced_rank``) composes the
    single-cell adaptive fusion into the embedding convolutions (exact rasterization). Returns (G,).
    """
    import scipy.sparse as sp  # lazy
    sparse = sp.issparse(X)
    if not sparse:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    N = len(xy)
    if not reduced_rank:
        return _direct_global(xy, X, radius_um, bin_size)[0]   # densifies per gene-chunk (sparse-safe)
    sv = svd or svd_embeddings(X, rank)   # svd_embeddings keeps sparse X sparse
    Ec, V = sv["Ec"], sv["V"]
    Mbar, _, _ = _embed_neighbor_mean(xy, Ec, radius_um, bin_size, fuse=fuse, tol=tol, tree=tree)
    Q = (Ec.T @ Mbar) / N                                  # k×k numerator
    S = (Ec.T @ Ec) / N                                    # k×k variance normalizer
    num = np.einsum("gk,kl,gl->g", V, Q, V)
    var = np.einsum("gk,kl,gl->g", V, S, V)
    return num / np.maximum(var, 1e-12)


def morans_local(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Per-cell local Moran maps for the selected ``genes`` (indices into X's columns).

    Returns (N, len(genes)). ``reduced_rank=True`` reconstructs each map from the rank-k gene
    (denoised); ``reduced_rank=False`` uses the actual gene. Same neighbourhood operator either way.
    """
    from .memory import gene_chunk  # lazy
    genes = np.atleast_1d(genes); N = len(xy)
    if not reduced_rank:                                   # direct: slice while sparse, densify subset only
        z = _standardize_cols(_dense_gene_subset(X, genes))
        nsum, ncnt, _ = _neighbor_sum_count(xy, z, radius_um, bin_size)
        meanN = nsum / np.maximum(ncnt, 1)[:, None]
        return z * meanN
    sv = svd or svd_embeddings(X, rank)                    # sparse-safe (no densify)
    Ec, V = sv["Ec"], sv["V"]
    A = V[genes]                                           # (m, k)
    nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, radius_um, bin_size)
    Mbar = nsum / np.maximum(ncnt, 1)[:, None]
    S = (Ec.T @ Ec) / N
    var = np.einsum("mk,kl,ml->m", A, S, A)                # rank-k gene variance
    from .memory import guard_shape  # lazy
    guard_shape((N, len(genes)))                           # M4: explicit error if the (N x m) output can't fit RAM
    out = np.empty((N, len(genes)))                        # built in memory-sized gene chunks
    cs = gene_chunk(N)
    for s in range(0, len(genes), cs):
        a = A[s:s + cs]
        out[:, s:s + cs] = (Ec @ a.T) * (Mbar @ a.T) / np.maximum(var[s:s + cs], 1e-12)
    return out


def _gini(v):
    """Gini coefficient of the non-negative part of a vector (focality of a local map)."""
    v = np.sort(np.clip(np.asarray(v, float), 0, None)); n = len(v)
    if n == 0:
        return 0.0
    c = np.cumsum(v)
    return float((n + 1 - 2 * (c / c[-1]).sum()) / n) if c[-1] > 0 else 0.0


def focality(xy, X, radius_um, genes=None, *, reduced_rank=True, rank=30, bin_size=None, svd=None):
    """Per-gene FOCALITY = Gini of the local Moran map (few tight hotspots -> high; diffuse -> low) -> (m,).

    Scales to whole-transcriptome panels automatically: with ``reduced_rank`` (the default) each gene's
    local map is a rank-k matrix product, and we STREAM over memory-sized gene chunks (from the live RAM
    budget) applying the Gini reduction per gene — the full (N × genes) map is never materialised. Gini is
    scale-invariant, so the per-gene normalisation is irrelevant and omitted. ``genes=None`` profiles all.
    """
    import scipy.sparse as sp  # lazy
    from .memory import gene_chunk  # lazy
    if bin_size is None:
        from .adaptive import autotune_grid  # lazy
        bin_size = float(autotune_grid(xy, radius_um)["bin_um"])
    G = X.shape[1]
    genes = np.arange(G) if genes is None else np.atleast_1d(genes)
    N = len(xy)
    if not reduced_rank:                                   # direct: keep sparse, densify one gene-chunk
        Xs = X.tocsc() if sp.issparse(X) else np.asarray(X, float)   # at a time -> WTA-scale memory
        if not sp.issparse(Xs) and Xs.ndim == 1:
            Xs = Xs[:, None]
        out = np.empty(len(genes)); cs = gene_chunk(N)
        for s in range(0, len(genes), cs):
            sub = Xs[:, genes[s:s + cs]]
            sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub, float)
            L = morans_local(xy, sub, radius_um, np.arange(sub.shape[1]), reduced_rank=False, bin_size=bin_size)
            out[s:s + L.shape[1]] = [_gini(L[:, j]) for j in range(L.shape[1])]
        return out
    sv = svd or svd_embeddings(X, rank)                    # sparse-safe
    Ec, V = sv["Ec"], sv["V"]
    nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, radius_um, bin_size)
    Mbar = nsum / np.maximum(ncnt, 1)[:, None]             # embedding neighbour-means (computed once)
    A = V[genes]; out = np.empty(len(genes)); cs = gene_chunk(N)
    for s in range(0, len(genes), cs):
        a = A[s:s + cs]
        L = (Ec @ a.T) * (Mbar @ a.T)                      # (N, chunk) local Moran (up to per-gene scale)
        for j in range(a.shape[0]):
            out[s + j] = _gini(L[:, j])
    return out


# ----------------------------------------------------------------- Getis-Ord Gi* (hotspots)
def getis_local(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Per-cell Getis-Ord Gi* hotspot z-scores for selected ``genes`` (N, len(genes)).

    Reduced-rank reconstructs the standardized neighbourhood sum of the rank-k gene."""
    from .memory import gene_chunk  # lazy
    genes = np.atleast_1d(genes); N = len(xy)
    if not reduced_rank:                                   # direct: slice while sparse, densify subset only
        z = _standardize_cols(_dense_gene_subset(X, genes))
        nsum, ncnt, _ = _neighbor_sum_count(xy, z, radius_um, bin_size)
        nsum = nsum + z; ncnt = ncnt + 1.0                 # Gi* INCLUDES the focal cell (was Gi)
        den = np.sqrt(np.maximum((N * ncnt - ncnt ** 2) / max(N - 1, 1), 1e-9))[:, None]
        return nsum / den
    sv = svd or svd_embeddings(X, rank)                    # sparse-safe (no densify)
    Ec, V = sv["Ec"], sv["V"]
    A = V[genes]
    nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, radius_um, bin_size)
    ncnt = ncnt + 1.0                                      # Gi* includes the focal cell
    S = (Ec.T @ Ec) / N
    sg = np.sqrt(np.maximum(np.einsum("mk,kl,ml->m", A, S, A), 1e-12))   # rank-k gene std
    den = np.sqrt(np.maximum((N * ncnt - ncnt ** 2) / max(N - 1, 1), 1e-9))[:, None]
    from .memory import guard_shape  # lazy
    guard_shape((N, len(genes)))                           # M4: explicit error if the (N x m) output can't fit RAM
    out = np.empty((N, len(genes)))                        # memory-sized gene chunks
    cs = gene_chunk(N)
    for s in range(0, len(genes), cs):
        a = A[s:s + cs]
        gval = Ec @ a.T                                    # the focal cell's own rank-k gene value
        out[:, s:s + cs] = ((nsum @ a.T + gval) / sg[None, s:s + cs]) / den
    return out


# ----------------------------------------------------------------- LOSH (local variance heterogeneity)
def losh(xy, X, radius_um, genes, *, a=2.0, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Per-cell LOSH — local spatial heteroscedasticity (Ord & Getis 2012) — for selected ``genes``.

    ``H_i = Σ_j w_ij |e_j|^a / (h_1 Σ_j w_ij)`` with ``e_j = x_j - x̄_j`` the residual against the
    neighbour mean AT j and ``h_1`` its global mean: > 1 where the neighbourhood is noisier than the
    tissue average, < 1 where it is locally smooth. Two neighbour reductions on the same radius.

    Unlike Moran and Getis, LOSH is not linear in the field, so the reduced-rank path cannot ride the
    embedding sums — it reconstructs the rank-k gene per chunk and applies the same formula (exact for
    that gene, as in :func:`local_geary`). No standardization: LOSH is invariant to affine rescaling.
    A gene with no residual variance (absent, or constant) is NaN for every cell: H is 0/0 there
    (:data:`spectrast.spatial.LOSH_RTOL`, changed in 0.13.2). Returns (N, len(genes))."""
    from .adaptive import autotune_grid  # lazy
    from .memory import gene_chunk, guard_shape  # lazy
    genes = np.atleast_1d(genes); N = len(xy)
    if bin_size is None:
        bin_size = float(autotune_grid(xy, radius_um)["bin_um"])   # field-independent (see _neighbor_sum_count)

    def _losh_block(Xg):
        nsx, ncnt = _nsum_chunked(xy, Xg, radius_um, bin_size)
        inv = 1.0 / np.maximum(ncnt, 1)[:, None]
        e = np.abs(Xg - nsx * inv) ** a                            # residual vs the neighbour mean at j
        e[ncnt < 0.5] = 0.0                                        # islands have no local mean
        nse, _ = _nsum_chunked(xy, e, radius_um, bin_size)
        h1 = e.mean(0)
        H = (nse * inv) / np.where(h1 == 0, np.nan, h1)
        if len(Xg):                                        # no residual variance: 0/0 (spatial.LOSH_RTOL)
            from .spatial import LOSH_RTOL  # lazy
            H[:, h1 <= (LOSH_RTOL * np.abs(Xg).max(0)) ** a] = np.nan
        return H

    if not reduced_rank:                                   # direct: slice while sparse, densify subset only
        return _losh_block(_dense_gene_subset(X, genes))
    sv = svd or svd_embeddings(X, rank)
    Ec, V = sv["Ec"], sv["V"]; A = V[genes]
    guard_shape((N, len(genes)))                           # explicit error if the (N x m) output can't fit RAM
    out = np.empty((N, len(genes)))
    cs = gene_chunk(N)
    for s in range(0, len(genes), cs):
        out[:, s:s + cs] = _losh_block(Ec @ A[s:s + cs].T)         # (N, cs) rank-k genes
    return out


# ----------------------------------------------------------------- local Geary's C (boundary detector)
def _nsum_chunked(xy, fields, radius_um, bin_size, col_chunk=128):
    """Neighbour SUM of many field columns, chunked over columns to bound grid memory.
    Returns (nsum (N, C), ncnt (N,))."""
    from .adaptive import _fft_neighbor_reduce  # lazy
    fields = np.asarray(fields, float)
    if fields.ndim == 1:
        fields = fields[:, None]
    N, C = fields.shape
    out = np.empty((N, C)); ncnt = None
    for s in range(0, C, col_chunk):
        ns, nc = _fft_neighbor_reduce(xy, fields[:, s:s + col_chunk], radius_um, bin_size)
        out[:, s:s + ns.shape[1]] = ns; ncnt = nc
    return out, ncnt


def local_geary(xy, X, radius_um, genes, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Per-cell local Geary's C for selected ``genes`` — a boundary/transition detector (high where a
    gene changes abruptly between neighbours). Standardized form ``c_i = Σ_j w_ij (z_i - z_j)²``.

    Reduced rank reconstructs it EXACTLY from the rank-k gene — but unlike Moran/Getis it needs the k²
    neighbour-sums of the embedding products ``Ẽ_e ⊙ Ẽ_e'`` (its per-cell Σ_j w_ij z_j² term is quadratic
    in the field). Returns (N, len(genes)). Streams over gene chunks: peak memory ~ N·chunk, not N·k².
    """
    from .adaptive import autotune_grid  # lazy
    from .memory import gene_chunk  # lazy
    genes = np.atleast_1d(genes); N = len(xy)
    if bin_size is None:
        bin_size = float(autotune_grid(xy, radius_um)["bin_um"])
    if not reduced_rank:                                   # direct: slice while sparse, densify subset only
        z = _standardize_cols(_dense_gene_subset(X, genes))
        nsz, ncnt = _nsum_chunked(xy, z, radius_um, bin_size)
        nsz2, _ = _nsum_chunked(xy, z ** 2, radius_um, bin_size)
        return ncnt[:, None] * z ** 2 - 2.0 * z * nsz + nsz2
    sv = svd or svd_embeddings(X, rank)
    Ec, V = sv["Ec"], sv["V"]; A = V[genes]                           # (m, k)
    # aᵀQa (neighbour-sum of x², with Q the k×k neighbour-summed embedding outer-products) equals the
    # neighbour-sum of x²=(Ec a)² directly — so we skip the N×k² product Q (a multi-GB spike at HD scale).
    # STREAM over memory-sized gene chunks so the N×m field arrays never all exist at once (else a whole-
    # transcriptome call holds ~5·N·G and OOMs); each chunk holds only ~5·N·cs.
    S = (Ec.T @ Ec) / N
    var = np.einsum("mk,kl,ml->m", A, S, A)                           # per-gene variance σ_g²
    from .memory import guard_shape  # lazy
    guard_shape((N, len(genes)))                           # M4: explicit error if the (N x m) output can't fit RAM
    out = np.empty((N, len(genes)))
    cs = gene_chunk(N)
    for s in range(0, len(genes), cs):
        Xg = Ec @ A[s:s + cs].T                                       # (N, cs) rank-k genes (centred)
        nsx, ncnt = _nsum_chunked(xy, Xg, radius_um, bin_size)        # neighbour sum of x
        nsx2, _ = _nsum_chunked(xy, Xg ** 2, radius_um, bin_size)     # neighbour sum of x²
        out[:, s:s + cs] = (ncnt[:, None] * Xg ** 2 - 2.0 * Xg * nsx + nsx2) / np.maximum(var[s:s + cs], 1e-12)
    return out


# ----------------------------------------------------------------- direct baseline (chunked)
def concordance(xy, X, radius_um, *, rank=None, bin_size=None, n_split=3, z=2.0, seed=0, genes=None,
                shift_tol=0.1):
    """Trust layer for reduced-rank spatial statistics: report the DIRECT and reduced-rank Moran's I
    per gene side by side and flag where they meaningfully disagree — so users know when to believe the
    reconstructed value (the ALRA/MAGIC adoption problem).

    Reduced rank systematically RAISES Moran's I (it removes noise attenuation), so absolute values differ
    almost everywhere — the trustworthy signal is the RANKING (which genes are spatially organised), which
    reduced rank largely preserves and recovers more accurately. So we flag on RANK MOVEMENT, not value: per
    gene we report I_direct, I_rr, detection, and the percentile-rank shift ``rank_shift`` (rr minus direct).
    A gene is ``concordant`` if ``|rank_shift| <= shift_tol`` (reduced rank agrees on the call → trust the
    ranking). Otherwise:
      * ``rr_promoted`` — reduced rank ranks it much higher (a low-detection gene here = candidate denoising
        rescue; a high-detection gene = inspect, could be borrowed structure),
      * ``rr_demoted``  — reduced rank ranks it lower (inspect; reduced rank discounts it).

    Returns a DataFrame (gene, I_direct, I_rr, detection, rank_direct, rank_rr, rank_shift, concordant,
    category). ``shift_tol`` is in percentile units (default 0.1 = 10 points). ``z``/``n_split`` retained
    for a per-gene direct SE (``se_direct``) reported for context.
    """
    import pandas as pd  # lazy
    import scipy.sparse as sp  # lazy
    from scipy.stats import rankdata  # lazy
    from .adaptive import autotune_grid  # lazy
    if genes is not None:                                    # slice while sparse, densify the subset only
        genes = np.atleast_1d(genes)
        X = _dense_gene_subset(X, genes)
    elif not sp.issparse(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    # else: sparse whole panel -> KEEP SPARSE (M5). morans_global / autotune_rank stream over gene blocks, so
    # we never materialize the full (N × 18k) dense matrix here (guard_densify let it through on a big-RAM node
    # and the downstream halves then made further ~50 GB copies).
    N, G = X.shape
    if bin_size is None:
        bin_size = float(autotune_grid(xy, radius_um)["bin_um"])
    if rank is None:
        rank = autotune_rank(xy, X, radius_um, bin_size=bin_size)["rank"]
    I_direct = morans_global(xy, X, radius_um, reduced_rank=False, bin_size=bin_size)
    I_rr = morans_global(xy, X, radius_um, reduced_rank=True, rank=rank, bin_size=bin_size)
    detection = np.asarray((X > 0).mean(0)).ravel()          # sparse- and dense-safe per-gene detection rate
    rng = np.random.default_rng(seed)
    halves = [morans_global(xy[i], X[i], radius_um, reduced_rank=False, bin_size=bin_size)
              for i in (rng.permutation(N)[:N // 2] for _ in range(n_split))]
    se = np.std(halves, axis=0) + 1e-9
    r_dir = (rankdata(I_direct) - 1) / max(G - 1, 1)          # 0..1 percentile rank
    r_rr = (rankdata(I_rr) - 1) / max(G - 1, 1)
    shift = r_rr - r_dir
    concordant = np.abs(shift) <= shift_tol
    cat = np.where(concordant, "concordant", np.where(shift > 0, "rr_promoted", "rr_demoted"))
    idx = np.arange(G) if genes is None else np.asarray(genes)
    return pd.DataFrame(dict(gene=idx, I_direct=I_direct, I_rr=I_rr, detection=detection,
                             se_direct=se, rank_direct=r_dir, rank_rr=r_rr, rank_shift=shift,
                             concordant=concordant, category=cat))


def _grid_gene_chunk(xy, bin_um, cap=256):
    """Gene-block size that bounds the (H·W, block) rasterized-grid FFT stack by the live RAM budget (M3).

    The peak of the direct/scalogram paths is the grid stack, whose H·W scales with the tissue extent / bin
    (not with N or gene count), so a fine ``bin_size`` override on a large section can blow it up. Capping the
    column block by ``memory.gene_chunk(H·W)`` keeps the stack bounded regardless of the bin."""
    from .adaptive import grid_index  # lazy
    from .memory import gene_chunk  # lazy
    HW = int(np.prod(grid_index(np.asarray(xy, float), float(bin_um))[2:4]))
    return int(min(cap, gene_chunk(HW)))


def _direct_global(xy, X, radius_um, bin_size, chunk=256):
    """Direct per-gene global Moran (mean-neighbour form), gene-chunked to bound memory. Sparse X stays
    sparse (CSC) and is densified one gene-chunk at a time, so peak memory is X + one dense chunk + the
    (H·W, chunk) grid stack (chunk bounded by the RAM budget via :func:`_grid_gene_chunk`)."""
    import scipy.sparse as sp  # lazy
    from .adaptive import autotune_grid  # lazy
    sparse = sp.issparse(X)
    if sparse:
        X = X.tocsc()
    elif X.ndim == 1:
        X = np.asarray(X, float)[:, None]
    N, G = X.shape
    if bin_size is None:                                  # autotune ONCE (field-independent bin), not per chunk
        bin_size = float(autotune_grid(np.asarray(xy, float), radius_um)["bin_um"])
    chunk = min(chunk, _grid_gene_chunk(xy, bin_size, chunk))    # M3: bound the (H·W, chunk) grid stack by RAM
    out = np.empty(G)
    for s in range(0, G, chunk):
        xc = X[:, s:s + chunk]
        z = _standardize_cols(xc.toarray() if sparse else xc)
        nsum, ncnt, _ = _neighbor_sum_count(xy, z, radius_um, bin_size)
        meanN = nsum / np.maximum(ncnt, 1)[:, None]
        out[s:s + z.shape[1]] = (z * meanN).mean(0)
    return out, bin_size


def _pick_bin(xy, radii, bin_size):
    """Fix ONE grid (shared across radii) via autotune at the smallest radius."""
    from .adaptive import autotune_grid  # lazy
    if bin_size is not None:
        return float(bin_size)
    return float(autotune_grid(np.asarray(xy, float), float(np.min(radii)))["bin_um"])


def _radius_bins(xy, radii, bin_size):
    """ONE convolution grid bin PER radius — the kernel size is chosen for each scale, not fixed.

    If ``bin_size`` is given it is used at every radius (user override). If ``None`` (default), each
    radius autotunes its own bin (~r/5, honouring the r/b≥5 rasterization envelope): fine grids for
    small radii (accuracy), coarse grids for large radii (speed) — instead of forcing the finest grid
    on every radius, which over-resolves the large ones (e.g. r/b≈40) and is needlessly slow.
    """
    from .adaptive import autotune_grid  # lazy
    radii = np.atleast_1d(radii).astype(float)
    if bin_size is None:                                    # auto: one autotuned kernel per radius (the
        xy = np.asarray(xy, float)                          # per-radius bin isn't a linear function of r —
        return [float(autotune_grid(xy, float(r))["bin_um"]) for r in radii]   # the occupancy cap makes it
        # r/b≈5 at small radii but finer at large — so we tune each; autotune_grid itself is made cheap
        # (occupancy-first candidate ordering) so the sweep no longer pays the fine-bin proxy per radius.
    b = np.atleast_1d(bin_size).astype(float)
    if len(b) == len(radii):                               # explicit PER-RADIUS bins (precomputed once)
        return [float(x) for x in b]
    return [float(b[0])] * len(radii)                      # scalar -> same bin at every radius


def _trunc(sv, k):
    """Truncate a precomputed SVD dict to rank k (reuse one SVD across candidate ranks)."""
    return dict(Ec=sv["Ec"][:, :k], V=sv["V"][:, :k], sigma=sv["sigma"][:k], mean=sv["mean"], rank=k)


def gavish_donoho_rank(X, max_cells=8000, seed=0):
    """Signal rank via the Gavish–Donoho optimal singular-value hard threshold (unknown noise,
    median rule). Fast heuristic from the singular spectrum (all singular values via the small Gram).
    Sparse input is subsampled to ``max_cells`` and densified (the signal spectrum is subsample-robust),
    so it never materialises the full sparse matrix."""
    import scipy.sparse as sp  # lazy
    if sp.issparse(X):
        if X.shape[0] > max_cells:
            X = X[np.random.default_rng(seed).choice(X.shape[0], max_cells, replace=False)]
        Xf = np.asarray(X.todense(), float)
    else:
        Xf = np.asarray(X, float)                          # once (was asarray'd twice)
    Xc = Xf - Xf.mean(0)
    m, n = Xc.shape
    ev = np.linalg.eigvalsh(Xc.T @ Xc if n <= m else Xc @ Xc.T)
    s = np.sqrt(np.clip(ev, 0, None))[::-1]                 # singular values, descending
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return int(max((s > omega * np.median(s)).sum(), 1))


def autotune_rank(xy, X, radius_um, *, candidate_ranks=None, bin_size=None,
                  method="reproducibility", frac=0.99, max_rank=64, seed=0):
    """Data-driven reduced rank for spatial statistics.

    ``method='reproducibility'`` (default, principled): split the cells in half and take the SMALLEST k
    whose split-half Spearman of per-gene Moran's I reaches ``frac`` of the maximum (an elbow rule —
    parsimony, robust to the noisy plateau real data shows; on a clean low-rank signal it coincides with
    the peak). Needs NO ground truth (reproducibility tracks accuracy; validated on the simulator). Also
    returns the Gavish–Donoho matrix-signal-rank estimate. ``method='mp'`` returns only Gavish–Donoho.

    ``max_rank`` caps the chosen (and candidate) rank — for spatial statistics a rank above ~70 is
    overkill (a handful of dozen spatial components suffice); the returned ``rank`` never exceeds it.

    Returns dict: ``rank``, ``candidate_ranks``, ``reproducibility`` (per candidate), ``gd_rank``.
    """
    import scipy.sparse as sp  # lazy
    if not sp.issparse(X):                                 # keep sparse X sparse (svd_embeddings + morans
        X = np.asarray(X, float)                           # are sparse-aware; only densify a dense 1-D vec)
        if X.ndim == 1:
            X = X[:, None]
    kmax_data = min(min(X.shape) - 1, int(max_rank))
    if candidate_ranks is None:
        candidate_ranks = [k for k in (2, 4, 8, 12, 16, 24, 32, 48, 64) if k <= kmax_data]
    else:
        candidate_ranks = [k for k in candidate_ranks if k <= kmax_data]
    gd = min(gavish_donoho_rank(X), int(max_rank))
    if method == "mp":
        return dict(rank=gd, candidate_ranks=None, reproducibility=None, gd_rank=gd, method="mp")
    from scipy.stats import spearmanr  # lazy
    bin_size = _pick_bin(xy, [radius_um], bin_size)
    rng = np.random.default_rng(seed); h = rng.permutation(len(xy))
    A, B = h[:len(xy) // 2], h[len(xy) // 2:]
    kmax = max(candidate_ranks)
    svA, svB = svd_embeddings(X[A], kmax), svd_embeddings(X[B], kmax)
    repro = []
    for k in candidate_ranks:
        mA = morans_global(xy[A], X[A], radius_um, reduced_rank=True, bin_size=bin_size, svd=_trunc(svA, k))
        mB = morans_global(xy[B], X[B], radius_um, reduced_rank=True, bin_size=bin_size, svd=_trunc(svB, k))
        repro.append(float(spearmanr(mA, mB).correlation))
    rep = np.asarray(repro)
    best = int(candidate_ranks[int(np.argmax(rep >= frac * rep.max()))])   # smallest k reaching frac*max
    return dict(rank=best, candidate_ranks=list(candidate_ranks), reproducibility=repro,
                gd_rank=gd, method="reproducibility")


# ================================================================= (1) SCALE SPECTRUM (scalogram)
def morans_scalogram(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Moran's I at each radius for every gene -> (n_radii, G) scalogram. ``reduced_rank=True``
    reconstructs every gene's whole scalogram from k embedding convolutions per radius (denoised)."""
    import scipy.sparse as sp  # lazy
    sparse = sp.issparse(X)
    if not sparse:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    elif not reduced_rank:            # keep sparse (CSC) and densify per chunk -> WTA-scale memory
        X = X.tocsc()
    radii = np.atleast_1d(radii).astype(float); R, G = len(radii), X.shape[1]
    bins = _radius_bins(xy, radii, bin_size)         # one kernel/bin per radius (auto ~r/5 if None)
    out = np.zeros((R, G))
    if not reduced_rank:
        # gene-chunk OUTER (densify + standardize each 256-col block ONCE, reuse across radii) so peak
        # memory is sparse X + one dense chunk, never a full dense copy -> the exact batched FFT scales to
        # whole-transcriptome panels (e.g. 700k cells x 18k genes). Per-column standardization is
        # gene-independent, so this is numerically identical to standardizing X upfront.
        cb = 256 if bin_size is None else _grid_gene_chunk(xy, bin_size, 256)  # M3: bound grid stack on bin override
        for s in range(0, G, cb):
            xc = X[:, s:s + cb]
            z = _standardize_cols(xc.toarray() if sparse else xc)
            for ri, r in enumerate(radii):
                nsum, ncnt, _ = _neighbor_sum_count(xy, z, r, bins[ri])
                meanN = nsum / np.maximum(ncnt, 1)[:, None]
                out[ri, s:s + z.shape[1]] = (z * meanN).mean(0)
        return out
    sv = svd or svd_embeddings(X, rank); Ec, V = sv["Ec"], sv["V"]; N = len(xy)
    S = (Ec.T @ Ec) / N; varg = np.einsum("gk,kl,gl->g", V, S, V)
    for ri, r in enumerate(radii):
        nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, r, bins[ri])
        Mbar = nsum / np.maximum(ncnt, 1)[:, None]
        Q = (Ec.T @ Mbar) / N
        out[ri] = np.einsum("gk,kl,gl->g", V, Q, V) / np.maximum(varg, 1e-12)
    return out


def correlation_length(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None, frac=0.5):
    """Per-gene characteristic spatial scale (radius where Moran's I falls to ``frac`` of its
    smallest-radius value), from the (optionally reduced-rank/denoised) scalogram. Returns (G,)."""
    from .spatial import correlation_length as _clen  # lazy
    sc = morans_scalogram(xy, X, radii, reduced_rank=reduced_rank, rank=rank, bin_size=bin_size)
    return _clen(sc, np.atleast_1d(radii).astype(float), frac=frac)


# ================================================================= (2) ALL-PAIRS GENE-GENE CROSS-CORR
def bivariate_matrix(xy, X, genes, radius_um, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Symmetric spatial cross-correlation (bivariate Moran) between all pairs of ``genes`` -> (m, m).

    ``reduced_rank=True`` reconstructs the whole matrix from one k×k embedding matrix as
    ``V Q V^T`` (normalized) -- the O(G^2) pairwise computation collapses to O(m^2 k), denoised.
    """
    genes = np.atleast_1d(genes); N = len(xy)
    if not reduced_rank:                                   # direct: slice while sparse, densify subset only
        z = _standardize_cols(_dense_gene_subset(X, genes))
        nsum, ncnt, _ = _neighbor_sum_count(xy, z, radius_um, bin_size)
        meanN = nsum / np.maximum(ncnt, 1)[:, None]
        C = (z.T @ meanN) / N
        return (C + C.T) / 2
    sv = svd or svd_embeddings(X, rank); Ec, V = sv["Ec"], sv["V"]
    A = V[genes]
    nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, radius_um, bin_size)
    Mbar = nsum / np.maximum(ncnt, 1)[:, None]
    Q = (Ec.T @ Mbar) / N; Qs = (Q + Q.T) / 2; S = (Ec.T @ Ec) / N
    num = A @ Qs @ A.T
    d = np.sqrt(np.maximum(np.einsum("mk,kl,ml->m", A, S, A), 1e-12))
    return num / np.outer(d, d)


# ================================================================= (3) GEARY'S C (global)
def general_g_global(xy, X, radius_um, *, bin_size=None):
    """Global Getis-Ord General G per gene on cells -> (G,) (Getis & Ord 1992).

    ``G = Σ_i x_i Σ_{j≠i} w_ij x_j / Σ_i Σ_{j≠i} x_i x_j`` with BINARY weights within ``radius_um`` --
    the form ``esda.G`` computes on a binary distance band. The numerator's neighbour sums come from the
    same FFT neighbour reduction as every other single-cell statistic here (self excluded); the
    denominator needs no neighbourhood and is exact, ``(Σ x)² − Σ x²``.

    The field is deliberately NOT centred: G is defined on the values themselves and reads "do HIGH
    values cluster" only for a non-negative variable, so negative input is refused rather than
    returning a number with no interpretation (cf. :func:`spectrast.spatial.general_g_multiscale`).
    """
    import scipy.sparse as sp  # lazy
    if bin_size is None:                                  # autotune ONCE (field-independent bin)
        from .adaptive import autotune_grid
        bin_size = float(autotune_grid(np.asarray(xy, float), radius_um)["bin_um"])
    if sp.issparse(X):
        X = X.tocsc()
    else:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    G = X.shape[1]
    out = np.empty(G)
    cb = _grid_gene_chunk(xy, bin_size, 256)              # bound the grid stack (M3)
    for s in range(0, G, cb):
        x = _col_block_dense(X, s, cb)
        if (x < -1e-9).any():
            raise ValueError("General G is defined on non-negative values; centre nothing and "
                             "pass expression, not z-scores")
        nsum, _ncnt, _ = _neighbor_sum_count(xy, x, radius_um, bin_size)
        num = (x * nsum).sum(0)                           # Σ_i x_i Σ_{j≠i, d_ij<=r} x_j
        den = x.sum(0) ** 2 - (x ** 2).sum(0)             # Σ_i Σ_{j≠i} x_i x_j
        out[s:s + x.shape[1]] = num / np.where(den > 0, den, np.nan)
    return out


def geary_global(xy, X, radius_um, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Global Geary's C per gene -> (G,); low = strong autocorrelation.

    ROW-STANDARDIZED weights (each cell's neighbours weighted 1/deg), consistent with the mean-neighbour
    Moran here and with squidpy (so C ≈ 1 − I). For that convention
    ``C = (N-1)/(2N) · Σ_i[z_i² − 2 z_i·meanN(z_i) + meanN(z_i²)] / Σ_i z_i²`` — note the neighbour mean of
    z² is needed, so the statistic is not a plain bilinear form. Reduced-rank reconstructs it from the
    embedding via k×k forms (the meanN(z²) term uses the neighbour-summed inverse-degree column weights).
    """
    import scipy.sparse as sp  # lazy
    N = len(xy)
    if bin_size is None:                                  # autotune ONCE (field-independent bin) up front,
        from .adaptive import autotune_grid               # not per 256-gene chunk / per neighbour-sum call
        bin_size = float(autotune_grid(np.asarray(xy, float), radius_um)["bin_um"])
    if not reduced_rank:                                  # keep sparse (CSC); densify + standardize per chunk
        if sp.issparse(X):
            X = X.tocsc()
        else:
            X = np.asarray(X, float)
            if X.ndim == 1:
                X = X[:, None]
        G = X.shape[1]; out = np.empty(G)
        cb = 256 if bin_size is None else _grid_gene_chunk(xy, bin_size, 256)  # M3: bound grid stack on bin override
        for s in range(0, G, cb):
            z = _standardize_cols(_col_block_dense(X, s, cb)); K = z.shape[1]
            nsum, ncnt, _ = _neighbor_sum_count(xy, np.concatenate([z, z ** 2], 1), radius_um, bin_size)
            has_nb = (ncnt > 0)[:, None]                       # a neighbourless cell has 0 pairs → 0 contribution
            inv = (1.0 / np.maximum(ncnt, 1))[:, None]
            meanN_z = nsum[:, :K] * inv; meanN_z2 = nsum[:, K:] * inv
            num = ((z ** 2 - 2.0 * z * meanN_z + meanN_z2) * has_nb).sum(0)   # Σ_i Σ_j w_rs_ij (z_i−z_j)²
            S0 = float((ncnt > 0).sum())                      # row-standardized total weight = cells w/ a neighbour
            # S0 = 0 (no cell has a neighbour within the radius): undefined, not a ZeroDivisionError
            out[s:s + K] = (((N - 1) / (2.0 * S0)) if S0 > 0 else np.nan) * num / np.maximum((z ** 2).sum(0), 1e-12)
        return out
    sv = svd or svd_embeddings(X, rank); Ec, V = sv["Ec"], sv["V"]
    nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, radius_um, bin_size)
    has_nb = ncnt > 0                                        # gate neighbourless cells out of the numerator
    inv = 1.0 / np.maximum(ncnt, 1)
    meanN_Ec = nsum * inv[:, None]                           # D^-1 W Ec (neighbourless rows already 0)
    csum, _, _ = _neighbor_sum_count(xy, (inv * has_nb)[:, None], radius_um, bin_size)  # colsum over cells w/ nb
    Ecg = Ec[has_nb]
    A_num = Ecg.T @ Ecg                                      # Σ_{has_nb} z_i²   (numerator, gated)
    B = Ecg.T @ meanN_Ec[has_nb]                             # Σ_{has_nb} z_i·(D^-1 W z)_i
    Mc = (Ec * csum).T @ Ec                                  # Σ_{has_nb} (D^-1 W z²)_i  (via the gated colsum)
    A = Ec.T @ Ec                                            # Σ_all z²  (variance denominator, NOT gated)
    num = np.einsum("gk,kl,gl->g", V, A_num - 2.0 * B + Mc, V)   # row-standardized (z_i−z_j)² sum, rank-k gene
    den = np.einsum("gk,kl,gl->g", V, A, V)                  # Σ z^2
    S0 = float(has_nb.sum())
    if S0 == 0:                                              # no cell has a neighbour: undefined, not 1/0
        return np.full(np.shape(num), np.nan)
    return (N - 1) / (2.0 * S0) * num / np.maximum(den, 1e-12)


# ================================================================= (4) VARIOGRAM (semivariance)
def variogram(xy, X, radii, *, reduced_rank=False, rank=30, bin_size=None, svd=None):
    """Disc-based semivariance gamma(r) = sill(variance) - covariance(r) per gene -> (n_radii, G).
    Reduced-rank removes the noise nugget (rank-k gene sheds unstructured variance)."""
    import scipy.sparse as sp  # lazy
    sparse = sp.issparse(X)
    if not sparse:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
    elif not reduced_rank:                                 # direct path: keep sparse (CSC), densify per chunk
        X = X.tocsc()
    radii = np.atleast_1d(radii).astype(float); R = len(radii)
    G = svd["V"].shape[0] if (svd is not None and reduced_rank) else X.shape[1]
    bins = _radius_bins(xy, radii, bin_size)         # one kernel/bin per radius (auto ~r/5 if None)
    out = np.zeros((R, G))
    if not reduced_rank:                             # chunk OUTER: densify+centre each block once, reuse radii
        cb = 256 if bin_size is None else _grid_gene_chunk(xy, bin_size, 256)  # M3: bound grid stack on bin override
        for s in range(0, G, cb):
            xc = _col_block_dense(X, s, cb)
            xc = xc - xc.mean(0); v = xc.var(0)
            for ri, r in enumerate(radii):
                nsum, ncnt, _ = _neighbor_sum_count(xy, xc, r, bins[ri])
                meanN = nsum / np.maximum(ncnt, 1)[:, None]
                out[ri, s:s + xc.shape[1]] = v - (xc * meanN).mean(0)
        return out
    sv = svd or svd_embeddings(X, rank); Ec, V = sv["Ec"], sv["V"]; N = len(xy)
    S = (Ec.T @ Ec) / N; varg = np.einsum("gk,kl,gl->g", V, S, V)
    for ri, r in enumerate(radii):
        nsum, ncnt, _ = _neighbor_sum_count(xy, Ec, r, bins[ri])
        Mbar = nsum / np.maximum(ncnt, 1)[:, None]
        Q = (Ec.T @ Mbar) / N
        out[ri] = varg - np.einsum("gk,kl,gl->g", V, Q, V)
    return out

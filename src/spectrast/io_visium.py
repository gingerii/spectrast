"""Read a Space Ranger output directory into an AnnData with micrometre coordinates.

Deliberately standalone and dependency-light. Spot coordinates come back in micrometres because every radius the toolkit
sweeps is in micrometres — a Visium spot is 55 um across by specification, and Space Ranger
records that diameter in full-resolution pixels, which is the only conversion factor needed.
"""
import gzip
import json
import os

import numpy as np

SPOT_DIAMETER_UM = 55.0        # 10x Visium specification; the fixed end of the pixel->um conversion


def _read_tissue_positions(path):
    """``(barcode, in_tissue, array_row, array_col, pxl_row, pxl_col)`` — header optional."""
    import pandas as pd
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres",
            "pxl_col_in_fullres"]
    with gzip.open(path, "rt") as f:
        first = f.readline().split(",")[0].strip()
    header = 0 if first == "barcode" else None
    df = pd.read_csv(path, header=header)
    df.columns = cols[:df.shape[1]]
    return df.set_index("barcode")


def read_spaceranger(dir_path, *, in_tissue_only=True, min_counts=1):
    """One Space Ranger directory -> AnnData; ``obsm['spatial']`` in micrometres.

    The directory must hold five gzipped files side by side, as in a flattened Space Ranger
    upload: ``matrix.mtx.gz``, ``barcodes.tsv.gz``, ``features.tsv.gz``,
    ``tissue_positions.csv.gz`` (with or without its header row) and
    ``scalefactors_json.json.gz``. The usual ``filtered_feature_bc_matrix/`` and ``spatial/``
    subfolders are not searched.

    Pixels are converted to micrometres with ``55 / spot_diameter_fullres``, because a Visium
    spot is 55 µm across by specification.

    Parameters
    ----------
    dir_path : str or path-like
        The directory holding the five files.
    in_tissue_only : bool, optional
        Keep only spots with ``in_tissue == 1``. Default ``True``.
    min_counts : float, optional
        Drop spots whose total count is below this. ``0`` or ``None`` keeps every spot.
        Default ``1``, which drops empty spots.

    Returns
    -------
    anndata.AnnData
        Spots × features. ``X`` holds the counts as stored, as float32 CSR. ``obs`` is indexed by
        barcode, with columns ``in_tissue``, ``array_row`` and ``array_col``. ``var`` is indexed
        by gene symbol (the second column of ``features.tsv.gz``, or the first if there is only
        one), made unique. ``obsm["spatial"]`` is ``(n_spots, 2)`` in µm, with x the full-resolution
        pixel column and y the pixel row. ``uns["spectrast_source"]`` records the directory and
        ``uns["spectrast_um_per_px"]`` the conversion factor.

    Examples
    --------
    >>> adata = spectrast.io_visium.read_spaceranger("GSM0000001")      # doctest: +SKIP
    """
    import anndata as ad
    import pandas as pd
    import scipy.io as sio
    d = str(dir_path)
    with gzip.open(os.path.join(d, "matrix.mtx.gz"), "rb") as f:
        X = sio.mmread(f).T.tocsr().astype("float32")                 # mtx is genes x spots
    with gzip.open(os.path.join(d, "barcodes.tsv.gz"), "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    feat = pd.read_csv(os.path.join(d, "features.tsv.gz"), sep="\t", header=None)
    symbols = feat[1].astype(str).to_numpy() if feat.shape[1] > 1 else feat[0].astype(str).to_numpy()
    pos = _read_tissue_positions(os.path.join(d, "tissue_positions.csv.gz")).reindex(barcodes)
    with gzip.open(os.path.join(d, "scalefactors_json.json.gz"), "rt") as f:
        um_per_px = SPOT_DIAMETER_UM / float(json.load(f)["spot_diameter_fullres"])
    a = ad.AnnData(X, obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")),
                   var=pd.DataFrame(index=pd.Index(symbols, name="symbol")))
    a.var_names_make_unique()
    a.obs["in_tissue"] = pos["in_tissue"].to_numpy()
    a.obs["array_row"] = pos["array_row"].to_numpy()
    a.obs["array_col"] = pos["array_col"].to_numpy()
    # x = pixel COLUMN, y = pixel ROW: the toolkit's xy is (horizontal, vertical) everywhere
    a.obsm["spatial"] = np.c_[pos["pxl_col_in_fullres"].to_numpy(float),
                              pos["pxl_row_in_fullres"].to_numpy(float)] * um_per_px
    keep = np.ones(a.n_obs, bool)
    if in_tissue_only:
        keep &= a.obs["in_tissue"].to_numpy().astype(int) == 1
    if min_counts:
        keep &= np.asarray(a.X.sum(1)).ravel() >= float(min_counts)
    a = a[keep].copy()
    a.uns["spectrast_source"] = d
    a.uns["spectrast_um_per_px"] = float(um_per_px)
    return a


def read_sections(root, gsm_ids, *, log=None, **kw):
    """``read_spaceranger`` over a list of GSM subdirectories, each labelled with its id.

    Parameters
    ----------
    root : str or path-like
        Parent directory of the section folders.
    gsm_ids : sequence of str
        Names of the subdirectories of ``root`` to read, one per section (for example GEO sample
        accessions). Each must be a directory :func:`read_spaceranger` accepts.
    log : callable, optional
        Called once per section with one line of text: spot and gene counts and the coordinate
        extent in µm. ``print`` works. ``None`` (default) prints nothing.
    **kw
        Passed to :func:`read_spaceranger` (``in_tissue_only``, ``min_counts``).

    Returns
    -------
    list of anndata.AnnData
        One per section, in ``gsm_ids`` order, each with ``obs["section_id"]`` set to its id. The
        sections are not concatenated and keep their own coordinate frames.

    Examples
    --------
    >>> secs = spectrast.io_visium.read_sections("data", ["GSM1", "GSM2"], log=print)  # doctest: +SKIP
    """
    out = []
    for g in gsm_ids:
        a = read_spaceranger(os.path.join(str(root), g), **kw)
        a.obs["section_id"] = g
        if log is not None:
            log(f"  {g}: {a.n_obs:,} spots x {a.n_vars:,} genes, "
                f"extent {np.ptp(a.obsm['spatial'], 0).astype(int)} um")
        out.append(a)
    return out

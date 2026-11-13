"""Standalone Space Ranger reader — coordinates in micrometres, no external project imported."""
import gzip
import json
import numpy as np

from spectrast import io_visium


def _write_toy_spaceranger(d):
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "barcodes.tsv.gz", "wt") as f:
        f.write("AAA-1\nCCC-1\nGGG-1\n")
    with gzip.open(d / "features.tsv.gz", "wt") as f:
        f.write("ENS1\tGeneA\tGene Expression\nENS2\tGeneB\tGene Expression\n")
    with gzip.open(d / "matrix.mtx.gz", "wt") as f:                 # 2 genes x 3 spots, 1-based
        f.write("%%MatrixMarket matrix coordinate integer general\n2 3 3\n1 1 5\n2 2 7\n1 3 2\n")
    with gzip.open(d / "tissue_positions.csv.gz", "wt") as f:       # barcode,in_tissue,row,col,pxl_row,pxl_col
        f.write("AAA-1,1,0,0,100,200\nCCC-1,1,0,2,100,400\nGGG-1,0,1,1,300,300\n")
    with gzip.open(d / "scalefactors_json.json.gz", "wt") as f:
        json.dump({"spot_diameter_fullres": 55.0}, f)               # 1 px = 1 um for the toy


def test_reader_returns_um_coordinates_and_symbols(tmp_path):
    d = tmp_path / "GSM_toy"
    _write_toy_spaceranger(d)
    a = io_visium.read_spaceranger(d)
    assert a.n_obs == 2 and list(a.var_names) == ["GeneA", "GeneB"]   # GGG-1 dropped: out of tissue
    xy = np.asarray(a.obsm["spatial"], float)
    # x is the pixel COLUMN, y the pixel ROW; 1 px = 1 um here
    np.testing.assert_allclose(xy, [[200.0, 100.0], [400.0, 100.0]])
    assert a.X.sum() == 12 and a.uns["spectrast_source"].endswith("GSM_toy")


def test_reader_scales_pixels_by_the_spot_diameter(tmp_path):
    d = tmp_path / "GSM_scaled"
    _write_toy_spaceranger(d)
    with gzip.open(d / "scalefactors_json.json.gz", "wt") as f:
        json.dump({"spot_diameter_fullres": 110.0}, f)               # 1 px = 0.5 um
    a = io_visium.read_spaceranger(d)
    np.testing.assert_allclose(np.asarray(a.obsm["spatial"], float), [[100.0, 50.0], [200.0, 50.0]])


def test_read_sections_labels_each_section(tmp_path):
    for g in ("GSM1", "GSM2"):
        _write_toy_spaceranger(tmp_path / g)
    secs = io_visium.read_sections(tmp_path, ["GSM1", "GSM2"])
    assert [s.obs["section_id"].iloc[0] for s in secs] == ["GSM1", "GSM2"]
    assert all(s.n_obs == 2 for s in secs)

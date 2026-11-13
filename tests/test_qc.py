"""spectrast.qc — the section report. Reports quantities; issues no verdict."""
import numpy as np
import pandas as pd
import pytest

from spectrast import qc


def _section(n=1500, n_org=4, n_noise=6, extent=900.0, sigma=120.0, seed=0):
    import anndata as ad
    r = np.random.default_rng(seed)
    xy = r.uniform(0, extent, size=(n, 2))
    d = np.exp(-((xy[:, 0] - extent / 2) ** 2 + (xy[:, 1] - extent / 2) ** 2) / (2 * sigma ** 2))
    org = r.poisson(np.tile((6 * d + 0.4)[:, None], (1, n_org)))
    noise = r.poisson(0.7, size=(n, n_noise))
    X = np.c_[org, noise].astype("float32")
    a = ad.AnnData(X, var=pd.DataFrame(index=[f"g{i}" for i in range(n_org + n_noise)]))
    a.obsm["spatial"] = xy
    return a


def test_report_has_the_blocks_and_their_arithmetic_agrees():
    rep = qc.section_report(_section())
    assert set(rep) >= {"lattice", "saturation", "provenance"}
    L, S = rep["lattice"], rep["saturation"]
    assert L["n_cells"] == 1500
    assert L["r_min_um"] <= L["r_max_um"]
    assert L["cells_per_bin"] == pytest.approx(L["n_cells"] / L["n_bins_occupied"], rel=1e-6)
    # the two saturation fractions are over different denominators and must both be fractions
    assert 0.0 <= S["frac_gene_radius_significant"] <= 1.0
    assert 0.0 <= S["frac_genes_significant_any_radius"] <= 1.0
    assert S["n_cells_tested"] == S["n_genes"] * L["n_radii"]
    # a gene significant at some radius is implied by any significant cell, so the per-gene
    # fraction can never be below the per-cell one
    assert S["frac_genes_significant_any_radius"] >= S["frac_gene_radius_significant"] - 1e-12


def test_it_issues_no_verdict_anywhere():
    """The design constraint, locked — tested on STRUCTURE, not on a word blacklist.

    The one gate this module could plausibly have offered — drop sections whose signal tracks a
    technical covariate — was MEASURED not to change calls (G5: 64->58,
    105->92, 200->200). So no block may carry a verdict field, and the rendered text must state the
    constraint rather than leave it to the reader's restraint.
    """
    rep = qc.section_report(_section(), n_units_A=3, n_units_B=3)

    # no block may contain a key that reads as a judgement on the section
    verdicty = {"pass", "passed", "fail", "failed", "ok", "verdict", "exclude", "drop",
                "usable", "quality", "grade", "score"}
    for name, block in rep.items():
        keys = set(block) if isinstance(block, dict) else set()
        assert not (keys & verdicty), f"block {name!r} carries a verdict field: {keys & verdicty}"

    # the flags that DO exist are statements about the measurement, not about the section's worth
    assert set(rep["lattice"]) >= {"r_min_below_pitch", "cells_per_bin_high"}
    assert "saturated" in rep["saturation"]

    txt = qc.format_report(rep).lower()
    assert "no section is passed or failed here" in txt
    assert "must not be used to discard sections" in txt
    assert "not a gate" in txt or "covariates" not in rep


def test_saturation_is_flagged_when_significance_stops_discriminating():
    """The block the module exists for: on a large section nearly everything is significant."""
    big = _section(n=9000, n_org=8, n_noise=2, sigma=300.0, seed=1)
    rep = qc.section_report(big)
    S = rep["saturation"]
    if S["frac_gene_radius_significant"] >= qc.SATURATION_WARN:
        assert S["saturated"] is True
        assert S["notes"], "a saturated section must carry the note that explains how to read it"
        assert "effect size" in S["notes"][0]
    # and the magnitude is reported beside the fraction, so "significant" can be read against it
    assert np.isfinite(S["median_abs_I_all"])


def test_a_radius_below_the_cell_spacing_is_flagged():
    """A neighbourhood smaller than the tissue's own spacing measures the lattice, not the tissue."""
    a = _section()
    rep = qc.section_report(a, radii=[1, 2, 4, 8], bin_um=4.0)   # 4-32 um, well under the spacing
    L = rep["lattice"]
    assert L["r_min_um"] < L["pitch_um"]
    assert L["r_min_below_pitch"] is True
    assert any("BELOW the median cell spacing" in n for n in L["notes"])
    assert "only a higher-resolution assay" in " ".join(L["notes"])


def test_design_block_is_arithmetic_and_reports_unreachability():
    rep = qc.section_report(_section(), n_units_A=3, n_units_B=3)
    d = rep["design"]
    assert d["n_partitions"] == 20 and d["p_floor"] == pytest.approx(0.1)
    assert d["fdr_reachable"] is False
    assert d["n_units_A"] == 3 and d["n_units_B"] == 3
    txt = qc.format_report(rep)
    assert "no test was performed" in txt
    # a design that CAN reach it says so without the warning clause
    ok = qc.section_report(_section(), n_units_A=14, n_units_B=14)
    assert ok["design"]["fdr_reachable"] is True
    assert "no test was performed" not in qc.format_report(ok)


def test_covariate_block_reports_a_shift_and_not_a_gate():
    a = _section(n=600, n_org=2, n_noise=1, seed=3)
    a.obs["total_counts"] = np.asarray(a.X.sum(1)).ravel()
    rep = qc.section_report(a, covariates=("total_counts",), n_perm=19,
                            covariate_genes=["g0", "g1"], radii=[2, 4], bin_um=30.0)
    assert rep["covariates"], "the covariate block ran but returned nothing"
    c = rep["covariates"][0]
    assert c["covariate"] == "total_counts"
    assert np.isfinite(c["median_frac_attributable"])
    txt = qc.format_report(rep)
    assert "not a gate" in txt.lower()

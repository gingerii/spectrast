# Tutorials

Two analysis tutorials, each on its own public dataset, and the two notebooks that download and
process those datasets. Run a preprocessing notebook once before its analysis tutorial.

```{toctree}
:maxdepth: 1

00a_preprocess_lymph_node
00b_preprocess_trem2
01_single_section
02_differential
```

**Preprocessing**

- **0a. Human lymph node (10x Xenium Prime 5K).** Stream the two files the analysis needs out of 10x
  Genomics' public output bundle, attach 10x's released cell types, quality-control the cells
  (low counts, nucleus-expansion segmentation, extreme sizes), normalise, and save an AnnData.
- **0b. Mouse brain, Trem2/5xFAD (MERFISH).** Download the cohort from the Brain Image Library,
  quality-control and volume-normalise it, and label the amyloid-bearing and amyloid-free sections.

**Analysis**

1. **One section.** How a single tissue is organized: a section report, whether each kind of
   organization is present at all, every gene placed by its characteristic scale and amplitude,
   per-gene curves against a calibrated null, a gene's six statistics and maps across radii, curve
   families with their enriched biology, and where each cell type sits on the tissue's length scale.
2. **Two conditions.** Which genes change their spatial organization between amyloid-bearing and
   amyloid-free brain, at what radii, on which statistic, and whether their characteristic scale
   moves — including changes that differential expression cannot see.

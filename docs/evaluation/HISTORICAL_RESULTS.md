# Historical 10-case results and failure audit

Audit date: 2026-10-06. These results were recovered from saved thesis-era nnU-Net summaries; no new inference is claimed in this document.

This document preserves the first, count-only reanalysis. Its evidence limitations concerning unavailable spatial masks are superseded by [the complete-volume spatial audit](FULL_VOLUME_RESULTS.md), which recovered and validated all 20 prediction files. Use that report and its full-volume CSV/JSON artifacts for manuscript results; the overlap values below remain valid and agree exactly.

## Evidence validation

The saved baseline folder `results_model_a` corresponds to packaged Model B; saved fine-tuned folder `results_model_b` corresponds to packaged Model C. Case-level confusion counts were recovered for both and verified against every complete local reference mask. All 10 case IDs, whole-volume voxel totals, label-1/label-2 reference counts and recomputed Dice values agree.

- Model B source summary: https://drive.google.com/file/d/1xZcWMEwuQpWONUGZtnyM_qMGqFDzSzks/view
- Model C source summary: https://drive.google.com/file/d/1L6e19YAbAt8ng0k7ZGrkMu9dJP00hmOQ/view
- Model B summary SHA-256: `8943528b32e237e99ee09c39361d79b179a195e81b629f9d4d66f6a780effce1`
- Model C summary SHA-256: `50f42fc056584f054785bea28404a0c55dce46ed7ce4205aaf996c7c63aa1195`
- Model B exact final split: https://drive.google.com/file/d/11ekse3GSJE-z07VZ_XjvjxHm9IHhaTLm/view
- Model C exact final split: https://drive.google.com/file/d/149moFPOTKZh91iP1JWKtnEk2QcTp8HdP/view

Original split file hashes and the explicit source mapping appear in `configs/evaluation/cohort_lits_121_130.json`. The split IDs show no observed training/validation overlap for these 10 cases. Actual patient identifiers and historical checkpoint linkage have not been independently verified. This is retrospective internal evidence, with `held_out_certified: false`.

## Aggregate results

Macro means and percentile 95% case-bootstrap intervals; n=10, 2,000 replicates, seed 20261006. All 10 references contain tumor. Complete misses stay in the Dice/recall denominator.

| Measure | Model B mean (95% CI) | Model C mean (95% CI) |
|---|---|---|
| Liver label-1 Dice | 0.8923 (0.8632–0.9189) | 0.9321 (0.9203–0.9450) |
| Tumor Dice | 0.5120 (0.2911–0.7242) | 0.4920 (0.2642–0.7229) |
| Tumor voxel recall | 0.5442 (0.3284–0.7447) | 0.5354 (0.3048–0.7526) |
| Tumor voxel precision | 0.5760 (0.3344–0.7986) | 0.6316 (0.4060–0.8277) |

Model C tumor precision is defined in nine cases; its zero-prediction case is explicitly undefined for precision. Precision means with differing defined denominators should not be interpreted as paired superiority.

Paired Model B minus C liver Dice is -0.0399 (-0.0666–-0.0197); the interval favors C in this cohort. Paired tumor Dice is 0.0199 (-0.0386–0.1004); its interval includes zero. Model B retains the higher tumor point estimate, but these data do not demonstrate that its tumor performance is superior to C. Both share B's original 120-case training lineage; C adds the five-case fine-tuning run.

## Every evaluated case

| Case | B liver Dice | B tumor Dice | C liver Dice | C tumor Dice |
|---|---|---|---|---|
| LITS_121 | 0.7933 | 0.1224 | 0.9290 | 0.1847 |
| LITS_122 | 0.8840 | 0.0000 | 0.9305 | 0.0000 |
| LITS_123 | 0.8516 | 0.3802 | 0.9233 | 0.5030 |
| LITS_124 | 0.8840 | 0.8311 | 0.9016 | 0.8505 |
| LITS_125 | 0.8826 | 0.8728 | 0.9296 | 0.8695 |
| LITS_126 | 0.9553 | 0.7969 | 0.9699 | 0.7344 |
| LITS_127 | 0.8917 | 0.4627 | 0.9097 | 0.1175 |
| LITS_128 | 0.9129 | 0.0000 | 0.9282 | 0.0000 |
| LITS_129 | 0.9342 | 0.7871 | 0.9490 | 0.8003 |
| LITS_130 | 0.9330 | 0.8662 | 0.9506 | 0.8601 |

Both models have zero tumor true-positive voxels on `LITS_122` and `LITS_128`: complete tumor misses. Model B has tumor Dice below 0.5 in five cases (121, 122, 123, 127, 128); Model C in four (121, 122, 127, 128). Model B's strongest tumor case is 125; its weakest positive-overlap case is 121. Model C's largest decrease relative to B is case 127 (0.4627 to 0.1175). These cases must appear in qualitative review, not only the attractive best-case screenshots.

The Model B lesion crop of case 121 produced Dice 0.7373, while this historical full-volume summary reports 0.1224. Different inputs and inference settings prevent a direct model-improvement claim. The cropped demonstration cannot replace full-volume evaluation.

## Component detection from the existing cropped acceptance case

The new metric pipeline reproduced the existing crop's tumor Dice 0.737259 exactly. Six 26-connected reference components were present, of which three matched prediction components at IoU >= 0.1. Component recall was 0.5 and component precision 1.0. Unmatched reference components contain 167, 3 and 124 voxels. Tumor HD95 was 6.0111 mm and ASSD 2.1656 mm.

These are technical crop measurements. The three-voxel island illustrates why connected components must be reviewed before being called separate clinical lesions. Separate full-volume spatial/component measurements are now complete in [FULL_VOLUME_RESULTS.md](FULL_VOLUME_RESULTS.md); expert interpretation of their failures remains required.

## Available reproducibility artifacts and remaining evidence

[historical_per_case_metrics.csv](historical_per_case_metrics.csv) contains the count-only confusion counts and recalculated voxel overlap/precision/recall, with model/case labels and failure flags. [historical_statistics.json](historical_statistics.json) records those summary/paired estimates. `scripts/summarize_historical_metrics.py` can reproduce them from the source summaries and a locked reference manifest. Component detection requires the separately recovered masks and is recorded in the newer full-volume artifacts.

The initial count-only reanalysis could not prove prediction/reference spatial alignment or reconstruct surfaces/components. A subsequent recovery of all 20 original prediction files now verifies full geometry and exact historical counts, supplies prediction hashes, and computes complete spatial/component measurements. See [FULL_VOLUME_RESULTS.md](FULL_VOLUME_RESULTS.md) and the associated full-volume CSV/JSON artifacts. Exact historical checkpoint association remains unknown. The current fast inference profile still needs its own 10-case pinned-checkpoint comparison. Expert review of small/missed lesions, patient identity checks and external validation remain required for broader clinical claims.

The active Model B plan uses CT normalization mean 82.8516, standard deviation 38.3680 and percentile bounds -32/174. Its packaged 120-case fingerprint instead records mean 90.7878, standard deviation 103.3213 and bounds -1000/198. The active plan matches the five-case C fingerprint. Keep the checkpoint's active plan for inference; the original `sourcePlans` provenance must be explained in the methods. The recovered C cases are within B's training/validation cohort, so this observation does not by itself demonstrate exposure to cases 121–130.

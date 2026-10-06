# Complete-volume spatial audit of historical Models B and C

Audit date: 2026-10-06. All 20 historical prediction labelmaps were recovered and evaluated on the same locked 10-case LiTS cohort (121–130). These are complete-volume historical predictions, not cropped acceptance inputs and not newly trained models.

This is the authoritative historical spatial-results report. It supersedes the evidence-availability limitations in the earlier [count-only reanalysis](HISTORICAL_RESULTS.md), while retaining the same validated overlap scores. The [completed current fast-profile cohort](CURRENT_FAST_RESULTS.md) is separate pinned-checkpoint evidence and must not be mixed with these historical predictions.

## Integrity and attribution

Every prediction matches its source CT/reference dimensions, spacing, origin and direction. Every TP/FP/FN/TN, reference/prediction count, Dice and IoU exactly matches the saved historical nnU-Net summary. Each prediction SHA-256 is recorded in `full_volume_statistics.json`; the fixed CT/reference hashes and source-case mapping are in `configs/evaluation/cohort_lits_121_130.json`.

The checkpoint hash for the historical runs remains unknown. Their reports explicitly use `checkpoint_attribution: historical_unpinned` and a null checkpoint hash. The checkpoint name or the old Colab output-folder name alone does not establish exact weights. Current Model B fast predictions are separately attributed to the pinned current checkpoint.

Cases have no observed ID overlap with the recovered B/C training or validation splits. They were already used in the thesis comparison and are retrospective internal evidence; unique patient identity and preprocessing-plan origin are not independently certified.

## Whole-volume results

Macro means with percentile 95% case-bootstrap intervals; 2,000 replicates, seed 20261006.

| Measure | Model B mean (95% CI) | Model C mean (95% CI) |
|---|---|---|
| Liver label-1 Dice, n=10 | 0.8923 (0.8632–0.9189) | 0.9321 (0.9203–0.9450) |
| Complete liver region Dice, n=10 | 0.8989 (0.8670–0.9277) | 0.9386 (0.9260–0.9518) |
| Tumor Dice, n=10 | 0.5120 (0.2911–0.7242) | 0.4920 (0.2642–0.7229) |
| Liver label-1 HD95, mm, n=10 | 85.68 (50.83–120.03) | 50.02 (29.16–70.54) |
| Tumor HD95, mm | 77.41 (20.95–157.74), n=10 | 67.56 (23.51–111.87), n=9 |
| Tumor ASSD, mm | 54.08 (4.94–129.00), n=10 | 23.23 (5.88–43.13), n=9 |

Model C predicts no tumor on case 128, making its tumor surface distance undefined. That complete miss remains in Dice, recall and component-failure reporting. Model B produces distant false tumor components on that case, so its surface distance is defined and very large. The unpaired surface means use different defined denominators and must not be advertised as evidence of superior tumor boundaries without the paired analysis and empty-case disclosure.

The paired B-minus-C tumor Dice difference is 0.0199 (-0.0386–0.1004); its interval contains zero. The paired complete-liver-region Dice difference is -0.0397 (-0.0660–-0.0196), favoring C in this cohort. B has the higher tumor point estimate, but demonstrated tumor superiority is not supported.

## Reference-component detection and failures

Detection uses 26-connected components and one-to-one IoU >= 0.1 matching. No small components are discarded.

| Count/measure | Model B | Model C |
|---|---:|---:|
| Reference components | 69 | 69 |
| Predicted components | 95 | 82 |
| Matched reference components | 43 | 42 |
| Unmatched reference components | 26 | 27 |
| Unmatched prediction components | 52 | 40 |
| Pooled component recall | 0.6232 | 0.6087 |
| Pooled component precision | 0.4526 | 0.5122 |

Component recall's 95% case-bootstrap interval is 0.4545–0.8388 for B and 0.4262–0.8215 for C. Precision's interval is 0.2745–0.6104 for B and 0.3662–0.6305 for C.

| Case | B tumor Dice | B matched/ref components | B unmatched predictions | C tumor Dice | C matched/ref components | C unmatched predictions |
|---|---:|---:|---:|---:|---:|---:|
| LITS_121 | 0.1224 | 3/6 | 8 | 0.1847 | 3/6 | 7 |
| LITS_122 | 0.0000 | 0/3 | 12 | 0.0000 | 0/3 | 3 |
| LITS_123 | 0.3802 | 8/17 | 4 | 0.5030 | 8/17 | 7 |
| LITS_124 | 0.8311 | 5/5 | 9 | 0.8505 | 5/5 | 7 |
| LITS_125 | 0.8728 | 6/14 | 0 | 0.8695 | 6/14 | 0 |
| LITS_126 | 0.7969 | 1/1 | 0 | 0.7344 | 1/1 | 1 |
| LITS_127 | 0.4627 | 1/1 | 2 | 0.1175 | 0/1 | 3 |
| LITS_128 | 0.0000 | 0/1 | 3 | 0.0000 | 0/1 | 0 |
| LITS_129 | 0.7871 | 9/11 | 7 | 0.8003 | 9/11 | 7 |
| LITS_130 | 0.8662 | 10/10 | 7 | 0.8601 | 10/10 | 5 |

Both models completely miss annotated tumor in cases 122 and 128. High global Dice also hides partial component misses: B case 125 has Dice 0.8728 while matching only six of 14 reference components. Tiny disconnected annotation islands may not correspond to independent clinical lesions, so these component counts need expert review before being described as clinical lesion sensitivity. `full_volume_component_details.csv` preserves each reference component's size and match result.

Publication failure figures should include cases 122 and 128 (complete misses), 121 (large false-positive burden), 125 (small-component misses despite strong global Dice), and 127 (fine-tuned C decrease). Include full CT context and the original reference alongside predictions.

## Reproduction and limitations

The included `full_volume_model_b_metrics.csv` and `full_volume_model_c_metrics.csv` contain all per-case overlap, physical surface and component measurements. `full_volume_statistics.json` includes the summary, paired confidence intervals, artifact hashes and per-case failure flags. The local canonical full reports are under `tmp/publication_evaluation/historical_full_model_b_metrics` and `historical_full_model_c_metrics`.

Run `scripts/evaluate_predictions.py` in historical mode to verify original saved counts and derive spatial metrics, then `scripts/compare_evaluations.py` for paired comparisons. The surface metric implementation uses exact nearest boundary-voxel coordinates and preserves anisotropic spacing; equivalence to a dense Euclidean distance transform passed an independent regression test.

This completes the historical per-case metrics, uncertainty and automated failure-analysis work. Remaining manuscript evidence includes expert qualitative review, explanation of patient identities and original preprocessing planning, exact old checkpoint provenance, and external validation if broader clinical generalization is claimed. The completed [current fast inference audit](CURRENT_FAST_RESULTS.md) retains separate pinned-checkpoint attribution and reports increased unmatched prediction components; it is not an equivalence claim.

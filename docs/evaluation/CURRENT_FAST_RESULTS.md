# Current pinned Model B: complete-volume fast-profile audit

Audit date: 2026-10-06. All ten locked LiTS cases (121–130) completed fresh full-volume CUDA inference and evaluation. No retraining or reference-derived inference ROI was used. Technical full-volume execution is now demonstrated; accuracy equivalence, tumor superiority and clinical readiness are not established.

These cases were previously used in the thesis. This is a reproducible prospective rerun of a retrospective internal cohort, not a new untouched or external test. There is no observed ID overlap with the recovered B/C training/validation splits; patient uniqueness and the original preprocessing-planning provenance remain unverified.

## Verified current provenance

The checkpoint SHA-256 is `280b1c6b2c8562b68803f269cb7a62fcd890430c2652af0daac578dd141e8b6b`. The local locked manifest SHA-256 is `f0d75a53a1c54c5423305ac5b504e66c7204efeaeb70390842571c3f9fc00e0e`.

Every case passed input/checkpoint/output byte-hash checks, full dimensions and physical geometry checks, explicit no-manual-ROI/no-reference-annotation claims, consistent inference settings, and exact membership in the completed batch report. Every prediction sidecar hash is retained in `current_fast_statistics.json`. Supplying an arbitrary checkpoint to the evaluator cannot invent current attribution. The authoritative evaluation reports `checkpoint_attribution: verified_prediction_sidecars`, `current_provenance_verified_all_cases: true` and `completed_cohort_run_verified: true`.

The candidate uses GTX 1660 Ti CUDA, float32 convolution, Gaussian sliding windows with tile step 0.5, no mirroring, CPU aggregation and two export threads, while preserving the trained checkpoint's active plans. Mean end-to-end time is 312.25 seconds (5.20 minutes), median 301.82 seconds; case times range from 110.84 to 432.85 seconds. See `current_fast_model_b_runtime.json` and `current_fast_model_b_latency.csv` for the separately validated latency/stage report. There is no completed same-case slow-profile benchmark from which to calculate a measured speed-up factor.

## Whole-volume accuracy and exploratory historical comparison

The comparator is the [historical full-volume Model B audit](FULL_VOLUME_RESULTS.md), whose exact checkpoint hash is unknown. Therefore the following differences describe observed artifacts; they are not a controlled same-weights fast-versus-reference experiment and cannot isolate the effect of removing mirroring or changing precision. No noninferiority margin, equivalence criterion or multiplicity correction was predeclared.

Macro means and paired current-minus-historical differences use percentile 95% case-bootstrap intervals, 2,000 replicates, seed 20261006. All displayed metrics have ten defined cases.

| Measure | Current fast mean (95% CI) | Historical B mean | Paired difference (95% CI) |
|---|---|---:|---|
| Liver label-1 Dice | 0.8840 (0.8545–0.9103) | 0.8923 | -0.0082 (-0.0110–-0.0057) |
| Complete liver-region Dice | 0.8907 (0.8575–0.9197) | 0.8989 | -0.0082 (-0.0110–-0.0056) |
| Tumor Dice | 0.5304 (0.3076–0.7341) | 0.5120 | 0.0184 (-0.0122–0.0615) |
| Tumor voxel precision | 0.5482 (0.3126–0.7577) | 0.5760 | -0.0279 (-0.0586–-0.0040) |
| Tumor voxel recall | 0.5906 (0.3727–0.7775) | 0.5442 | 0.0464 (0.0050–0.1118) |
| Tumor HD95, mm | 106.87 (47.08–179.33) | 77.41 | 29.47 (8.15–58.89) |
| Tumor ASSD, mm | 58.95 (11.08–132.89) | 54.08 | 4.86 (1.63–8.55) |

The tumor Dice interval is wide and its paired difference includes zero. A slightly higher mean Dice does not offset the observed decrease in voxel precision, larger boundary distances and substantially more unmatched prediction components. The faster profile remains opt-in research functionality, not a quality-preserving replacement certified by this audit.

Both runs predict nonempty tumor masks in every case, including the two cases with no overlap with annotated tumor. Surface distances are consequently defined for those complete misses; zero Dice is not hidden by an undefined-distance exclusion.

## Components and failure cases

Matching uses 26-connected components, maximum-cardinality one-to-one IoU >= 0.1, with no small-component filtering. These are engineering annotation/prediction components, not expert-adjudicated clinical lesion counts.

| Count/measure | Current fast B | Historical B |
|---|---:|---:|
| Reference components | 69 | 69 |
| Predicted components | 138 | 95 |
| Matched reference components | 42 | 43 |
| Unmatched reference components | 27 | 26 |
| Unmatched prediction components | 96 | 52 |
| Pooled component recall | 0.6087 | 0.6232 |
| Pooled component precision | 0.3043 | 0.4526 |

Current component recall's 95% case-bootstrap interval is 0.4339–0.8388; precision's is 0.1808–0.4103. Component-ratio intervals resample whole cases, not individual components.

| Case | Current tumor Dice | Historical B tumor Dice | Current matched/ref components | Current unmatched predictions |
|---|---:|---:|---:|---:|
| LITS_121 | 0.1179 | 0.1224 | 3/6 | 18 |
| LITS_122 | 0.0000 | 0.0000 | 0/3 | 15 |
| LITS_123 | 0.4482 | 0.3802 | 7/17 | 8 |
| LITS_124 | 0.8315 | 0.8311 | 5/5 | 8 |
| LITS_125 | 0.8598 | 0.8728 | 6/14 | 17 |
| LITS_126 | 0.7639 | 0.7969 | 1/1 | 4 |
| LITS_127 | 0.6508 | 0.4627 | 1/1 | 5 |
| LITS_128 | 0.0000 | 0.0000 | 0/1 | 1 |
| LITS_129 | 0.7652 | 0.7871 | 9/11 | 11 |
| LITS_130 | 0.8663 | 0.8662 | 10/10 | 9 |

Cases 122 and 128 remain complete tumor misses. Four cases have tumor Dice below 0.5. Case 125 illustrates why a high whole-tumor Dice is insufficient: only six of 14 reference components match, and there are 17 unmatched prediction components, versus zero in the historical prediction. Expert review should determine whether tiny reference islands are separate lesions and whether prediction components are anatomical false positives.

## Reproduction and public artifacts

`current_fast_statistics.json` contains sanitized aggregate/per-case metrics, bootstrap results, artifact hashes, fixed protocol and explicit current/historical attribution. `current_fast_per_case_metrics.csv`, `current_fast_failures.csv` and `current_fast_component_details.csv` provide numerical tables. They contain public LiTS case IDs, not raw scans, masks, coordinates, patient names or absolute local paths. The raw local reports retain full sidecar provenance and must not be shipped in the source archive.

Run `scripts/evaluate_predictions.py` with the locked manifest, all ten current outputs, the pinned checkpoint and the completed `run_settings.json`; then `scripts/compare_evaluations.py` against the unpinned historical report. Use `scripts/export_public_evaluation.py` to generate the allowlisted public artifacts; it refuses incomplete/current-unverified or falsely pinned historical comparators and validates the paired source-report hashes.

This supersedes the pending-cohort status in the [initial full-volume pilot](FULL_VOLUME_PILOT.md). The initial pilot remains preserved as its own serialized-input audit; it has not been retroactively relabeled as passing the strict canonical-input provenance checks. The canonical case-121 rerun produced the identical prediction hash.

Next scientific requirements are expert failure adjudication, explanation of original patient/planning provenance, a controlled pinned same-weights accuracy comparison if equivalence is claimed, and genuinely independent validation for broader generalization. No change to trained normalization or postprocessing was made based on these test results.

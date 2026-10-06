# Full-volume Model B pilot

Date: 2026-10-06. Purpose: engineering feasibility on the previously evaluated LITS_121 case. This is one case, with no cohort confidence interval.

This first pilot preserves its original results and serialized-input equivalence audit. The evaluator subsequently added stricter sidecar-backed checkpoint/input attribution; this older evaluation is not retroactively relabeled as passing those guards. The later locked-cohort case 121 uses the exact canonical input-file hash and produces an identical prediction hash. Its current authoritative accuracy result belongs to the [completed ten-case current cohort evaluation](CURRENT_FAST_RESULTS.md). See [the historical full-volume report](FULL_VOLUME_RESULTS.md) for the separate B/C baseline evidence.

The complete input and output are 512 x 512 x 424. Spacing, origin and direction match the original CT; labels are 0/background, 1/liver and 2/tumor. No manual ROI or reference annotation selected the input. Model weights and their active training plans were preserved.

## Settings and provenance

- Model B fold-0 `checkpoint_best.pth` SHA-256: `280b1c6b2c8562b68803f269cb7a62fcd890430c2652af0daac578dd141e8b6b`.
- Output labelmap SHA-256: `f709c1aecd12ca99339a694a2c674101b7220ba7a2cd2a49aa0b0f3dd228a7b8`.
- Frozen raw CT compressed-file SHA-256: `2e01564b21b7017e871f7bb37363e8f47025a72625d9ce70ccbe53343164b286`.
- Pilot's serialized CT compressed-file SHA-256: `30c53253b98ace6bd6b02380ebcbb8b9ed173c885495fd74aee539796b550c50`.
- The two CT files have exactly identical voxel values, scalar dtype, dimensions, spacing, origin and direction. Their compressed/header serializations differ; array comparison gives maximum difference 0. The canonical cohort remains locked to the raw file.
- Device: NVIDIA GeForce GTX 1660 Ti, CUDA; software PyTorch 2.7.1+cu118, nnU-Net 2.6.2.
- Fast candidate: float32 network, no mirroring, Gaussian tiled inference, tile step 0.5, CPU aggregation, 567 tiles.
- End-to-end time including export: 363.34 seconds.
- Neural-network prediction stage: 250.56 seconds.
- Preprocessing: 87.72 seconds; reconstruction: 22.95 seconds.
- Peak CUDA allocation: 4.04 GiB.

## Quantitative result

| Measure | Liver label 1 | Tumor label 2 | Combined liver region 1 or 2 |
|---|---|---|---|
| Dice | 0.7810 | 0.1179 | 0.7726 |
| Precision | 0.6496 | 0.0638 | 0.6377 |
| Recall | 0.9790 | 0.7819 | 0.9799 |
| HD95, mm | 186.74 | 165.76 | 187.07 |
| ASSD, mm | 35.21 | 103.86 | 34.52 |

Tumor reference volume is 4.345 mL; predicted tumor volume is 53.275 mL, with 60,049 false-positive voxels and 1,141 missed voxels. The main failure is excess segmentation away from the reference tumor, despite substantial voxel recall.

Three of six 26-connected reference components match at IoU >= 0.1. There are 21 predicted components, 18 unmatched; component recall 0.5000 and precision 0.1429. Expert review must determine which disconnected annotation islands represent distinct clinical lesions.

The historical full-volume tumor Dice for this case is 0.122409; current fast Dice is 0.117918 (difference -0.004491). Liver label Dice changes from 0.793277 to 0.780989. Historical checkpoint identity is not yet pinned, so this is a comparison with historical evidence rather than an exact same-checkpoint reference benchmark.

## Acceptance decision

Full-volume execution, output geometry and saved artifacts pass this engineering pilot. Faster execution is feasible without retraining. The subsequent complete-cohort audit does not establish accuracy equivalence and finds substantially more unmatched prediction components than the unpinned historical baseline. Keep this profile opt-in research functionality. The model's large full-volume false-positive error requires explicit failure discussion and limits broader deployment claims.

The cropped acceptance screenshot's tumor Dice 0.7373 is not representative of this complete-volume result. A publication should show this whole-CT output and its errors, alongside any tumor close-up.

The local canonical results are `tmp/publication_evaluation/full_volume_pilot_metrics/evaluation.json`, `per_case_metrics.csv` and `failures.csv`. The full generated mask and runtime record are under `tmp/publication_audit/fast_model_b`; these CT-derived files are excluded from the public source package.

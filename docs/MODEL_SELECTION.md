# Model Selection Decision

Decision date: 2026-09-23. Evidence updated: 2026-10-06.

## Decision

Use `nnunet_liver_modelb` (Model B) as the operational default for tumor-focused inference and UAT. Keep Model C available for the same-case research comparison, especially when complete-liver segmentation is the objective. This default is not a claim that B has demonstrated tumor superiority.

The saved comparison gives B a slightly higher tumor Dice point estimate; the uncertainty is too wide to establish a tumor advantage. C inherits B's original training and then uses a five-case fine-tuning set, rather than being trained from scratch on five cases.

## Evidence

| Model | Training evidence | Available evaluation evidence | Interpretation |
|---|---|---|---|
| A | 10 cases, 3D low-resolution experiment; best training EMA pseudo-Dice about 0.7499 | Local 27-case summary: liver Dice 0.7588, tumor Dice 0.2083 | Functional historical model, not the recommended default. |
| B | 120 cases; recorded split 96 train / 24 validation; best training EMA pseudo-Dice 0.7468 | Full historical 10-case masks: liver-label Dice 0.8923 (95% CI 0.8632–0.9189), tumor Dice 0.5120 (0.2911–0.7242) | Operational default with higher tumor point estimate; superiority unproven. |
| C | Fine-tuned from B using five cases; recorded split four train / one validation; best training EMA pseudo-Dice 0.5310 | Full historical 10-case masks: liver-label Dice 0.9321 (95% CI 0.9203–0.9450), tumor Dice 0.4920 (0.2642–0.7229) | Stronger liver performance in this cohort; tumor estimate slightly lower but not demonstrably different. |

The recovered exact B split contains IDs 001–120; C fine-tunes on IDs 010/017/020/025 with 005 for validation. The historical test cohort is exactly IDs 121–130, with no observed split-ID overlap. All 20 full-volume prediction files now pass exact geometry and saved TP/FP/FN/TN/Dice/IoU validation. Historical checkpoint hashes remain unpinned and patient-identity/preprocessing provenance is not independently certified.

The paired B-minus-C tumor Dice difference is 0.0199 (95% CI -0.0386–0.1004), including zero. Complete liver-region Dice is 0.8989 for B and 0.9386 for C; its paired difference -0.0397 (CI -0.0660–-0.0196) favors C. Both models completely miss annotated tumor on cases 122 and 128. See `evaluation/FULL_VOLUME_RESULTS.md` for physical boundary errors, component detection and every failure case.

## 2026-09-23 acceptance result

Model B was run through 3D Slicer and the CUDA MONAI Label server on a fixed `192 x 160 x 128` LITS_121 crop containing the complete annotated tumor bounding box and surrounding liver context.

| Measure | Result |
|---|---:|
| End-to-end Slicer inference | 216.93 seconds |
| Liver label Dice | 0.9649 |
| Tumor label Dice | 0.7373 |
| Combined liver-region Dice | 0.9657 |
| Tumor precision | 0.7281 |
| Tumor recall | 0.7467 |

The prediction matched the input dimensions, contained labels `[0, 1, 2]`, loaded in Slicer, and produced axial, coronal, sagittal, and 3D views.

These crop metrics are acceptance-test evidence, not a claim of clinical generalization. The crop was selected using ground-truth location so that tumor behavior could be tested. The complete-volume historical evidence supersedes this crop for assessing accuracy. The crop matched only three of six reference components at IoU >= 0.1; disconnected annotation islands require expert review before clinical lesion interpretation.

## Full-volume latency finding

The unmodified default predictor uses overlapping tiles and test-time mirroring. On the full `512 x 512 x 424` LITS_121 volume:

- CPU reached `3/567` steps at about 29 seconds per step, projecting approximately 4.5 hours.
- CUDA reached `4/567` steps after about one minute and stabilized near 11–13 seconds per step, projecting roughly two hours.

Both original full-volume attempts were intentionally stopped after the performance gate clearly failed; these are projected times, not completed benchmark measurements.

On 2026-10-06, an opt-in fast profile completed the same full-volume case in 363.34 seconds on GTX 1660 Ti, preserving the checkpoint and its active plans. It uses float32 convolution, Gaussian sliding-window inference at tile step 0.5, no mirroring and CPU aggregation. Output size/spacing/origin/direction all match the complete CT. The subsequent locked ten-case fresh batch completed with mean end-to-end latency 312.25 seconds and passed all input/checkpoint/output hash, geometry and completed-batch provenance guards. Full-volume execution is feasible without retraining.

The current full-volume pilot tumor Dice is 0.1179 and precision 0.0638, versus historical full-volume Dice 0.1224 and cropped Dice 0.7373. Liver-label Dice is 0.7810, and gross distant false positives persist. The crop cannot represent complete-scan performance. See `evaluation/FULL_VOLUME_PILOT.md`. Do not declare this fast profile clinically ready or advertise the system as real-time from these measurements.

The completed current fast cohort has tumor Dice 0.5304 (95% CI 0.3076–0.7341), versus historical B 0.5120; paired difference 0.0184 (CI -0.0122–0.0615) does not establish superiority or equivalence. Liver-label Dice decreases to 0.8840; unmatched prediction components increase from 52 to 96, while matched reference components decrease from 43 to 42 of 69. Cases 122 and 128 remain complete tumor misses. Keep the fast profile opt-in research functionality; the historical comparator's checkpoint is unpinned, so this comparison cannot isolate runtime-setting effects. See [the complete current audit](evaluation/CURRENT_FAST_RESULTS.md).

Packaged active plans match the plans embedded in each checkpoint. B's active normalization differs from its separately packaged dataset fingerprint; this is a historical planning-method question rather than deployment drift. Preserve the trained checkpoint's active plans and explain their original source in the methods.

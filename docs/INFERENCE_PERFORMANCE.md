# Inference performance audit — 2026-10-06

No retraining, checkpoint edits, normalization changes or annotation-guided full-input cropping occurred.

## Why the GPU was slow

An actual Model B patch-shape benchmark on this GTX 1660 Ti measured approximately 1.104 s for contiguous CUDA autocast float16 versus 0.364 s for contiguous float32 (three timed repeats after warmup). Channels-last float16 was approximately 1.268 s. Channels-last float32 was interrupted during slow cuDNN setup; no fourth result or complete benchmark JSON is claimed. Synthetic tile timings are diagnostics, not end-to-end or accuracy evidence.

Default nnU-Net mirroring performs multiple network evaluations per tile. Removing it changes predictions. Float32/no-mirroring is an experimental candidate, not a transparent precision-only replacement.

## Profiles

| Setting | `reference` (default) | `fast` (opt-in) |
|---|---|---|
| Checkpoint/plans | Best fold 0, unchanged | Same |
| Tile step / Gaussian | 0.5 / yes | Same |
| Mirroring | Yes | No |
| CUDA network precision | Normal autocast float16 | Float32 |
| Aggregation | nnU-Net CPU float16 | Same |
| Output extent | Complete supplied input geometry | Same |
| Inference annotation access | None | None |

CPU aggregation avoids retaining the whole high-resolution logits volume in the 6 GB GPU. nnU-Net's internal nonzero bounding-box/resampling is restored to original input geometry; it is not a manual tumor ROI. Two threads, including export, limit oversubscription on this 16 GB RAM machine.

## Measurements

Hardware/software: GTX 1660 Ti 6 GB, PyTorch 2.7.1+cu118, nnU-Net 2.6.2, MONAI Label 0.8.5, SimpleITK 2.5.3, NumPy 1.26.4.

| Test/stage | Extent / endpoint | Time |
|---|---|---:|
| Fast full pilot | 512 × 512 × 424, cold CLI/export included | 363.34 s |
| Preprocessing | 567 tiles | 87.72 s |
| Network | Float32/no mirroring | 250.56 s |
| Reconstruction | Original geometry | 22.95 s |
| Fast Slicer-extension smoke | Fixed 192 × 160 × 128 technical crop, HTTP round trip | 33.84 s |
| Earlier reference Slicer | Same crop, earlier runtime | 216.93 s |

The later fresh-source-archive Slicer test completed the default reference profile on the same technical crop in 140.80 s, with the exact same input/checkpoint hashes as the 33.84 s fast test. Tumor Dice was 0.737259 reference versus 0.735250 fast; tumor false-positive voxels were 1,459 versus 1,918. This is a one-case annotation-selected crop comparison, not a whole-volume quality-equivalence or generalized speedup claim. Runtime/model files in that tested source snapshot match the final candidate; later changes are documentation/evidence and packaging allowlist only.

Pilot peak allocation: 4,340,167,680 bytes (4.04 GiB). It started before the latest finite-output guards/explicit export-thread limit; the locked cohort includes those updates. Slicer loaded the full pilot separately; 363.34 s is not a full-scan Slicer-client timing.

Crop timings are exploratory because runtime/profile changed. Earlier complete reference runs were stopped, projecting about two hours CUDA / 4.5 hours CPU. These were not completed timings and cannot be speedup denominators.

Pilot and locked CT compressed-file hashes differ, despite exactly identical voxels/dtype/physical geometry. Its original sidecar is preserved; canonical batch does not silently reuse it.

## Completed locked ten-case run

All ten canonical CTs completed on 2026-10-06, in 52.04 minutes of recorded per-case end-to-end time (52.04 minutes between report start/completion timestamps). No cases were resumed/reused, no reference masks were read for inference, and every prediction hash agrees with its sidecar and completed report. These are CLI timings with runtime initialization/file export, not interpreter import/startup time; full-scan Slicer-client latency is not claimed.

| Measure | Seconds |
|---|---:|
| Mean end-to-end | 312.25 |
| Median end-to-end | 301.82 |
| Minimum / maximum | 110.84 / 432.85 |
| First cold case (LITS_121) | 362.61 |
| Nine cached-runtime cases, mean / median | 306.65 / 299.10 |
| Mean preprocessing | 76.56 |
| Mean network prediction | 208.45 |
| Mean reconstruction | 26.57 |

Cold/cached cases have different scan sizes and tile counts; their timings do not isolate a causal cache speedup. The first case peak CUDA allocation was 4.04 GiB; the cached cases recorded 1.95 GiB. The strict latency summarizer verifies manifest membership/order, completion, settings, geometry, software/hardware and prediction-byte hashes without reading CTs/references.

| Case | End-to-end seconds | Tiles |
|---|---:|---:|
| LITS_121 | 362.61 | 567 |
| LITS_122 | 419.40 | 720 |
| LITS_123 | 299.10 | 504 |
| LITS_124 | 348.32 | 630 |
| LITS_125 | 283.81 | 504 |
| LITS_126 | 285.79 | 504 |
| LITS_127 | 275.19 | 504 |
| LITS_128 | 432.85 | 720 |
| LITS_129 | 304.54 | 504 |
| LITS_130 | 110.84 | 168 |

Runtime JSON SHA-256: `728ab0bef303d9dc38c63f95f7a15ba5b5f6fdbbed2e777a24a4831a69887cf4`. Per-case CSV SHA-256: `0fe04852d4b47550015f3d58cf6011478736d5dd99e1de3655a13fd3439f292d`. Local complete records are under `tmp/publication_evaluation/current_fast_model_b_runtime/`.

## Accuracy gate (not just runtime)

Full pilot tumor Dice is 0.1179 (historical 0.1224), despite crop Dice 0.7373. Principal error: whole-volume false positives. Speed does not repair this error.

The locked accuracy evaluation is complete: tumor macro Dice 0.5304 (95% CI 0.3076–0.7341), liver label-1 Dice 0.8840, and two complete tumor misses (122/128). Against historical B, tumor Dice delta is +0.0184 (CI −0.0122–0.0615), but unmatched tumor components increase from 52 to 96 and mean tumor HD95 from 77.41 to 106.87 mm. Historical checkpoint/settings are unpinned, so this does not isolate a profile-only effect or establish equivalence/superiority. A small Dice gain cannot justify ignoring greater false-positive burden. See [current full results](evaluation/CURRENT_FAST_RESULTS.md).

Keep `fast` experimental, not promoted. Same-checkpoint reference validation and agreed accuracy criteria remain necessary. Retain default `reference` and original CPU fallback. Any later postprocessing/parameter calibration must use training/validation data and be frozen before suitable untouched evaluation, not tuned on these examined cases.

# Current UAT summary — 2026-10-06

The earlier bundle passed UAT-01 through UAT-08 on a 96 × 96 × 32 crop. Model C's GPU test was about 27% slower than CPU for that small input. This does not describe current full-volume Model B performance.

Model B's earlier 192 × 160 × 128 lesion Slicer UAT completed in 216.93 s with tumor Dice 0.7373. Reference annotation selected that crop, so it is a technical integration test, not independent full-volume accuracy evidence.

## Current staged-source checks

- Model B `fast` completed all ten hash-locked full CTs. Mean CLI end-to-end latency was 312.25 s, median 301.82 s, range 110.84–432.85 s; geometry and saved hashes are verified. The first canonical 512 × 512 × 424 case completed in 362.61 s and exactly reproduced the pilot mask.
- A new Slicer 5.10.0 instance used its installed MONAI Label extension against the updated staged Model B server. The same lesion crop completed in 33.84 s; geometry matched, both segments loaded, and Four-Up capture/editable segmentation were saved.
- The canonical whole-volume case 121 loaded separately in Slicer and passed full geometry/four-view export. Final numerical viewport checks confirm all CT corner projections lie within each axial/coronal/sagittal view. Offset-only slice navigation retains CT-centered context; 3D fitting includes prediction islands instead of hiding them. An initial viewport-check API error was repaired; its failed test instance alone was stopped.
- Runtime, batch, evaluation and release-integrity regression tests check implementation; they do not measure clinical quality.

A fresh source-only ZIP was extracted separately and installed with all twelve locally verified model artifacts. Its positional Conda launcher registered A/B/C on localhost8006. Model B's default reference profile passed a real Slicer-extension HTTP test on the technical crop in 140.80 s, preserving geometry/labels and numerically fitting the complete supplied image in all three slice viewports. Four-Up PNG and editable segmentation were saved. Tumor Dice 0.737259 reproduced the earlier reference crop result; fast on the same crop/checkpoint had Dice 0.735250. This verifies installation/default integration, not uncropped reference accuracy. The temporary server was stopped, and no existing Slicer scene was replaced.

The combined synthetic regression suite passes 83 tests, including inference/batch, evaluation/provenance, physical metrics, source/privacy packaging and latency checks. All twelve model artifacts verify, preparation passes, and publication deliberately fails the seven unresolved owner/revision fields.

The full pilot's tumor Dice is 0.1179 and precision 0.0638. Integration/geometry pass, but segmentation-quality acceptance remains open. Historical B/C full-volume evaluation includes complete misses in two of ten cases per model.

The completed current ten-case fast cohort has mean tumor Dice 0.5304 (95% CI 0.3076–0.7341), still completely missing tumors in 122/128. Unmatched components increase from 52 historically to 96, and mean tumor HD95 77.41→106.87 mm. Historical weights/settings are unpinned: this is exploratory, not controlled equivalence. Fast is not promoted. See [current results](evaluation/CURRENT_FAST_RESULTS.md).

The fast profile remains experimental; `reference` is the default. Original CPU environment, legacy application and checkpoints are unchanged. Historical masks lack exact checkpoint attribution, so same-checkpoint reference equivalence still needs validation. No retraining, upload, push or license selection occurred. CT-derived artifacts remain local, excluded from source release.

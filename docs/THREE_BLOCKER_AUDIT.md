# Three-blocker feasibility and implementation audit

Date: 2026-10-06. Scope: inference and reproducibility, without retraining. Original checkpoints, legacy working application and CPU environment were preserved. No remote push/upload occurred.

## 1. Whole-volume evaluation instead of cropped demonstrations

Feasible locally. Ten full CT/reference pairs for LITS_121–130 were hash-locked. Dataset407 files match original source files (`LITS_121` maps to `volume-120`). Recovered fold-0 B split has 96 training / 24 validation cases; C has four fine-tuning / one validation case. These ten IDs are absent from those splits.

Implemented:

- `configs/evaluation/cohort_lits_121_130.json`: hashes, geometry, split evidence, source mapping and retrospective limitations. CTs/references are not packaged.
- Full-input CLI and cohort runner: original plans, best checkpoint, fold 0, no reference access or annotation-selected bounding boxes during inference.
- Strict preflight/resume: hashes, settings, geometry and provenance must match; no silent overwrite. Evaluation rejects geometry mismatch rather than silently resampling.
- All ten canonical full CTs completed with the pinned current checkpoint, mean 312.25 s / median 301.82 s, range 110.84–432.85 s. Canonical case 121 finished in 362.61 s and exactly reproduced the separate pilot mask; neither input files nor the pilot sidecar were rewritten. It loaded in Slicer with full 512 × 512 × 424 geometry, and all three 2D viewports numerically contain the complete CT bounds.

Opt-in `fast` makes this hardware feasible: float32 network calls/no mirroring, retaining Gaussian weighting and 0.5 tile step. Default `reference` retains mirroring and normal CUDA autocast. This changes predictions; speed alone is not accuracy acceptance. See [performance](INFERENCE_PERFORMANCE.md).

Remaining scientific boundary: these are previously examined internal cases, not pristine external data. Patient-level provenance and independent validation require additional evidence/data. Historical masks are checkpoint-unpinned and cannot certify exact reference-profile equivalence.

## 2. Metrics, confidence intervals and failure analysis

Feasible and implemented. All twenty recovered B/C complete masks passed size/physical-geometry checks and exactly reproduced their saved confusion counts. Historical checkpoint attribution is explicitly unknown.

Evaluator exports Dice/IoU, voxel precision/recall, HD95/ASSD in millimetres, combined liver region, physical volumes, connected-component matches and failures. It retains misses, defines empty-mask policy, uses seeded case-bootstrap intervals and supports paired comparisons. Regression tests cover metric definitions and geometry errors.

| Ten-case retrospective tumor result | Model B | Model C |
|---|---:|---:|
| Macro Dice | 0.5120 | 0.4920 |
| 95% bootstrap CI | 0.2911–0.7242 | 0.2642–0.7229 |
| Zero tumor true-positive cases | 2/10 | 2/10 |

Paired B-minus-C Dice is 0.0199, CI −0.0386 to 0.1004: no established tumor superiority. For case 121, crop Dice was 0.7373, historical full Dice 0.1224, current fast full pilot 0.1179. Tumor precision is 0.0638 with 60,049 false-positive voxels. This is a model error, not a viewer cropping issue.

The current pinned-checkpoint ten-case run also passed strict sidecar/batch evaluation. Mean tumor Dice is 0.5304 (95% CI 0.3076–0.7341), liver Dice 0.8840, with two complete misses. Current-minus-historical tumor Dice delta is +0.0184 (CI −0.0122–0.0615), but unmatched prediction components increase 52→96 and mean tumor HD95 77.41→106.87 mm. Historical weights/settings are unpinned: this does not establish quality equivalence. The candidate is not promoted. See [current results](evaluation/CURRENT_FAST_RESULTS.md).

Detection counts are 26-connected annotation islands, not clinician-confirmed lesions. Tiny islands and matching thresholds need expert review. This repairs missing evidence, not model accuracy. See [historical results](evaluation/FULL_VOLUME_RESULTS.md) and [protocol](EVALUATION_PROTOCOL.md).

## 3. Licensed GitHub/Hugging Face release

Technically feasible; prepared but owner-gated. The three checkpoints total 742,575,921 bytes (about 708 MiB), plus the runtime JSON metadata, making twelve locked runtime files. Source/configs/docs belong on GitHub; checkpoint packages on Hugging Face. Scans, labels, derived patient images, credentials and local environments remain private.

Implemented:

- SHA-256/size artifact lock for A/B/C weights and metadata.
- Immutable-commit downloader with staged verification; no mutable `main` release installs.
- Preparation/publication gates: unresolved licenses, rights, dataset evidence and remote revision identities cannot pass publication.
- Model card, citation draft, MONAI notices, publishing checklist, CPU fallback and reproduction/user manual.
- Source-only ZIP builder excludes weights, medical data, caches, scratch and runtime outputs. A ZIP is not a Git revision.

Fresh local installation was verified from a separately extracted source ZIP plus the twelve hash-checked model artifacts: Conda launcher, A/B/C registration, default-reference Model B inference through Slicer's extension, physical geometry, Four-Up and editable segmentation all passed on a technical crop. The code used by that test matches final runtime/Slicer/launcher code; final audit/evidence docs are updated separately. This is not a download from published immutable remote revisions or full reference-cohort equivalence. Combined regression suite: 83 passing tests.

Cannot resolve automatically: source/checkpoint licenses, redistribution rights (university/dataset constraints), final repository IDs/visibility, authentication and immutable revisions. No permission/license was invented. See [release feasibility](RELEASE_FEASIBILITY.md).

## Decision

Whole-volume execution/evaluation and uncertainty/failure reporting are implemented and tested. Release tooling is prepared, but a licensed hosted release remains owner-gated. Fast passes engineering execution, not demonstrated quality-preserving acceptance. Keep it opt-in and retain reference/CPU fallback; deployment/publication claims remain research-only.

Without retraining, next feasible quality work is validation-set calibration of inference/postprocessing, then a frozen controlled reference comparison and suitable independent evaluation. Do not tune these test cases retrospectively. Expert lesion adjudication, rights and external validation require people/data, not merely another script.

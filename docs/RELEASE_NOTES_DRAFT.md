# Research release notes — draft, not published

Original source license: **Apache-2.0**, with upstream notices preserved; see [license scope](LICENSE_SCOPE.md). Release tag, final source commit, immutable model commit and checkpoint license: **not assigned**. Do not describe this document as a published release or replace unresolved fields with sample identities.

Current priority is a [source-only GitHub candidate](SOURCE_ONLY_RELEASE.md), not the final model-backed release described below. A source clone contains no weights or CTs and cannot yet perform inference without separately authorized artifacts. Model hosting, final immutable revision fields and fresh-install inference acceptance remain unfinished.

## Included

- Inference-only MONAI Label / nnU-Net integration with Models A/B/C, using locked best checkpoints and preprocessing metadata.
- Full-input Model B runtime with original-geometry reconstruction, strict label/hash/provenance guards, a default reference profile and experimental opt-in fast profile.
- Locked internal-cohort evaluation, per-case metrics, seeded bootstrap intervals, paired exploratory comparisons and component/failure analysis.
- 3D Slicer integration and axial/coronal/sagittal/3D visualization with saved editable segmentation and fitted whole-input views.
- Source/model split, immutable download verification, deterministic source packages, local publication inventories/exports and a read-only GitHub safety workflow.
- Operator manual, notices, provenance and reviewed anonymous numerical evidence; no CTs, masks or patient figures in release payloads.

## Verification scope

The pinned current Model B fast profile completed ten full internal LiTS scans, averaging 312.25 seconds of recorded CLI inference/export per case. Mean tumor Dice was 0.5304 (95% bootstrap CI 0.3076–0.7341), with two complete tumor misses and substantial false positives. Timing excludes the complete Slicer transport workflow. This is previously examined retrospective internal evidence, not an untouched external test.

A separately extracted source package with verified local model copies passed default-reference Model B inference through the installed Slicer extension on a technical crop, including full supplied-input geometry, all four views and editable segmentation. It reused an existing working Conda environment. A fresh remote dependency/model installation remains required.

Historical B/C masks reproduce saved counts, but their checkpoint identities are unpinned. Point estimates do not establish Model B superiority or fast/reference equivalence. Connected components are engineering annotation islands, not expert-adjudicated lesions. Keep reference as default and preserve the working CPU fallback.

## Intended use

Academic reproducibility and engineering evaluation only. Not clinically validated and not established for diagnosis, treatment decisions or unsupervised patient use. Expert review, independent validation and quality acceptance remain separate from software release readiness.

## Finish before publication

Confirm institutional/dataset redistribution terms, choose permitted weight terms, finalize model destinations/visibility, verify immutable remote downloads, complete fresh-install UAT, run the combined publication gate, and record actual baseline/final source and model commits. See [release workflow](RELEASE_HANDOFF.md), [publishing checklist](PUBLISHING_CHECKLIST.md) and [current audit](THREE_BLOCKER_AUDIT.md).

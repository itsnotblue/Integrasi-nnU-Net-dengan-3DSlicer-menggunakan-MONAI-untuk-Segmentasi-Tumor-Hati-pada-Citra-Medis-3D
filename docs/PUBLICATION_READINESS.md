# Readiness summary

## Software

The source contains the MONAI Label/Slicer integration, inference CLI, cohort evaluation, model-integrity tools and synthetic tests. Original source is Apache-2.0.

Weights are not published. A fresh source clone can run source checks, but cannot perform inference until the exact model artifacts are installed. A fresh dependency installation and immutable remote model download still need verification.

Local integration checks registered A/B/C and ran default-reference Model B through Slicer's MONAI Label extension on a technical crop. Geometry, four views and editable segmentation passed. Those checks reused existing dependencies and local verified weights; see the [UAT summary](UAT_SUMMARY.md).

## Segmentation evidence

The current pinned Model B fast run completed ten full LiTS scans:

| Measure | Result |
| --- | --- |
| Mean tumor Dice | 0.5304; 95% bootstrap CI 0.3076–0.7341 |
| Mean liver-label Dice | 0.8840 |
| Complete tumor misses | 2 of 10 scans |
| Mean CLI inference/export latency | 312.25 seconds |

These are retrospective internal results, not external validation. Historical B/C masks have unpinned checkpoints, so comparisons cannot isolate a profile-only effect. Fast/reference quality equivalence is not established.

See [current results](evaluation/CURRENT_FAST_RESULTS.md), [historical results](evaluation/FULL_VOLUME_RESULTS.md) and the [evaluation protocol](EVALUATION_PROTOCOL.md). Expert failure review and independent evaluation are still needed. The software is not clinically validated.

## Distribution

Code and model publication are separate. Weight terms, dataset/institutional rights, immutable model revisions and a fresh model-backed installation remain open. Follow the [publishing checklist](PUBLISHING_CHECKLIST.md).

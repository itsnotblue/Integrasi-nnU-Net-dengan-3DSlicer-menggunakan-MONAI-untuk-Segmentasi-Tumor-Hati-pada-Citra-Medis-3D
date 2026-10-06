# Documentation

## Using the project

| Guide | Contents |
| --- | --- |
| [User manual](USER_MANUAL.md) | Environment and model installation, Slicer inference, four-view capture, saving and troubleshooting. |
| [Model selection](MODEL_SELECTION.md) | Models A/B/C, recovered training splits and comparison limits. |
| [Inference performance](INFERENCE_PERFORMANCE.md) | CPU/CUDA measurements and reference/fast profiles. |
| [Evaluation protocol](EVALUATION_PROTOCOL.md) | Cohort locking, provenance, metrics, failures and statistical comparisons. |
| [UAT summary](UAT_SUMMARY.md) | Local integration checks and their scope. |
| [Current full-volume results](evaluation/CURRENT_FAST_RESULTS.md) | Pinned Model B fast run, per-case accuracy, failures and latency. |
| [Historical full-volume results](evaluation/FULL_VOLUME_RESULTS.md) | Recovered B/C predictions; checkpoint identities are unpinned. |

## Maintaining a release

| Guide | Contents |
| --- | --- |
| [Source distribution](SOURCE_ONLY_RELEASE.md) | What source Git includes and what remains separate. |
| [License scope](LICENSE_SCOPE.md) | Code, third-party notices, weights and data. |
| [Publishing checklist](PUBLISHING_CHECKLIST.md) | Source publication and later model-release checks. |
| [Release workflow](RELEASE_HANDOFF.md) | Exact payloads, source/model revisions and verification order. |
| [Release feasibility](RELEASE_FEASIBILITY.md) | Artifact sizes, tooling and earlier preparation findings. |
| [Readiness summary](PUBLICATION_READINESS.md) | Current engineering and accuracy limitations. |
| [Three-blocker audit](THREE_BLOCKER_AUDIT.md) | Full-volume execution, evaluation evidence and model distribution. |
| [Release-notes draft](RELEASE_NOTES_DRAFT.md) | Summary for the later model-backed release. |

The `evaluation/` directory contains reviewed numerical evidence, not CTs, masks or patient screenshots. Earlier test counts and preparation states in dated reports describe those snapshots; use CI and the current checkout for source checks.

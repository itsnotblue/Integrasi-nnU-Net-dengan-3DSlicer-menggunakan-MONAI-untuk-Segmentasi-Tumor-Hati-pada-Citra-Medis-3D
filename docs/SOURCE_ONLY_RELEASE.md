# Source distribution

This repository publishes the software before the pretrained models.

Included: application code, scripts, synthetic tests, environment specifications, documentation, license/notices, model hashes and reviewed numerical evaluation tables.

Not included: checkpoints, CTs, annotation masks, patient figures, environments, credentials, private logs or training caches. `radiology_portable/` is only a compatibility link directory.

A fresh clone can run the source checks in the [README](../README.md). Inference requires the exact checkpoints and preprocessing files listed in `configs/model_artifacts.lock.json`. The public model repository and immutable download revision are not available yet.

[Original source is Apache-2.0](LICENSE_SCOPE.md). Weight/data terms remain separate. Source publication does not mean a complete pretrained-model release or a validated clinical system.

The GitHub workflow checks documentation links, source privacy, release-tool tests and Python syntax on Windows and Linux. The complete synthetic suite needs the inference/evaluation dependencies; real-model UAT and accuracy assessment are separate.

For the later model release, follow the [checklist](PUBLISHING_CHECKLIST.md) and [release workflow](RELEASE_HANDOFF.md). The combined `validate_release.py --publication` gate stays blocked until model rights and exact source/model revisions are recorded.

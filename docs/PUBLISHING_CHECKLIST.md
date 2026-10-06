# Publishing checklist

## Source repository

- [x] Preserve existing Git history in a separate clean checkout.
- [x] Separate app, scripts, configuration, tests and documentation.
- [x] Add Apache-2.0 license, citation and retained upstream notices.
- [x] Exclude weights, medical data, figures, secrets, environments and private logs.
- [x] Provide model-free source checks and read-only Windows/Linux CI.
- [x] State that pretrained weights are not yet available publicly.
- [ ] Run final tests, source privacy, documentation links and exact staged-file review.
- [ ] Commit/push without force or visibility changes.
- [ ] Verify the remote revision and hosted CI.

## Later model-backed release

- [ ] Confirm permitted checkpoint terms and data/institutional redistribution requirements.
- [ ] Add weight license, model card, provenance and checksums.
- [ ] Choose and authenticate the model host and repository visibility.
- [ ] Upload only the exact verified artifact/document inventory.
- [ ] Record the baseline source commit and immutable model revision.
- [ ] Verify a separate download against all twelve locked runtime artifacts.
- [ ] Pass the combined publication gate.
- [ ] Test fresh dependencies, model installation, server registration and A/B/C inference.
- [ ] Verify Slicer geometry, four views and saved editable segmentation.
- [ ] Record CPU/CUDA latency with its measurement scope.
- [ ] Tag the tested source/model pair and finalize release notes.

Source checks and integration UAT do not establish segmentation accuracy. Keep `reference` as the default; the experimental `fast` profile still needs a controlled accuracy comparison. See the [release workflow](RELEASE_HANDOFF.md) and [evaluation protocol](EVALUATION_PROTOCOL.md).

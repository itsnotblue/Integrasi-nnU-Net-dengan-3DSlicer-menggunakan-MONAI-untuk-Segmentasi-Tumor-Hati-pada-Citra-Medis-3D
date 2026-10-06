# Release feasibility and owner decisions

Audit date: 2026-10-06.

## Feasibility

GitHub can host the clean source tree. The three `.pth` files are individually 235-238 MiB and exceed GitHub's ordinary 100 MiB file limit, so they should stay outside source Git. [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Hugging Face is feasible for these nnU-Net checkpoints. The three weights total 742575921 bytes (about 708 MiB), plus approximately 72 KiB of runtime JSON metadata. Current Hub documentation lists 100 GB private storage for free accounts and best-effort public storage; these artifacts are far smaller than its recommended large-file size. Available account quota and permission must still be checked at upload time. Hosting a model repository stores artifacts; it does not provide a running MONAI Label/Slicer service automatically. [Hugging Face storage limits](https://huggingface.co/docs/hub/storage-limits).

The local Hugging Face tooling is installed (`huggingface_hub==1.2.3`, `hf` available in both Conda environments), but no Hugging Face credential is configured in the tested GPU environment. The check inspected only credential presence and did not display any credential. Authentication and actual external publication remain pending.

The clean `github_repo` directory is staging, without `.git`. The older nested repository has the project's existing GitHub remote and untracked legacy application/data. Publish from a clean checkout of that remote or initialize a separate clean repository after deciding whether to preserve history. Do not copy the legacy untracked data into the release.

## Implemented release controls

- `configs/model_artifacts.lock.json` pins SHA-256 and sizes for all 12 runtime files.
- `scripts/model_artifacts.py` verifies local artifacts or downloads a full immutable 40-character Hugging Face commit, then validates all files before installing them.
- `scripts/download_models.ps1` wraps the verified downloader; the former mutable `main` default was removed.
- `scripts/validate_release.py` checks the source/model staging, unexpected artifacts, credentials, machine-specific paths, GitHub-size limits, and the publication decisions.
- The Hugging Face model card separates historical evaluation, annotation-selected crop UAT, and full-volume evidence.
- `CITATION.cff`, third-party notices, and the supplied MONAI Apache license text are retained.

Preparation checks can pass while publication checks fail. This records unfinished owner decisions explicitly rather than assigning permissions automatically.

## Verify the local candidate

From `github_repo`, with the selected Conda environment active:

```powershell
python scripts/model_artifacts.py verify --model-root ../huggingface_model_repo
python scripts/validate_release.py --model-root ../huggingface_model_repo
python scripts/validate_release.py --model-root ../huggingface_model_repo --publication
```

The first two should pass. The third must fail until the owner decisions and remote identities below are recorded. The release validator is a preparation/identity gate; it does not replace full-volume UAT, cohort leakage checks, accuracy evaluation, or a fresh installation test.

Verification performed on 2026-10-06: all 12 actual model artifacts passed byte-size and SHA-256 checks; preparation validation passed with zero errors; public validation rejected the seven unresolved decision/identity fields. All eight synthetic integrity tests passed, covering corruption, missing metadata, path traversal, incomplete locks, mutable revisions, preservation of existing files on rejected downloads, and successful verified installation. The PowerShell wrapper and citation/model-card YAML parsed successfully. The tests used tiny synthetic data and did not alter the actual model directories. No remote download/upload was tested because no model repository has been published.

The full-volume runtime and cohort runner also passed 31 tests using mocked predictors and tiny images. These checks cover physical geometry, invalid labels/logits/settings, nonoverwrite, unique output artifacts, and resume verification of hashes, geometry, settings and report identity. The cohort runner reads CT inputs and manifest geometry without opening reference masks. Its `{case_id}.nii.gz` output naming matches the evaluation script. Source layout checks now require the new runtime, batch, cohort-locking and evaluation tools plus all necessary model metadata.

`summarize_inference_run.py` requires a completed locked cohort and verifies the exact case order, manifest/input/checkpoint/output hashes, profile/device, sidecar geometry, software/hardware identity and finite timing stages. It reads metadata and prediction bytes, not CT/reference files. Its synthetic tests cover incomplete/mismatched/tampered artifacts, cache-state classification, CPU/GPU metadata and nonoverwrite. The completed ten-case CUDA run passed these strict checks; the reviewed public-LiTS latency JSON/CSV retain no machine-specific paths. Reports for other user cases stay private until their identifiers and rights are reviewed.

## Operator and evaluation workflow

The launchers now accept environment, profile and port, using CPU/reference/8002 by default. `start_server.bat liver-seg-gpu fast 8002` selects the separate GPU environment; it does not replace the CPU fallback. `reference` retains Model B mirroring and ordinary nnU-Net precision; `fast` disables mirroring and uses float32 network calls while retaining full-volume tiling and geometry reconstruction. Fast remains an experimental accuracy candidate. Models A/C retain their historical inference implementations.

Use `run_inference.py` for a complete scan and `run_cohort_inference.py` for a locked cohort. Freeze inputs with `lock_cohort.py`, evaluate saved labelmaps with `evaluate_predictions.py`, then compare equivalent cohort/protocol reports with `compare_evaluations.py`. File hashes, geometry and settings are retained; references are used for preparation/evaluation rather than inference or manual ROI selection. The detailed commands and failure definitions are in the user manual and evaluation protocol.

Run `summarize_inference_run.py` on the completed run report and locked manifest to obtain stage/end-to-end/tiling/peak-GPU statistics. It separates the first model-loading case, later cached-runtime cases and reused prior measurements with unknown original cache status. This is latency/integrity evidence, not an accuracy or independently decoded image-geometry evaluation. The latter remains the evaluation script's responsibility.

`requirements-evaluation.txt` pins SciPy 1.15.3 and Matplotlib 3.10.8 plus the NumPy/SimpleITK versions observed in the working environment. Install it before quantitative evaluation or `create_failure_panels.py`. This isolates explicit evaluation/figure dependencies from reliance on unpinned transitive packages. Figure selection may use annotations retrospectively; inference input/ROI selection does not.

`slicer_view_result.py` validates and captures a complete saved result in Four-Up layout in a new Slicer process. `slicer_inference_smoke.py` additionally invokes the installed MONAI Label extension against a localhost server. Both save private CT-derived artifacts; the source archive includes their code, not the CT images, segmentations, screenshots or response logs.

## Deterministic source candidate archive

Build an archive after reviewing the final source/documents:

```powershell
python scripts/package_source.py --model-root ../huggingface_model_repo --output ../distributions/liver-tumor-segmentation-source-candidate-2026-10-06.zip
```

The generator verifies preparation and all 12 model-artifact hashes, then packages only allowlisted source/configuration/documents/tests/license files. It checks archive CRCs and every packaged byte against the captured source snapshot. Fixed ZIP timestamps, file order and permissions make repeated unchanged snapshots deterministic. Separate manifest and SHA-256 files record the inventory and archive identity. Existing outputs are never replaced; use a new candidate name after changes.

The evaluation agent reviewed the explicitly allowlisted `docs/evaluation` tables and summaries as privacy-safe: public LiTS IDs, scalar metrics, failure flags, attribution and hashes only. They contain no images or absolute private paths. All other evaluation results, checkpoints, CT/reference volumes, generated labelmaps, screenshots, logs, caches, environments, credentials and test scratch are excluded. Six synthetic packaging tests passed, including deterministic archive identity, exclusion rules, credential rejection and nonoverwrite. The manifest marks this as a local candidate and carries unresolved publication decisions.

The reviewed evidence now includes the completed current pinned Model B fast cohort, its strict stage/latency JSON and CSV, and the historical full-volume B/C analysis. `export_public_evaluation.py` applies an explicit numerical/provenance allowlist rather than copying raw local evaluation reports, which can contain sidecar paths. Keep technical execution and observed retrospective accuracy separate: the candidate is opt-in, historical weights are not pinned, and expert failure review/independent validation remain outstanding. See `docs/evaluation/CURRENT_FAST_RESULTS.md` for the exact current results and limitations.

## Owner decisions needed before publication

1. Choose terms for custom source additions. Existing MONAI sample headers remain Apache 2.0; compatible terms for custom work require an owner decision. Add `LICENSE` and record the identifier in `configs/release_manifest.json`.
2. Confirm checkpoint redistribution rights against actual dataset acquisition and institutional terms; choose the weight license, add the model-repository `LICENSE`, and record the decision and evidence reference.
3. Select the Hugging Face destination account/repository. Log in on the trusted computer using `hf auth login`; never put a token in a command, file, screenshot, chat, or Git commit.
4. Decide whether the clean source should continue the existing GitHub repository history or use a new source repository. Record exact source/model commits once published.

## Publication sequence after decisions

The new local-only `prepare_publication.py` planner/exporter provides exact source and model/document inventories plus a source-only export without Git initialization, login or upload. The reviewed read-only GitHub safety workflow checks source privacy and synthetic release tools without patient data, weights or secrets. See [the detailed handoff](RELEASE_HANDOFF.md) for owner decisions, installed-versus-new CLI authentication capabilities, history-preserving source preparation and the staged publication order. This is preparation, not hosted CI or a completed remote release.

Review the complete candidate, run full-volume evaluation and fresh-install UAT, and validate the local preparation gate. In the selected clean Git repository, commit the reviewed source, upload only the allowed model-repository files with `hf upload OWNER/MODEL_REPOSITORY ../huggingface_model_repo . --repo-type model`, and record the returned immutable model commit. The CLI supports commit-specific downloads. [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

Fill `configs/release_manifest.json` with the actual repository, model commit, licensing decisions, and the source evaluation commit. Run `--publication`, commit the final manifest, tag the final source release, and record the manifest-bearing commit in the release notes to avoid a self-referential commit hash. Test a clean download of the recorded model revision with the locked checksums before publishing the final tag. Older source ZIPs predate this work; use the allowlisted generator for the final reviewed commit. A candidate ZIP does not by itself confirm a fresh installation or grant redistribution rights.

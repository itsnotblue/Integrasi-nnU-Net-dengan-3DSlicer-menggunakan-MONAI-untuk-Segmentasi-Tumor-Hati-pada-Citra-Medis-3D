# Release workflow

## Payloads

| Destination | Contents |
| --- | --- |
| GitHub | Source, tests, environment files, documentation, licenses/notices and reviewed numerical evidence. |
| Model host | A/B/C best checkpoints, their preprocessing metadata, model card, provenance, checksums and permitted weight terms. |
| Private workspace | CTs, annotations, generated segmentations/figures, environments, training caches and private logs. |

Never upload the entire research workspace. Preserve the existing Git history and repository visibility.

## Prepare a source snapshot

Run from the repository root, with verified local models staged separately:

```powershell
python scripts/prepare_publication.py check-source
python scripts/prepare_publication.py plan --model-root ../huggingface_model_repo
python scripts/prepare_publication.py export --model-root ../huggingface_model_repo --output-dir ../publication-preparation-NEW-NAME
```

The exporter verifies all twelve runtime artifacts but copies only source. Existing export destinations are never overwritten. Regenerate inventories after edits; a snapshot digest is not a Git commit.

Review the exact paths and staged contents before committing. Keep `scripts/start_server.sh` executable in Git. Push normally, without force, then verify the remote revision and Windows/Linux CI.

## Publish models later

1. Document checkpoint terms, data-acquisition/privacy requirements and institutional or collaborator redistribution rights. Source licensing does not resolve these.
2. Select the model-host repository and visibility. Keep publisher tools separate from the working inference environments; never put tokens in code or command arguments.
3. Record the clean source baseline in `configs/release_manifest.json`. Upload only the freshly verified fifteen-file model/document selection, adding the chosen weight terms as needed.
4. Record the actual full model commit. Download that immutable revision into a separate directory and verify all locked hashes/sizes.
5. Complete the manifest and model card, then run the publication gate:

   ```powershell
   python scripts/validate_release.py --model-root ../huggingface_model_repo --publication
   ```

6. Test a fresh dependency/model installation, server registration, permitted-input inference and Slicer four-view/save workflow.
7. Commit the final manifest and tag the tested source revision. Record the manifest-bearing commit separately from its baseline hash to avoid self-reference.

The model host stores files; it does not run the MONAI Label server. Local tests with existing dependencies/models do not replace a fresh remote installation.

See the [publishing checklist](PUBLISHING_CHECKLIST.md) and [readiness summary](PUBLICATION_READINESS.md).

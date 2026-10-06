# Liver Tumor Segmentation

nnU-Net v2 liver and tumor segmentation through MONAI Label and 3D Slicer. Developed for a university final project; this repository runs saved models, not training.

The repository contains source code, tests and evaluation reports. **Model weights and CT data are not included.** A public model download is not available yet, so a new clone can run source checks but cannot run inference without the weights.

Research use only. The models are not clinically validated.

## Models

| Model | Slicer key | Configuration | Recorded training pool |
| --- | --- | --- | --- |
| A | `nnunet_liver` | 3D low resolution | 10 cases |
| B | `nnunet_liver_modelb` | 3D full resolution | 120 cases; 96 train / 24 validation |
| C | `nnunet_liver_modelc` | Fine-tuned from B | 5 cases; 4 train / 1 validation |

Output labels: `0` background, `1` liver, `2` tumor.

Model B is the current operational default, not a proven winner. Training pools and recovered splits are described in [model selection](docs/MODEL_SELECTION.md). The [full-volume results](docs/evaluation/CURRENT_FAST_RESULTS.md) include missed tumors and false positives.

## Run source checks

Python 3.10+ is sufficient; these checks need no models or inference dependencies.

```bash
python scripts/prepare_publication.py check-source
python -m unittest discover -s tests -p test_documentation.py -v
python -m unittest discover -s tests -p test_release_artifacts.py -v
python -m unittest discover -s tests -p test_source_package.py -v
python -m unittest discover -s tests -p test_prepare_publication.py -v
python -m unittest discover -s tests -p test_inference_latency.py -v
```

## Run inference in Slicer

The tested setup is Windows, Conda and 3D Slicer 5.10.0 with the MONAI Label extension. Run commands from the repository root.

1. Create the CPU environment:

   ```powershell
   conda env create -f environment-cpu.yml
   conda activate liver-seg-cpu
   ```

2. Install the exact A/B/C checkpoints and preprocessing files in `app/radiology/lib/models/`. Each `model_a`, `model_b` and `model_c` directory needs `dataset.json`, `plans.json`, `dataset_fingerprint.json` and `fold_0/checkpoint_best.pth`. Verify them before starting:

   ```powershell
   python scripts/model_artifacts.py verify
   .\scripts\verify_layout.ps1 -RequireModels
   ```

   The [user manual](docs/USER_MANUAL.md) covers installation and the verified downloader. Its download command will need a published model repository and an immutable revision.

3. Start the server:

   ```powershell
   .\scripts\start_server.bat liver-seg-cpu reference 8002
   ```

4. In Slicer's MONAI Label module, connect to `http://127.0.0.1:8002`, load a CT and select `nnunet_liver_modelb`. Run inference, inspect the axial/coronal/sagittal/3D views, and save the segmentation.

For NVIDIA CUDA, use `environment-gpu.yml` and launch with `liver-seg-gpu`. Keep `reference` as the default profile. Model B's `fast` profile is experimental and has not passed accuracy-equivalence testing. Full-volume reference inference can be slow; see [performance notes](docs/INFERENCE_PERFORMANCE.md). A shell launcher is also included; only the Windows workflow has been integration-tested.

## Repository layout

- `app/radiology/`: MONAI Label app and model tasks.
- `scripts/`: launchers, CLI inference, evaluation, Slicer capture and release tools.
- `configs/`: model hashes, labels, recovered splits and cohort metadata.
- `tests/`: synthetic regression tests.
- `docs/`: [user manual, methods, results and maintainer guides](docs/README.md).
- `licenses/`: retained upstream license text.

`radiology_portable/` contains links for the old repository paths, not another app. Local models, data, outputs and environments are ignored by Git.

## Documentation and license

Start with the [user manual](docs/USER_MANUAL.md) for inference, saving results and troubleshooting. For quantitative evaluation, use the [evaluation protocol](docs/EVALUATION_PROTOCOL.md).

Original code is [Apache-2.0](LICENSE). Upstream attributions are in [third-party notices](THIRD_PARTY_NOTICES.md); [weights and data have separate terms](docs/LICENSE_SCOPE.md). Citation metadata is in [CITATION.cff](CITATION.cff).

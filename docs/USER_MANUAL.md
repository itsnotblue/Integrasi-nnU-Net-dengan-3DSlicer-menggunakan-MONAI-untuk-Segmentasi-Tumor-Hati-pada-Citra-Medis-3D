# User Manual — Liver Tumor Segmentation

This package performs research inference with saved nnU-Net checkpoints. It preserves the complete supplied CT volume and original physical geometry. It does not retrain models. The source and model artifacts are a local release candidate; original source is Apache-2.0, while public hosting and checkpoint licensing remain pending. See [license scope](LICENSE_SCOPE.md).

The current GitHub distribution is a source-only candidate: a fresh clone contains no checkpoints or medical data and cannot perform inference yet. You can run the [lightweight source checks](SOURCE_ONLY_RELEASE.md) without installing the inference environment or connecting Hugging Face. The steps below describe runtime use after the exact authorized models are available; source publication alone does not complete them.

## Models and output

| Slicer model key | Model | Recorded dataset pool |
|---|---|---:|
| nnunet_liver | A, historical low-resolution experiment | 10 |
| nnunet_liver_modelb | B, provisional tumor-focused choice | 120; recovered 96 train / 24 validation |
| nnunet_liver_modelc | C, fine-tuned from B | 5; recovered 4 train / 1 validation |

Labels are 0/background, 1/liver and 2/lesion (tumor). An anatomical liver-region metric combines labels 1 and 2. The counts above describe dataset pools and their recovered splits, not an independently verified external validation cohort.

Each new Model B inference produces a complete-volume labelmap and JSON provenance record. Inspect the entire organ and all tumor components in axial, coronal, sagittal and 3D views. Some slices naturally contain no tumor. Use a separate labeled close-up for small lesions while retaining a whole-CT screenshot.

## 1. Prepare the environment

Install Conda, 3D Slicer and the MONAI Label Slicer extension separately. Run shell commands from the extracted source root. The tested Slicer capture used version 5.10.0; the package pins its Python inference dependencies in the environment/requirements files.

Create the CPU fallback:

    conda env create -f environment-cpu.yml
    conda activate liver-seg-cpu
    python -m pip install -r requirements-release.txt

For NVIDIA CUDA experiments, create the separate GPU environment:

    conda env create -f environment-gpu.yml
    conda activate liver-seg-gpu
    python -m pip install -r requirements-release.txt
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

The supplied GPU requirements use PyTorch 2.7.1+cu118. A compatible NVIDIA driver is required. Keep the CPU environment available. CUDA wheel support does not imply that every GPU has a faster float16 implementation.

## 2. Install and verify model artifacts

The model repository has not yet been published. Until publication, use only the locally reviewed staged artifacts, or have the owner provide the exact authorized repository and model commit. Do not substitute an unrelated checkpoint.

After publication, replace the two quoted placeholders with the actual repository ID and full 40-character commit:

    .\scripts\download_models.ps1 -RepoId "OWNER/MODEL_REPOSITORY" -Revision "FULL_40_CHARACTER_COMMIT"
    python scripts/model_artifacts.py verify
    .\scripts\verify_layout.ps1 -RequireModels

On other platforms use the same Python downloader:

    python scripts/model_artifacts.py download --repo-id OWNER/MODEL_REPOSITORY --revision FULL_40_CHARACTER_COMMIT

Each model directory must contain:

    app/radiology/lib/models/model_x/dataset.json
    app/radiology/lib/models/model_x/plans.json
    app/radiology/lib/models/model_x/dataset_fingerprint.json
    app/radiology/lib/models/model_x/fold_0/checkpoint_best.pth

The downloader verifies all requested files before installing them. SHA-256 and byte sizes are pinned in configs/model_artifacts.lock.json. It rejects mutable revisions such as main. Authenticate with the official Hugging Face login flow only if the destination requires it; never place credentials in scripts or source Git.

## 3. Choose a Model B inference profile

| Profile | Model B behavior | Interpretation |
|---|---|---|
| reference, default | Mirroring enabled; normal nnU-Net CUDA autocast or float32 CPU calls; Gaussian tiling, step 0.5 | Retained reference settings; full scans can be slow |
| fast, experimental | Mirroring disabled; float32 network calls on GPU/CPU; same Gaussian tiling and step 0.5 | Changes predictions and must pass accuracy comparison |

Both profiles reconstruct the full output. Both retain nnU-Net CPU aggregation. The profile option applies to the updated Model B task; A and C retain their historical task implementations.

The experimental full-volume Model B cohort completed all ten locked LiTS scans on a GTX 1660 Ti, averaging 312.25 seconds per scan (median 301.82; range 110.84–432.85). Mean tumor Dice was 0.5304, with two complete tumor misses and increased unmatched prediction components compared with historical B. Historical checkpoint identity is unpinned, so this does not establish quality equivalence or a controlled profile comparison. The earlier 363.34-second single-case pilot is retained as separate evidence. Reference CPU/GPU runs on comparable whole scans may take hours; no same-case whole-cohort reference speedup factor is claimed. Crop timings and crop metrics cannot stand in for complete-volume performance.

Select acceleration settings using a designated development/validation case. Then freeze them before measuring the final test cohort. Record any settings selected after reviewing test outputs as exploratory analysis.

## 4. Start the MONAI Label server

The Windows launcher accepts environment, profile and port. With no arguments it uses liver-seg-cpu, reference and 8002:

    .\scripts\start_server.bat liver-seg-cpu reference 8002

Experimental CUDA environment:

    .\scripts\start_server.bat liver-seg-gpu fast 8002

The shell launcher accepts the same arguments:

    bash scripts/start_server.sh liver-seg-cpu reference 8002

The launchers use conda run and register all three keys. They bind to this computer at http://127.0.0.1:8002. Keep the terminal open and confirm registration from a second terminal:

    Invoke-WebRequest http://127.0.0.1:8002/info

For a server restricted to Model B with an explicit device, activate the chosen environment and run:

    python -m monailabel.main start_server --app app/radiology --studies data/studies --host 127.0.0.1 --port 8002 --conf models nnunet_liver_modelb --conf sam2 false --conf scribbles false --conf skip_trainers true --conf inference_profile fast --conf device cuda --conf inference_threads 2

Use reference/cpu for an explicit CPU fallback. If CUDA is requested but unavailable, the updated task raises an error; it does not silently change the comparison device.

## 5. Run a new case interactively in Slicer

1. Open Slicer and the MONAI Label module.
2. Set the same local server URL/port used above and refresh the server settings.
3. Confirm nnunet_liver_modelb is registered and select it for Auto Segmentation.
4. Load a complete scalar 3D CT. If required, upload/add it through the local MONAI Label module.
5. Run inference and wait for completion. Avoid cropping the liver/tumor using a reference annotation.
6. Select Four-Up layout and inspect axial, coronal, sagittal and 3D panels.
7. Enable the segmentation's 3D visibility and create its closed-surface representation if the 3D panel is empty.
8. Fit the slice views and reset the 3D focal point so organ boundaries are visible. Save a whole-volume view before taking a tumor close-up.
9. Save the returned labelmap and an editable segmentation in a private output directory.

The overlays are predictions. Review false positives, missed components and incomplete boundaries across the volume. A working application and an attractive screenshot do not demonstrate clinical accuracy.

## 6. Run full-volume inference from the CLI

From the chosen environment:

    python scripts/run_inference.py --model-dir app/radiology/lib/models/model_b --input data/images/CASE_0000.nii.gz --output outputs/case_reference/CASE.nii.gz --profile reference --device cpu --threads 2

For the experimental GPU profile, use a new output name and fast/cuda:

    python scripts/run_inference.py --model-dir app/radiology/lib/models/model_b --input data/images/CASE_0000.nii.gz --output outputs/case_fast/CASE.nii.gz --profile fast --device cuda --threads 2

Supported outputs are .nii.gz and .nrrd. The CLI refuses to replace an existing output or sidecar. The sidecar records input/checkpoint/output SHA-256, full dimensions, spacing, origin, direction, labels, settings, software, hardware and stage timings. Inference receives the CT alone; no reference mask is read.

## 7. Capture four-view evidence using a saved result

Run the capture inside a new Slicer instance, not the Conda Python interpreter. It preserves existing user scenes by refusing an instance that already contains a CT. The script loads the complete CT and saved prediction, verifies matching physical geometry, creates colored liver/tumor surfaces, saves results, then exits that new instance.

In PowerShell, replace the Slicer executable placeholder:

    $env:LIVER_AUDIT_IMAGE=(Resolve-Path "data/images/CASE_0000.nii.gz").Path
    $env:LIVER_AUDIT_PREDICTION=(Resolve-Path "outputs/case_fast/CASE.nii.gz").Path
    $env:LIVER_AUDIT_OUTPUT_DIR=[System.IO.Path]::GetFullPath("outputs/slicer_case_fast")
    $slicerExecutable="REPLACE_WITH_SLICER_EXECUTABLE"
    $captureScript=(Resolve-Path "scripts/slicer_view_result.py").Path
    $slicerArguments='--python-script "{0}"' -f $captureScript
    Start-Process -FilePath $slicerExecutable -ArgumentList $slicerArguments -WindowStyle Hidden -Wait

Use a new output directory. It saves:

- slicer_full_volume_four_up.png: axial, coronal, sagittal and 3D panels.
- model_b_full_volume.seg.nrrd: editable Slicer segmentation.
- slicer_visualization.json: geometry, labels, Slicer version and capture details.

Slice locations are chosen from the prediction, not from a reference mask. A screenshot of selected slices cannot show every tumor. Review the entire saved labelmap separately.

## 8. Test inference through the actual Slicer extension

With the local Model B server running, use another new Slicer process:

    $env:LIVER_AUDIT_IMAGE=(Resolve-Path "data/images/CASE_0000.nii.gz").Path
    $env:LIVER_AUDIT_OUTPUT_DIR=[System.IO.Path]::GetFullPath("outputs/slicer_http_case")
    $env:LIVER_AUDIT_SERVER="http://127.0.0.1:8002"
    $slicerExecutable="REPLACE_WITH_SLICER_EXECUTABLE"
    $inferenceScript=(Resolve-Path "scripts/slicer_inference_smoke.py").Path
    $slicerArguments='--python-script "{0}"' -f $inferenceScript
    Start-Process -FilePath $slicerExecutable -ArgumentList $slicerArguments -WindowStyle Hidden -Wait

The script invokes the installed extension's inference client, saves its labelmap and response/timing record, then calls the four-view viewer. It accepts only a localhost server. This is an integration test; its response file can contain machine paths and must remain private.

## 9. Freeze a cohort, infer and evaluate

Install the pinned evaluation and plotting dependencies in the selected environment:

    python -m pip install -r requirements-evaluation.txt

These add SciPy 1.15.3 and Matplotlib 3.10.8 alongside the verified NumPy/SimpleITK versions. They are independent of model training and do not require CUDA for metric/figure computation.

Put legally obtained inputs under data/images and references under data/reference. The source includes portable LiTS cohort metadata and recovered split files under configs/evaluation, but no images or masks. The recovered test IDs are LITS_121 through LITS_130; see EVALUATION_PROTOCOL.md for their source-index mapping, historical exposure and remaining provenance limitations.

Create a new lock before inference:

    python scripts/lock_cohort.py --image-dir data/images --reference-dir data/reference --cases CASE_A,CASE_B --cohort-id my_cohort_v1 --purpose technical_full_volume --output outputs/cohort.json

For a retrospective test, use the correct checkpoint-linked split and explicit purpose:

    python scripts/lock_cohort.py --image-dir data/images --reference-dir data/reference --cases LITS_121,LITS_122,LITS_123,LITS_124,LITS_125,LITS_126,LITS_127,LITS_128,LITS_129,LITS_130 --cohort-id lits_121_130_retrospective_full_volume_v1 --purpose retrospective_test --split-file configs/evaluation/model_b_fold0_split.json --output outputs/lits_cohort.json

Run the complete cohort:

    python scripts/run_cohort_inference.py --manifest outputs/cohort.json --model-dir app/radiology/lib/models/model_b --output-dir outputs/model_b_reference --profile reference --device cpu --threads 2

Experimental CUDA cohort:

    python scripts/run_cohort_inference.py --manifest outputs/cohort.json --model-dir app/radiology/lib/models/model_b --output-dir outputs/model_b_fast --profile fast --device cuda --threads 2

Add --resume to reuse only verified matching artifacts. The runner checks input/output hashes, geometry, profile, threads, resolved device and report identity. It refuses incomplete, mismatched or overwritten results. A failed partial output must be investigated; use a new run directory rather than editing provenance metadata. Different compressed serializations of equal voxels still have different file hashes, so preserve input identity.

Evaluate after every case has completed:

    python scripts/evaluate_predictions.py --manifest outputs/cohort.json --prediction-dir outputs/model_b_fast --output-dir outputs/model_b_fast_evaluation --model model_b --checkpoint app/radiology/lib/models/model_b/fold_0/checkpoint_best.pth --inference-settings outputs/model_b_fast/run_settings.json

The output contains evaluation.json, per_case_metrics.csv and failures.csv. Geometry mismatch is rejected; evaluation does not silently resample. Definitions include liver label 1, tumor label 2, combined liver region, physical boundary distances, empty-mask policy, component matching and case-level confidence intervals.

Generate the completed-cohort latency report into a new private directory:

    python scripts/summarize_inference_run.py --manifest outputs/cohort.json --run-settings outputs/model_b_fast/run_settings.json --output-dir tmp/model_b_fast_runtime

This tool needs only Python's standard library; it reads the locked manifest, run report, sidecars and prediction bytes for hash checking, not CT/reference files or checkpoints. It rejects partial runs, changed/reordered cases, mismatched hashes/settings/devices, incomplete sidecar geometry and invalid timings. It writes inference_latency.json and per_case_latency.csv with stage/end-to-end times, tile counts, peak allocated GPU memory and count/mean/median/min/max. The first new case includes model loading; later new cases share the cached runtime. Reused results retain prior timings and are marked original cache status unknown. Stage timings do not include cohort preflight or Slicer/server transport. Reports preserve case IDs and need privacy review before publication.

To compare two completed runs:

    python scripts/compare_evaluations.py --left outputs/model_b_fast_evaluation/evaluation.json --right outputs/model_b_reference_evaluation/evaluation.json --output outputs/profile_comparison.json

Both runs must use the same cohort and metric protocol. Historical predictions have separate unpinned-checkpoint attribution and must not be presented as a newly reproduced same-checkpoint reference run. Follow EVALUATION_PROTOCOL.md for those options.

Use `scripts/export_public_evaluation.py --help` for the strict current/historical numerical exporter. It requires verified completed current provenance and an explicitly unpinned historical comparator, validates paired report identity, and uses an output allowlist. The included `docs/evaluation/current_fast_*` tables were reviewed as public-LiTS numerical evidence. Raw local reports can retain private sidecar paths; never copy them wholesale into source Git. Reports from a different dataset require a new privacy and rights review.

For a retrospective failure figure using completed evaluated masks:

    python scripts/create_failure_panels.py --manifest outputs/cohort.json --prediction-dirs outputs/model_b_fast --model-labels "Model B fast" --evaluations outputs/model_b_fast_evaluation/evaluation.json --case-ids CASE_A,CASE_B --output outputs/failure_panels.png

Use case IDs actually present in the manifest. The figure retains each complete axial CT field of view and shows CT, reference and prediction separately. It selects a reference-rich slice for illustration after inference, labels the reported Dice as a whole-volume metric, and saves figure provenance. The supplied renderer requires LPS-aligned axes rather than silently reorienting data. Keep the CT-derived figure outside source Git until image redistribution rights are confirmed.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| No model keys or connection refused | Check the running terminal, /info, localhost URL and matching port |
| Missing model or checksum mismatch | Verify exact artifacts and metadata; do not rename/flatten fold_0 or replace plans arbitrarily |
| CUDA unavailable | Confirm the GPU environment and driver, or select the retained CPU fallback |
| Reference inference is very slow | Record whole-volume timing; use fast only as an evaluated candidate with new output paths |
| Empty overlay on one slice | Review all planes and slice locations, label visibility and saved labels |
| Only one panel or empty 3D view | Select Four-Up, create closed surface and enable 3D visibility |
| Crop boundaries clip the organ | Use the original complete CT and rerun; fitting the camera cannot restore voxels omitted from input |
| Resume rejects a result | Preserve it; compare hashes, source input, settings and physical geometry, then use a new directory if needed |
| Capture reports an existing scene/output | Launch a new Slicer instance and new output directory |
| Slicer capture fails under normal Python | Run it with Slicer's --python-script option; qt/slicer/vtk are Slicer APIs |

## 11. Shutdown and reproducibility

Stop the server with Ctrl+C. Keep input data, generated masks, Slicer response files/screenshots and runtime logs private. Reviewed anonymous metric tables may accompany a scientific release; the actual scans and segmentations are separate artifacts with their own redistribution terms.

Run synthetic regression tests and release preparation checks:

    python -m unittest discover -s tests -p "test_*.py" -v
    python scripts/validate_release.py --model-root ../huggingface_model_repo

Source and checkpoint licenses, dataset/institutional rights, authenticated hosting destinations, exact source/model revisions, fresh installation, and full-volume accuracy acceptance remain separate publication gates. Refer to RELEASE_FEASIBILITY.md and PUBLISHING_CHECKLIST.md.

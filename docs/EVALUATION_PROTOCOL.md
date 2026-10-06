# Reproducible full-volume evaluation

Protocol version: 2026-10-06. No model retraining is required.

## Cohort and provenance

The recovered final Model B (`Dataset406_LiT`, fold 0) split contains 96 training IDs and 24 validation IDs, together `LITS_001` through `LITS_120`. Model C (`Dataset409_FnT`, fold 0) fine-tuned on `LITS_010`, `LITS_017`, `LITS_020`, `LITS_025`, with `LITS_005` for validation. Model C also inherits the exposure of its pretrained Model B. The 10 historical evaluation cases are exactly `LITS_121` through `LITS_130`; none of these IDs appears in either recovered split.

The earlier root `splits_final_131.json` belongs to a different 131-case experiment and includes these cases in training. It must not be substituted for the final B split. The Model A `Dataset408_LTS` split is also separate.

The packaged Model B fingerprint contains 120 spacings/shapes, C contains five and A contains 10. Model B's active `sourcePlans` normalization statistics differ from its packaged 120-case fingerprint and equal the five-case C fingerprint's statistics. Preserve the active plans used with the trained checkpoint. Recover the notebook's source-plan copying/planning history before asserting that preprocessing statistics were estimated exclusively from a particular training subset. This discrepancy alone does not prove test leakage.

This supports a retrospective reevaluation of the original internal test cohort. These cases and their results have already influenced model choice and demonstrations, so they cannot be represented as a new untouched external test. Source filenames are evidence of case separation, not proof of unique patient identity across scans. The scripts therefore retain `held_out_certified: false` rather than automatically certify patient-level leakage absence.

The local conversion script numbers source cases from one: `LITS_121` maps to original `volume-120` / `segmentation-120`, through `LITS_130` mapping to source `129`. SHA-256 comparisons verified exact image and reference identity for all 10 pairs. Do not infer that renamed `LITS_121` means original `volume-121`.

Portable cohort hashes, geometry and IDs are in `configs/evaluation/cohort_lits_121_130.json`. Place legally obtained, correspondingly converted files in `data/images` and `data/reference` or create your own manifest with the command below. These data directories are local and must not be published with the source.

## Freeze the inputs before inference

From the source repository:

```powershell
python scripts/lock_cohort.py --image-dir data/images --reference-dir data/reference --cases LITS_121,LITS_122,LITS_123,LITS_124,LITS_125,LITS_126,LITS_127,LITS_128,LITS_129,LITS_130 --cohort-id lits_121_130_retrospective_full_volume_v1 --purpose retrospective_test --split-file configs/evaluation/model_b_fold0_split.json --output results/cohort.json
```

The manifest contains input/reference file hashes and complete size, spacing, origin and direction. It refuses to overwrite an existing lock. It also records overlap against a supplied split. No case selection, ROI selection, cropping or model inference uses the reference annotation. An explicit `technical_crop` purpose is available only for acceptance testing and remains marked as cropped evidence.

Freeze the checkpoint and inference settings before examining this cohort's new accuracy. Compare settings using a designated development/validation case (for example `LITS_026` for Model B). Record any settings selected using test results as post hoc exploratory analysis.

## Evaluate completed predictions

```powershell
python scripts/evaluate_predictions.py --manifest results/cohort.json --prediction-dir results/model_b --output-dir results/model_b_evaluation --model model_b --checkpoint app/radiology/lib/models/model_b/fold_0/checkpoint_best.pth --inference-settings results/model_b/run_settings.json
```

Current evaluation requires a locked full-volume manifest and the provenance sidecar saved beside every prediction (`{prediction_filename}.json`). Supplying `--checkpoint` alone cannot establish attribution: its file hash must match each sidecar, along with the locked input hash and actual output hash. Sidecars must explicitly declare no manual ROI and no reference-annotation use, preserve complete input/output dimensions and physical geometry, and specify valid consistent settings, labels and timing. Actual output headers are checked before loading full images. Evaluation rejects changed files, invalid labels and any size/spacing/origin/direction mismatch; it never silently resamples predictions.

`--inference-settings` is optional in current mode but, when supplied, must be the **completed** `run_cohort_inference.py` report. Its completion flag/time, exact manifest/cohort/checkpoint IDs, case membership, resolved device, settings, per-case output hashes and timings must agree with all sidecars. Partial batches and missing/empty predictions are rejected before expensive CT/reference reads. Successful current reports use `checkpoint_attribution: verified_prediction_sidecars`. Historical mode accepts the old inference-arguments JSON and retains unknown checkpoint attribution.

Default prediction filenames are `{case_id}.nii.gz`; change `--prediction-pattern` for NRRD files. Use a new output directory for each model/settings version. Older cropped acceptance artifacts without sidecars can be measured only as explicitly unpinned historical evidence, never current verified full-volume results.

Outputs:

- `evaluation.json`: protocol, hashes, settings, software versions, per-case measurements, aggregate estimates and confidence intervals.
- `per_case_metrics.csv`: numerical data for supplementary tables and figures.
- `failures.csv`: all cases ranked by tumor Dice, with missed/false-positive components and review flags.

## Metric definitions

Report liver-only label 1, tumor-only label 2 and combined liver region labels 1 or 2 separately. The liver-only metric changes when a voxel is relabeled as tumor, so it is not interchangeable with complete anatomical liver-region Dice.

For each binary class report Dice, IoU, voxel precision, voxel recall, volumes and confusion counts. Report both all-case tumor metrics and the reference-positive subset to expose effects of empty cases.

Empty policy: both masks empty gives Dice/IoU 1 and surface distances 0. A zero precision/recall denominator gives `null`, never an invented perfect detection score. Exactly one empty mask gives undefined surface distances (`null`) and is explicitly counted. Cases with missing surfaces must remain in failure reporting; surface summary reports how many measurements were undefined.

HD95 is the 95th percentile of pooled bidirectional distances between 6-connected boundary voxels; ASSD is their pooled mean. Both use physical millimetres and retain anisotropic spacing. Rotated orthonormal directions are supported. Nonorthonormal axes are rejected. Temporary foreground bounding boxes are used only to reduce metric memory allocation, after whole-volume overlap/count calculation.

Tumor detection uses 26-connected annotation/prediction components, no small-component removal, and one-to-one maximum-cardinality matching at component IoU >= 0.1. A merged prediction cannot count as detection of multiple reference components. This is an explicit engineering criterion, not an expert lesion-adjudication claim: disconnected annotation islands may not correspond to distinct clinical lesions. Tiny components need expert review and sensitivity analysis under any justified exclusion rule.

Use case-level percentile bootstrap (2,000 replicates, seed 20261006) for the mean's 95% interval; report mean/sample SD, median and quartiles. Undefined values are excluded from that metric and counted. A single case has no estimated confidence interval. Pooled component sensitivity/precision intervals resample complete cases to preserve within-case clustering. If multiple scans belong to one patient, supply patient grouping and adapt the bootstrap to patient units before publication.

## Recover historical evidence without claiming new inference

```powershell
python scripts/summarize_historical_metrics.py --summary historical/model_b_summary.json --comparison-summary historical/model_c_summary.json --manifest results/cohort.json --model model_b --comparison-model model_c --output-dir results/historical_analysis
```

This independently recomputes Dice/IoU/precision/recall from saved confusion counts. It verifies full-volume voxel totals and each reference-label count against the locked local references, then produces macro statistics, paired B-minus-C intervals, and a ranked failure report. It cannot reconstruct surfaces, geometry alignment, combined liver-region Dice or component detection without historical masks. The historical checkpoint identity requires separate authoritative provenance.

When original historical masks are available, evaluate them with explicit unpinned attribution and verify their counts against the original summary:

```powershell
python scripts/evaluate_predictions.py --manifest results/cohort.json --prediction-dir historical/model_b --output-dir results/historical_model_b_spatial --model model_b --historical-predictions --historical-summary historical/model_b_summary.json --inference-settings historical/model_b/predict_from_raw_data_args.json
```

This hashes each mask, validates full geometry, computes surfaces/components and requires exact agreement with all saved TP/FP/FN/TN, reference/prediction counts, Dice and IoU. Its checkpoint hash remains `null` because historical filenames do not establish the actual weights. Supplying `--checkpoint` with `--historical-predictions` is rejected. Evaluate Model C separately with the same locked cohort and definitions.

## Validation completed

Numerical regressions cover empty cases, physical anisotropic distances, equivalence of sparse boundary queries to an independent dense distance transform, one-to-one lesion matching, misses/false positives, deterministic/singleton confidence intervals, paired case identity checks, geometry rejection, manifest integrity and historical count consistency. Provenance regressions reject invented checkpoint/input/output attribution, absent or changed sidecars, false ROI/reference claims, inconsistent settings, altered physical headers and partial/mismatched batch records. The existing Model B lesion crop reproduced tumor Dice 0.737259 exactly and revealed three unmatched reference components among six; its older metadata is technical crop evidence and is not promoted to verified current attribution.

```powershell
python -m unittest discover -s tests -v
```

All 20 historical B/C masks passed full geometry and exact saved confusion-count validation; complete spatial/component results are in [the historical report](evaluation/FULL_VOLUME_RESULTS.md). All ten new current fast-profile labelmaps also completed and passed strict sidecar/completed-batch provenance validation; [the current report](evaluation/CURRENT_FAST_RESULTS.md) records accuracy and an exploratory paired comparison against the unpinned historical B artifacts. Increased unmatched prediction components and wide tumor Dice uncertainty prevent an equivalence or clinical-readiness claim. External cohort validation, patient-identity verification, expert failure review and actual historical checkpoint linkage remain separate evidence requirements.

## Paired comparison and safe public export

```powershell
python scripts/compare_evaluations.py --left results/model_b_evaluation/evaluation.json --right results/historical_model_b_spatial/evaluation.json --output results/current_fast_vs_historical_b.json
python scripts/export_public_evaluation.py --current results/model_b_evaluation/evaluation.json --historical results/historical_model_b_spatial/evaluation.json --paired results/current_fast_vs_historical_b.json --output-dir results/public_metrics
```

The comparison requires the identical locked manifest, complete case membership and metric protocol. It retains the current pinned checksum and historical unknown checksum, rather than label the latter as the supplied current weights. A confidence interval containing zero is not evidence of equivalence; a controlled same-weights comparison with a justified predeclared tolerance is required for that claim. These retrospective comparisons are exploratory.

The public exporter validates paired source-report hashes and verified current/completed-cohort flags. It explicitly allowlists numerical metrics, public LiTS IDs, artifact hashes and settings, excluding local input/output paths and raw sidecar contents. It produces `current_fast_statistics.json`, `current_fast_per_case_metrics.csv`, `current_fast_failures.csv` and `current_fast_component_details.csv`, and refuses to overwrite them. Raw `evaluation.json` can embed machine paths in the inference sidecars and must stay local. Expert-reviewed CT-derived figures require separate image-publication permission.

## Retrospective failure figures

```powershell
python scripts/create_failure_panels.py --manifest results/cohort.json --prediction-dirs historical/model_b historical/model_c --model-labels "Model B" "Model C" --evaluations results/historical_model_b_spatial/evaluation.json results/historical_model_c_spatial/evaluation.json --output outputs/historical_failure_panels.png
```

The default fixed cases are 121 (false positives), 122 (complete miss), 125 (strong Dice with small-component misses), and 128 (complete miss). `--case-ids` can supply all 10 IDs. Each plane is selected by the maximum reference tumor voxel count solely for retrospective figure preparation; it never affects input extent or inference. The figure preserves the full axial CT field of view and uses the same HU window for CT, reference and model overlays. Displayed Dice is the previously validated whole-volume score, not a selected-slice score. Figure provenance is saved beside the PNG, including source geometry, plane selection and exact prediction hashes.

This generator expects native LPS-aligned images and rejects other directions rather than silently mislabel an anatomical view. A separate phantom regression verified plane selection, full field-of-view dimensions and saved PNG/provenance. CT-derived figures stay in local ignored outputs until image publication rights are confirmed; only the generator ships in the source release.

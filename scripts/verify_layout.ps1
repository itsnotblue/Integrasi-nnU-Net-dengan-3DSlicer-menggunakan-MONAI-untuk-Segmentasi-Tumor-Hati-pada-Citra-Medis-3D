param(
    [string]$ReleaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$RequireModels
)

$root = (Resolve-Path -LiteralPath $ReleaseRoot).Path
$required = @(
    "app\radiology\main.py",
    "app\radiology\lib\configs\nnunet_liver.py",
    "app\radiology\lib\configs\nnunet_liver_modelb.py",
    "app\radiology\lib\configs\nnunet_liver_modelc.py",
    "app\radiology\lib\infers\nnunet_runtime.py",
    "scripts\run_inference.py",
    "scripts\run_cohort_inference.py",
    "scripts\summarize_inference_run.py",
    "scripts\lock_cohort.py",
    "scripts\evaluation_common.py",
    "scripts\evaluate_predictions.py",
    "scripts\compare_evaluations.py",
    "scripts\export_public_evaluation.py",
    "scripts\create_failure_panels.py",
    "scripts\slicer_view_result.py",
    "scripts\slicer_inference_smoke.py",
    "scripts\model_artifacts.py",
    "configs\model_artifacts.lock.json"
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_)) })
if ($missing.Count -gt 0) {
    Write-Error ("Missing required source files:`n" + ($missing -join "`n"))
    exit 1
}

$modelFiles = foreach ($modelName in @("model_a", "model_b", "model_c")) {
    foreach ($modelFile in @("fold_0\checkpoint_best.pth", "dataset.json", "plans.json", "dataset_fingerprint.json")) {
        "app\radiology\lib\models\$modelName\$modelFile"
    }
}
$missingModels = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_)) })
if ($RequireModels -and $missingModels.Count -gt 0) {
    Write-Error ("Missing required model files; run download_models.ps1 first:`n" + ($missingModels -join "`n"))
    exit 1
}
if ($missingModels.Count -gt 0) {
    Write-Output "Source layout verified; model artifacts are not present (expected for the GitHub source-only tree)."
} else {
    Write-Output "Release layout verified: source and all three models with preprocessing metadata are present."
    Write-Output "Run scripts/model_artifacts.py verify to check exact byte sizes and SHA-256 hashes."
}

param(
    [Parameter(Mandatory = $true)]
    [string]$RepoId,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-fA-F0-9]{40}$')]
    [string]$Revision,
    [string]$Python = "python",
    [ValidateSet("model_a", "model_b", "model_c")]
    [string[]]$Models = @("model_a", "model_b", "model_c")
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ErrorActionPreference = "Stop"
& $Python (Join-Path $PSScriptRoot "model_artifacts.py") download `
    --repo-id $RepoId --revision $Revision --models @Models
if ($LASTEXITCODE -ne 0) {
    throw "Model download or checksum verification failed."
}
Write-Output "Exact pinned model artifacts downloaded and verified."

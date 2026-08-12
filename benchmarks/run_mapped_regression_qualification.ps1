param(
    [Parameter(Mandatory = $true)][string]$CurrentAnymeshSha,
    [Parameter(Mandatory = $true)][string]$CurrentAnygeometrySha,
    [Parameter(Mandatory = $true)][string]$HarnessSha256,
    [Parameter(Mandatory = $true)][string]$ComparatorSha256,
    [string]$FinalReport = 'C:\Github\ANYmesh\reports\native_hybrid\mapped_regression_comparison.json',
    [string]$FinalCleanupReport = 'C:\Github\ANYmesh\reports\native_hybrid\mapped_regression_cleanup.json'
)

$ErrorActionPreference = 'Stop'
$baselineAnymeshSha = 'e31f8c700b91796b93a8d2b21a6d44f70145eaed'
$baselineAnygeometrySha = 'f2d7793d7d32a6dcd772c7ed8701aca11b459288'
$currentAnymesh = 'C:\Github\ANYmesh'
$currentAnygeometry = 'C:\Github\ANYgeometry'
$runRoot = Join-Path $env:TEMP 'anymesher-mapped-regression-e31f8c7-f2d7793'
$baselineAnymesh = Join-Path $runRoot 'ANYmesh'
$baselineAnygeometry = Join-Path $runRoot 'ANYgeometry'
$samples = Join-Path $runRoot 'samples'
$temporaryReport = Join-Path $runRoot 'mapped_regression_comparison.json'
$harness = Join-Path $currentAnymesh 'benchmarks\mapped_regression_baseline.py'
$comparator = Join-Path $currentAnymesh 'benchmarks\compare_mapped_regression.py'
$createdAnymesh = $false
$createdAnygeometry = $false
$comparisonExitCode = 1
$cleanup = [System.Collections.Generic.List[object]]::new()

function Remove-RegisteredWorktree {
    param([string]$Repository, [string]$Target, [string]$Name)
    if (-not (Test-Path -LiteralPath $Target)) {
        $cleanup.Add([pscustomobject]@{ name = $Name; path = $Target; status = 'absent' })
        return
    }
    $resolved = (Resolve-Path -LiteralPath $Target).Path
    $registered = @(
        & git -C $Repository worktree list --porcelain |
            Where-Object { $_ -like 'worktree *' } |
            ForEach-Object { [System.IO.Path]::GetFullPath($_.Substring(9)) }
    )
    if ($registered -notcontains [System.IO.Path]::GetFullPath($resolved)) {
        throw "Refusing to remove unregistered worktree path: $resolved"
    }
    & git -C $Repository worktree remove --force $resolved
    if ($LASTEXITCODE -ne 0) {
        throw "git worktree remove failed for $resolved"
    }
    $cleanup.Add([pscustomobject]@{ name = $Name; path = $resolved; status = 'removed' })
}

try {
    if (Test-Path -LiteralPath $baselineAnymesh) {
        throw "Baseline ANYmesh target already exists: $baselineAnymesh"
    }
    if (Test-Path -LiteralPath $baselineAnygeometry) {
        throw "Baseline ANYgeometry target already exists: $baselineAnygeometry"
    }
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

    & git -C $currentAnymesh worktree add --detach $baselineAnymesh $baselineAnymeshSha
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create baseline ANYmesh worktree' }
    $createdAnymesh = $true

    & git -C $currentAnygeometry worktree add --detach $baselineAnygeometry $baselineAnygeometrySha
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create baseline ANYgeometry worktree' }
    $createdAnygeometry = $true

    & python $comparator `
        --baseline-anymesh-root $baselineAnymesh `
        --baseline-anygeometry-root $baselineAnygeometry `
        --current-anymesh-root $currentAnymesh `
        --current-anygeometry-root $currentAnygeometry `
        --baseline-anymesh-sha $baselineAnymeshSha `
        --baseline-anygeometry-sha $baselineAnygeometrySha `
        --current-anymesh-sha $CurrentAnymeshSha `
        --current-anygeometry-sha $CurrentAnygeometrySha `
        --expected-harness-sha256 $HarnessSha256 `
        --expected-comparator-sha256 $ComparatorSha256 `
        --elements 10000 `
        --leg-repeats 7 `
        --warmup 1 `
        --samples-dir $samples `
        --output $temporaryReport
    $comparisonExitCode = $LASTEXITCODE
}
finally {
    try {
        if ($createdAnygeometry) {
            Remove-RegisteredWorktree $currentAnygeometry $baselineAnygeometry 'ANYgeometry'
        } else {
            $cleanup.Add([pscustomobject]@{ name = 'ANYgeometry'; path = $baselineAnygeometry; status = 'not-created' })
        }
    } catch {
        $cleanup.Add([pscustomobject]@{ name = 'ANYgeometry'; path = $baselineAnygeometry; status = 'cleanup-failed'; error = $_.Exception.Message })
        $comparisonExitCode = 1
    }
    try {
        if ($createdAnymesh) {
            Remove-RegisteredWorktree $currentAnymesh $baselineAnymesh 'ANYmesh'
        } else {
            $cleanup.Add([pscustomobject]@{ name = 'ANYmesh'; path = $baselineAnymesh; status = 'not-created' })
        }
    } catch {
        $cleanup.Add([pscustomobject]@{ name = 'ANYmesh'; path = $baselineAnymesh; status = 'cleanup-failed'; error = $_.Exception.Message })
        $comparisonExitCode = 1
    }

    $cleanupDocument = [ordered]@{
        schema = 'anymesher.mapped_regression_cleanup.v1'
        run_root = $runRoot
        comparison_exit_code = $comparisonExitCode
        entries = $cleanup
    }
    $cleanupDirectory = Split-Path -Parent $FinalCleanupReport
    New-Item -ItemType Directory -Path $cleanupDirectory -Force | Out-Null
    $cleanupDocument | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $FinalCleanupReport
    if (Test-Path -LiteralPath $temporaryReport) {
        $reportDirectory = Split-Path -Parent $FinalReport
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
        Copy-Item -LiteralPath $temporaryReport -Destination $FinalReport -Force
    }
}

exit $comparisonExitCode

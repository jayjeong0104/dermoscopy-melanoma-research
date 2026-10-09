$ErrorActionPreference = 'Stop'
$taskProject = $PSScriptRoot
$taskOutput = Join-Path $taskProject 'artifacts\precision_rebuild_20261006\head_study'
$taskFeatureMarker = Join-Path $taskProject 'artifacts\precision_rebuild_20261006\features_matched_runtime\train_complete.json'
if (-not (Test-Path -LiteralPath $taskFeatureMarker)) { throw 'Training feature extraction is not complete.' }
$taskPython = (Get-Command python).Source
$taskAttempt = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$taskStandardLog = Join-Path $taskOutput "driver_$taskAttempt.log"
$taskErrorLog = Join-Path $taskOutput "driver_$taskAttempt.error.log"
$taskPrevious = @{}
foreach ($taskName in @('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')) {
    $taskPrevious[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
    [Environment]::SetEnvironmentVariable($taskName, '2', 'Process')
}
try {
    $taskChild = Start-Process -FilePath $taskPython -ArgumentList @('run_precision_rebuild.py') `
        -WorkingDirectory $taskProject -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $taskStandardLog `
        -RedirectStandardError $taskErrorLog
    Write-Output "Precision head study PID: $($taskChild.Id)"
    Write-Output "Log: $taskStandardLog"
    # Keep the owning shell alive: managed execution environments may terminate
    # its child processes when an otherwise detached launcher exits.
    $taskChild.WaitForExit()
    if ($taskChild.ExitCode -ne 0) { throw "Precision head study exited $($taskChild.ExitCode). See $taskErrorLog" }
} finally {
    foreach ($taskName in $taskPrevious.Keys) {
        [Environment]::SetEnvironmentVariable($taskName, $taskPrevious[$taskName], 'Process')
    }
}

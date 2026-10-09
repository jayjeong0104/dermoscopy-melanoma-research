$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$output = Join-Path $project 'artifacts\convergence_recovery_20261001'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$python = (Get-Command python -ErrorAction Stop).Source
$attempt = '{0}_{1}' -f (Get-Date -Format 'yyyyMMdd_HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
$stdout = Join-Path $output "$attempt.stdout.log"
$stderr = Join-Path $output "$attempt.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList @('-u', 'run_convergence_study.py') `
    -WorkingDirectory $project -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Seconds 3
$process.Refresh()
if ($process.HasExited -and $process.ExitCode -ne 0) {
    throw "Study exited with code $($process.ExitCode); inspect $stderr"
}
[PSCustomObject]@{ PID = $process.Id; StandardOutput = $stdout; StandardError = $stderr }

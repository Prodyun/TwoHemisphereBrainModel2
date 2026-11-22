# Start the real-data pipeline as a Windows scheduled task, so it runs on its own:
#  - independent of VS Code, Claude and the internet
#  - started again automatically every time you log in, until it has finished
#    (scripts/run_pipeline.py skips finished steps and finished experiment runs)
#
#   powershell -ExecutionPolicy Bypass -File scripts/start_pipeline_task.ps1          # start
#   powershell -ExecutionPolicy Bypass -File scripts/start_pipeline_task.ps1 -Remove  # remove the task
#
# Progress: results/pipeline/status.txt   Full log: results/pipeline/log.txt
param([switch]$Remove)

$name = "BrainModelPipeline"
$root = Split-Path -Parent $PSScriptRoot
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed scheduled task $name"
    return
}

# pythonw = no console window, so it can't be closed by accident
$python = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $python)) { throw "Python not found at $python" }

$action = New-ScheduledTaskAction -Execute $python -Argument "scripts\run_pipeline.py" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 2) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Two-hemisphere brain model: real-data pipeline (resumes after reboot)" -Force | Out-Null
Start-ScheduledTask -TaskName $name
Write-Output "Started scheduled task $name. Progress: results\pipeline\status.txt"

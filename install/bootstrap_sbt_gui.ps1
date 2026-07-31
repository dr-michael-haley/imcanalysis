param(
    [string]$EnvironmentName = $(
        if ($env:SBT_GUI_ENV) { $env:SBT_GUI_ENV } else { "sbt-gui" }
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $ScriptDirectory
$EnvironmentFile = Join-Path $RepositoryRoot "Local_envs\sbt_gui_env.yml"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Run this from Anaconda PowerShell Prompt."
}

$EnvironmentList = conda env list --json | ConvertFrom-Json
$EnvironmentExists = @(
    $EnvironmentList.envs | Where-Object {
        (Split-Path -Leaf $_) -eq $EnvironmentName
    }
).Count -gt 0

if ($EnvironmentExists) {
    conda env update --file $EnvironmentFile --name $EnvironmentName --prune
} else {
    conda env create --file $EnvironmentFile --name $EnvironmentName
}
if ($LASTEXITCODE -ne 0) {
    throw "Conda could not create or update environment '$EnvironmentName'."
}

conda run -n $EnvironmentName python -m pip install -e $RepositoryRoot --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) {
    throw "The editable Project Console installation failed."
}

Write-Host "Installed the SBT Project Console in '$EnvironmentName'."
Write-Host "Next: run 'sbt gui project --project path\to\project'."

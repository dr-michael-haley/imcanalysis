param([string]$OutputDirectory = (Join-Path $PSScriptRoot 'dist'))
$ErrorActionPreference = 'Stop'
$compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compilerPath)) {
    $compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compilerPath)) {
    throw 'The Windows .NET Framework C# compiler was not found.'
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputPath = Join-Path ([IO.Path]::GetFullPath($OutputDirectory)) 'NapariSBT.exe'
& $compilerPath /nologo /target:winexe /platform:anycpu /optimize+ `
    /reference:System.Windows.Forms.dll /reference:System.Drawing.dll `
    "/win32manifest:$(Join-Path $PSScriptRoot 'launcher.manifest.xml')" `
    "/out:$outputPath" (Join-Path $PSScriptRoot 'LauncherCore.cs') (Join-Path $PSScriptRoot 'Launcher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.html') -Destination $OutputDirectory -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README_DISTRIBUTOR.html') -Destination $OutputDirectory -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Start_NapariSBT.bat') -Destination $OutputDirectory -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start_naparisbt.py') -Destination $OutputDirectory -Force
Write-Output "Built $outputPath"

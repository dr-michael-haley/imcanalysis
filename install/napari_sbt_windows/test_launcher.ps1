param([string]$PythonEnvironment = 'C:\Anaconda3\envs\sbt-release')
$ErrorActionPreference = 'Stop'
$compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compilerPath)) {
    $compilerPath = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ('NapariSBT-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDirectory | Out-Null
$testRunner = Join-Path $testDirectory 'tests.exe'
try {
    & $compilerPath /nologo /target:exe "/out:$testRunner" `
        (Join-Path $PSScriptRoot 'LauncherCore.cs') (Join-Path $PSScriptRoot 'tests\LauncherTests.cs')
    if ($LASTEXITCODE -ne 0) { throw 'Launcher test compilation failed.' }
    & $testRunner $testDirectory $PythonEnvironment (Join-Path $PSScriptRoot 'tests\fixture.py')
    if ($LASTEXITCODE -ne 0) { throw 'Launcher tests failed.' }
} finally {
    $resolvedTestDirectory = [IO.Path]::GetFullPath($testDirectory)
    $expectedParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
    if ((Split-Path -Parent $resolvedTestDirectory) -eq $expectedParent -and
        (Split-Path -Leaf $resolvedTestDirectory).StartsWith('NapariSBT-tests-')) {
        Remove-Item -LiteralPath $resolvedTestDirectory -Recurse -Force
    }
}

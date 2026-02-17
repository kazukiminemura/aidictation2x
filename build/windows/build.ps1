param(
    [string]$Version = "0.1.0",
    [string]$AppName = "AIDictation2x"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$OutputRoot = Join-Path $ProjectRoot "dist"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        throw "PyInstaller is not installed. Run: pip install pyinstaller"
    }

    pyinstaller `
        --noconfirm `
        --clean `
        --windowed `
        --name $AppName `
        --collect-all openvino `
        --collect-all openvino_genai `
        --collect-submodules vosk `
        --add-data "config;config" `
        --paths "." `
        main.py

    $DistAppDir = Join-Path $OutputRoot $AppName
    if (-not (Test-Path $DistAppDir)) {
        throw "Build output not found: $DistAppDir"
    }

    $Iscc = Get-Command iscc -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        Write-Host "Inno Setup (iscc) is not installed. EXE build is complete at: $DistAppDir"
        exit 0
    }

    & $Iscc.Path `
        "/DMyAppName=$AppName" `
        "/DMyAppVersion=$Version" `
        "/DMySourceDir=$DistAppDir" `
        (Join-Path $PSScriptRoot "aidictation2x.iss")

    Write-Host "Installer build finished."
}
finally {
    Pop-Location
}

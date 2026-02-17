param(
    [string]$Version = "0.1.0",
    [string]$AppName = "AIDictation2x"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$OutputRoot = Join-Path $ProjectRoot "dist"
$StagingRoot = Join-Path $OutputRoot "staging"
$DistPath = Join-Path $StagingRoot "$AppName-$Version"
$WorkPath = Join-Path $ProjectRoot "build\pyinstaller"

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $DistPath -Force | Out-Null
    New-Item -ItemType Directory -Path $WorkPath -Force | Out-Null

    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $PythonExe)) {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCmd) {
            $PythonExe = $PythonCmd.Source
        }
    }
    if (-not (Test-Path $PythonExe)) {
        throw "Python was not found. Create venv or install Python first."
    }

    & $PythonExe -m PyInstaller --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is not installed for $PythonExe. Run: `"$PythonExe`" -m pip install pyinstaller"
    }

    & $PythonExe -m PyInstaller `
        --noconfirm `
        --clean `
        --noupx `
        --windowed `
        --name $AppName `
        --distpath $DistPath `
        --workpath $WorkPath `
        --collect-all openvino `
        --collect-all openvino_genai `
        --collect-all openvino_tokenizers `
        --collect-all huggingface_hub `
        --add-data "config;config" `
        --paths "." `
        main.py

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed. Close running app/explorer windows under dist and retry."
    }

    $DistAppDir = Join-Path $DistPath $AppName
    if (-not (Test-Path $DistAppDir)) {
        throw "Build output not found: $DistAppDir"
    }

    $Iscc = Get-Command iscc -ErrorAction SilentlyContinue
    if (-not $Iscc) {
        $DefaultIsccPaths = @(
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 6\ISCC.exe",
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
        )
        foreach ($Candidate in $DefaultIsccPaths) {
            if (Test-Path $Candidate) {
                $Iscc = @{ Path = $Candidate }
                break
            }
        }
    }

    if (-not $Iscc) {
        Write-Host "Inno Setup (iscc) is not installed. EXE build is complete at: $DistAppDir"
        Write-Host "Install it with: winget install --id JRSoftware.InnoSetup -e"
        exit 0
    }

    & $Iscc.Path `
        "/DMyAppName=$AppName" `
        "/DMyAppVersion=$Version" `
        "/DMySourceDir=$DistAppDir" `
        (Join-Path $PSScriptRoot "aidictation2x.iss")

    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup build failed."
    }

    Write-Host "Installer build finished."
}
finally {
    Pop-Location
}

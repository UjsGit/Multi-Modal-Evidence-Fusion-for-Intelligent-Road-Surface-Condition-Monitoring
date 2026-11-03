# PowerShell launcher for Road Quality Map Server
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Road Quality Map Backend & Interactive GIS" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$venvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Host "Using virtual environment: $venvPython" -ForegroundColor Green
    & $venvPython "backend/api.py"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "Using system Python 3.13 launcher (py -3.13)..." -ForegroundColor Green
    py -3.13 "backend/api.py"
} else {
    Write-Host "Using default python executable..." -ForegroundColor Yellow
    python "backend/api.py"
}

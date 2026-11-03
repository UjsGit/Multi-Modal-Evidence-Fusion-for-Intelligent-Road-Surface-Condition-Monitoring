@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo Starting Road Quality Map Backend & Interactive GIS
echo ===================================================

if exist ".venv\Scripts\python.exe" (
    echo Using virtual environment at .venv...
    ".venv\Scripts\python.exe" backend\api.py
) else (
    echo Virtual environment not found, launching with Python 3.13...
    py -3.13 backend\api.py
)

if errorlevel 1 (
    echo.
    echo Backend server exited with an error.
    pause
)

@echo off
REM MarketPulse — Backend only (Windows)

setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
for %%I in ("%ROOT%\..") do set "PROJECT_ROOT=%%~fI"

echo.
echo   +------------------------------------------------+
echo   ^|     MarketPulse -- Backend (FastAPI)           ^|
echo   +------------------------------------------------+
echo.

REM ── Locate virtual environment ────────────────────────────
if exist "%ROOT%\venv\Scripts\activate.bat" (
    set "VENV=%ROOT%\venv"
) else if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    set "VENV=%PROJECT_ROOT%\venv"
) else if exist "%USERPROFILE%\ml_env\Scripts\activate.bat" (
    set "VENV=%USERPROFILE%\ml_env"
) else (
    echo   No virtual environment found -- creating one at %ROOT%\venv ...
    python -m venv "%ROOT%\venv"
    set "VENV=%ROOT%\venv"
    echo   Virtual env created.
)
echo   Virtual env : %VENV%

REM ── Install dependencies ──────────────────────────────────
echo.
echo   Installing / verifying dependencies...
call "%VENV%\Scripts\activate.bat"
pip install -r "%ROOT%\requirements.txt" -q --disable-pip-version-check

echo.
echo   Starting FastAPI server --^>  http://localhost:8000
echo   API docs                --^>  http://localhost:8000/docs
echo   NOTE: First start loads the Keras model (may take ~30s)
echo   Press Ctrl+C to stop.
echo.

cd /d "%ROOT%"
uvicorn app.main:app --host 0.0.0.0 --port 8000
pause

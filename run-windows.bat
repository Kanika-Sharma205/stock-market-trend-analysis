@echo off
REM MarketPulse — Start backend + frontend together (Windows)

setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
REM Remove trailing backslash
set "ROOT=%ROOT:~0,-1%"

echo.
echo   +------------------------------------------------+
echo   ^|       MarketPulse -- Starting Up               ^|
echo   +------------------------------------------------+
echo.

REM ── Locate virtual environment ────────────────────────────
if exist "%ROOT%\backend\venv\Scripts\activate.bat" (
    set "VENV=%ROOT%\backend\venv"
) else if exist "%ROOT%\venv\Scripts\activate.bat" (
    set "VENV=%ROOT%\venv"
) else if exist "%USERPROFILE%\ml_env\Scripts\activate.bat" (
    set "VENV=%USERPROFILE%\ml_env"
) else (
    echo   No virtual environment found -- creating one at %ROOT%\venv ...
    python -m venv "%ROOT%\venv"
    set "VENV=%ROOT%\venv"
    echo   Virtual env created.
)
echo   Virtual env : %VENV%

REM ── Install backend dependencies ──────────────────────────
echo.
echo   Installing backend dependencies...
call "%VENV%\Scripts\activate.bat"
pip install -r "%ROOT%\backend\requirements.txt" -q --disable-pip-version-check

REM ── Check npm ─────────────────────────────────────────────
where npm >nul 2>&1
if errorlevel 1 (
    echo   ERROR: npm not found.
    echo   Install Node.js ^(v18+^) from https://nodejs.org
    echo.
    pause
    exit /b 1
)

REM ── Install frontend deps if needed ───────────────────────
if not exist "%ROOT%\frontend\node_modules\" (
    echo   Installing frontend dependencies...
    cd /d "%ROOT%\frontend"
    npm install --silent
    cd /d "%ROOT%"
)

echo.
echo   [1/2] Opening backend  window  --^>  http://localhost:8000
echo   [2/2] Opening frontend window  --^>  http://localhost:5173
echo.
echo   Two console windows will open, one per service.
echo   Close both windows to stop the project.
echo.

REM ── Open backend in a new console window ──────────────────
start "MarketPulse — Backend  (port 8000)" cmd /k ^
  "cd /d "%ROOT%\backend" ^
   && call "%VENV%\Scripts\activate.bat" ^
   && echo. ^
   && echo   Backend running at http://localhost:8000 ^
   && echo   API docs at      http://localhost:8000/docs ^
   && echo   NOTE: First start loads the Keras model (may take ~30s) ^
   && echo. ^
   && uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM Small pause so backend binds first
timeout /t 2 /nobreak >nul

REM ── Open frontend in a new console window ─────────────────
start "MarketPulse — Frontend (port 5173)" cmd /k ^
  "cd /d "%ROOT%\frontend" ^
   && echo. ^
   && echo   Frontend running at http://localhost:5173 ^
   && echo. ^
   && npm run dev"

echo   Both windows launched. Press any key to close this launcher.
echo   (Backend and frontend will keep running in their own windows.)
echo.
pause >nul

@echo off
REM MarketPulse — Frontend only (Windows)

setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo.
echo   +------------------------------------------------+
echo   ^|     MarketPulse -- Frontend (Vite)             ^|
echo   +------------------------------------------------+
echo.

REM ── Check npm ─────────────────────────────────────────────
where npm >nul 2>&1
if errorlevel 1 (
    echo   ERROR: npm not found.
    echo   Install Node.js ^(v18+^) from https://nodejs.org
    echo.
    pause
    exit /b 1
)

REM ── Install deps if needed ────────────────────────────────
if not exist "%ROOT%\node_modules\" (
    echo   Installing frontend dependencies...
    cd /d "%ROOT%"
    npm install --silent
)

echo.
echo   Starting Vite dev server --^>  http://localhost:5173
echo   Make sure the backend is running on port 8000.
echo   Press Ctrl+C to stop.
echo.

cd /d "%ROOT%"
npm run build
npm run dev
pause

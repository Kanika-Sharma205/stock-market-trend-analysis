#!/usr/bin/env bash
# MarketPulse — Start backend + frontend together (Linux / macOS)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; CYAN=$'\033[0;36m'
YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

echo ""
echo -e "${BOLD}  ┌──────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}  │        MarketPulse — Starting Up             │${RESET}"
echo -e "${BOLD}  └──────────────────────────────────────────────┘${RESET}"
echo ""

# ── Locate virtual environment ────────────────────────────
if   [ -d "$ROOT/backend/venv" ];    then VENV="$ROOT/backend/venv"
elif [ -d "$ROOT/venv" ];            then VENV="$ROOT/venv"
elif [ -d "$HOME/ml_env" ];          then VENV="$HOME/ml_env"
else
  echo -e "${YELLOW}  No virtual environment found — creating one at $ROOT/venv ...${RESET}"
  python3 -m venv "$ROOT/venv"
  VENV="$ROOT/venv"
  echo -e "${GREEN}  Virtual env created at $ROOT/venv${RESET}"
fi
echo -e "  Virtual env : ${CYAN}$VENV${RESET}"

# ── Install / upgrade backend deps ────────────────────────
echo ""
echo -e "${CYAN}  Installing backend dependencies...${RESET}"
source "$VENV/bin/activate"
pip install -r "$ROOT/backend/requirements.txt" -q --disable-pip-version-check

# ── Locate npm (handles nvm installs not on PATH) ─────────
if ! command -v npm &>/dev/null; then
  for NVM_NODE in "$HOME/.nvm/versions/node"/*/bin; do
    [ -x "$NVM_NODE/npm" ] && export PATH="$NVM_NODE:$PATH" && break
  done
fi
if ! command -v npm &>/dev/null; then
  echo -e "${RED}  ERROR: npm not found. Install Node.js (v18+) from https://nodejs.org${RESET}"
  echo ""
  exit 1
fi

# ── Install frontend deps if needed ──────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo -e "${CYAN}  Installing frontend dependencies...${RESET}"
  (cd "$ROOT/frontend" && npm install --silent)
fi

# ── Free port 8000 if already in use ─────────────────────
if lsof -ti:8000 &>/dev/null; then
  echo -e "${YELLOW}  Port 8000 in use — stopping existing process...${RESET}"
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# ── Start backend in background ───────────────────────────
echo ""
echo -e "${GREEN}  [1/2] Starting backend  → http://localhost:8000${RESET}"
echo -e "${GREEN}        API docs          → http://localhost:8000/docs${RESET}"
echo ""

(
  cd "$ROOT/backend"
  source "$VENV/bin/activate"
  uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 \
    | sed "s/^/  ${YELLOW}[backend]${RESET} /"
) &
BACKEND_PID=$!

# ── Trap Ctrl+C — kill both processes cleanly ─────────────
cleanup() {
  echo ""
  echo -e "${CYAN}  Shutting down...${RESET}"
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  echo -e "${GREEN}  Done.${RESET}"
  exit 0
}
trap cleanup INT TERM

# Give backend a moment to bind
sleep 1

# ── Start frontend in foreground ──────────────────────────
echo -e "${GREEN}  [2/2] Starting frontend → http://localhost:5173${RESET}"
echo ""
echo -e "  ${BOLD}Both servers running. Press Ctrl+C to stop all.${RESET}"
echo ""

(
  cd "$ROOT/frontend"
  npm run dev 2>&1 | sed "s/^/  ${CYAN}[frontend]${RESET} /"
)

# Frontend exited — shut everything down
cleanup

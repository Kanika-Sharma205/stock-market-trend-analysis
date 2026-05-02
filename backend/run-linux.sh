#!/usr/bin/env bash
# MarketPulse — Backend only (Linux / macOS)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"

# ── Colours ───────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; CYAN=$'\033[0;36m'
YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

echo ""
echo -e "${BOLD}  ┌──────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}  │      MarketPulse — Backend (FastAPI)         │${RESET}"
echo -e "${BOLD}  └──────────────────────────────────────────────┘${RESET}"
echo ""

# ── Locate virtual environment ────────────────────────────
if   [ -d "$ROOT/venv" ];            then VENV="$ROOT/venv"
elif [ -d "$PROJECT_ROOT/venv" ];    then VENV="$PROJECT_ROOT/venv"
elif [ -d "$HOME/ml_env" ];          then VENV="$HOME/ml_env"
else
  echo -e "${YELLOW}  No virtual environment found — creating one at $ROOT/venv ...${RESET}"
  python3 -m venv "$ROOT/venv"
  VENV="$ROOT/venv"
  echo -e "${GREEN}  Virtual env created.${RESET}"
fi
echo -e "  Virtual env : ${CYAN}$VENV${RESET}"

# ── Install / upgrade deps ────────────────────────────────
echo ""
echo -e "${CYAN}  Installing / verifying dependencies...${RESET}"
source "$VENV/bin/activate"
pip install -r "$ROOT/requirements.txt" -q --disable-pip-version-check

# ── Free port 8000 if busy ────────────────────────────────
if lsof -ti:8000 &>/dev/null; then
  echo -e "${YELLOW}  Port 8000 in use — stopping existing process...${RESET}"
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

echo ""
echo -e "${GREEN}  Starting FastAPI server → http://localhost:8000${RESET}"
echo -e "${GREEN}  API docs                → http://localhost:8000/docs${RESET}"
echo -e "  ${BOLD}Press Ctrl+C to stop.${RESET}"
echo ""

cd "$ROOT"
uvicorn app.main:app --host 0.0.0.0 --port 8000

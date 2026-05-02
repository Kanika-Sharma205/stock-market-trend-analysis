#!/usr/bin/env bash
# MarketPulse — Frontend only (Linux / macOS)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; CYAN=$'\033[0;36m'
YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

echo ""
echo -e "${BOLD}  ┌──────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}  │      MarketPulse — Frontend (Vite)           │${RESET}"
echo -e "${BOLD}  └──────────────────────────────────────────────┘${RESET}"
echo ""

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
echo -e "  Node : ${CYAN}$(node --version)${RESET}   npm : ${CYAN}$(npm --version)${RESET}"

# ── Install deps if needed ────────────────────────────────
if [ ! -d "$ROOT/node_modules" ]; then
  echo ""
  echo -e "${CYAN}  Installing frontend dependencies...${RESET}"
  (cd "$ROOT" && npm install --silent)
fi

echo ""
echo -e "${GREEN}  Starting Vite dev server → http://localhost:5173${RESET}"
echo -e "  ${BOLD}Make sure the backend is running on port 8000.${RESET}"
echo -e "  ${BOLD}Press Ctrl+C to stop.${RESET}"
echo ""

cd "$ROOT"
npm run build
npm run dev

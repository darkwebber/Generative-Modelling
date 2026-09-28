#!/usr/bin/env bash
# Cache the Google Fonts used by the episodes into video/.fonts so render.mjs can serve them offline.
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p .fonts
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
curl -sS -A "$UA" "https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;1,6..72,400&family=Karla:wght@400;600&family=JetBrains+Mono:wght@400&display=swap" -o .fonts/gf.css
for u in $(grep -o "https://fonts.gstatic.com[^)]*" .fonts/gf.css | sort -u); do curl -sS "$u" -o ".fonts/$(basename "$u")"; done
echo "cached $(ls .fonts | wc -l) files in .fonts/"

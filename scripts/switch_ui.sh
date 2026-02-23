#!/usr/bin/env bash
# switch_ui.sh — Toggle VirtualFermLab UI between v1 and v2
# Usage: ./scripts/switch_ui.sh v1   # original Bootstrap dark
#        ./scripts/switch_ui.sh v2   # Preprint-v1 gradient palette

set -euo pipefail

VERSION="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="$BASE_DIR/src/virtualfermlab/web"

if [[ "$VERSION" != "v1" && "$VERSION" != "v2" ]]; then
    echo "Usage: $0 {v1|v2}"
    echo "  v1  — Original Bootstrap dark navbar"
    echo "  v2  — Preprint-v1 gradient colour palette"
    exit 1
fi

echo "Switching UI to $VERSION ..."

cp "$WEB_DIR/static/style_${VERSION}.css"     "$WEB_DIR/static/style.css"
cp "$WEB_DIR/templates/base_${VERSION}.html"   "$WEB_DIR/templates/base.html"
cp "$WEB_DIR/templates/index_${VERSION}.html"  "$WEB_DIR/templates/index.html"

echo "Done. Active UI version: $VERSION"
echo "  style.css  <- style_${VERSION}.css"
echo "  base.html  <- base_${VERSION}.html"
echo "  index.html <- index_${VERSION}.html"

#!/usr/bin/env bash
# Bootstrap a virtualenv for caveman-compression engines.
# Usage: caveman-bootstrap.sh [nlp|mlm|llm]
#
# Idempotent: skips work that is already done.
# Resolves the plugin root from the script's own location so it works whether
# the plugin is run from a Claude Code plugin install dir or a git clone.

set -euo pipefail

ENGINE="${1:-nlp}"
case "$ENGINE" in
    nlp|mlm|llm) ;;
    *) echo "Unknown engine: $ENGINE (expected nlp|mlm|llm)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PLUGIN_ROOT/.venv"
REQS="$PLUGIN_ROOT/requirements-${ENGINE}.txt"
[ "$ENGINE" = "llm" ] && REQS="$PLUGIN_ROOT/requirements.txt"

if [ ! -f "$REQS" ]; then
    echo "Requirements file not found: $REQS" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python3 not found on PATH. Install Python 3.8+ first." >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

MARKER="$VENV/.installed-${ENGINE}"
if [ ! -f "$MARKER" ]; then
    echo "Installing $ENGINE requirements"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$REQS"

    if [ "$ENGINE" = "nlp" ] || [ "$ENGINE" = "mlm" ]; then
        if ! python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
            echo "Downloading spaCy en_core_web_sm"
            python -m spacy download en_core_web_sm
        fi
    fi

    touch "$MARKER"
fi

echo "Bootstrap complete for engine: $ENGINE"
echo "Activate with: source $VENV/bin/activate"

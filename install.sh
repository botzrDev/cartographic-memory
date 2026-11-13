#!/usr/bin/env bash
# Cartographic Memory — install script
# Run from the repository root: bash install.sh
#
# Creates .venv/ in the repository root and installs dependencies there.
# Nothing is written to the system Python.
# Activate with: source .venv/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

echo "=== Cartographic Memory — install ==="
echo ""

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.9 or newer is required. Install from https://python.org"
    exit 1
fi

echo "Using: $($PYTHON --version) ($(command -v $PYTHON))"

echo ""
echo "Creating virtual environment at .venv/"
if [ -d "$VENV" ]; then
    echo "  .venv/ already exists, reusing it."
else
    "$PYTHON" -m venv "$VENV"
fi

VPY="$VENV/bin/python"
if [ ! -x "$VPY" ]; then
    VPY="$VENV/Scripts/python.exe"   # Windows / Git Bash
fi
if [ ! -x "$VPY" ]; then
    echo "ERROR: could not create a working virtual environment."
    echo "On Debian/Ubuntu you may need: sudo apt install python3-venv"
    exit 1
fi

echo ""
echo "Installing dependencies (numpy, scikit-learn)..."
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet numpy scikit-learn

echo ""
echo "Verifying imports..."
"$VPY" -c "import numpy, sklearn; print('  numpy:', numpy.__version__); print('  scikit-learn:', sklearn.__version__)"

echo ""
echo "=== Install complete ==="
echo ""
echo "Activate the environment first:"
echo "  source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)"
echo ""
echo "Run the conformance harness (under 30s, fully offline):"
echo "  cd code && python conformance.py"

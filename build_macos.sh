#!/usr/bin/env bash
set -euo pipefail

# Build helper for macOS/Linux.
# Run from project root: ./build_macos.sh

python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -U pyinstaller pyside6

python build.py --name PTZoptics_JS --entry main1.py

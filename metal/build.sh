#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "The current MPS backend requires macOS and Apple Silicon." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
echo "Verifying the custom Metal kernel through PyTorch MPS..."
PYTHONPATH="${PROJECT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/verify_mps.py"

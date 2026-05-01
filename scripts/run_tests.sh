#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".vendor_py37" ]]; then
  echo "Missing .vendor_py37 test dependencies."
  echo "Set up the local pytest wheels first, then rerun this script."
  exit 1
fi

PYTHONPATH=".vendor_py37${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest -q "$@"

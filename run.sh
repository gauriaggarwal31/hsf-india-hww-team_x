#!/bin/bash
# The reference pipeline: run.sh <dataset_dir> <output_dir> -> <output_dir>/RESULT.json (fit.py, numpy + scipy).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$2"
python3 "$HERE/fit.py" "$1" "$2"

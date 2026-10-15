#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python src/evaluate.py --case 1 "$@"

python src/evaluate.py --case 2 "$@"

python src/evaluate.py --case 3 "$@"

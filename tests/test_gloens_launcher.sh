#!/usr/bin/env bash
# Run GLOENS one-day launcher tests from any working directory.
# Usage:
#   bash tests/test_gloens_launcher.sh                  # mocked launcher test
#   bash tests/test_gloens_launcher.sh --live-read 20251106  # live S3 read
#   bash tests/test_gloens_launcher.sh --run 20251106        # real 28-day Parcels run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"

case "${1:-}" in
  "")
    "$PYTHON" -m unittest discover -s tests -p 'test_gloens_launcher.py' -v
    "$PYTHON" -m unittest discover -s tests -p 'test_forecast_inputs.py' -v
    ;;
  --drifter-files|--drifters|--model|--inputs)
    DATE="${2:?Provide a forecast initialization date YYYYMMDD}"
    case "$1" in
      --drifter-files) "$PYTHON" tests/check_gloens_inputs.py --date "$DATE" --drifter-files ;;
      --drifters) "$PYTHON" tests/check_gloens_inputs.py --date "$DATE" --drifters ;;
      --model) "$PYTHON" tests/check_gloens_inputs.py --date "$DATE" --model ;;
      --inputs) "$PYTHON" tests/check_gloens_inputs.py --date "$DATE" --drifters --model ;;
    esac
    ;;
  --year|--year-2025)
    MODE="$1"
    shift
    if [[ "$MODE" == "--year-2025" ]]; then
      "$PYTHON" tests/run_gloens_2025.py --year 2025 "$@"
    else
      YEAR="${1:?Provide a year, e.g. 2024}"
      shift
      "$PYTHON" tests/run_gloens_2025.py --year "$YEAR" "$@"
    fi
    ;;
  --compare-10)
    DATE="${2:?Provide a forecast initialization date YYYYMMDD}"
    "$PYTHON" tests/compare_gloens_10_drifters.py --date "$DATE"
    ;;
  --parcels-smoke)
    "$PYTHON" tests/smoke_parcels.py
    ;;
  --live-read)
    DATE="${2:?Provide a forecast initialization date YYYYMMDD}"
    "$PYTHON" tests/smoke_gloens_zarr.py --run-date "$DATE"
    ;;
  --run)
    DATE="${2:?Provide a forecast initialization date YYYYMMDD}"
    if ! [[ "$DATE" =~ ^[0-9]{8}$ ]]; then
      echo "Expected YYYYMMDD forecast initialization date" >&2
      exit 2
    fi
    echo "Launching GLOENS for $DATE with 28-day particle dispersion."
    echo "Requires observed drifter files and a working Parcels adapter for GLOENS grid."
    "$PYTHON" Lagrangian_tool.py \
      --date1 "$DATE" --date2 "$DATE" \
      --lead FCST --sys GLOENS --scheme AdvectionRK4 --engine parcels
    ;;
  *)
    echo "Usage: bash tests/test_gloens_launcher.sh [--drifter-files YYYYMMDD | --drifters YYYYMMDD | --model YYYYMMDD | --inputs YYYYMMDD | --year YYYY [--continue-on-error] | --year-2025 [--step-days N] | --compare-10 YYYYMMDD | --parcels-smoke | --live-read YYYYMMDD | --run YYYYMMDD]" >&2
    exit 2
    ;;
esac

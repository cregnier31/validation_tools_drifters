#!/usr/bin/env bash
# Real GLOENS / Parcels integration tests. Run from anywhere.
# Examples:
#   bash tests/test_gloens_realizations.sh --day 20240104
#   bash tests/test_gloens_realizations.sh --year 2024
#   bash tests/test_gloens_realizations.sh --year 2024 --continue-on-error
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
usage() {
  echo "Usage: bash tests/test_gloens_realizations.sh --day YYYYMMDD | --year YYYY [--continue-on-error]" >&2
  exit 2
}
[[ $# -ge 2 ]] || usage
MODE="$1"
VALUE="$2"
shift 2
case "$MODE" in
  --day)
    [[ "$VALUE" =~ ^[0-9]{8}$ && $# -eq 0 ]] || usage
    ;;
  --year)
    [[ "$VALUE" =~ ^[0-9]{4}$ ]] || usage
    ;;
  *) usage ;;
esac
if [[ "$MODE" == "--day" ]]; then
  echo "GLOENS Parcels: initialization $VALUE, all valid drifters, 28 days"
  exec "$PYTHON" Lagrangian_tool.py \
    --date1 "$VALUE" --date2 "$VALUE" --lead FCST --sys GLOENS \
    --scheme AdvectionRK4 --engine parcels --typeobs COPERNICUS_DAILY
else
  echo "GLOENS Parcels: discover weekly Zarr initializations in $VALUE"
  exec "$PYTHON" tests/run_gloens_2025.py --year "$VALUE" "$@"
fi

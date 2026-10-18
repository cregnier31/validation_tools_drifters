#!/bin/bash
#SBATCH -J DRIFTERS_GLOENS
#SBATCH -A cmems
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH -o Drifters_GLOENS-%j.out
#SBATCH -e Drifters_GLOENS-%j.err

# Annual GLOENS forecast-initialization batch, like Launch_drifters.ksh.
# Change these dates to select the desired period (inclusive).
set -euo pipefail
DATE1="20240101"
DATE2="20241231"
SCHEME="AdvectionRK4"
OBSTYPE="COPERNICUS_DAILY"
PYTHON="/home/cregnier/.conda/envs/py312_parcels_test/bin/python"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep the batch and aggregation diagnostics in files, not on screen.
mkdir -p "outputs/gloens_${DATE1}_${DATE2}"
exec >> "outputs/gloens_${DATE1}_${DATE2}/batch.log" 2>&1

# Only available GLOENS initialization cycles are processed, sequentially.
# Each initialization runs its own 28-day forecast and writes a separate log.
YEAR="${DATE1:0:4}"
if [[ "${DATE2:0:4}" != "$YEAR" ]]; then
  echo "This launcher currently requires DATE1 and DATE2 in the same year." >&2
  exit 2
fi
GLOENS_DEFER_PLOTS=1 "$PYTHON" tests/run_gloens_2025.py \
  --year "$YEAR" --start "$DATE1" --end "$DATE2" \
  --continue-on-error \
  --output "outputs/gloens_${DATE1}_${DATE2}"

# Aggregate all successful forecast initializations by lead day, once.
"$PYTHON" tests/aggregate_gloens_period.py \
  --summary "outputs/gloens_${DATE1}_${DATE2}/summary.csv" \
  --start "$DATE1" --end "$DATE2" --days 28

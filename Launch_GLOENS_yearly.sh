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
YEAR="${1:?Usage: sbatch Launch_GLOENS_yearly.sh YEAR (e.g. 2024)}"
if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
  echo "YEAR must be four digits, e.g. 2024" >&2
  exit 2
fi
DATE1="${YEAR}0101"
DATE2="${YEAR}1231"
SCHEME="AdvectionRK4"
OBSTYPE="COPERNICUS_DAILY"
PYTHON="/home/cregnier/.conda/envs/py312_parcels_test/bin/python"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep the batch and aggregation diagnostics in files, not on screen.
mkdir -p "outputs/gloens_${DATE1}_${DATE2}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Batch mode: statuses go to SLURM stdout; diagnostics go to batch.log.
  exec 3>&1
  exec >> "outputs/gloens_${DATE1}_${DATE2}/batch.log" 2>&1
else
  # Interactive mode: statuses go directly to the terminal in real time.
  exec 3>&1
  exec >> "outputs/gloens_${DATE1}_${DATE2}/batch.log" 2>&1
fi

# Single GLOENS Python process: discover Zarr cycles, run trajectories,
# append all per-cycle scores to results_stats, aggregate and plot once.
"$PYTHON" Lagrangian_tool_GLOENS.py --year "$YEAR" --continue-on-error --status-fd 3

#!/bin/bash
#SBATCH -J DRIFTERS_PARCELS
#SBATCH -A cmems
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH -o Drifters_Parcels-%j.out
#SBATCH -e Drifters_Parcels-%j.err
set -euo pipefail

usage() {
  echo "Usage: bash Launch_Parcels.sh DATE1 DATE2 SYSTEM [--console yes|no] [--lead FCST|HDCT] [--scheme AdvectionRK4|AdvectionEE|K_smag] [--typeobs COPERNICUS_DAILY|AOML] [--drift-days DAYS] [--continue-on-error] [--delete-failed-particles] [--no-plots] [--plots-only] [--audit-liu] [--pickle-file PATH] [--postprocess-only] [--trajectory-tag TAG] [--remove-old-trajectories]" >&2
}
if (( $# < 3 )); then usage; exit 2; fi
DATE1="$1"; DATE2="$2"; SYSTEM="$3"; shift 3
if [[ ! "$DATE1" =~ ^[0-9]{8}$ || ! "$DATE2" =~ ^[0-9]{8}$ ]]; then usage; exit 2; fi
CONSOLE="yes"
ARGS=(--date1 "$DATE1" --date2 "$DATE2" --sys "$SYSTEM")
while (( $# )); do
  case "$1" in
    --console)
      if (( $# < 2 )); then usage; exit 2; fi
      CONSOLE="$2"; shift 2 ;;
    --lead|--scheme|--typeobs|--drift-days|--pickle-file|--trajectory-tag)
      if (( $# < 2 )); then usage; exit 2; fi
      ARGS+=("$1" "$2"); shift 2 ;;
    --continue-on-error|--delete-failed-particles|--no-plots|--plots-only|--audit-liu|--postprocess-only|--remove-old-trajectories)
      ARGS+=("$1"); shift ;;
    *) usage; exit 2 ;;
  esac
done
if [[ "$CONSOLE" != yes && "$CONSOLE" != no ]]; then
  echo "--console must be yes or no" >&2; exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/cregnier/.conda/envs/py312_parcels_test/bin/python"
DRIFT_DAYS="default"
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--drift-days" ]]; then
    DRIFT_DAYS="${ARGS[i+1]}"
  fi
done
if [[ "$DRIFT_DAYS" == "default" && "$SYSTEM" == "GLOENS" ]]; then DRIFT_DAYS=28; fi
LOGDIR="$ROOT/outputs/parcels_${SYSTEM}_drift${DRIFT_DAYS}_${DATE1}_${DATE2}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run.log"
if [[ "$CONSOLE" == yes ]]; then
  "$PYTHON" -u "$ROOT/Lagrangian_tool_Parcels.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
else
  "$PYTHON" -u "$ROOT/Lagrangian_tool_Parcels.py" "${ARGS[@]}" > "$LOG" 2>&1
fi

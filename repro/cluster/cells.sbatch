#!/bin/bash
#SBATCH -A naiss2026-4-1816-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 03:00:00
#SBATCH -J wp-cells
#SBATCH -o logs/cell_%A_%a.log
# One array task = one line of cells.txt (comment lines skipped), N test images, B images per GPU pass.
#   sbatch --array=1-76 repro/cluster/cells.sbatch 100 16
# Resumable: finished images are skipped, so a timed-out task can simply be resubmitted.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source repro/cluster/env.sh
N=${1:-5} B=${2:-16}
args=$(grep -v '^#' repro/cluster/cells.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$args" ] || { echo "no cell ${SLURM_ARRAY_TASK_ID}"; exit 1; }
echo "cell ${SLURM_ARRAY_TASK_ID}: $args --n $N --batch $B on $(hostname)"
HF_HUB_OFFLINE=1 $PY repro/run.py $args --n "$N" --batch "$B" 2>&1 | sed -E '/warn|deprecat/Id'   # sed exits 0, so pipefail reports run.py's status

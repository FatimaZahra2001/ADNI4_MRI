#!/bin/bash
#SBATCH --job-name=phase1_ml
#SBATCH --qos=bbdefault
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase1_ml_%j.out
#SBATCH --error=logs/phase1_ml_%j.err

set -euo pipefail

echo "===== START PHASE 1 ML BASELINES ====="
date
hostname

module purge
module load bear-apps/2024a
module load Miniforge3/25.3.0-3

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

echo "Python: $(which python)"

python - <<EOF
import sys, sklearn, pandas, numpy
print("Python:", sys.version)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
EOF

PROJECT_DIR=/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/RECENT/adni4_mri_codebase/baselines/tabular
SCRIPT=${PROJECT_DIR}/train_tabular_ml_baselines.py

OUTDIR=/rds/projects/j/jouaitim-mri-test/fatima/outputs/phase1_ml_baselines

mkdir -p logs
mkdir -p "${OUTDIR}"

cd "${PROJECT_DIR}"

python "${SCRIPT}" \
  --outdir "${OUTDIR}" \
  --folds 5 \
  --inner_folds 3 \
  --seed 42 \
  --n_jobs 8 \
  --scoring roc_auc \
  --k_values 4 6 8 10 12 16 20 30 40 50 80 120

echo "===== END PHASE 1 ML BASELINES ====="
date

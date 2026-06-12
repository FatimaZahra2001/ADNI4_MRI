#!/bin/bash
#SBATCH --job-name=t1radio
#SBATCH --qos=bbdefault
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/t1radio_%j.out
#SBATCH --error=logs/t1radio_%j.err

echo "===== START T1 RADIOMICS + NORMATIVE RUN ====="
date
hostname

module purge
module load bluebear
module load bear-apps/2024a
module load Miniforge3/25.3.0-3

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

CODE_DIR=/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/RECENT/adni4_mri_codebase/baselines/tabular
SCRIPT=$CODE_DIR/t1_normative.py

mkdir -p "$CODE_DIR/logs"
cd "$CODE_DIR"

echo "Python: $(which python)"
python "$SCRIPT"

echo "===== END T1 RADIOMICS + NORMATIVE RUN ====="
date



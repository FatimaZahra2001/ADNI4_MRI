#!/bin/bash
#SBATCH --job-name=add_demog
#SBATCH --qos=bbshort
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/add_demog_%j.out
#SBATCH --error=logs/add_demog_%j.err

echo "===== START ADD DEMOGRAPHICS ====="
date
hostname

# ========================
# ENV SETUP
# ========================
module purge
module load bluebear
module load bear-apps/2024a
module load Miniforge3/25.3.0-3

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

echo "Python: $(which python)"

# ========================
# PATHS
# ========================
CODE_DIR=/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI
SCRIPT=$CODE_DIR/add_demographics.py

cd "$CODE_DIR"

# ========================
# RUN SCRIPT
# ========================
python "$SCRIPT"

echo "===== DONE ADD DEMOGRAPHICS ====="
date

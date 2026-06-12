#!/bin/bash
#SBATCH --job-name=phase2_embed
#SBATCH --qos=bbgpu
#SBATCH --partition=icelake-gpua100-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase2_embed_%j.out
#SBATCH --error=logs/phase2_embed_%j.err

set -euo pipefail

echo "===== START PHASE 2 MEDICALNET EMBEDDINGS + ML ====="
date
hostname

module purge
module load bear-apps/2024a
module load Miniforge3/25.3.0-3

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

PROJECT_DIR=/rds/projects/j/jouaitim-mri-test/fatima/code/MONAI/RECENT/adni4_mri_codebase/deep_learning/medicalnet_embeddings
SCRIPT=${PROJECT_DIR}/train_medicalnet_embedding_classifier.py

CSV=/rds/projects/j/jouaitim-mri-test/fatima/code/DINO/preprocessing/adni4_subject_splits_mni.csv

ROI_ROOT=/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm

OUTDIR=/rds/projects/j/jouaitim-mri-test/fatima/outputs/phase2_medicalnet_embeddings_ml

MEDICALNET=/rds/projects/j/jouaitim-mri-test/ADNI4/pretrained/medicalnet/resnet_18_23dataset.pth

mkdir -p logs
mkdir -p "${OUTDIR}"

# KEEP existing embeddings
# rm -f "${OUTDIR}/medicalnet_roi_embeddings.csv"

cd "${PROJECT_DIR}"

python "${SCRIPT}" \
  --csv "${CSV}" \
  --roi_root "${ROI_ROOT}" \
  --outdir "${OUTDIR}" \
  --medicalnet_ckpt "${MEDICALNET}" \
  --patch_shape 64 64 64 \
  --margin 8 \
  --batch_size 4 \
  --num_workers 4 \
  --folds 5 \
  --inner_folds 3 \
  --seed 42 \
  --n_jobs 1 \
  --scoring roc_auc \
  --k_values 8 16 32 64 \
  --reuse_embeddings

echo "===== END PHASE 2 MEDICALNET EMBEDDINGS + ML ====="
date

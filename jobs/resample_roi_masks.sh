#!/bin/bash
#SBATCH --job-name=roi_resample
#SBATCH --qos=bbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/roi_resample_%j.out

set -euo pipefail

module purge
module load bear-apps/2024a
module load FSL/6.0.7.19

source "$FSLDIR/etc/fslconf/fsl.sh"
export FSLOUTPUTTYPE=NIFTI_GZ

echo "===== START ROI RESAMPLE ====="
date

IMG_ROOT=/rds/projects/j/jouaitim-mri-test/UNIFIED_PREPROC_MNI/adni
ROI_ROOT=/rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm

mkdir -p logs

count=0
missing=0

for subj in "$IMG_ROOT"/*; do
    ptid=$(basename "$subj")

    img="$IMG_ROOT/$ptid/T1_MNI.nii.gz"
    roi="$ROI_ROOT/$ptid/MTL_roi_mask.nii.gz"
    out="$IMG_ROOT/$ptid/MTL_roi_mask_aligned.nii.gz"

    if [ -f "$img" ] && [ -f "$roi" ]; then
        echo "Processing $ptid"

        flirt \
          -in "$roi" \
          -ref "$img" \
          -out "$out" \
          -applyxfm \
          -usesqform \
          -interp nearestneighbour

        count=$((count + 1))
    else
        echo "Missing image or ROI for $ptid"
        missing=$((missing + 1))
    fi
done

echo "Resampled: $count"
echo "Missing: $missing"
echo "===== END ROI RESAMPLE ====="
date

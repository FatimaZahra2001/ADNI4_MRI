#!/bin/bash
#SBATCH --job-name=ho_masks
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ho_masks_%j.out
#SBATCH --error=logs/ho_masks_%j.err

echo "===== START HO ROI MASK CREATION ====="
date
hostname

module purge
module load bear-apps/2024a
module load Miniforge3/25.3.0-3
module load FSL/6.0.7.19

source activate /rds/projects/j/jouaitim-mri-test/envs/adni4_gpu

echo "Python: $(which python)"
echo "FSLDIR: ${FSLDIR}"
echo "FSLOUTPUTTYPE before: ${FSLOUTPUTTYPE}"

# Important for FSL
if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
    source "${FSLDIR}/etc/fslconf/fsl.sh"
fi

export FSLOUTPUTTYPE=NIFTI_GZ

echo "FSLOUTPUTTYPE after: ${FSLOUTPUTTYPE}"

cd /rds/projects/j/jouaitim-mri-test/fatima/code/MONAI

python create_ho_roi_masks.py

echo "===== CHECK OUTPUT FOR ONE SUBJECT ====="
ls -lh /rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm/002_S_6053/*hippocampus* \
       /rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm/002_S_6053/*amygdala* \
       /rds/projects/j/jouaitim-mri-test/ADNI4/MRI_preproc_norm/002_S_6053/*parahippocampal* 2>/dev/null || true

echo "===== DONE HO ROI MASK CREATION ====="
date

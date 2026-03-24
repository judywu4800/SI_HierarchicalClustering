#!/bin/bash
#SBATCH --job-name=validity_K
#SBATCH --output=../logs/fig3_batch/validity_K%a.out
#SBATCH --error=../logs/fig3_batch/validity_K%a.err
#SBATCH --array=0-99
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

K=3
TRIAL_ID=${SLURM_ARRAY_TASK_ID}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering

mkdir -p logs/fig3_batch
mkdir -p results/raw/fig3_batch/K${K}

export PYTHONPATH=$PWD/src:$PYTHONPATH

NEW_NAME="validity_K${K}_t${TRIAL_ID}"
scontrol update JobID=$SLURM_JOB_ID JobName=$NEW_NAME

echo "============================================="
echo "  Running validity (one trial) for K=$K"
echo "  TRIAL_ID=$TRIAL_ID"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python simulations/test_validity_batch.py --K "$K" --trial_id "$TRIAL_ID"

echo "Job completed for K=$K trial=$TRIAL_ID at $(date)"

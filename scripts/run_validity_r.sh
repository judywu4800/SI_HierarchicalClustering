#!/bin/bash
#SBATCH --job-name=validity_K
#SBATCH --output=../logs/validity_K%a.out
#SBATCH --error=../logs/validity_K%a.err
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

K_VALUES=(2 3)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH


NEW_NAME="validity_K${K}"
scontrol update JobID=$SLURM_JOB_ID JobName=$NEW_NAME

echo "============================================="
echo "  Running validity check for K=$K"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python simulations/check_validity.py --K "$K"

touch results/raw/fig3/validity_done.txt

echo "Job completed for K=$K at $(date)"
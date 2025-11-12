#!/bin/bash
#SBATCH --job-name=fig6_randomizedK2_batch
#SBATCH --output=../../logs/fig6/fig6_randomizedK2_%a.log
#SBATCH --error=../../logs/fig6/fig6_randomizedK2_%a.err
#SBATCH --array=0-3
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

# ==== Define linkage list ====
LINKAGES=("complete" "average" "single" "minimax")
LINKAGE=${LINKAGES[$SLURM_ARRAY_TASK_ID]}

# ==== Environment setup ====
module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1
conda activate rpy2_env

echo "============================================="
echo "  Starting fig6_randomized_batch for LINKAGE=$LINKAGE"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

echo "Python executable:"
which python
python -c "import sys; print('Python executable:', sys.executable)"
echo "R HOME:"
R RHOME

# ==== Navigate to repo ====
cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results/raw/fig6

# Ensure Python can find src/
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "PYTHONPATH set to: $PYTHONPATH"
echo "============================================="

# ==== Run experiment for this linkage ====
python experiments/power_experiments/fig6_rand_K2.py "$LINKAGE"

echo "Done for LINKAGE=$LINKAGE at $(date)"

#!/bin/bash
#SBATCH --job-name=validity_fig5_batch
#SBATCH --output=../../logs/validity_fig5_%a.log
#SBATCH --error=../../logs/validity_fig5_%a.err
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

# ==== Define parameter list ====
K_values=(2 3)
K=${K_values[$SLURM_ARRAY_TASK_ID]}

# ==== Environment setup ====
module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1
conda activate rpy2_env

echo "============================================="
echo "  Starting validity_fig5_batch for K=$K"
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
mkdir -p results/raw/fig5

export PYTHONPATH=$PWD/src:$PYTHONPATH

# ==== Run experiment for this K ====
python experiments/fig_code/fig5_simulations.py --K $K

echo "Done for K=$K at $(date)"

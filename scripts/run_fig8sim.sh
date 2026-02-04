#!/bin/bash
#SBATCH --job-name=validity_fig8_batch
#SBATCH --output=../logs/fig5_all/validity_fig8_%a.log
#SBATCH --error=../logs/fig5_all/validity_fig8_%a.err
#SBATCH --array=0-1
#SBATCH --time=06:00:00
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
mkdir -p results/raw/fig5

export PYTHONPATH="$PWD/src:$PYTHONPATH"

# ==== Run experiment for this K ====
python simulations/fig5_simulations.py --K $K --num_trials 2000

echo "Done for K=$K at $(date)"

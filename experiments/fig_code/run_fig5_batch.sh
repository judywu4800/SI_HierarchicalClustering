#!/bin/bash
#SBATCH --job-name=validity_fig5_1000
#SBATCH --output=../../logs/fig5_batch/validity_fig5_%a.log
#SBATCH --error=../../logs/fig5_batch/validity_fig5_%a.err
#SBATCH --array=0-11
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

# ============================================================
#  Section 1: Define parameter grids
# ============================================================
K_values=(2 3)
linkage_values=("complete" "single" "average")
methods=("randomized" "gao_barber")
# ---- Index mapping ----
# total combinations = len(K_values) * len(linkage_values) * len(methods)
num_K=${#K_values[@]}
num_link=${#linkage_values[@]}
num_method=${#methods[@]}

# Compute indices
K_index=$(( SLURM_ARRAY_TASK_ID % num_K ))
link_index=$(( (SLURM_ARRAY_TASK_ID / num_K) % num_link ))
method_index=$(( (SLURM_ARRAY_TASK_ID / (num_K * num_link)) % num_method ))

K=${K_values[$K_index]}
linkage=${linkage_values[$link_index]}
method=${methods[$method_index]}


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1
conda activate rpy2_env

echo "============================================="
echo "  Starting validity_fig5_full_batch"
echo "  SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "  K = $K"
echo "  Linkage = $linkage"
echo "  Method = $method"
echo "============================================="

which python
python -c "import sys; print('Python executable:', sys.executable)"
R RHOME

# ============================================================
#  Section 3: Navigate to repo and prepare folders
# ============================================================
cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs results/raw/fig5_batch
export PYTHONPATH=$PWD/src:$PYTHONPATH

# ============================================================
#  Section 4: Run the experiment
# ============================================================
python experiments/fig_code/fig5_batch.py \
    --K $K \
    --linkage $linkage \
    --method $method \
    --num_trials 1000 \
    --n_jobs 32

echo "Done for K=$K, linkage=$linkage, method=$method at $(date)"

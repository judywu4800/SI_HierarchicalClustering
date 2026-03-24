#!/bin/bash
#SBATCH --job-name=power_randomized_batch
#SBATCH --output=../logs/power/power_randomized_%A_%a.log
#SBATCH --error=../logs/power/power_randomized_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

# ==== Define linkage list ====
LINKAGES=("complete" "average" "single" "minimax")
KS=(2 3)
# build all 8 combinations
COMBOS=()
for l in "${LINKAGES[@]}"; do
  for k in "${KS[@]}"; do
    COMBOS+=("${l}_${k}")
  done
done

PAIR=${COMBOS[$SLURM_ARRAY_TASK_ID]}
LINKAGE=${PAIR%_*}
K=${PAIR#*_}
# ==== Environment setup ====
module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1
conda activate rpy2_env

# dynamically rename job for clarity in squeue
NEW_NAME="fig4_${LINKAGE}_K${K}"
scontrol update JobID=$SLURM_JOB_ID JobName=$NEW_NAME

echo "============================================="
echo "  Starting $NEW_NAME (Array ID: $SLURM_ARRAY_TASK_ID)"
echo "============================================="

echo "Python executable:"
which python
python -c "import sys; print('Python executable:', sys.executable)"
echo "R HOME:"
R RHOME

# ==== Navigate to repo ====
cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results/raw/fig6_es

export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo "============================================="

# ==== Run experiment ====
python simulations/rand_power_es.py --num_trials 2000 --K "$K" --linkage "$LINKAGE"

touch results/raw/fig4_es/power_rand_done.txt
echo "Done for LINKAGE=$LINKAGE, K=$K at $(date)"
#!/bin/bash
#SBATCH --job-name=validity
#SBATCH --output=../../logs/validity/validity_output.log
#SBATCH --error=../../logs/validity/validity_%a.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

linkage_list=("single" "average" "complete" "minimax")
linkage=${linkage_list[$SLURM_ARRAY_TASK_ID]}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1

conda activate rpy2_env

which python
python -c "import sys; print('Python executable:', sys.executable)"

export R_HOME=$(R RHOME)

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "============================================="
echo " Running validity simulations linkage=$linkage"
echo " SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python experiments/validity_experiments/validity_linkages.py \
    --linkage $linkage \
    --K 3 \
    --num_trials 2000

echo "Done linkage=$linkage at $(date)"


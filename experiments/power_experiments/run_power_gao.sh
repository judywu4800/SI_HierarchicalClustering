#!/bin/bash
#SBATCH --job-name=power_gao
#SBATCH --output=../../logs/power/power_gao_output.log
#SBATCH --error=../../logs/power/power_gao_%a.err
#SBATCH --array=0-1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

linkage_list=("single" "average" "complete")
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
echo " Running Gao ES linkage=$linkage"
echo " SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python experiments/power_experiments/gao_power_es.py \
    --linkage $linkage \
    --K 3 \
    --num_trials 2000

echo "Done linkage=$linkage at $(date)"


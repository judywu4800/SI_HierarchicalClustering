#!/bin/bash
#SBATCH --job-name=validity_gao_barber
#SBATCH --output=../../logs/validity2_K%a.out
#SBATCH --error=../../logs/validity2_K%a.err
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

K_VALUES=(2 3)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

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


NEW_NAME="validity_gao_barber_K${K}"
scontrol update JobID=$SLURM_JOB_ID JobName=$NEW_NAME

echo "============================================="
echo "  Running validity check for K=$K"
echo "  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python experiments/validity_experiments/check_validity2.py --K "$K"

echo "Job completed for K=$K at $(date)"

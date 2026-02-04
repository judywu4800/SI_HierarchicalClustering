#!/bin/bash
#SBATCH --job-name=khat_array
#SBATCH --output=../logs/chooseK/chooseK_%a.out
#SBATCH --error=../logs/chooseK/chooseK_%a.err
#SBATCH --array=0-599
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

DELTA_LIST=(4 6 8 10 12 14)
N_REP=100

delta_idx=$(( SLURM_ARRAY_TASK_ID / N_REP ))
trial=$(( SLURM_ARRAY_TASK_ID % N_REP ))

delta=${DELTA_LIST[$delta_idx]}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "Starting trial $SLURM_ARRAY_TASK_ID at $(date)"
python simulations/Khat_delta.py --delta $delta --trial $trial
echo "Finished trial $SLURM_ARRAY_TASK_ID at $(date)"
touch results/raw/fig5/khat_delta_done.txt
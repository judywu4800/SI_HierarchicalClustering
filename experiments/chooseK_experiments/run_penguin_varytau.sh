#!/bin/bash
#SBATCH --job-name=penguins_array_gamma
#SBATCH --output=../../logs/penguins_gamma_array.out
#SBATCH --error=../../logs/penguins_gamma_array.err
#SBATCH --array=0-699
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

GAMMA_LIST=(0.9 0.8 0.7 0.6 0.5 0.4 0.3)
N_REP=100

gamma_idx=$(( SLURM_ARRAY_TASK_ID / N_REP ))
trial=$(( SLURM_ARRAY_TASK_ID % N_REP ))

gamma=${GAMMA_LIST[$gamma_idx]}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "Starting trial $SLURM_ARRAY_TASK_ID at $(date)"
python experiments/chooseK_experiments/penguin_varytau.py --gamma $gamma --trial_id $trial
echo "Finished trial $SLURM_ARRAY_TASK_ID at $(date)"
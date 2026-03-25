#!/bin/bash
#SBATCH --job-name=typeI_random
#SBATCH --output=../logs/typeI_r_output_%a.log
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-1

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

NUM_TRIALS=200
NUM_REPEATS=100
ALPHA=0.05

K_LIST=(2 3)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

python simulations/type1_error.py \
  --K ${K} \
  --num_trials ${NUM_TRIALS} \
  --num_repeats ${NUM_REPEATS}

#!/bin/bash
#SBATCH --job-name=typeI_random
#SBATCH --output=../logs/typeI_r_output.log
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

K=3
NUM_TRIALS=200
NUM_REPEATS=100
ALPHA=0.05

python simulations/type1_error.py \
  --K ${K} \
  --num_trials ${NUM_TRIALS} \
  --num_repeats ${NUM_REPEATS}

touch results/raw/fig3/type1_done.txt


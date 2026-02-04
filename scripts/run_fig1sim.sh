#!/bin/bash
#SBATCH --job-name=findK_batch
#SBATCH --output=../logs/fig1/findK_batch_%A_%a.out
#SBATCH --error=../logs/fig1/findK_batch_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

export BATCH_ID=$SLURM_ARRAY_TASK_ID
export NUM_BATCHES=10
export REPS_PER_BATCH=10


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

python simulations/fig1_simulations.py
touch results/raw/fig1/fig1sim_done.txt

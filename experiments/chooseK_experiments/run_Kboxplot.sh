#!/bin/bash
#SBATCH --job-name=K_true_boxplot
#SBATCH --output=../../logs/K_true_output.log
#SBATCH --time=00:01:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

module purge
module load python3.10-anaconda
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results/chooseK_experiments

export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/chooseK_experiments/Khat_test.py


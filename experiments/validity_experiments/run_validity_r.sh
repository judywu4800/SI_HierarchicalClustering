#!/bin/bash
#SBATCH --job-name=validity_test
#SBATCH --output=../../logs/validity_output.log
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32


module purge
module load python3.10-anaconda
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results/validity_experiments

export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/validity_experiments/check_validity.py


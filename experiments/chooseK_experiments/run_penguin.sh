#!/bin/bash
#SBATCH --job-name=penguin_chooseK
#SBATCH --output=../../logs/penguin_K_output.log
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/chooseK_experiments/penguin_chooseK.py

#!/bin/bash
#SBATCH --job-name=fig1_sim
#SBATCH --output=../../logs/fig1_sim_output.log
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32


module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/fig_code/fig1_simulations.py


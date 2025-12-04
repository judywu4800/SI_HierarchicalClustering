#!/bin/bash
#SBATCH --job-name=validity_varyTau
#SBATCH --output=../../logs/validity_varyTau/validity_varyTau.out
#SBATCH --error=../../logs/validity_varyTau/validity_varyTau.err
#SBATCH --array=0-199
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/validity_experiments/validity_tau.py \
    --trial_id $SLURM_ARRAY_TASK_ID \
    --n 30 \
    --p 10 \
    --sigma 1 \
    --K 3 \
    --epsilon 0.01
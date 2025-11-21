#!/bin/bash
#SBATCH --job-name=fwer_tau
#SBATCH --output=../../logs/fwer/fwer_tau_%a.out
#SBATCH --error=../../logs/fwer/fwer_tau_%a.err
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

TAUS=(0 0.025 0.05 0.1 0.25 0.5 1 5)
TAU=${TAUS[$SLURM_ARRAY_TASK_ID]}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

python experiments/chooseK_experiments/fwer_tau.py \
    --tau $TAU \
    --num_trials 1000 \
    --outdir results/raw/fwer_90
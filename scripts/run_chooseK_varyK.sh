#!/bin/bash
#SBATCH --job-name=khat_array_Ks
#SBATCH --output=../logs/chooseK_varyK/chooseKs.out
#SBATCH --error=../logs/chooseK_varyK/chooseKs.err
#SBATCH --array=0-999
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

K_LIST=(1 2 3 4 5 6 7 8 9 10)
N_REP=100
N=200
P=2
DELTA=6

K_idx=$(( SLURM_ARRAY_TASK_ID / N_REP ))
trial=$(( SLURM_ARRAY_TASK_ID % N_REP ))

K=${K_LIST[$K_idx]}

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "Starting trial $SLURM_ARRAY_TASK_ID at $(date)"
python simulations/fig9_sim.py \
    --K $K \
    --n $N \
    --p $P \
    --delta $DELTA \
    --trial $trial
echo "Finished trial $SLURM_ARRAY_TASK_ID at $(date)"
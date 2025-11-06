#!/bin/bash
#SBATCH --job-name=penguins_array
#SBATCH --output=../../logs/penguins_array.out
#SBATCH --error=../../logs/penguins_array.err
#SBATCH --array=0-49
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate rac-env

cd /home/judydw/SI_HierarchicalClustering
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "Starting trial $SLURM_ARRAY_TASK_ID at $(date)"
python experiments/chooseK_experiments/penguin_K_batch.py --trial_id $SLURM_ARRAY_TASK_ID
echo "Finished trial $SLURM_ARRAY_TASK_ID at $(date)"
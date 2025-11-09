#!/bin/bash
#SBATCH --job-name=linkage_batch
#SBATCH --output=logs/linkage_%a.out
#SBATCH --error=logs/linkage_%a.err
#SBATCH --array=0-3
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

module load python3.10-anaconda/2023.03
source activate rac-env

LINKAGES=("complete" "average" "single" "minimax")
LINKAGE=${LINKAGES[$SLURM_ARRAY_TASK_ID]}

echo "Running linkage: $LINKAGE"
python experiments/power_experiments/run_linkage_batch.py "$LINKAGE"
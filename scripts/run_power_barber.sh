#!/bin/bash
#SBATCH --job-name=power_barber
#SBATCH --output=../logs/power/power_barber_output.log
#SBATCH --error=../logs/power/power_barber_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

linkage_list=("single" "average" "complete")
KS=(2 3)

i=$SLURM_ARRAY_TASK_ID
linkage=${linkage_list[$((i / 2))]}
K=${KS[$((i % 2))]}

if [ $((i / 2)) -ge ${#linkage_list[@]} ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$i"
  exit 1
fi

module purge
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
module load R/4.3.1

conda activate rpy2_env

which python
python -c "import sys; print('Python executable:', sys.executable)"

export R_HOME=$(R RHOME)

cd /home/judydw/SI_HierarchicalClustering
mkdir -p logs
mkdir -p results

export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "============================================="
echo " Running Barber ES linkage=$linkage, K=$K"
echo " SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "============================================="

python simulations/barber_power_es.py \
    --linkage $linkage \
    --K $K \
    --num_trials 2000
touch results/raw/fig4_es/power_barber_done.txt
echo "Done linkage=$linkage K=$K at $(date)"


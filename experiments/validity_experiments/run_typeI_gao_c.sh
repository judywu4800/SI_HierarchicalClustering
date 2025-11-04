#!/bin/bash
#SBATCH --job-name=typeI_gao_clu
#SBATCH --output=../../logs/typeI_gao_c_output.log
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

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

python experiments/validity_experiments/type1_gao_clustered.py

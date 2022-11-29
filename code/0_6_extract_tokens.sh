#!/bin/bash
#SBATCH -p fat
#SBATCH -t 24:00:00
#SBATCH -C scratch
#SBATCH --mem 200G

module purge

module load anaconda3
source $ANACONDA3_ROOT/etc/profile.d/conda.sh

for i in $(seq ${CONDA_SHLVL}); do
    conda deactivate
done

conda activate myenv

python3 ./0_6_extract_tokens.py

#!/bin/bash
#SBATCH -p medium
#SBATCH -t 24:00:00
#SBATCH -C scratch
#SBATCH --mem 100G

module purge

module load anaconda3

conda activate myenv

python3 ./0_6_extract_tokens.py

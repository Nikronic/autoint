#!/usr/bin/sh

#SBATCH -p gpu20
#SBATCH -t 1-00:00:00
#SBATCH -J ImgIntegral
#SBATCH -D /HPS/deep_topopt/work/autoint
#SBATCH -o /HPS/deep_topopt/work/autoint/logs/slurm/siren/slurm-%x-%j.log
#SBATCH --gres gpu:1

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate autoint


# call your program here
python3 /HPS/deep_topopt/work/autoint/training/train_siren.py --jid ${SLURM_JOBID}
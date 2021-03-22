#!/usr/bin/sh

#SBATCH -p cpu20
#SBATCH -t 1-23:00:00
#SBATCH -J ImgIntegral
#SBATCH -D /HPS/deep_topopt/work/autoint
#SBATCH -o /HPS/deep_topopt/work/autoint/logs/slurm/siren/slurm-%x-%j.log
#SBATCH -c 2
#SBATCH --mem-per-cpu 32000

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate autoint


# call your program here
python3 /HPS/deep_topopt/work/autoint/training/train_siren.py --jid ${SLURM_JOBID}

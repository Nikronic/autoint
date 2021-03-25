#!/usr/bin/sh

#SBATCH -p cpu20
#SBATCH -t 0-05:00:00
#SBATCH -J CT
#SBATCH -D /HPS/deep_topopt/work/autoint
#SBATCH -o /HPS/deep_topopt/work/autoint/logs/slurm/ct/slurm-%x-%j.log
#SBATCH -c 2

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate autoint


# call your program here
python3 /HPS/deep_topopt/work/autoint/training/train_ct.py --jid ${SLURM_JOBID}

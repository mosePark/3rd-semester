#!/bin/bash
#SBATCH --job-name=AL
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=001
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu3

hostname
date

module add ANACONDA/2020.11

/home1/mose1103/anaconda3/envs/fine-koactive/bin/python main.py --task active_learning

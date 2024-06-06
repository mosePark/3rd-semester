#!/bin/bash
#SBATCH --job-name=example
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu1

hostname
date

module add ANACONDA/2020.11

/home1/mose1103/anaconda3/envs/fine-koactive/bin/python 24-06-06.py

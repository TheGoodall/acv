#!/bin/bash
#SBATCH -N 1
#SBATCH -o stdout.out
#SBATCH -c 4
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH -t 2:00:00
#SBATCH --job-name=ACV
#SBATCH --mem=6G

source /etc/profile
singularity exec --nv ../cv.tif python -u cv.py

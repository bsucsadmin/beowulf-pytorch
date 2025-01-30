#!/usr/bin/bash
#SBATCH -J torch-benchmarks
#SBATCH -o %x.o%j
#SBATCH --gres=gpu:L40:1
#SBATCH -p gpu

source /opt/ohpc/pub/Anaconda3/bin/activate torch
~/.conda/envs/torch/bin/python main.py
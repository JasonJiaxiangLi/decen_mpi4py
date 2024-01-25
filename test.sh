#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=62gb
#SBATCH --output=log/slurm/%j.out                              
#SBATCH --error=log/slurm/%j.out
#SBATCH --job-name=mezo
#SBATCH -n 12
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-8

# Benchmark info
echo "TIMING - Starting jupyter at: $(date)"

conda activate dec_mpi
cd ~/jobsubmit/decen_mpi4py || exit
echo "Job is starting on $(hostname)"
which python3
nvidia-smi

mpirun -np 8 python run.py --data='cifar' --model='resnet' --updates=10001 --report=100 --comm_pattern='ring' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt'

exit
#!/bin/bash
#SBATCH --job-name="Milestone1_Unroll"
#SBATCH --output="Milestone1_Unroll.out"
#SBATCH --error="Milestone1_Unroll.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun nsys profile --trace=cuda,nvtx -o issue_c ./m1_unroll 10000


#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
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

srun ./m2 100 > m2.out

# Generate nsys profile information, output will be in profile.out and report.nsys-rep
#srun nsys profile --stats=true -o q9sys ./m2 > profile.out

# Generate ncu profile information, output will be in analysis_file.ncu-rep
#srun ncu --set full -f -o analysis_file ./m2
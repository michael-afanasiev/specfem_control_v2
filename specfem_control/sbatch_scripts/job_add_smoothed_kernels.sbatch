#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="kernel_sum"
#SBATCH --nodes=3
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=./sbatch_logs/job_add.o
#SBATCH --error=./sbatch_logs/job_add.e

# Load modules.
module load slurm

# Parameters.
optimization_dir=$1
max_percentage=$2

echo "--"
echo "Summing kernels: `date`"
echo "--"
 
cd $optimization_dir
aprun -B ./bin/xadd_model_tiso $max_percentage

echo "--"
echo "Done: `date`"
echo "--"

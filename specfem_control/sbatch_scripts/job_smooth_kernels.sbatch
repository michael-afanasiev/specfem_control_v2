#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="kernel_smooth"
#SBATCH --nodes=3
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=./sbatch_logs/job_smooth.%A.o
#SBATCH --error=./sbatch_logs/job_smooth.%A.e

# Load modules.
module load slurm

# Parameters.
horizontal_smoothing=$1
vertical_smoothing=$2
kernel_name=$3
kernel_dir=$4
databases=$5
optimization_dir=$6

echo "--"
echo "Summing kernels: `date`"
echo "--"
 
cd $optimization_dir
aprun -B ./bin/xsmooth_sem $horizontal_smoothing $vertical_smoothing $kernel_name $kernel_dir $databases 

echo "--"
echo "Done: `date`"
echo "--"

#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="calculate misfit"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/calculate_misfit.%A.o
#SBATCH --error=./sbatch_logs/calculate_misfit.%A.e

export MV2_ENABLE_AFFINITY=0
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=16

lasif_scratch_dir=$1
iteration_name=$2
current_dir=`pwd`

cd $lasif_scratch_dir

aprun -B $current_dir/python_scripts/evaluate_misfit_for_iteration.py $iteration_name
#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="adjoint_source"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/adjoint_source.%A.%a.o
#SBATCH --error=./sbatch_logs/adjoint_source.%A.%a.e

if [ "$2" == "" ]; then
  echo "Usage: ./select_windows_parallel.sh" 
  exit
fi

cd /users/afanasm/specfem_control/african_office
aprun -B oval_office select_windows -fj $1 -lj $2

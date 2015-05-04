#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="preprocessing"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=./sbatch_logs/preprocess_data.%A.o
#SBATCH --error=./sbatch_logs/preprocess_data.%A.e

export MV2_ENABLE_AFFINITY=0
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1

if [ "$3" == "" ]; then
  echo "Usage: ./select_windows_parallel.sh [lasif_scratch_dir] 
      [lasif_project_dir] [iteration_name]"
  exit
fi

lasif_scratch_dir=$1; event=$2; freqmin=$3; freqmax=$4; origin_time=$5
dt=$6; npts=$7; lasif_project_dir=$8

# Copy project to scratch
echo "Copying data..., $event"
rsync -av $lasif_project_dir/DATA/$event/ $lasif_scratch_dir/DATA/$event
echo "Done copying data..."


# Run the code.
aprun -B ./python_scripts/preprocess_data.py $lasif_scratch_dir $event $freqmin $freqmax $origin_time $dt $npts

# Copy back to project.
rsync -av $lasif_scratch_dir/DATA/$event/*preprocessed* $lasif_project_dir/DATA/$event/
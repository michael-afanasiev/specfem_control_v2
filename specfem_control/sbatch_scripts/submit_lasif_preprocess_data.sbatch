#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="preprocessing"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=./sbatch_logs/preprocess_data.%A.%a.o
#SBATCH --error=./sbatch_logs/preprocess_data.%A.%a.e

export MV2_ENABLE_AFFINITY=0
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=1

if [ "$3" == "" ]; then
  echo "Usage: ./select_windows_parallel.sh [lasif_scratch_dir] 
      [lasif_project_dir] [iteration_name]"
  exit
fi

lasif_scratch_dir=$1
lasif_project_dir=$2
iteration_name=$3

# Get all event names.
cd $lasif_scratch_dir
shopt -s nullglob
array=($lasif_scratch_dir/EVENTS/*)
my_event=${array[$SLURM_ARRAY_TASK_ID]}
my_event=${my_event%.xml}
my_event=${my_event##*/}

# Copy data to scratch.
rsync -av $lasif_project_dir/DATA/$my_event $lasif_scratch_dir/DATA/

# Untar data.
cd $lasif_scratch_dir/DATA/$my_event
for dir in ./*; do
 if [ -d "$dir" ]; then
   cd $dir
   tar -xvf data.tar
   rm -f data.tar
   cd ../
 fi
done

# Run the code.
aprun -B lasif preprocess_data $iteration_name $my_event --read_only_caches

# Retar data.
cd $lasif_scratch_dir/DATA/$my_event
for dir in ./*; do
 if [ -d "$dir" ]; then
   cd $dir
   tar -cvf data.tar *.mseed
   rm -f *.mseed
   cd ../
 fi
done

# Copy back to project.
rsync -av $lasif_scratch_dir/DATA/$my_event/*preprocessed* $lasif_project_dir/DATA/$my_event
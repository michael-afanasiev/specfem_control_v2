#!/bin/bash -l

#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="kernel_sum"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=./sbatch_logs/job_vtk.o
#SBATCH --error=./sbatch_logs/job_vtk.e

# Load modules.
module load slurm

# Parameters.
optimization_dir=$1
slices_file=./VTK_FILES/SLICES_ALL.txt
databases_mpi=../mesh/DATABASES_MPI
input_kernels=./PROCESSED_KERNELS
output_dir=./VTK_FILES
resolution=0
region=1

# Kernel names
names=( bulk_betah_kernel_smooth bulk_betav_kernel_smooth bulk_c_kernel_smooth eta_kernel_smooth )

echo "--"
echo "Generating .vtk files kernels: `date`"
echo "--"
 
cd $optimization_dir
for i in "${names[@]}"; do
        aprun -B ./bin/xcombine_vol_data_vtk $slices_file $i $databases_mpi $input_kernels $output_dir $resolution $region
done
    
echo "--"
echo "Done: `date`"
echo "--"

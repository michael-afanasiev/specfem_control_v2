#!/bin/bash -l

#!/bin/bash -l

#SBATCH --account=ch1
#SBATCH --job-name="model_vtk"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=./sbatch_logs/job_vtk.o
#SBATCH --error=./sbatch_logs/job_vtk.e

# Load modules.
module load slurm

# Parameters.
mesh_dir=$1
slices_file=./VTK_FILES/SLICES_ALL.txt
databases_mpi=./DATABASES_MPI
output_dir=./VTK_FILES
resolution=0
region=1

# Kernel names
names=( vsh vsv vph vpv rho eta )

echo "--"
echo "Generating .vtk files kernels: `date`"
echo "--"
 
cd $mesh_dir
cp ../OPTIMIZATION/bin/xcombine_vol_data_vtk ./bin/
for i in "${names[@]}"; do
        aprun -B ./bin/xcombine_vol_data_vtk $slices_file $i $databases_mpi $databases_mpi $output_dir $resolution $region
done
    
echo "--"
echo "Done: `date`"
echo "--"

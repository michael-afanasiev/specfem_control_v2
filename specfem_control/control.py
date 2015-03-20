#!/usr/bin/env python

import os
import utils
import shutil
import subprocess
import tarfile

import data
import seismograms

from multiprocessing import Pool, cpu_count

def _copy_files_and_tar((source, dest)):
    
    utils.print_ylw("Moving: " + source.split('/')[-2])
    
    tar_name = os.path.join(dest, 'data.tar')

    if os.path.exists(tar_name):
        os.remove(tar_name)
        
    if not os.path.exists(dest):
        utils.mkdir_p(dest)
    
    source_files = [x for x in os.listdir(source) if x.endswith('.sac')]
    with tarfile.open(tar_name, "w") as tar:
        for file in source_files:
            tar.add(os.path.join(source, file), arcname=file)

def _setup_dir_tree(event, forward_run_dir):
    """
    Sets up the simulation directory structure for one event.
    """
    event_path = os.path.join(forward_run_dir, event)
    utils.mkdir_p(event_path)
    utils.mkdir_p(os.path.join(event_path, 'bin'))
    utils.mkdir_p(os.path.join(event_path, 'DATA'))
    utils.mkdir_p(os.path.join(event_path, 'DATA/GLL'))
    utils.mkdir_p(os.path.join(event_path, 'DATA/cemRequest'))
    utils.mkdir_p(os.path.join(event_path, 'OUTPUT_FILES'))
    utils.mkdir_p(os.path.join(event_path, 'DATABASES_MPI'))
    utils.mkdir_p(os.path.join(event_path, 'SEM'))


def _copy_input_files(event, forward_run_dir, lasif_path, iteration_name,
                      specfem_root, mesh=False):
    """
    Copies the pre-generated input files from LASIF. Don't copy the Par_file or
    STF. Par_file will be copied later, from the LASIF/SUBMISSION directory.
    """
    compile_data = os.path.join(specfem_root, 'DATA')
    mesh_data = os.path.join(forward_run_dir, 'mesh', 'DATA')
    event_data = os.path.join(forward_run_dir, event, 'DATA')
    lasif_output = os.path.join(lasif_path, 'OUTPUT')
    for dir in os.listdir(lasif_output):
        if iteration_name in dir and event in dir and 'input_files' in dir:
            source = os.path.join(lasif_output, dir)
            utils.copy_directory(source, event_data, exc=['Par_file', 'STF'])
            # This flag also copies the first event's data to the solver base
            # directory for compilation, and the mesh directory for meshing.
            if mesh:
                utils.copy_directory(
                    source,
                    mesh_data,
                    exc=[
                        'Par_file',
                        'STF'])
                utils.copy_directory(source, compile_data,
                                     exc=['Par_file, STF'])


def _change_job_and_par_file(params, run_type):
    """
    Runs and changes the simulation_type and simulation time in the submission
    files.
    """
    forward_run_dir = params['forward_run_dir']
    forward_stage_dir = params['forward_stage_dir']
    event_list = params['event_list']
    if run_type == 'adjoint_run':
        simulation_type = '= 3'
        save_forward = '= .false.\n'
        sbatch_time = '#SBATCH --time=04:00:00\n'
    elif run_type == 'first_iteration' or run_type == 'update_mesh' \
            or run_type == 'forward_run':
        simulation_type = '= 1'
        save_forward = '= .true.\n'
        sbatch_time = '#SBATCH --time=02:00:00\n'

    utils.print_ylw("Modifying Par_files for run type...")
    os.chdir(forward_run_dir)
    for dir in os.listdir('./'):
        if dir not in event_list and dir != 'mesh':
            continue
        par_path = os.path.join(dir, 'DATA', 'Par_file')
        par_path_new = os.path.join(dir, 'DATA', 'Par_file_new')
        new_file = open(par_path_new, 'w')
        with open(par_path, 'r') as file:
            for line in file:
                fields = line.split('=')
                if 'SIMULATION_TYPE' in fields[0]:
                    new_file.write(fields[0] + simulation_type + fields[1][2:])
                elif 'SAVE_FORWARD' in fields[0]:
                    new_file.write(fields[0] + save_forward)
                elif fields[0].startswith('MODEL') and \
                        run_type == 'update_mesh':
                    new_file.write(fields[0] + '= CEM_GLL\n')
                elif fields[0].startswith('MODEL') and \
                        run_type == 'first_iteration':
                    new_file.write(fields[0] + '= CEM_ACCEPT\n')
                else:
                    new_file.write(line)
        os.rename(par_path_new, par_path)

    utils.print_ylw("Modifying .sbatch file for run type...")
    os.chdir(forward_stage_dir)
    job_array = os.path.join('./', 'jobArray_solver_daint.sbatch')
    new_job_array_name = os.path.join('./', 'jobArray_solver_daint.sbatch_new')
    new_job_array = open(new_job_array_name, 'w')
    with open(job_array, 'r') as file:
        for line in file:
            if '--time' in line:
                new_job_array.write(sbatch_time)
            else:
                new_job_array.write(line)

    new_job_array.close()
    os.remove(job_array)
    os.rename(new_job_array_name, job_array)
    
def _copy_relevant_lasif_files(params):
    """
    Rsyncs the CACHE, EVENTS, FUNCTIONS, INTERATIONS, OUTPUT, STATIONS, 
    directories in LASIF to /scratch.
    """
    lasif_path = params['lasif_path']
    lasif_scratch_path = params['lasif_scratch_path']
    
    utils.mkdir_p(lasif_scratch_path)
    
    # Copy directory tree.
    subprocess.Popen(
        ["rsync -av --include='*/' --exclude='*' " + lasif_path + "/ " 
         + lasif_scratch_path], shell=True).wait()
    
    folders = ['CACHE', 'STATIONS', 'EVENTS', 'FUNCTIONS', 'ITERATIONS',
               'OUTPUT', 'config.xml', 'LOGS', 'MODELS', 
               'WAVEFIELDS', 'ADJOINT_SOURCES_AND_WINDOWS', 'KERNELS']
    
    for folder in folders:
        subprocess.Popen(['rsync', '-av', os.path.join(lasif_path, folder), 
                         lasif_scratch_path]).wait()


def setup_solver(params):
    """
    It's dylan, you know the drill.
    """
    # Setup local parameters.
    forward_run_dir = params['forward_run_dir']
    forward_stage_dir = params['forward_stage_dir']
    event_list = params['event_list']
    lasif_path = params['lasif_path']
    iteration_name = params['iteration_name']
    specfem_root = params['specfem_root']
    compiler_suite = params['compiler_suite']
    project_name = params['project_name']

    # Set up the mesh directory.
    _setup_dir_tree('mesh', forward_run_dir)

    # Set up the optimization directory.
    optimization_base = os.path.join(forward_run_dir, 'OPTIMIZATION')
    utils.mkdir_p(optimization_base)
    utils.mkdir_p(os.path.join(optimization_base, 'bin'))
    utils.mkdir_p(os.path.join(optimization_base, 'PROCESSED_KERNELS'))
    utils.mkdir_p(os.path.join(optimization_base, 'GRADIENT_INFO'))
    utils.mkdir_p(os.path.join(optimization_base, 'LOGS'))
    utils.mkdir_p(os.path.join(optimization_base, 'DATA'))
    utils.mkdir_p(os.path.join(optimization_base, 'VTK_FILES'))

    # Create the forward modelling directories. Also copy relevant parameter
    # files from the LASIF project. _copy_input_files also copies the input
    # files to the specfem_root directory if mesh == True.
    utils.print_ylw("Creating forward modelling directories...")
    mesh = True
    for i, event in enumerate(event_list):
        _setup_dir_tree(event, forward_run_dir)
        _copy_input_files(event, forward_run_dir, lasif_path, iteration_name,
                          specfem_root, mesh=mesh)
        mesh = False

    # Copy the files in SUBMISSION to the specfem root directory.
    par_file = os.path.join(lasif_path, 'SUBMISSION', project_name,
                            'Par_file')
    dest = os.path.join(specfem_root, 'DATA')
    utils.safe_copy(par_file, dest)

    # Change to specfem root directory and compile.
    utils.print_ylw("Compiling...")
    os.chdir(specfem_root)
    with open('compilation_log.txt', 'w') as output:
        proc = subprocess.Popen(['./mk_daint.sh', compiler_suite, 'adjoint'],
                                stdout=output, stderr=output)
        proc.communicate()
        proc.wait()

    # Distribute binaries and Par_file to directories.
    utils.print_ylw('Copying compiled binaries...')
    bin_directory = os.path.join('./bin')
    opt_bin_directory = os.path.join(optimization_base, 'bin')
    opt_dat_directory = os.path.join(optimization_base, 'DATA')
    utils.copy_directory(bin_directory, opt_bin_directory)
    for event in os.listdir(forward_run_dir):
        event_bin = os.path.join(forward_run_dir, event, 'bin')
        event_dat = os.path.join(forward_run_dir, event, 'DATA')
        compile_par = os.path.join(specfem_root, 'DATA', 'Par_file')
        utils.safe_copy(compile_par, event_dat)
        utils.copy_directory(bin_directory, event_bin,
                             only=['xspecfem3D', 'xmeshfem3D'])

    # Also copy to the optimization directory. Recompile with vectorized cray
    # compiler.
    utils.print_ylw("Recompiling for vectorized smoother CRAY smoother...")
    with open('compilation_log_tomo.txt', 'w') as output:
        proc = subprocess.Popen(['./mk_daint.sh', 'cray.tomo', 'adjoint'],
                                stdout=output, stderr=output)
        proc.communicate()
        proc.wait()
    utils.copy_directory(bin_directory, opt_bin_directory)
    compile_par = os.path.join(specfem_root, 'DATA', 'Par_file')
    utils.safe_copy(compile_par, opt_dat_directory)

    # Copy jobarray script to base directory.
    utils.print_ylw('Copying jobarray sbatch script...')
    source = os.path.join(lasif_path, 'SUBMISSION', project_name,
                          'jobArray_solver_daint.sbatch')
    utils.safe_copy(source, forward_stage_dir)
    utils.mkdir_p(os.path.join(forward_stage_dir, 'logs'))

    # Copy mesh submission script.
    source = os.path.join(lasif_path, 'SUBMISSION', project_name,
                          'job_mesher_daint.sbatch')
    dest = os.path.join(forward_run_dir, 'mesh')
    utils.safe_copy(source, dest)

    # Copy topography information to mesh directory.
    utils.print_ylw('Copying topography information...')
    master_topo_path = os.path.join(specfem_root, 'DATA', 'topo_bathy')
    mesh_topo_path = os.path.join(forward_run_dir, 'mesh', 'DATA',
                                  'topo_bathy')
    utils.mkdir_p(mesh_topo_path)
    utils.copy_directory(master_topo_path, mesh_topo_path)

    utils.print_blu('Done.')


def prepare_solver(params):
    """
    Sets up symbolic link to generated mesh files.
    """
    forward_run_dir = params['forward_run_dir']
    event_list = params['event_list']
    utils.print_ylw("Preparing solver directories...")
    databases_mpi = os.path.join(params['forward_run_dir'], 'mesh',
                                 'DATABASES_MPI')
    output_files = os.path.join(params['forward_run_dir'], 'mesh',
                                'OUTPUT_FILES')
    for dir in sorted(os.listdir(params['forward_run_dir'])):
        if dir not in event_list:
            continue

        utils.print_ylw('Linking ' + dir + '...')
        dest_mpi = os.path.join(forward_run_dir, dir, 'DATABASES_MPI')
        dest_out = os.path.join(forward_run_dir, dir, 'OUTPUT_FILES')
        utils.sym_link_directory(databases_mpi, dest_mpi)
        utils.sym_link_directory(output_files, dest_out)

    utils.print_blu('Done.')


def submit_solver(params, first_job, last_job, run_type):
    """
    Submits the jobarray script for jobs first_job to last_job.
    """
    _change_job_and_par_file(params, run_type)

    forward_stage_dir = params['forward_stage_dir']
    iteration_name = params['iteration_name']
    job_array = 'jobArray_solver_daint.sbatch'
    os.chdir(forward_stage_dir)
    subprocess.Popen(['sbatch', '--array=%s-%s' % (first_job, last_job),
                      job_array, iteration_name]).wait()


def submit_mesher(params, run_type):
    """
    Submits the jobarray script in the mesh directory.
    """

    _change_job_and_par_file(params, run_type)

    forward_run_dir = params['forward_run_dir']
    os.chdir(os.path.join(forward_run_dir, 'mesh'))
    subprocess.Popen(['sbatch', 'job_mesher_daint.sbatch']).wait()


def submit_window_selection(params, first_job, last_job):
    """
    Submits the window selection job for jobs first_job to last_job.
    """
    lasif_project_dir = params['lasif_path']
    scratch_path = params['scratch_path']
    iteration_name = params['iteration_name']
    lasif_project_name = os.path.basename(lasif_project_dir)
    lasif_scratch_dir = os.path.join(scratch_path, lasif_project_name)

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'select_windows_parallel.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(['sbatch', '--array=%s-%s' % (first_job, last_job),
                      sbatch_file, lasif_scratch_dir,
                      lasif_project_dir, iteration_name]).wait()


def distribute_adjoint_sources(params):
    """
    Searches through the OUTPUT directory of the lasif project, and distributes
    the adjoint sources to their appropriate simulated directories. Also writes
    the STATIONS_ADJOINT file.
    """
    lasif_output_dir = os.path.join(params['lasif_path'], 'OUTPUT')
    forward_run_dir = params['forward_run_dir']
    for dir in sorted(os.listdir(lasif_output_dir)):

        if 'adjoint' not in dir:
            continue

        adjoint_source_dir = os.path.join(lasif_output_dir, dir)
        event_name = adjoint_source_dir[adjoint_source_dir.find('GCMT'):]
        solver_data_dir = os.path.join(forward_run_dir, event_name, 'DATA')
        solver_sem_dir = os.path.join(forward_run_dir, event_name, 'SEM')
        adjoint_stations = os.listdir(adjoint_source_dir)

        utils.print_ylw("Copying adjoint sources for " + event_name + "...")
        utils.mkdir_p(solver_sem_dir)
        # This is necessary because the adjoint source names expected by
        # specfem3d_globe are opposite of what lasif puts out. Could just fix
        # this in lasif instead. This line would then reduce to a
        # 'copy_directory'.
        for old_name in os.listdir(adjoint_source_dir):
            fields = old_name.split('.')
            new_name = fields[1] + '.' + fields[0] + '.' + fields[2] + '.' + \
                fields[3]
            utils.safe_copy_file(os.path.join(adjoint_source_dir, old_name),
                                 os.path.join(solver_sem_dir, new_name))

        adjoint_stations = sorted(list(set([fields0.split('.')[0] +
                                            '.' + fields0.split('.')[1]
                                            for fields0 in adjoint_stations])))
        with open(os.path.join(solver_data_dir, 'STATIONS_ADJOINT'), 'w') as \
                sta_adj:
            sta = open(os.path.join(solver_data_dir, 'STATIONS'), 'r')
            for line in sta:
                if line.split()[0] + '.' + line.split()[1] in adjoint_stations:
                    sta_adj.write(line)

    utils.print_blu('Done.')


def sum_kernels(params, first_job, last_job):
    """
    Goes through the output solver directory, and runs the summing
    commands on the kernels that were output. Kernels will end up in the
    optimization/processed kernels directory.
    """
    forward_run_dir = params['forward_run_dir']
    event_list = params['event_list']
    optimization_dir = os.path.join(forward_run_dir, 'OPTIMIZATION')
    gradient_info_dir = os.path.join(optimization_dir, 'GRADIENT_INFO')
    file_name = os.path.join(gradient_info_dir, 'kernels_list.txt')

    # First, make sure that kernels exist in the directories.
    event_kernels = []
    for event in os.listdir(forward_run_dir):
        if event not in event_list[first_job:last_job + 1]:
            continue
        databases_mpi = os.path.join(forward_run_dir, event, 'DATABASES_MPI')
        if any('_kernel.bin' in s for s in os.listdir(databases_mpi)):
            event_kernels.append(event)
        else:
            utils.print_red("Kernel not found for event: " + event)

    # Write the existing kernels_list file.
    with open(file_name, 'w') as outfile:
        for event in event_kernels:
            full_path = os.path.join(forward_run_dir, event, 'DATABASES_MPI')
            outfile.write(full_path + '\n')

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'job_sum_kernels.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(['sbatch', sbatch_file, optimization_dir]).wait()

def lasif_preprocess_data(params, first_job, last_job):
    """
    Calls the lasif preprocessing function.
    """
    lasif_scratch_path = params['lasif_scratch_path']
    iteration_name = params['iteration_name']
    lasif_project_path = params['lasif_path']
    
    _copy_relevant_lasif_files(params)
    
    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(
        this_script, 'sbatch_scripts', 'submit_lasif_preprocess_data.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(
        ['sbatch', '--array=%s-%s' % (first_job, last_job), sbatch_file, 
        lasif_scratch_path, lasif_project_path, iteration_name]).wait()

def smooth_kernels(params, horizontal_smoothing, vertical_smoothing):
    """
    Smoothes the summed kernels (requires that sum_kernels has already been
    run.
    """
    optimization_dir = os.path.join(params['forward_run_dir'], 'OPTIMIZATION')
    kernel_names = ['bulk_c_kernel', 'bulk_betah_kernel', 'bulk_betav_kernel',
                    'eta_kernel']
    kernel_dir = './PROCESSED_KERNELS'
    databases_mpi = '../mesh/DATABASES_MPI'

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'job_smooth_kernels.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    for kernel_name in kernel_names:
        subprocess.Popen(
            ['sbatch', sbatch_file, horizontal_smoothing, vertical_smoothing,
             kernel_name, kernel_dir, databases_mpi, optimization_dir]).wait()


def setup_new_iteration(params, old_iteration, new_iteration):
    """
    Sets up a new iteration, and links the mesh files from the old iteration to
    the new one.
    """
    forward_stage_dir = params['forward_stage_dir']
    event_list = params['event_list']
    params.update({'iteration_name': new_iteration})
    new_forward_run_dir = os.path.join(forward_stage_dir, new_iteration)
    params.update({'forward_run_dir': new_forward_run_dir})
    setup_solver(params)

    old_database_dir = os.path.join(forward_stage_dir, old_iteration, 'mesh',
                                    'DATABASES_MPI')
    new_database_dir = os.path.join(forward_stage_dir, new_iteration, 'mesh',
                                    'DATABASES_MPI')

    old_optimization_dir = os.path.join(forward_stage_dir, old_iteration,
                                        'OPTIMIZATION', 'PROCESSED_KERNELS')
    new_optimization_dir = os.path.join(forward_stage_dir, new_iteration,
                                        'OPTIMIZATION', 'PROCESSED_KERNELS')

    utils.print_ylw("Copying mesh information...")
    utils.copy_directory(old_database_dir, new_database_dir)
    utils.print_ylw("Copying kernels...")
    utils.copy_directory(old_optimization_dir, new_optimization_dir,
                         ends='smooth.bin')
    utils.print_ylw("Copying DATA files...")
    for event in event_list:
        old_data_dir = os.path.join(forward_stage_dir, old_iteration, event,
                                    'DATA')
        new_data_dir = os.path.join(forward_stage_dir, new_iteration, event,
                                    'DATA')
        utils.copy_directory(old_data_dir, new_data_dir,
                             only=['Par_file', 'CMTSOLUTION', 'STATIONS'])
    old_mesh_dir = os.path.join(forward_stage_dir, old_iteration, 'mesh',
                                'DATA')
    new_mesh_dir = os.path.join(forward_stage_dir, new_iteration, 'mesh',
                                'DATA')
    utils.copy_directory(old_mesh_dir, new_mesh_dir,
                         only=['Par_file', 'CMTSOLUTION', 'STATIONS'])
    utils.print_blu('Done.')


def add_smoothed_kernels(params, max_perturbation):
    """
    Adds the transversely isotropic kernels back to the original model, and
    puts the results in the ../mesh/DATA/GLL directory.
    """
    optimization_dir = os.path.join(params['forward_run_dir'], 'OPTIMIZATION')
    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'job_add_smoothed_kernels.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(['sbatch', sbatch_file, optimization_dir,
                      max_perturbation]).wait()


def clean_attenuation_dumps(params):
    """
    Goes through the simulation directories for an iteration, and cleans out
    the massive adios attenuation snapshot files.
    """
    forward_run_dir = params['forward_run_dir']
    event_list = params['event_list']

    for dir in sorted(os.listdir(forward_run_dir)):
        if dir in event_list:
            utils.print_ylw("Cleaning " + dir + "...")
            databases_mpi = os.path.join(forward_run_dir, dir, 'DATABASES_MPI')
            for file in os.listdir(databases_mpi):
                if file.startswith('save_frame_at'):
                    os.remove(os.path.join(databases_mpi, file))
    utils.print_blu('Done.')


def clean_event_kernels(params):
    """
    Goes through the simulation directories for an iteration, and cleans out
    the individual event kernels (make sure you've summed them already!).
    """
    forward_run_dir = params['forward_run_dir']
    event_list = params['event_list']

    for dir in sorted(os.listdir(forward_run_dir)):
        if dir in event_list:
            utils.print_ylw("Cleaning " + dir + "...")
            databases_mpi = os.path.join(forward_run_dir, dir, 'DATABASES_MPI')
            for file in os.listdir(databases_mpi):
                if file.endswith('kernel.bin'):
                    os.remove(os.path.join(databases_mpi, file))
    utils.print_blu('Done.')


def clean_failed(params):
    """
    Deletes error files after a failed run.
    """
    forward_run_dir = params['forward_run_dir']
    for dir in os.listdir(forward_run_dir):
        utils.print_ylw("Cleaning " + dir + "...")
        if os.path.exists(os.path.join(forward_run_dir, dir, 'OUTPUT_FILES')):
            output_files = os.path.join(forward_run_dir, dir, 'OUTPUT_FILES')
            for file in os.listdir(output_files):
                if 'error' in file:
                    os.remove(os.path.join(output_files, file))
        if os.path.exists(os.path.join(forward_run_dir, dir, 'core')):
            os.remove(os.path.join(forward_run_dir, dir, 'core'))
    utils.print_blu('Done.')


def generate_kernel_vtk(params, num_slices):
    """
    Generates .vtk files for the smoothed and summed kernels, and puts them
    in the OPTIMIZATION/VTK_FILES directory.
    """
    optimization_dir = os.path.join(params['forward_run_dir'], 'OPTIMIZATION')
    # Write slices file.
    with open(os.path.join(optimization_dir, 'VTK_FILES',
                           'SLICES_ALL.txt'), 'w') as f:
        for i in range(0, num_slices):
            f.write(str(i) + '\n')

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'job_generate_smoothed_kernels_vtk.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(['sbatch', sbatch_file, optimization_dir]).wait()


def delete_adjoint_sources_for_iteration(params):
    """
    Deletes the directories on both /scratch and /project which contain the
    adjoint sources for this iteration. As well, cleans the SEM and
    STATIONS_ADJOINT files for the solver.
    """
    forward_run_dir = params['forward_run_dir']
    lasif_scratch_dir_output = os.path.join(params['lasif_scratch_path'],
                                            'OUTPUT')
    lasif_dir_output = os.path.join(params['lasif_path'], 'OUTPUT')
    lasif_scratch_dir_adjoint = os.path.join(
        params['lasif_scratch_path'], 'ADJOINT_SOURCES_AND_WINDOWS')
    lasif_project_dir_adjoint = os.path.join(
        params['lasif_path'], 'ADJOINT_SOURCES_AND_WINDOWS')
    iteration_name = params['iteration_name']

    utils.print_ylw("Cleaning forward runs...")
    for dir in os.listdir(forward_run_dir):
        if not os.path.exists(os.path.join(forward_run_dir, dir, 'SEM')):
            continue
        for file in os.listdir(os.path.join(forward_run_dir, dir, 'SEM')):
            os.remove(os.path.join(forward_run_dir, dir, 'SEM', file))

    utils.print_ylw("Cleaning LASIF scratch...")
    for dir in os.listdir(lasif_scratch_dir_output):
        if iteration_name and 'adjoint_sources' in dir:
            shutil.rmtree(os.path.join(lasif_scratch_dir_output, dir))
    for dir in os.listdir(lasif_scratch_dir_adjoint):
        for sub_dir in os.listdir(os.path.join(lasif_scratch_dir_adjoint, dir)):
            shutil.rmtree(os.path.join(lasif_scratch_dir_adjoint, dir, sub_dir))            

    utils.print_ylw("Cleaning LASIF project...")
    for dir in os.listdir(lasif_dir_output):
        if iteration_name and 'adjoint_sources' in dir:
            shutil.rmtree(os.path.join(lasif_dir_output, dir))
    for dir in os.listdir(lasif_project_dir_adjoint):
        for sub_dir in os.listdir(os.path.join(lasif_project_dir_adjoint, dir)):
            shutil.rmtree(os.path.join(lasif_project_dir_adjoint, dir, sub_dir))            

    utils.print_blu('Done.')
    
def process_synthetics(params, first_job, last_job):
    """
    Processes the synthetic seismograms in parallel.
    """
        
    event_list = params['event_list']
    forward_run_dir = params['forward_run_dir']
    lasif_path = params['lasif_path']
    iteration_name = params['iteration_name']
    chosen_event = event_list[first_job:last_job+1]
    source_dirs = [os.path.join(forward_run_dir, event, 'OUTPUT_FILES') 
                   for event in chosen_event]
    dest_dirs = [os.path.join(
        lasif_path, 'SYNTHETICS', event, 'ITERATION_%s' % (iteration_name)) 
        for event in chosen_event]
    
    if __name__ == 'specfem_control.control':
        pool = Pool(processes=cpu_count())
        print "Using %d cpus..." % (cpu_count())
        pool.map(_copy_files_and_tar, zip(source_dirs, dest_dirs))
        
    utils.print_blu('Done.')
            
def download_data(params, station_list, with_waveforms, recording_time, 
                  padding_time):
    """
    Downloads data from IRIS.
    """        
    data.download_data(params, station_list, with_waveforms, recording_time, 
                       padding_time)    

def prefilter_data(params):
    """
    Rotates the downloaded data in the "DOWNLOADED" folder. Will fail if no
    StationXML files exist, so make sure you've done this first.
    """
    data.prefilter_data(params)

def plot_seismogram(params, file_name):
    """
    Plots a single seismogram.
    """
    seismogram = seismograms.Seismogram(file_name)
    seismogram.plot_seismogram()
    
def plot_two_seismograms(params, file_1, file_2):
    """
    Plots two sesimograms on top of each other.
    """
    s1 = seismograms.Seismogram(file_1)
    s2 = seismograms.Seismogram(file_2)
    seismograms.plot_two(s1, s2)


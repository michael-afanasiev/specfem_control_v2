#!/usr/bin/env python

import os
import utils
import subprocess

def _setup_dir_tree(event, forward_run_dir):
    """
    Sets up the simulation directory structure for one event.
    """
    event_path = os.path.join(forward_run_dir, event)
    utils.mkdir_p(event_path)
    utils.mkdir_p(event_path + '/bin')
    utils.mkdir_p(event_path + '/DATA')
    utils.mkdir_p(event_path + '/DATA/GLL')
    utils.mkdir_p(event_path + '/DATA/cemRequest')
    utils.mkdir_p(event_path + '/OUTPUT_FILES')
    utils.mkdir_p(event_path + '/DATABASES_MPI')
 
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
        if iteration_name in dir and event in dir:
            source = os.path.join(lasif_output, dir)
            utils.copy_directory(source, event_data, exc=['Par_file', 'STF'])
            # This flag also copies the first event's data to the solver base
            # directory for compilation, and the mesh directory for meshing.
            if mesh:
                utils.copy_directory(source, mesh_data, exc=['Par_file', 'STF'])
                utils.copy_directory(source, compile_data, 
                    exc=['Par_file, STF'])

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
    
    # Set up the mesh directory.
    _setup_dir_tree('mesh', forward_run_dir)
    
    # Set up the optimization directory.
    optimization_base = os.path.join(forward_run_dir, 'OPTIMIZATION')
    utils.mkdir_p(optimization_base)
    utils.mkdir_p(os.path.join(optimization_base, 'bin'))
    utils.mkdir_p(os.path.join(optimization_base, 'PROCESSED_KERNELS'))
    utils.mkdir_p(os.path.join(optimization_base, 'GRADIENT_INFO'))
    utils.mkdir_p(os.path.join(optimization_base, 'LOGS'))
            
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
    par_file = os.path.join(lasif_path, 'SUBMISSION', iteration_name, 
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
    utils.copy_directory(bin_directory, opt_bin_directory)
    for event in os.listdir(forward_run_dir):
        event_bin = os.path.join(forward_run_dir, event, 'bin')
        event_dat = os.path.join(forward_run_dir, event, 'DATA')
        compile_par = os.path.join(specfem_root, 'DATA', 'Par_file')
        utils.safe_copy(compile_par, event_dat)
        utils.copy_directory(bin_directory, event_bin, 
                             only=['xspecfem3D', 'xmeshfem3D'])
        
    # Copy jobarray script to base directory.
    utils.print_ylw('Copying jobarray sbatch script...')
    source = os.path.join(lasif_path, 'SUBMISSION', iteration_name,
                          'jobArray_solver_daint.sbatch')
    utils.safe_copy(source, forward_stage_dir)
    utils.mkdir_p(os.path.join(forward_stage_dir, 'logs'))
    
    # Copy mesh submission script.
    source = os.path.join(lasif_path, 'SUBMISSION', iteration_name, 
                          'job_mesher_daint.sbatch')
    dest = os.path.join(forward_run_dir, 'mesh')
    utils.safe_copy(source,dest)
    
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
    forward_stage_dir = params['forward_stage_dir']
    iteration_name = params['iteration_name']
    job_array = 'jobArray_solver_daint.sbatch'
    os.chdir(forward_stage_dir)
    subprocess.Popen(['sbatch', '--array=%s-%s' % (first_job, last_job),
                      job_array, iteration_name]).wait()
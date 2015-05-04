#!/usr/bin/env python

import os
import utils
import shutil
import subprocess
import tarfile
import random

import data
import seismograms

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from itertools import repeat


def _copy_files_and_tar(xxx_todo_changeme):

    (source, dest, tar) = xxx_todo_changeme
    utils.print_ylw("Moving: " + source.split('/')[-2])

    tar_name = os.path.join(dest, 'data.tar')

    if os.path.exists(tar_name):
        os.remove(tar_name)

    if not os.path.exists(dest):
        utils.mkdir_p(dest)

    source_files = [x for x in os.listdir(source) if x.endswith('.sac')]

    if tar:
        with tarfile.open(tar_name, "w") as tar:
            for file in source_files:
                tar.add(os.path.join(source, file), arcname=file)
    else:
        utils.move_directory(source, dest, ends='.sac')


def _tar_seismograms(dir):

    utils.print_ylw("Tarring: " + dir + "...")

    tar_files = []
    tar_name = os.path.join(dir, "data.tar")
    for file in os.listdir(dir):
        if file != "data.tar":
            tar_files.append(os.path.join(dir, file))

    if not tar_files:
        return

    with tarfile.open(tar_name, "w") as tar:
        for file in tar_files:
            tar.add(file, arcname=os.path.basename(file))

    for file in tar_files:
        os.remove(file)


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
        undo_attenuation = '= .true.\n'
    elif run_type == 'first_iteration' or run_type == 'update_mesh' \
            or run_type == 'forward_run':
        simulation_type = '= 1'
        save_forward = '= .true.\n'
        sbatch_time = '#SBATCH --time=02:00:00\n'
        undo_attenuation = '= .true.\n'
    elif run_type == 'line_search':
        simulation_type = '= 1'
        save_forward = '= .false.\n'
        sbatch_time = '#SBATCH --time=02:00:00\n'
        undo_attenuation = '= .false.\n'

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
                if len(fields) == 1:
                    new_file.write(line)
                    continue
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
                elif fields[0].startswith('UNDO_ATTENUATION'):
                    new_file.write(fields[0] + undo_attenuation)
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
        ["rsync -av --include='*/' --exclude='*' " + lasif_path + "/ " +
         lasif_scratch_path], shell=True).wait()

    folders = ['CACHE', 'STATIONS', 'EVENTS', 'FUNCTIONS', 'ITERATIONS',
               'OUTPUT', 'config.xml', 'LOGS', 'MODELS']

    for folder in folders:
        subprocess.Popen(['rsync', '-av', os.path.join(lasif_path, folder),
                          lasif_scratch_path]).wait()


def _copy_synthetics_for_iteration(params):

    lasif_path = params['lasif_path']
    lasif_scratch_path = params['lasif_scratch_path']

    iteration_synthetics = os.path.join(lasif_path, 'SYNTHETICS')
    scratch_synthetics = os.path.join(lasif_scratch_path, 'SYNTHETICS')
    utils.mkdir_p(lasif_scratch_path)
    rsync_string = "rsync -rav --include=" + iteration_synthetics + \
        "/*/*01_globe_a --exclude=* " + iteration_synthetics + " " + \
        scratch_synthetics
    subprocess.Popen(
        [rsync_string], shell=True).wait()


def get_quadratic_steplength(p1, m1, p2, m2, p3, m3):
    """
    Finds the minimum of a parabola that passes through the three points
    (pn, mn).
    """
    # Define coefficient matrix.
    A = np.matrix([[p1**2, p1, 1],
                   [p2**2, p2, 1],
                   [p3**2, p3, 1]])

    # Define right hand side (y-values).
    rhs = np.matrix([[m1],
                     [m2],
                     [m3]])

    # Solve linear system for coefficients.
    a, b, c = np.linalg.solve(A, rhs).flat

    # Minimum where dy/dx = 0.
    minimum = -b / (2 * a)
    min_fit = a * minimum**2 + b * minimum + c
    print "Minimum in the quadratic approximation: " + str(minimum)
    print "Estimated misfit value at minimum: " + str(min_fit)

    # Plot
    minplot = minimum - 0.1
    maxplot = minimum + 0.1
    grid = np.linspace(minplot, maxplot, 10000)
    fx = [a * x**2 + b * x + c for x in grid]
    plt.plot(grid, fx)
    plt.title("Minimum in the quadratic approximation.")
    plt.xlabel("Step length")
    plt.ylabel("Misfit")
    plt.xlim(minplot, maxplot)
    plt.show()


def calculate_cumulative_misfit(params):
    """
    Goes through an iteration and calculates the cumulative misfit.
    """
    iteration_name = params['iteration_name']
    lasif_scratch_path = params['lasif_scratch_path']
    print "Calculating misfit for iteration %s." % (iteration_name)
    _copy_synthetics_for_iteration(params)

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'submit_misfit_calculation.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(
        ['sbatch', sbatch_file, lasif_scratch_path, iteration_name]).wait()


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
    utils.print_ylw("Cleaning old mesh files...")
    for file in os.listdir("DATABASES_MPI"):
        os.remove(os.path.join("DATABASES_MPI", file))

    subprocess.Popen(['sbatch', 'job_mesher_daint.sbatch']).wait()


def _unpack_if_needed(possible_tar_file):

    if os.path.exists(possible_tar_file):
        utils.print_ylw("Unpacking %s..." % (possible_tar_file))
        tar = tarfile.open(possible_tar_file)
        tar.extractall(path=os.path.dirname(possible_tar_file))
        os.remove(possible_tar_file)


def plot_random_seismograms(params, num, two_iterations):
    """
    Plots num randomly selected seismograms.
    """
    lasif_scratch_path = params['lasif_scratch_path']
    lasif_windows_path = os.path.join(
        lasif_scratch_path, 'ADJOINT_SOURCES_AND_WINDOWS', 'WINDOWS')
    all_events = sorted(os.listdir(lasif_windows_path))
    num_events = len(all_events)

    # Choose random events
    chosen_events = []
    for i in range(num):
        chosen_events.append(all_events[random.randint(0, num_events)])

    # # Choose one trace per event.
    # chosen_traces = []
    # for e in chosen_events:
    #    window_path = os.path.join(
    #        lasif_windows_path, e, 'ITERATION_' + iteration_name)
    #    all_traces = os.listdir(window_path)
    #    chosen_traces.append(
    #        all_traces[random.randint(0,len(all_traces)-1)].split('_')[1][:-4])

    chosen_events = ['GCMT_event_BERING_SEA_Mag_6.5_2010-4-30-23']
    chosen_traces = ['II.ESK.MXZ']
    # Find all the synthetics you need.
    chosen_synthetics = []
    for x, e in zip(chosen_traces, chosen_events):
        for itr in ['00_globe', '01_globe_b']:

            # Check to see if data needs to be untarred.
            possible_tar_file = os.path.join(
                lasif_scratch_path, 'SYNTHETICS', e, 'ITERATION_' + itr,
                'data.tar')
            # _unpack_if_needed(possible_tar_file)

            # Add files to list.
            chosen_synthetics.append(
                os.path.join(
                    os.path.dirname(possible_tar_file),
                    '.'.join(x.split('.')[0:2]) + '.MX%s.sem.sac' % (x[-1])))

    # Find all the data you need.
    chosen_data = []
    for x, e in zip(chosen_traces, chosen_events):

        # Check to see if data needs to be untarred.
        possible_tar_file = os.path.join(
            lasif_scratch_path, 'DATA', e,
            'preprocessed_hp_0.00833_lp_0.01667_npts_38850_dt_0.142500',
            'data.tar')
        _unpack_if_needed(possible_tar_file)

        possible_file = os.path.join(
            lasif_scratch_path, 'DATA', e,
            'preprocessed_hp_0.00833_lp_0.01667_npts_38850_dt_0.142500',
            '.'.join(x.split('.')[0:2]) + '.00.BH%s.mseed' % (x[-1]))
        if not os.path.exists(possible_file):
            possible_file = os.path.join(
                lasif_scratch_path, 'DATA', e,
                'preprocessed_hp_0.00833_lp_0.01667_npts_38850_dt_0.142500',
                '.'.join(x.split('.')[0:2]) + '..BH%s.mseed' % (x[-1]))

        chosen_data.append(possible_file)

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row',
                            figsize=(ncols * 8, nrows * 2.5))
    try:
        axs_flat = axs.flat
    except:
        axs_flat = [axs]

    print chosen_synthetics
    for x, y, z, ax in zip(
            chosen_data, chosen_synthetics[0::2],
            chosen_synthetics[1::2], axs_flat):
        plot_two_seismograms(
            params, x, y, process_s1=False, process_s2=True, plot=False,
            ax=ax, legend=False, third=z)

    # for x, y, ax in zip(chosen_data, chosen_synthetics, axs_flat):
    #     plot_two_seismograms(
    #         params, x, y, process_s1=False, process_s2=True,
    #         plot=False, ax=ax, legend=False)

    from string import ascii_lowercase
    for i, a in enumerate(axs_flat):
        a.text(0.965, 0.9, ascii_lowercase[i] + ')', transform=a.transAxes)

    if len(axs_flat) > 1:
        for i, row in enumerate(axs):
            for j, cell in enumerate(row):
                if i == len(axs) - 1:
                    cell.set_xlabel('Time (minutes)')
                if j == 0:
                    cell.set_ylabel('Amplitude (normalized)')
    else:
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Amplitude (normalized)')

    if not two_iterations:
        fig.suptitle("Data (black) and synthetics (red)")
    else:
        fig.suptitle("Data (black) and synthetics (red/blue)")
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig('seismo.pdf', bbox_inches='tight')
    plt.show()

    for x, y in zip(chosen_synthetics, chosen_data):
        print x, y


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
    for dir in sorted(os.listdir(lasif_output_dir))[::-1]:

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
        kerns = []
        for x in os.listdir(databases_mpi):
            if 'hess' in x:
                kerns.append(x)
        if len(kerns) == 24:
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
    lasif_project_path = params['lasif_path']
    lowpass_f = params['lowpass_f']
    highpass_f = params['highpass_f']
    event_list = params['event_list']
    dt = params['dt']
    npts = params['npts']
    communicator = params['lasif_communicator']

    _copy_relevant_lasif_files(params)

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(
        this_script, 'sbatch_scripts', 'submit_lasif_preprocess_data.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)

    for event in event_list[first_job:last_job + 1]:
        starttime = communicator.comm.events.get(event)['origin_time']
        subprocess.Popen(
            ['sbatch', sbatch_file, lasif_scratch_path, event, str(highpass_f),
             str(lowpass_f), str(starttime), str(dt), str(npts),
             lasif_project_path]).wait()


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
                if 'sac' in file:
                    os.remove(os.path.join(output_files, file))
        if os.path.exists(os.path.join(forward_run_dir, dir, 'core')):
            os.remove(os.path.join(forward_run_dir, dir, 'core'))
    utils.print_blu('Done.')


def pack_up_all_seismograms(params):
    """
    Goes through the /project directory and packs up any loose seismograms.
    """
    lasif_path = params['lasif_path']
    lasif_scratch_path = params['lasif_scratch_path']

    data_path_1 = os.path.join(lasif_path, 'DATA')
    synthetic_path_1 = os.path.join(lasif_path, 'SYNTHETICS')

    for data_path in [data_path_1]:  # , data_path_2]:

        # Data.
        tar_dirs = []
        for dir in sorted(os.listdir(data_path)):
            for sub_dir in os.listdir(os.path.join(data_path, dir)):
                sub_dir_path = os.path.join(data_path, dir, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    continue
                tar_dirs.append(sub_dir_path)

        if __name__ == "specfem_control.control":
            pool = Pool(processes=cpu_count())
            print "Using %d cpus..." % (cpu_count())
            pool.map(_tar_seismograms, tar_dirs)

    for synthetic_path in [synthetic_path_1]:  # , synthetic_path_2]:

        # Synthetics.
        tar_dirs = []
        for dir in sorted(os.listdir(synthetic_path)):
            for sub_dir in os.listdir(os.path.join(synthetic_path, dir)):
                sub_dir_path = os.path.join(synthetic_path, dir, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    continue
                tar_dirs.append(sub_dir_path)

        if __name__ == "specfem_control.control":
            pool = Pool(processes=cpu_count())
            print "Using %d cpus..." % (cpu_count())
            pool.map(_tar_seismograms, tar_dirs)


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


def generate_model_vtk(params, num_slices):
    """
    Generates .vtk files for the model, and puts them
    in the mesh/VTK_FILES directory.
    """
    mesh_dir = os.path.join(params['forward_run_dir'], 'mesh')

    # Make VTK folder.
    utils.mkdir_p(os.path.join(mesh_dir, 'VTK_FILES'))

    # Write slices file.
    with open(os.path.join(mesh_dir, 'VTK_FILES',
                           'SLICES_ALL.txt'), 'w') as f:
        for i in range(0, num_slices):
            f.write(str(i) + '\n')

    # Get path of this script and .sbatch file.
    this_script = os.path.dirname(os.path.realpath(__file__))
    sbatch_file = os.path.join(this_script, 'sbatch_scripts',
                               'job_generate_model_vtk.sbatch')

    # Change to the directory above this script, and submit the job.
    os.chdir(this_script)
    subprocess.Popen(['sbatch', sbatch_file, mesh_dir]).wait()


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
        for sub_dir in os.listdir(
            os.path.join(
                lasif_scratch_dir_adjoint,
                dir)):
            shutil.rmtree(
                os.path.join(
                    lasif_scratch_dir_adjoint,
                    dir,
                    sub_dir))

    utils.print_ylw("Cleaning LASIF project...")
    for dir in os.listdir(lasif_dir_output):
        if iteration_name and 'adjoint_sources' in dir:
            shutil.rmtree(os.path.join(lasif_dir_output, dir))
    for dir in os.listdir(lasif_project_dir_adjoint):
        for sub_dir in os.listdir(
            os.path.join(
                lasif_project_dir_adjoint,
                dir)):
            shutil.rmtree(
                os.path.join(
                    lasif_project_dir_adjoint,
                    dir,
                    sub_dir))

    utils.print_blu('Done.')


def process_synthetics(params, first_job, last_job):
    """
    Processes the synthetic seismograms in parallel.
    """

    event_list = params['event_list']
    forward_run_dir = params['forward_run_dir']
    lasif_path = params['lasif_path']
    lasif_scratch_path = params['lasif_scratch_path']
    iteration_name = params['iteration_name']
    chosen_event = event_list[first_job:last_job + 1]
    source_dirs = [os.path.join(forward_run_dir, event, 'OUTPUT_FILES')
                   for event in chosen_event]
    dest_dirs = [os.path.join(
        lasif_path, 'SYNTHETICS', event, 'ITERATION_%s' % (iteration_name))
        for event in chosen_event]
    dest_dirs_scratch = [
        os.path.join(
            lasif_scratch_path,
            'SYNTHETICS',
            event,
            'ITERATION_%s' %
            (iteration_name)) for event in chosen_event]

    if __name__ == 'specfem_control.control':
        pool = Pool(processes=cpu_count())
        print "Using %d cpus..." % (cpu_count())
        #pool.map(_copy_files_and_tar, zip(source_dirs, dest_dirs, repeat(True)))
        pool.map(
            _copy_files_and_tar, zip(
                source_dirs, dest_dirs_scratch, repeat(False)))

    utils.print_blu('Done.')


def download_data(params, station_list, with_waveforms, recording_time,
                  padding_time):
    """
    Downloads data from IRIS.
    """
    data.download_data(params, station_list, with_waveforms, recording_time,
                       padding_time)


def station_statistics(params, station_list):
    """
    Looks through the station_list file and makes some statistics on the data.
    Helpful for deciding downloading parameters.
    """
    data.station_statistics(params, station_list)


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


def plot_two_seismograms(
        params, file_1, file_2, process_s1=False, process_s2=True, ax=None,
        plot=True, legend=True, third=None):
    """
    Plots two sesimograms on top of each other.
    """
    s1 = seismograms.Seismogram(file_1)
    s2 = seismograms.Seismogram(file_2)
    s3 = None
    if third:
        s3 = seismograms.Seismogram(third)
    seismograms.plot_two(s1, s2, process_s1=process_s1, process_s2=process_s2,
                         ax=ax, plot=plot, legend=legend, third=s3)

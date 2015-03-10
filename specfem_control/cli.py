#!/usr/bin/env python

import os
import sys
import utils
import argparse
import xml.etree.ElementTree as ET

import control

FCT_PREFIX = "office_"


class FileNotFoundError(Exception):
    pass


class ParameterError(Exception):
    pass


def command_group(group_name):
    """
    Decorator to be able to logically group commands.
    """
    def wrapper(func):
        func.group_name = group_name
        return func
    return wrapper


@command_group("Setup")
def office_setup_solver(parser, args, params):
    """
    Sets up the solver directories on the scratch filesystem. Compiles and
    copies all necessary parameter files and binaries.
    """
    parser.parse_known_args(args)
    control.setup_solver(params)


@command_group("Setup")
def office_prepare_solver(parser, args, params):
    """
    Sets up symbolic links to the files generated by the mesher.
    """
    parser.parse_known_args(args)
    control.prepare_solver(params)


@command_group("Submit")
def office_submit_mesher(parser, args, params):
    """
    Submits the mesher to the batch filesystem in the /mesh directory.
    """
    parser.add_argument('--run_type', type=str,
                        help='Specify either first_iteration or update_mesh',
                        metavar='', required=True)
                        
    local_args = parser.parse_known_args(args)
    run_type = local_args[0].run_type
    
    if run_type != 'first_iteration' and run_type != 'update_mesh':
        raise ParameterError("Must specifiy either first_iteration or "
                             "update_mesh")
                             
    control.submit_mesher(params, run_type)
    

@command_group("Submit")
def office_submit_solver(parser, args, params):
    """
    Submits solver jobs -fj to -lj.
    """
    parser.add_argument('-fj', type=int, help='First event to submit.',
                        metavar='', required=True)
    parser.add_argument('-lj', type=int, help='Last event to submit.',
                        metavar='', required=True)
    parser.add_argument('--run_type', type=str,
                        help='Specify either adjoint_run or forward_run.',
                        metavar='', required=True)

    local_args = parser.parse_known_args(args)
    first_job = local_args[0].fj
    last_job = local_args[0].lj
    run_type = local_args[0].run_type

    if run_type != 'adjoint_run' and run_type != 'forward_run':
        raise ParameterError("Must specifiy either forward_run or "
                             "adjoint_run")

    control.submit_solver(params, first_job, last_job, run_type)


@command_group("Submit")
def office_submit_window_selection(parser, args, params):
    """
    Submits the .sbatch script to select the windows and create the adjoint
    sources.
    """
    parser.add_argument('-fj', type=int, help='First event to submit.',
                        metavar='', required=True)
    parser.add_argument('-lj', type=int, help='Last event to submit.',
                        metavar='', required=True)

    local_args = parser.parse_known_args(args)
    first_job = local_args[0].fj
    last_job = local_args[0].lj

    control.submit_window_selection(params, first_job, last_job)


@command_group("Setup")
def office_distribute_adjoint_sources(parser, args, params):
    """
    Searches through the OUTPUT directory of the lasif project, and distributes
    the adjoint sources to their appropriate simulated directories. Also writes
    the STATIONS_ADJOINT file.
    """
    parser.parse_known_args(args)
    control.distribute_adjoint_sources(params)


@command_group("Submit")
def office_sum_kernels(parser, args, params):
    """
    Goes through the output solver directory, and runs the summing and
    smoothing commands on the kernels that were output. Kernels will
    end up in the optimization/processed kernels directory.
    """
    parser.add_argument('-fj', type=int, help='First event to submit.',
                        metavar='', required=True)
    parser.add_argument('-lj', type=int, help='Last event to submit.',
                        metavar='', required=True)

    local_args = parser.parse_known_args(args)
    first_job = local_args[0].fj
    last_job = local_args[0].lj

    control.sum_kernels(params, first_job, last_job)


@command_group("Submit")
def office_smooth_kernels(parser, args, params):
    """
    Smoothes the summed kernels (requires that sum_kernels has already been
    run.
    """
    parser.add_argument('-h_length', type=str,
                        help='Variance of Gaussian [horizontal] (default=25)',
                        metavar='', default='25')
    parser.add_argument('-v_length', type=str,
                        help='Variance of Gaussian [vertical] (default=5)',
                        metavar='', default='5')
    local_args = parser.parse_known_args(args)
    h_length = local_args[0].h_length
    v_length = local_args[0].v_length

    control.smooth_kernels(params, h_length, v_length)
    
    
@command_group("Submit")
def office_add_smoothed_kernels(parser, args, params):
    """
    Adds the transversely isotropic kernels back to the original model, and 
    puts the results in the ../mesh/DATA/GLL directory.
    """
    parser.add_argument('--max_perturbation', type=str,
                         help='Amount by which to scale the velocity updates',
                         metavar='', required=True)
    local_args = parser.parse_known_args(args)
    max_perturbation = local_args[0].max_perturbation
    
    control.add_smoothed_kernels(params, max_perturbation)
    
@command_group("Setup")
def office_clean_attenuation_dumps(parser, args, params):
    """
    Goes through the simulation directories for an iteration, and cleans out
    the massive adios attenuation snapshot files.
    """
    parser.parse_known_args(args)
    control.clean_attenuation_dumps(params)
    
@command_group("Setup")
def office_clean_event_kernels(parser, args, params):
    """
    Goes through the simulation directories for an iteration, and cleans out
    the individual event kernels (make sure you've summed them already!).
    """
    parser.parse_known_args(args)
    control.clean_event_kernels(params)
    
@command_group("Setup")
def office_setup_new_iteration(parser, args, params):
    """
    Sets up a new iteration, and links the mesh files from the old iteration to
    the new one.
    """
    parser.add_argument('--old_iteration', type=str,
                        help='Name of old iteration',
                        metavar='', required=True)
    parser.add_argument('--new_iteration', type=str,
                        help='Name of new iteration',
                        metavar='', required=True)
    local_args = parser.parse_known_args(args)
    old_iteration = local_args[0].old_iteration
    new_iteration = local_args[0].new_iteration
    
    control.setup_new_iteration(params, old_iteration, new_iteration)
    
@command_group("Setup")
def office_clean_failed(parser, args, params):
    """
    Deletes error files after a failed run.
    """
    parser.parse_known_args(args)
    control.clean_failed(params)

@command_group("Visualize")
def office_generate_kernel_vtk(parser, args, params):
    """
    Generates .vtk files for the smoothed and summed kernels, and puts them
    in the OPTIMIZATION/VTK_FILES directory.
    """
    parser.add_argument('--num_slices', type=int,
                        help='Number of slices (processors)',
                        metavar='', required=True)
    local_args = parser.parse_known_args(args)
    num_slices = local_args[0].num_slices
    
    control.generate_kernel_vtk(params, num_slices)
    
@command_group("Setup")
def office_delete_adjoint_sources_for_iteration(parser, args, params):
    """
    Deletes the directories on both /scratch and /project which contain the 
    adjoint sources for this iteration. As well, cleans the SEM and
    STATIONS_ADJOINT files for the solver.
    """
    parser.parse_known_args(args)
    control.delete_adjoint_sources_for_iteration(params)
    
@command_group("Visualize")
def office_plot_seismogram(parser, args, params):
    """
    Plots a single seismogram.
    """
    parser.add_argument('--file_name', type=str,
                        help='File name',
                        metavar='', required=True)
    local_args = parser.parse_known_args(args)
    file_name = local_args[0].file_name
    
    control.plot_seismogram(params, file_name)

def _read_parameter_file():
    """
    Reads the parameter file and populates the parameter dictionary.
    """
    if not os.path.exists("./config.txt"):
        raise FileNotFoundError("Can't find the configuration file "
                                "./config.txt")

    required = ['compiler_suite', 'project_name', 'scratch_path',
                'specfem_root', 'lasif_path', 'iteration_name']

    # Read parameters into dictionary.
    parameters = {}
    file = open("./config.txt", "r")
    for line in file:
        if line.startswith("#"):
            continue
        fields = line.split()
        parameters.update({fields[0]: fields[1]})

    # Ensure all parameters are there.
    for param in required:
        if param not in parameters.keys():
            raise ParameterError("Parameter " + param + " not in parameter "
                                 "file")

    # Build full paths.
    parameters['scratch_path'] = os.path.abspath(parameters['scratch_path'])
    parameters['specfem_root'] = os.path.abspath(parameters['specfem_root'])
    parameters['lasif_path'] = os.path.abspath(parameters['lasif_path'])

    # Derived parameters.
    forward_stage_dir = os.path.join(
        parameters['scratch_path'],
        parameters['project_name'])
    forward_run_dir = os.path.join(
        forward_stage_dir,
        parameters['iteration_name'])

    # Get list of all event names.
    iteration_xml_path = os.path.join(
        parameters['lasif_path'],
        'ITERATIONS',
        'ITERATION_%s.xml' %
        (parameters['iteration_name']))
    tree = ET.parse(iteration_xml_path)
    root = tree.getroot()
    event_list = []
    for name in root.findall('event'):
        for event in name.findall('event_name'):
            event_list.append(event.text)
    lasif_scratch_path = os.path.join(parameters['scratch_path'],
        os.path.basename(parameters['lasif_path']))

    parameters.update({'forward_stage_dir': forward_stage_dir})
    parameters.update({'forward_run_dir': forward_run_dir})
    parameters.update({'iteration_xml_path': iteration_xml_path})
    parameters.update({'event_list': sorted(event_list)})
    parameters.update({'lasif_scratch_path': lasif_scratch_path})

    return parameters


def _get_cmd_description(fct):
    """
    Exctracts the first line of a function docstring.
    """
    try:
        return fct.__doc__.strip('\n')
    except:
        return ""


def _get_argument_parser(fct):
    """
    Helper function to create a proper argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="oval_office %s" % fct.__name__.replace("office_", ""),
        description=_get_cmd_description(fct))
    return parser


def _get_functions():
    """
    Gets the names of all the functions defined in this file.
    """

    # Get all functions that start with _office.
    fcts = {fct_name[len(FCT_PREFIX):]: fct for (fct_name, fct) in
            globals().iteritems() if fct_name.startswith(FCT_PREFIX) and
            hasattr(fct, "__call__")}

    return fcts


def _print_help(fcts):
    """
    Prints the master help.
    """
    utils.print_ylw("\n\nWelcome to the oval office. There's no clever acronym "
        "here, I was just reading the Hunt for the Red October while writing "
        "this.\n\n")
    fct_groups = {}
    for fct_name, fct in fcts.iteritems():
        group_name = fct.group_name if hasattr(fct, "group_name") else "Misc"
        fct_groups.setdefault(group_name, {})
        fct_groups[group_name][fct_name] = fct
        
    for group_name in sorted(fct_groups.iterkeys()):
        utils.print_red(("{0:=>25s} Functions".format(" " + group_name)))
        current_functions = fct_groups[group_name]
        for name in sorted(current_functions.keys()):
            utils.print_cyn(name)
            utils.print_gry(_get_cmd_description(fcts[name]))

def main():
    """
    Main driver program for oval office. Does all the delegations.
    """

    # Read parameter file.
    params = _read_parameter_file()

    # Get function names.
    fcts = _get_functions()

    # Get command line arguments.
    args = sys.argv[1:]

    # Print help.
    if not args or args == ["help"] or args == ["--help"]:
        _print_help(fcts)
        sys.exit()

    # Use lowercase to increase tolerance.
    fct_name = args[0].lower()

    # Get argument parser.
    parser = _get_argument_parser(fcts[fct_name])

    fcts[fct_name](parser, args, params)

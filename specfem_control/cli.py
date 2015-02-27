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


def office_submit_mesher(parser, args, params):
    """
    Submits the mesher to the batch filesystem in the /mesh directory.
    """
    pass


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

def office_distribute_adjoint_sources(parser, args, params):
    """
    Searches through the OUTPUT directory of the lasif project, and distributes
    the adjoint sources to their appropriate simulated directories. Also writes
    the STATIONS_ADJOINT file.
    """
    parser.parse_known_args(args)
    control.distribute_adjoint_sources(params)
    
def office_sum_kernels(parser, args, params):
    """
    Goes through the output solver directory, and runs the summing and smoothing
    commands on the kernels that were output. Kernels will end up in the 
    optimization/processed kernels directory.
    """
    parser.add_argument('-fj', type=int, help='First event to submit.',
                        metavar='', required=True)
    parser.add_argument('-lj', type=int, help='Last event to submit.',
                        metavar='', required=True)
                        
    local_args = parser.parse_known_args(args)
    first_job = local_args[0].fj
    last_job = local_args[0].lj
    
    control.sum_kernels(params, first_job, last_job)
    
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

    parameters.update({'forward_stage_dir': forward_stage_dir})
    parameters.update({'forward_run_dir': forward_run_dir})
    parameters.update({'iteration_xml_path': iteration_xml_path})
    parameters.update({'event_list': sorted(event_list)})

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
        print "\nList of possible commands:"
        for fct_name in fcts:
            utils.print_blu(fct_name)
            utils.print_ylw(_get_cmd_description(fcts[fct_name]))
        sys.exit()

    # Use lowercase to increase tolerance.
    fct_name = args[0].lower()

    # Get argument parser.
    parser = _get_argument_parser(fcts[fct_name])

    fcts[fct_name](parser, args, params)
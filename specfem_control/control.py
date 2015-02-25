#!/usr/bin/env python

import utils

def _setup_dir_tree(event_path):
    """
    Sets up the simulation directory structure for one event.
    """
    utils.mkdir_p(event_path)
    utils.mkdir_p(event_path + '/bin')
    utils.mkdir_p(event_path + '/DATA')
    utils.mkdir_p(event_Path + '/DATA/GLL')
    utils.mkdir_p(event_path + '/DATA/cemRequest')
    utils.mkdir_p(event_path + '/OUTPUT_FILES')
    utils.mkdir_p(event_path + '/DATABASES_MPI')
    

def setup_solver(params):
    
    
    
    
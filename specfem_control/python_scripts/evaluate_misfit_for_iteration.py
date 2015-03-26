#!/usr/bin/env python

import os
import sys

from itertools import repeat
from multiprocessing import Pool, cpu_count
from lasif.components import project

def _evaluate_misfits_run((event, iteration)):

    # Get thread specific comm.
    proj = project.Project('./')
    
    # Copied this from Andreas/Saule's scripts.
    misfit = 0
    for trace in proj.comm.windows.get(event, iteration).list():
        for window in range(
            len(proj.comm.windows.get(event, iteration).get(trace).windows)):
            try:
                misfit += proj.comm.windows.get(
                    event, iteration).get(trace).windows[window].misfit_value
            except:
                pass
    
    return misfit

# Make sure iteration name comes in.
if len(sys.argv) < 2:
    sys.exit("No iteration name given.")
iteration = str(sys.argv[1])

# Define iteration name and initialize master communicator.
master = project.Project('./')

# Get event list.
event_list = sorted(master.comm.iterations.get(iteration).events.keys())

# Farm out misfit evaluation to all cores.
print "Running in parallel on %d cores." % (cpu_count())
pool = Pool(processes=cpu_count())
all_misfits = pool.map(
    _evaluate_misfits_run, zip(event_list, repeat(iteration)))

# Sum total misfits.
total_misfit = sum(all_misfits)
print "Total misfit for iteration %s: %f" % (iteration, total_misfit)

# Write file.
fname = os.path.join("./", "ITERATIONS", "MISFIT_%s.txt" % (iteration))
with open(fname, "w") as file:
    file.write("Total misfit for iteration %s: %f" % (iteration, total_misfit))

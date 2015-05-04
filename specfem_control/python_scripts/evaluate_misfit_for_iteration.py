#!/usr/bin/env python

import os
import sys

from itertools import repeat
from multiprocessing import Pool, cpu_count
from lasif.components import project

def _evaluate_misfits_run((event, iteration)):

    print "Calculating for event: " + event
    
    # Get thread specific comm.
    proj = project.Project('./', read_only_caches=True)
    
    # Copied this from Andreas/Saule's scripts.
    misfit = 0
    failed = 0
    passed = 0
    failed_traces = []
    for trace in proj.comm.windows.get(event, iteration).list():
        for window in range(
            len(proj.comm.windows.get(event, iteration).get(trace).windows)):
            skip = False
            for i in ['01_globe', '01_globe_a', '01_globe_b']:
                try:
                    proj.comm.windows.get(event, i).get(trace).windows[window].misfit_value
                except:
                    skip = True
                    break
            if skip:
                continue
            try:
                misfit = proj.comm.windows.get(
                    event, iteration).get(trace).windows[window].misfit_value
                print misfit
                passed += 1
            except:
                print proj.comm.windows.get(
                    event, iteration).get(trace).windows[window].misfit_value
                sys.exit("DID NOT WORK")
    
    return (misfit, passed)

# Make sure iteration name comes in.
if len(sys.argv) < 2:
    sys.exit("No iteration name given.")
iteration = str(sys.argv[1])

# Define iteration name and initialize master communicator.
master = project.Project('./', read_only_caches=True)

# Get event list.
event_list = sorted(master.comm.iterations.get(iteration).events.keys())

print master.comm.iterations.get(iteration).get_process_params()
print "HEP"
sys.exit()

# Farm out misfit evaluation to all cores.
print "Running in parallel on %d cores." % (cpu_count())
pool = Pool(processes=1)#cpu_count())
misfits_and_windows = pool.map(
    _evaluate_misfits_run, zip(event_list, repeat(iteration)))

# Unpack from tuple.
misfits = [x[0] for x in misfits_and_windows]
passed = [x[1] for x in misfits_and_windows]

# Sum total misfits.
total_misfit = sum(misfits)
print "Total misfit for iteration %s: %f" % (iteration, total_misfit)

# Sum total failed windows.
total_passed = sum(passed)
print "Total passed windows for iteration %s: %d" % (iteration, total_passed)

# Write file.
fname = os.path.join("./", "ITERATIONS", "MISFIT_%s.txt" % (iteration))
with open(fname, "w") as file:
    file.write("Total misfit for iteration %s: %f\n" % (iteration, total_misfit))
    file.write("Total passed for iteration %s: %d\n" % (iteration, total_passed))
    file.write("Normalized misfit: %f\n" % (total_misfit / float(total_passed)))

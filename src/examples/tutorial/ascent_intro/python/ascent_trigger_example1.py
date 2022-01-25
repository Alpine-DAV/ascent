###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import conduit.blueprint
import ascent
import numpy as np

from ascent_tutorial_py_utils import tutorial_gyre_example

# Use triggers to render when conditions occur
a = ascent.Ascent()
a.open()

# setup actions
actions = conduit.Node()

# declare a question to ask 
add_queries = actions.append()
add_queries["action"] = "add_queries"

# add our entropy query (q1)
queries = add_queries["queries"] 
queries["q1/params/expression"] = "entropy(histogram(field('gyre'), num_bins=128))"
queries["q1/params/name"] = "entropy"

# declare triggers 
add_triggers = actions.append()
add_triggers["action"] = "add_triggers"
triggers = add_triggers["triggers"] 

# add a simple trigger (t1_ that fires at cycle 500
triggers["t1/params/condition"] = "cycle() == 500"
triggers["t1/params/actions_file"] = "cycle_trigger_actions.yaml"

# add trigger (t2) that fires when the change in entroy exceeds 0.5

# the history function allows you to access query results of previous
# cycles. relative_index indicates how far back in history to look.

# Looking at the plot of gyre entropy in the previous notebook, we see a jump
# in entropy at cycle 200, so we expect the trigger to fire at cycle 200
triggers["t2/params/condition"] = "entropy - history(entropy, relative_index = 1) > 0.5"
triggers["t2/params/actions_file"] = "entropy_trigger_actions.yaml"

# view our full actions tree
print(actions.to_yaml())

# gyre time varying params
nsteps = 10
time = 0.0
delta_time = 0.5

for step in range(nsteps):
    # call helper that generates a double gyre time varying example mesh.
    # gyre ref :https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
    mesh = tutorial_gyre_example(time)
    
    # update the example cycle
    cycle = 100 + step * 100
    mesh["state/cycle"] = cycle
    print("time: {} cycle: {}".format(time,cycle))

    # publish mesh to ascent
    a.publish(mesh)

    # execute the actions
    a.execute(actions)
    
    # update time
    time = time + delta_time

# retrieve the info node that contains the trigger and query results
info = conduit.Node()
a.info(info)

# close ascent
a.close()

# this will render:
#  cycle_trigger_out_500.png
#  entropy_trigger_out_200.png

#
# We can also examine when the triggers executed by looking at the expressions
# results in the output info
#
print(info["expressions"].to_yaml())

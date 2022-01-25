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

# Use Ascent to extract mesh cycle and entropy of a time varying mesh
a = ascent.Ascent()
a.open()

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_queries"

# declare a queries to ask some questions
queries = add_act["queries"] 

# add a simple query expression (q1)
queries["q1/params/expression"] = "cycle()"
queries["q1/params/name"] = "cycle"

# add a more complex query expression (q2)
queries["q2/params/expression"] = "entropy(histogram(field('gyre'), num_bins=128))"
queries["q2/params/name"] = "entropy_of_gyre"

# declare a scene to render the dataset
add_scenes = actions.append()
add_scenes["action"] = "add_scenes"
scenes = add_scenes["scenes"] 
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "gyre"
# Set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_gyre"

# view our full actions tree
print(actions.to_yaml())

# gyre time varying params
nsteps = 10
time = 0.0
delta_time = 0.5

info = conduit.Node()
for step in range(nsteps):
    # call helper that generates a gyre time varying example mesh.
    # gyre ref :https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
    mesh = tutorial_gyre_example(time)
    
    # update the example cycle
    cycle = 100 + step * 100
    mesh["state/cycle"] = cycle
    print("time: {} cycle: {}".format(time,cycle))
    
    # publish mesh to ascent
    a.publish(mesh)
    
    # update image name
    scenes["s1/image_name"] = "out_gyre_%04d" % step;
    
    # execute the actions
    a.execute(actions)

    # retrieve the info node that contains the query results
    ts_info = conduit.Node()
    a.info(ts_info)

    # add to our running info
    info["expressions"].update(ts_info["expressions"])

    # update time
    time = time + delta_time

# close ascent
a.close()

# view the results of the cycle query
print(info["expressions/cycle"].to_yaml())
# Note that query results can be indexed by cycle

# view the results of the cycle query
print(info["expressions/entropy_of_gyre"].to_yaml())

# create an array with the entropy values from all 
# cycles
entropy = np.zeros(nsteps)
# get the node that has the time history
gyre = info["expressions/entropy_of_gyre"]

# transfer conduit data to our summary numpy array
for i in range(gyre.number_of_children()):
    entropy[i] = gyre[i]["value"]

print("Entropy Result")
print(entropy)

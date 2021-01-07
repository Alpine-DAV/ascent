###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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

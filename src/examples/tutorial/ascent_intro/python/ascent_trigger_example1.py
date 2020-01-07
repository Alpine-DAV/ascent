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

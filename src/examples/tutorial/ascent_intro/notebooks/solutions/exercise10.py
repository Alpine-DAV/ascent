"""
# Exercise 10 prompts:

First, copy the code from notebook 8 and paste below. Re-run this code to refresh your memory of the output.

Second refactor this code so that you get the same output as notebook 8 but without using the trigger files "cycle_trigger_actions.yaml" or "entropy_trigger_actions.yaml".

"""
# cleanup any old results
!./cleanup.sh

# ascent + conduit imports
import conduit
import conduit.blueprint
import ascent

import numpy as np

# helpers we use to create tutorial data
from ascent_tutorial_jupyter_utils import img_display_width
from ascent_tutorial_jupyter_utils import tutorial_gyre_example

import matplotlib.pyplot as plt


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
# triggers["t1/params/actions_file"] = "cycle_trigger_actions.yaml" # replace this file:
trigger_actions = triggers["t1/params/actions"].append()
# plot when trigger fires
trigger_actions["action"] = "add_scenes"
scenes = trigger_actions["scenes"]
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "gyre"
# set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "cycle_trigger_out_"

# add trigger (t2) that fires when the change in entroy exceeds 0.5

# the history function allows you to access query results of previous
# cycles. relative_index indicates how far back in history to look.

# Looking at the plot of gyre entropy in the previous notebook, we see a jump
# in entropy at cycle 200, so we expect the trigger to fire at cycle 200
triggers["t2/params/condition"] = "entropy - history(entropy, relative_index = 1) > 0.5"
# triggers["t2/params/actions_file"] = "entropy_trigger_actions.yaml" # replace with the following:
trigger2_actions = triggers["t2/params/actions"].append()
trigger2_actions["action"] = "add_scenes"
scenes2 = trigger2_actions["scenes"]
scenes2["s1/plots/p1/type"] = "pseudocolor"
scenes2["s1/plots/p1/field"] = "gyre"
# set the output file name (ascent will add ".png")
scenes2["s1/image_name"] = "entropy_trigger_out_"

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

# show the result image from the cycle trigger
ascent.jupyter.AscentImageSequenceViewer(["cycle_trigger_out_500.png", "entropy_trigger_out_200.png"]).show()
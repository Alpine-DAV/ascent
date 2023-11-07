"""
# Exercise 8 prompts:

Create your own trigger that sets up a scene with two renders (as in notebook 4) at cycle 1000.

**First**, refactor the code from Trigger Example 1 to have a single trigger at cycle 1000.

**Second**, create a new actions file or prepare to edit `cycle_trigger_actions.yaml`. Make sure your trigger is using the correct actions file.

**Third**, edit your actions .yaml file to include two renders, as in notebook 4's "Scene Example 3". One render can use an azimuth angle of 10 degrees and the other a 3x zoom.

**Fourth**, use `ascent.jupyter.AscentImageSequenceViewer` as above to plot the two .png files created by your trigger.

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

# For the exercise, update `t1` to fire at cycle 1000.
# Create a new actions file or edit cycle_trigger_actions.yaml
# to specify two renders as in `solutions_notebook8.yaml`
triggers["myt1/params/condition"] = "cycle() == 1000"
triggers["myt1/params/actions_file"] = "solution_notebook8.yaml"

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

# show the result image from the cycle trigger -- 
# include whatever filenames you declared for your renders in your yaml file
ascent.jupyter.AscentImageSequenceViewer(["cycle_trigger_out_r1.png", "cycle_trigger_out_r2.png"]).show()
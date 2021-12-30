###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import conduit.blueprint
import ascent
import numpy as np

from ascent_tutorial_py_utils import tutorial_tets_example

mesh = conduit.Node()
# (call helper to create example tet mesh as in blueprint example 2)
tutorial_tets_example(mesh)

# Use Ascent to render with views with different camera parameters
a = ascent.Ascent()
a.open()
a.publish(mesh)

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_scenes"

# declare a scene to render the dataset
scenes = add_act["scenes"]
# add a pseudocolor plot (`p1`)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "var1"
# add a mesh plot (`p1`) 
# (experiment with commenting this out)
scenes["s1/plots/p2/type"] = "mesh"
scenes["s1/image_name"] = "out_scene_ex2_render_two_plots"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

a.close()


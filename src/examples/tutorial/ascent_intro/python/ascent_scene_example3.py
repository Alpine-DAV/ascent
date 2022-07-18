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

# setup our actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_scenes"

# declare a scene to render the dataset
scenes = add_act["scenes"]

#
# You can define renders to control the parameters of a single output image.
# Scenes support multiple renders.
# 
# See the Renders docs for more details:
# https://ascent.readthedocs.io/en/latest/Actions/Scenes.html#renders-optional
#

# setup our scene (s1) with two renders (r1 and r2)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "var1"

# render a view (r1) with a slight adjustment to camera azimuth
scenes["s1/renders/r1/image_name"] = "out_scene_ex3_view1"
scenes["s1/renders/r1/camera/azimuth"] = 10.0

# render a view (r2) that zooms in from the default camera
scenes["s1/renders/r2/image_name"] = "out_scene_ex3_view2"
scenes["s1/renders/r2/camera/zoom"] = 3.0

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

a.close()


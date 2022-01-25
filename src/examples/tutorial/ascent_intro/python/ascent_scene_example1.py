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

# Use Ascent with multiple scenes to render different variables
a = ascent.Ascent()
a.open()
a.publish(mesh);

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_scenes"

# declare two scenes (s1 and s2) to render the dataset
scenes = add_act["scenes"]
# our first scene (named 's1') will render the field 'var1'
# to the file out_scene_ex1_render_var1.png
scenes["s1/plots/p1/type"] = "pseudocolor";
scenes["s1/plots/p1/field"] = "var1";
scenes["s1/image_name"] = "out_scene_ex1_render_var1";

# our second scene (named 's2') will render the field 'var2'
# to the file out_scene_ex1_render_var2.png
scenes["s2/plots/p1/type"] = "pseudocolor";
scenes["s2/plots/p1/field"] = "var2";
scenes["s2/image_name"] = "out_scene_ex1_render_var2";

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

a.close()


###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

"""
 file: ascent_first_light_example.py

 description:
   Demonstrates using ascent to render a pseudocolor plot.

"""

import conduit
import conduit.blueprint
import ascent

# print details about ascent
print(ascent.about().to_yaml())

# create conduit node with an example mesh using conduit blueprint's braid function
# ref: https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid

# things to explore:
#  changing the mesh resolution

mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      50,
                                      50,
                                      50,
                                      mesh)

# create an Ascent instance
a = ascent.Ascent()

# set options to allow errors propagate to python
ascent_opts = conduit.Node()
ascent_opts["exceptions"] = "forward"

#
# open ascent
#
a.open(ascent_opts)

#
# publish mesh data to ascent
#
a.publish(mesh)

#
# Ascent's interface accepts "actions" 
# that to tell Ascent what to execute
#
actions = conduit.Node()

# Create an action that tells Ascent to:
#  add a scene (s1) with one plot (p1)
#  that will render a pseudocolor of 
#  the mesh field `braid`
add_act = actions.append()
add_act["action"] = "add_scenes"

# declare a scene (s1) and pseudocolor plot (p1)
scenes = add_act["scenes"]

# things to explore:
#  changing plot type (mesh)
#  changing field name (for this dataset: radial)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"

# set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_first_light_render_3d"

# view our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# close ascent
a.close()

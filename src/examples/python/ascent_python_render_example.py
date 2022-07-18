###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


"""
 file: ascent_python_render_example.py

 description:
   Demonstrates using ascent to render a pseudocolor plot.

"""

import conduit
import conduit.blueprint
import ascent

# print details about ascent
print(ascent.about())


# open ascent
a = ascent.Ascent()
a.open()


# create example mesh using conduit blueprint
n_mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      10,
                                      10,
                                      10,
                                      n_mesh)
# publish mesh to ascent
a.publish(n_mesh)

# declare a scene to render the dataset
scenes  = conduit.Node()
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"
# Set the output file name (ascent will add ".png")
scenes["s1/image_prefix"] = "out_ascent_render_3d"

# setup actions to
actions = conduit.Node()
add_act =actions.append()
add_act["action"] = "add_scenes"
add_act["scenes"] = scenes

# execute
a.execute(actions)

# close alpine
a.close()





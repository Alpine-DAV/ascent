###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import conduit.blueprint
import ascent
import numpy as np


mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      25,
                                      25,
                                      25,
                                      mesh)

a = ascent.Ascent()
a.open()

# publish mesh to ascent
a.publish(mesh);

# setup actions
actions = conduit.Node()
add_act = actions.append();
add_act["action"] = "add_pipelines"
pipelines = add_act["pipelines"]

# create a  pipeline (pl1) with a contour filter (f1)
pipelines["pl1/f1/type"] = "contour"

# extract contours where braid variable
# equals 0.2 and 0.4
contour_params = pipelines["pl1/f1/params"]
contour_params["field"] = "braid"
iso_vals = np.array([0.2, 0.4],dtype=np.float32)
contour_params["iso_values"].set(iso_vals)

# declare a scene to render several angles of
# the pipeline result (pl1) to a Cinema Image
# database

add_act2 = actions.append()
add_act2["action"] = "add_scenes"
scenes = add_act2["scenes"]

scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/plots/p1/field"] = "braid"
# select cinema path
scenes["s1/renders/r1/type"] = "cinema"
# use 5 renders in phi
scenes["s1/renders/r1/phi"] = 5
# and 5 renders in theta
scenes["s1/renders/r1/theta"] = 5
# setup to output database to:
#  cinema_databases/out_extract_cinema_contour
# you can view using a webserver to open:
#  cinema_databases/out_extract_cinema_contour/index.html
scenes["s1/renders/r1/db_name"] = "out_extract_cinema_contour"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

a.close()

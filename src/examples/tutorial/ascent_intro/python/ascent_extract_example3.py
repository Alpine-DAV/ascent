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

# Use Ascent to calculate and render contours
a = ascent.Ascent()
a.open()

# publish our mesh to ascent
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

# add an extract to capture the pipeline result
add_act2 = actions.append()
add_act2["action"] = "add_extracts"
extracts = add_act2["extracts"]

# add an relay extract (e1) to export the pipeline result
# (pl1) to blueprint hdf5 files
extracts["e1/type"] = "relay"
extracts["e1/pipeline"]  = "pl1"
extracts["e1/params/path"] = "out_extract_braid_contour"
extracts["e1/params/protocol"] = "blueprint/mesh/hdf5"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

a.close()

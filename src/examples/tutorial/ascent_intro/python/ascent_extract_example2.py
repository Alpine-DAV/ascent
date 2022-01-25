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

# Use Ascent to export our mesh to blueprint flavored hdf5 files
a = ascent.Ascent()
a.open()

# publish mesh to ascent
a.publish(mesh);

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_extracts"

# add a relay extract that will write mesh data to 
# blueprint hdf5 files
extracts = add_act["extracts"]
extracts["e1/type"] = "relay"
extracts["e1/params/path"] = "out_export_braid_one_field"
extracts["e1/params/protocol"] = "blueprint/mesh/hdf5"

#
# add fields parameter to limit the export to the just 
# the `braid` field
#
extracts["e1/params/fields"].append().set("braid")

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

a.close()

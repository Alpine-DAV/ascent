"""
# Exercise 6 prompts:

Use and modify the code from Relay Extract Example 2 above.

**First**, add a second field -- "radial" -- to be saved alongside "braid".

**Second** change the protocol from hdf5 to yaml and change the path of the file created from `out_export_braid_one_field` to `out_export_braid_two_fields`

"""

import conduit
import conduit.blueprint
import ascent

import numpy as np

!./cleanup.sh

# create example mesh using the conduit blueprint braid helper
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
# blueprint yaml file
extracts = add_act["extracts"]
extracts["e1/type"] = "relay"
# Change the path of the file created
extracts["e1/params/path"] = "out_export_braid_two_fields"
# Update the protocol from hdf5 to yaml
extracts["e1/params/protocol"] = "blueprint/mesh/yaml"

#
# Original: add fields parameter to limit the export to
# just the `braid` field
#
extracts["e1/params/fields"].append().set("braid")
# For exercise, add the "radial" field
extracts["e1/params/fields"].append().set("radial")


# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

# show details
ascent.jupyter.AscentViewer(a).show()
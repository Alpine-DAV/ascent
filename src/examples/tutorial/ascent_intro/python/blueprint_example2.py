###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import conduit.blueprint
import ascent
import numpy as np
#
# Create a 3D mesh defined on an explicit set of points,
# composed of two tets, with two element associated fields
#  (`var1` and `var2`)
#

mesh = conduit.Node()

# create an explicit coordinate set
x = np.array( [-1.0, 0.0, 0.0, 0.0, 1.0 ], dtype=np.float64 )
y = np.array( [0.0, -1.0, 0.0, 1.0, 0.0 ], dtype=np.float64 )
z = np.array( [ 0.0, 0.0, 1.0, 0.0, 0.0 ], dtype=np.float64 )

mesh["coordsets/coords/type"] = "explicit";
mesh["coordsets/coords/values/x"].set_external(x)
mesh["coordsets/coords/values/y"].set_external(y)
mesh["coordsets/coords/values/z"].set_external(z)

# add an unstructured topology
mesh["topologies/mesh/type"] = "unstructured"
# reference the coordinate set by name
mesh["topologies/mesh/coordset"] = "coords"
# set topology shape type
mesh["topologies/mesh/elements/shape"] = "tet"
# add a connectivity array for the tets
connectivity = np.array([0, 1, 3, 2, 4, 3, 1, 2 ],dtype=np.int64)
mesh["topologies/mesh/elements/connectivity"].set_external(connectivity)
    
var1 = np.array([0,1],dtype=np.float32)
var2 = np.array([1,0],dtype=np.float32)

# create a field named var1
mesh["fields/var1/association"] = "element"
mesh["fields/var1/topology"] = "mesh"
mesh["fields/var1/values"].set_external(var1)

# create a field named var2
mesh["fields/var2/association"] = "element"
mesh["fields/var2/topology"] = "mesh"
mesh["fields/var2/values"].set_external(var2)

# print the mesh we created
print(mesh.to_yaml())

# make sure the mesh we created conforms to the blueprint
verify_info = conduit.Node()
if not conduit.blueprint.mesh.verify(mesh,verify_info):
    print("Mesh Verify failed!")
    print(verify_info.to_yaml())
else:
    print("Mesh verify success!")

# now lets look at the mesh with Ascent
a = ascent.Ascent()
a.open()

# publish mesh to ascent
a.publish(mesh)

# setup actions
actions = conduit.Node()
add_act = actions.append();
add_act["action"] = "add_scenes"

# declare a scene (s1) with one plot (p1) 
# to render the dataset
scenes = add_act["scenes"]
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "var1"
# Set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_ascent_render_tets"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# close ascent
a.close()


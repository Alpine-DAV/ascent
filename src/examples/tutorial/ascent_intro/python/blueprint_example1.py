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
# Create a 3D mesh defined on a uniform grid of points
# with a single vertex associated field named `alternating`
#

mesh = conduit.Node()

# create the coordinate set
num_per_dim = 9
mesh["coordsets/coords/type"] = "uniform";
mesh["coordsets/coords/dims/i"] = num_per_dim
mesh["coordsets/coords/dims/j"] = num_per_dim
mesh["coordsets/coords/dims/k"] = num_per_dim

# add origin and spacing to the coordset (optional)
mesh["coordsets/coords/origin/x"] = -10.0
mesh["coordsets/coords/origin/y"] = -10.0
mesh["coordsets/coords/origin/z"] = -10.0

distance_per_step = 20.0/(num_per_dim-1)

mesh["coordsets/coords/spacing/dx"] = distance_per_step
mesh["coordsets/coords/spacing/dy"] = distance_per_step
mesh["coordsets/coords/spacing/dz"] = distance_per_step

# add the topology
# this case is simple b/c it's implicitly derived from the coordinate set
mesh["topologies/topo/type"] = "uniform";
# reference the coordinate set by name
mesh["topologies/topo/coordset"] = "coords";

# create a vertex associated field named alternating
num_vertices = num_per_dim * num_per_dim * num_per_dim
vals = np.zeros(num_vertices,dtype=np.float32)
for i in range(num_vertices):
    if i%2:
        vals[i] = 0.0
    else:
        vals[i] = 1.0
mesh["fields/alternating/association"] = "vertex";
mesh["fields/alternating/topology"] = "topo";
mesh["fields/alternating/values"].set_external(vals)

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
add_act["action"] = "add_scenes";

# declare a scene (s1) with one plot (p1) 
# to render the dataset
scenes = add_act["scenes"]
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "alternating"
# Set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_ascent_render_uniform"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# close ascent
a.close()

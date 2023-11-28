
"""
# Exercise 3 prompts:

Use/alter the code from Mesh Blueprint Example 1 to create a scene with two plots: two versions of the alternating field on a uniform grid -- one with an origin at (-10,-10,-10) and one with an origin at (0,0,0).

**First**, add a second coordinate set to `mesh` and call it `mycoords`. `mycoords` will have the same properties as `coords` except for the difference in origin.

**Second**, add a second topology to `mesh` and call it `mytopo`. `mytopo` will have the same properties as `topo` except that its coordinate set will be `mycoords` instead of `coords`.

**Third**, add a second field to `mesh` and call it `myalternating`. `myalternating` will have the same properties as `alternating` except that its topology will be `mytopo` instead of `topo`.

**Fourth** add a second plot (`p2`) to the scene `s1`. `p1` will still plot the field `alternating` and `p2` should plot `myalternating`.

Finally, use AscentViewer to plot the result.
"""

# conduit + ascent imports
import conduit
import conduit.blueprint
import ascent

import math
import numpy as np

# cleanup any old results
!./cleanup.sh

#
# Create a 3D mesh defined on a uniform grid of points
# with two vertex associated fields named `alternating` and `myalternating`
#

mesh = conduit.Node()

# Create the coordinate sets

# Create the first, original coordinate set
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

# Create the second coordinate set, `mycoords`, with the same dimensions and spacing
mesh["coordsets/mycoords/type"] = "uniform";
mesh["coordsets/mycoords/dims/i"] = num_per_dim
mesh["coordsets/mycoords/dims/j"] = num_per_dim
mesh["coordsets/mycoords/dims/k"] = num_per_dim

mesh["coordsets/mycoords/spacing/dx"] = distance_per_step
mesh["coordsets/mycoords/spacing/dy"] = distance_per_step
mesh["coordsets/mycoords/spacing/dz"] = distance_per_step

# add an origin at (0,0,0) instead of (-10,-10,-10)
mesh["coordsets/mycoords/origin/x"] = 0.0
mesh["coordsets/mycoords/origin/y"] = 0.0
mesh["coordsets/mycoords/origin/z"] = 0.0

# Create topologies

# add the topologies that will be used for each field
# Create the first topology, topo, which uses the coordinate set coords
mesh["topologies/topo/type"] = "uniform";
# reference the coordinate set by name, coords
mesh["topologies/topo/coordset"] = "coords";

# Create the second topology, mytopo, which uses the coordinate set mycoords
mesh["topologies/mytopo/type"] = "uniform";
# reference the secoond coordinate set by name, mycoords
mesh["topologies/mytopo/coordset"] = "mycoords";

# Create two fields named alternating and myalternating

# Generate the data that will be plotted at the vertices on the mesh
num_vertices = num_per_dim * num_per_dim * num_per_dim
vals = np.zeros(num_vertices,dtype=np.float32)
for i in range(num_vertices):
    if i%2:
        vals[i] = 0.0
    else:
        vals[i] = 1.0

# Create the first vertex-associated field, alternating, 
# using the original topology, topo
mesh["fields/alternating/association"] = "vertex";
mesh["fields/alternating/topology"] = "topo";
mesh["fields/alternating/values"].set_external(vals)

# Create the second vertex-associated field, alternating, 
# using mytopo (and thereby mycoords)
mesh["fields/myalternating/association"] = "vertex";
mesh["fields/myalternating/topology"] = "mytopo";
mesh["fields/myalternating/values"].set_external(vals)

# Print the mesh
print(mesh.to_yaml())

# Verify the mesh conforms to the blueprint
verify_info = conduit.Node()
if not conduit.blueprint.mesh.verify(mesh,verify_info):
    print("Mesh Verify failed!")
    print(verify_info.to_yaml())
else:
    print("Mesh verify success!")

# Finally, let's plot both fields using Ascent
# We'll create one scene with two plots
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
# First plot alternating
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "alternating"
# Add a second plot for myalternating
scenes["s1/plots/p2/type"] = "pseudocolor"
scenes["s1/plots/p2/field"] = "myalternating"

# Set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_ascent_render_uniform"

# execute the actions
a.execute(actions)

# show the result using the AscentViewer widget
ascent.jupyter.AscentViewer(a).show()
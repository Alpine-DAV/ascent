###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

import math
import conduit
import ascent

# create example mesh using the conduit blueprint braid helper
mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      20,
                                      20,
                                      20,
                                      mesh)

################################################
### Add a new Spherical topology to the mesh ###
################################################

radius = 10.0
num_lat = 12
num_lon = 24

### Defining the Topology Coordinate Set ###

# Initialize with north pole values
x_vals, y_vals, z_vals = [0.0], [0.0], [radius]

# Add lat, lon coordinates
for lat in range(1, num_lat):
    theta = math.pi * float(lat) / float(num_lat)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    for lon in range(num_lon):
        phi = 2.0 * math.pi * float(lon) / float(num_lon)

        x_vals.append(radius * sin_theta * math.cos(phi))
        y_vals.append(radius * sin_theta * math.sin(phi))
        z_vals.append(radius * cos_theta)

# Add south pole values
x_vals.append(0.0)
y_vals.append(0.0)
z_vals.append(-radius)

# Add new sphere coordset to mesh
coords = mesh["coordsets/sample_sphere_coords"]
coords["type"] = "explicit"
coords["values/x"] = x_vals
coords["values/y"] = y_vals
coords["values/z"] = z_vals

### Defining the Topology Conectivity ###

ring_idx = (lambda lat, lon: int(1 + (lat - 1) * num_lon + (lon % num_lon)))

conn = []
for lon in range(num_lon):
    conn.extend([0, ring_idx(1, lon + 1), ring_idx(1, lon)])

for lat in range(1, num_lat - 1):
    for lon in range(num_lon):
        lower_left = ring_idx(lat, lon)
        lower_right = ring_idx(lat, lon + 1)
        upper_left = ring_idx(lat + 1, lon)
        upper_right = ring_idx(lat + 1, lon + 1)

        conn.extend([lower_left, lower_right, upper_left,
                     lower_right, upper_right, upper_left])

for lon in range(num_lon):
    conn.extend([ring_idx(num_lat - 1, lon), ring_idx(num_lat - 1, lon + 1), int(len(x_vals))])

# Add sphere topology connectivity to mesh
topo = mesh["topologies/sample_sphere"]
topo["type"] = "unstructured"
topo["coordset"] = "sample_sphere_coords"
topo["elements/shape"] = "tri"
topo["elements/connectivity"] = conn

print(mesh.to_yaml())

######################################
### Plot over Topology with Ascent ###
######################################

# Use Ascent to bin an input mesh in a few ways
a = ascent.Ascent()

# open ascent
a.open()

# publish mesh to ascent
a.publish(mesh)

# setup actions
actions = conduit.Node()

# Add a sampling pipeline
add_sample_act = actions.append()
add_sample_act["action"] = "add_pipelines"

sample_pipe = add_sample_act["pipelines"]
sample_pipe["pl1/f1/type"] = "sample"
sample_pipe["pl1/f1/params/fields"] = ["braid"]

# Define the topology to sample onto
sample_pipe["pl1/f1/params/topology"] = "sample_sphere"
sample_pipe["pl1/f1/params/invalid_value"] = -10.0

# Add a scene that renders the sampled result
add_act = actions.append()
add_act["action"] = "add_scenes"

scenes = add_act["scenes"]
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/image_name"] = "sample_spherical_topology"

# view our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# close ascent
a.close()

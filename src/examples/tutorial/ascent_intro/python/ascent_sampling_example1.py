###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


import conduit
import ascent

# create example mesh using the conduit blueprint braid helper
mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      20,
                                      20,
                                      20,
                                      mesh)

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

# Define the output uniform grid
sample_pipe["pl1/f1/params/uniform_grid/dims/i"] = 10
sample_pipe["pl1/f1/params/uniform_grid/dims/j"] = 10
sample_pipe["pl1/f1/params/uniform_grid/dims/k"] = 10

sample_pipe["pl1/f1/params/uniform_grid/origin/x"] = -10
sample_pipe["pl1/f1/params/uniform_grid/origin/y"] = -10
sample_pipe["pl1/f1/params/uniform_grid/origin/z"] = -10

sample_pipe["pl1/f1/params/uniform_grid/spacing/dx"] = 1
sample_pipe["pl1/f1/params/uniform_grid/spacing/dy"] = 1
sample_pipe["pl1/f1/params/uniform_grid/spacing/dz"] = 1

sample_pipe["pl1/f1/params/invalid_value"] = -10.0

# Add a scene that renders the sampled result.
add_act = actions.append()
add_act["action"] = "add_scenes"

scenes = add_act["scenes"]
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/image_name"] = "sample_uniform_grid"

# view our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# close ascent
a.close()

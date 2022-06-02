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

# Use Ascent to create and render multiple pipelines
a = ascent.Ascent()
a.open()

# publish mesh to ascent
a.publish(mesh);

# setup actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_pipelines"
pipelines = add_act["pipelines"]

# create our first pipeline (pl1) 
# with a contour filter (f1)
pipelines["pl1/f1/type"] = "contour"
# extract contours where braid variable
# equals 0.2 and 0.4
contour_params = pipelines["pl1/f1/params"]
contour_params["field"] = "braid"
iso_vals = np.array([0.2, 0.4],dtype=np.float32)
contour_params["iso_values"].set(iso_vals)

# create our second pipeline (pl2) with a threshold filter (f1)
# and a clip filter (f2)

# add our threshold (pl2 f1)
pipelines["pl2/f1/type"] = "threshold"
thresh_params = pipelines["pl2/f1/params"]
# set threshold parameters
# keep elements with values between 0.0 and 0.5
thresh_params["field"]  = "braid"
thresh_params["min_value"] = 0.0
thresh_params["max_value"] = 0.5


# add our clip (pl2 f2)
pipelines["pl2/f2/type"]   = "clip"
clip_params = pipelines["pl2/f2/params"]
# set clip parameters
# use spherical clip
clip_params["sphere/center/x"] = 0.0
clip_params["sphere/center/y"] = 0.0
clip_params["sphere/center/z"] = 0.0
clip_params["sphere/radius"]   = 12

# declare a scene to render our pipeline results
add_act2 = actions.append()
add_act2["action"] = "add_scenes"
scenes = add_act2["scenes"]

# add a scene (s1) with two pseudocolor plots 
# (p1 and p2) that will render the results 
# of our pipelines (pl1 and pl2)## Pipeline Example 2:

# plot (p1) to render our first pipeline (pl1)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/plots/p1/field"] = "braid"
# plot (p2) to render our second pipeline (pl2)
scenes["s1/plots/p2/type"] = "pseudocolor"
scenes["s1/plots/p2/pipeline"] = "pl2"
scenes["s1/plots/p2/field"] = "braid"
# set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_pipeline_ex3_two_plots"

# print our full actions tree
print(actions.to_yaml())

# execute
a.execute(actions)

# close ascent
a.close()

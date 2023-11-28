"""
# Exercise 5 prompts:

Use and refactor the code in Pipeline Example 3.

**First** break the second pipeline `pl2` with two filters into two pipelines (`pl2` and `pl3`) -- one with a single filter each. 

**Second** create separate plots in `s1` for each of the three pipelines.

You should end with a single scene and three plots that you can toggle between.

"""

# ascent + conduit imports
import conduit
import conduit.blueprint
import ascent

import numpy as np

# cleanup any old results
!./cleanup.sh

# Create example mesh

# create example mesh using the conduit blueprint braid helper
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

# create our second pipeline (pl2) with a threshold filter (f2)

# add our threshold (pl2 f1)
pipelines["pl2/f2/type"] = "threshold"
thresh_params = pipelines["pl2/f2/params"]
# set threshold parameters
# keep elements with values between 0.0 and 0.5
thresh_params["field"]  = "braid"
thresh_params["min_value"] = 0.0
thresh_params["max_value"] = 0.5


# create our third pipeline (pl3) with a clip filter (f3)
pipelines["pl3/f3/type"]   = "clip"
clip_params = pipelines["pl3/f3/params"]
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

# scene (s1) to render our first pipeline (pl1)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/plots/p1/field"] = "braid"
# scene (s2) to render our second pipeline (pl2)
scenes["s2/plots/p1/type"] = "pseudocolor"
scenes["s2/plots/p1/pipeline"] = "pl2"
scenes["s2/plots/p1/field"] = "braid"
# scene (s3) to render our third pipeline (pl3)
scenes["s3/plots/p1/type"] = "pseudocolor"
scenes["s3/plots/p1/pipeline"] = "pl3"
scenes["s3/plots/p1/field"] = "braid"
# set the output file names (ascent will add ".png")
scenes["s1/image_name"] = "out_pipeline_ex3_1st_plot"
scenes["s2/image_name"] = "out_pipeline_ex3_2nd_plot"
scenes["s3/image_name"] = "out_pipeline_ex3_3rd_plot"

# print our full actions tree
print(actions.to_yaml())

# execute
a.execute(actions)

# show the resulting image
ascent.jupyter.AscentViewer(a).show()
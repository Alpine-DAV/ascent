import conduit

#create a pipeline with a single filter
pipelines = conduit.Node()
pipelines["pl1/f1/type"] = "contour"
pipelines["pl1/f1/params/field"] = "energy"
pipelines["pl1/f1/params/levels"] = 5

#create a scene from the
scenes = conduit.Node()
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/plots/p1/field"] = "energy"
# Set the output file name (ascent will add ".png")
scenes["s1/image_prefix"] = "energy_contour"

# setup actions to
actions = conduit.Node()
add_pipelines = actions.append()
add_pipelines["action"] = "add_pipelines"
add_pipelines["pipelines"] = pipelines

add_scene = actions.append()
add_scene["action"] = "add_scenes"
add_scene["scenes"] = scenes

# save example actions to a json and yaml file
actions.save("ascent_actions.json",protocol="json")
actions.save("ascent_actions.yaml",protocol="yaml")

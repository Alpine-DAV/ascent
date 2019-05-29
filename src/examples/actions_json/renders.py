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
######################################
# color table
######################################
scenes["s1/plots/p1/color_table/name"] = "viridis"

renders = conduit.Node()
# simple renders example controlling basic parameters
renders["r1/image_width"] = 1024
renders["r1/image_height"] = 1024
# Set the output file name (ascent will add ".png")
renders["r1/image_name"] = "simple_render"

######################################
# image resolution
######################################
renders["r2/image_width"] = 1024
renders["r2/image_height"] = 1024

######################################
# image names
######################################
# Set the output file name (ascent will add ".png")
renders["r2/image_name"] = "advanced_render"

######################################
# annotations
######################################
#renders["r2/annotations"] = "false" # no annotations
renders["r2/annotations"] = "true" # annotations (defualt)

######################################
# foreground and background colors
######################################
# default is black backgournd and white forground.
# This code reverses the defaults
renders["r2/fg_color"].append().set(0.0)
renders["r2/fg_color"].append().set(0.0)
renders["r2/fg_color"].append().set(0.0)

renders["r2/bg_color"].append().set(1.0)
renders["r2/bg_color"].append().set(1.0)
renders["r2/bg_color"].append().set(1.0)

# render bg controls if the background is
# rendered. If disabled, the background
# color will be completetly transparent.
# This is often useful for presentation
# graphics
#renders["r2/render_bg"] = "false"

######################################
# common camera parameters
######################################
renders["r2/camera/look_at"].append().set(5.0)
renders["r2/camera/look_at"].append().set(5.0)
renders["r2/camera/look_at"].append().set(5.0)

renders["r2/camera/position"].append().set(5.0)
renders["r2/camera/position"].append().set(5.0)
renders["r2/camera/position"].append().set(18.0)

renders["r2/camera/up"].append().set(0.0)
renders["r2/camera/up"].append().set(1.0)
renders["r2/camera/up"].append().set(0.0)

renders["r2/camera/fov"] = 60
renders["r2/camera/zoom"] = 0.0 #no zoom

scenes["s1/renders"] = renders;
scenes["s1/renders/r2/camera/azimuth"] = 10.0
scenes["s1/renders/r2/camera/elevation"] = -10.0
scenes["s1/renders/r2/camera/near_plane"] = 0.1
scenes["s1/renders/r2/camera/far_plane"] = 100.1


# setup actions to
actions = conduit.Node()
add_pipelines = actions.append()
add_pipelines["action"] = "add_pipelines"
add_pipelines["pipelines"] = pipelines

add_scene = actions.append()
add_scene["action"] = "add_scenes"
add_scene["scenes"] = scenes

# execute and reset the actions
actions.append()["action"] = "execute"
actions.append()["action"] = "reset"

# save example actions to a json and yaml file
actions.save("ascent_actions.json",protocol="json")
actions.save("ascent_actions.yaml",protocol="yaml")


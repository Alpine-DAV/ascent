"""
# Exercise 4 prompts:

Use and modify the code from Scene Example 3 ("Adjusting camera parameters").
Change the color scheme to Viridis and rotate the view of the tet example
360 degrees. 

**First**, update the name of the color table as in Example 4.

**Second**, create 37 renders of `s1` with azimuth angles [0, 10, 20, 30, .... 360]

Note: the following Python syntax for string interpolation may be helpful:

```
a = "world"
print(f"Hello {a}")
```

"""

# ascent + conduit imports
import conduit
import conduit.blueprint
import ascent

import numpy as np

# helper for creating tutorial data
from ascent_tutorial_jupyter_utils import tutorial_tets_example

# cleanup any old results
!./cleanup.sh

# Prepare tet mesh
mesh = conduit.Node()
tutorial_tets_example(mesh)

# Create Ascent instance and publish tet mesh
a = ascent.Ascent()
a.open()
a.publish(mesh)

# Set up our actions
actions = conduit.Node()
add_act = actions.append()
add_act["action"] = "add_scenes"

# Declare a scene to render the dataset
scenes = add_act["scenes"]

# Set up our scene (s1)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "var1"
scenes["s1/plots/p1/color_table/name"] = "Viridis"

# Add renders of the field at different angles
# The view rotates 360 degrees in 10 degree increments,
# starting at and returning to 0 degrees
for i in range(37):
    scenes[f"s1/renders/r{i}/image_name"] = f"out_scene_ex3_view{i}"
    scenes[f"s1/renders/r{i}/camera/azimuth"] = 10.0*i

# Without a loop, creating the first few renders would have looked
# like this:
    
# scenes["s1/renders/r0/image_name"] = "out_scene_ex3_view0"
# scenes["s1/renders/r0/camera/azimuth"] = 0.0

# scenes["s1/renders/r1/image_name"] = "out_scene_ex3_view1"
# scenes["s1/renders/r1/camera/azimuth"] = 10.0

# scenes["s1/renders/r2/image_name"] = "out_scene_ex3_view2"
# scenes["s1/renders/r2/camera/azimuth"] = 20.0

# scenes["s1/renders/r3/image_name"] = "out_scene_ex3_view3"
# scenes["s1/renders/r3/camera/azimuth"] = 30.0

# scenes["s1/renders/r4/image_name"] = "out_scene_ex3_view4"
# scenes["s1/renders/r4/camera/azimuth"] = 40.0

# (...)

a.execute(actions)

ascent.jupyter.AscentViewer(a).show()

"""
# Exercise 1 prompts:
First, change the "field" name for this dataset from "braid" to "radial" and re-plot the image.

Second, observe how the image generated changes as you decrease the number of points on the mesh. For example, move from a 50x50x50 mesh to a 15x15x15 mesh.


# Changes to the code:
(1) `scenes["s1/plots/p1/field"] = "braid"` is updated to `scenes["s1/plots/p1/field"] = "radial"
`

(2)
The line below is updated from a 50x50x50 mesh to a 15x15x15 mesh.

```
conduit.blueprint.mesh.examples.braid("hexs",
                                       15,
                                       15,
                                       15,
                                       mesh)
```

"""

# conduit + ascent imports
import conduit
import conduit.blueprint
import ascent

# cleanup any old results
!./cleanup.sh

mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      15,
                                      15,
                                      15,
                                      mesh)


# create an Ascent instance
a = ascent.Ascent()

# set options to allow errors propagate to python
ascent_opts = conduit.Node()
ascent_opts["exceptions"] = "forward"

#
# open ascent
#
a.open(ascent_opts)

#
# publish mesh data to ascent
#
a.publish(mesh)

#
# Ascent's interface accepts "actions" 
# that to tell Ascent what to execute
#
actions = conduit.Node()

# Create an action that tells Ascent to:
#  add a scene (s1) with one plot (p1)
#  that will render a pseudocolor of 
#  the mesh field `braid`
add_act = actions.append()
add_act["action"] = "add_scenes"

# declare a scene (s1) and pseudocolor plot (p1)
scenes = add_act["scenes"]

# things to explore:
#  changing plot type (mesh)
#  changing field name (for this dataset: radial)
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "radial"

# set the output file name (ascent will add ".png")
scenes["s1/image_name"] = "out_first_light_render_3d"

# view our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions)

# show the result from our scene using the AscentViewer widget
ascent.jupyter.AscentViewer(a).show()

## close ascent
# a.close()
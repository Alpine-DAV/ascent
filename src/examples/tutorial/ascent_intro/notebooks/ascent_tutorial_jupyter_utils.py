###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
#
# file: ascent_tutorial_jupyter_utils.py
#
###############################################################################

import ipywidgets as widgets
import numpy as np
import math
import conduit


def img_display_width():
    """
    returns common size for displaying rendered results in
    our notebooks
    """
    return 400

class ImageSeqViewer(object):
    """
    helper for displaying sequences of images
    in a jupyter notebook (thanks to Tom Stitt @ LLNL)
    """
    def __init__(self,fnames):
        self.data = []
        for fname in fnames:
            with open(fname, "rb") as f:
                self.data.append(f.read())
        self.image = widgets.Image(value=self.data[0],
                                    width=img_display_width(),
                                    height=img_display_width(),
                                    format="png")
        self.slider = widgets.IntSlider()
        self.play = widgets.Play(value=0,
                                 min=0,
                                 max=len(self.data)-1,
                                 step=1,
                                 interval=500) 

        widgets.jslink((self.play, "min"), (self.slider, "min"))
        widgets.jslink((self.play, "max"), (self.slider, "max"))
        widgets.jslink((self.play, "value"), (self.slider, "value"))

    def update(self,change):
        index = change.owner.value
        self.image.value = self.data[index]
 
    def show(self):
        v = widgets.VBox([self.image, self.slider, self.play])
        self.slider.observe(self.update)
        self.play.observe(self.update)
        return v


def tutorial_tets_example(mesh):
    """
     (creates example tet mesh from blueprint example 2)
     
     Create a 3D mesh defined on an explicit set of points,
     composed of two tets, with two element associated fields
      (`var1` and `var2`)
    
    """
    # create an explicit coordinate set
    x = np.array( [-1.0, 0.0, 0.0, 0.0, 1.0 ], dtype=np.float64 )
    y = np.array( [0.0, -1.0, 0.0, 1.0, 0.0 ], dtype=np.float64 )
    z = np.array( [ 0.0, 0.0, 1.0, 0.0, 0.0 ], dtype=np.float64 )

    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set(x)
    mesh["coordsets/coords/values/y"].set(y)
    mesh["coordsets/coords/values/z"].set(z)

    # add an unstructured topology
    mesh["topologies/mesh/type"] = "unstructured"
    # reference the coordinate set by name
    mesh["topologies/mesh/coordset"] = "coords"
    # set topology shape type
    mesh["topologies/mesh/elements/shape"] = "tet"
    # add a connectivity array for the tets
    connectivity = np.array([0, 1, 3, 2, 4, 3, 1, 2 ],dtype=np.int64)
    mesh["topologies/mesh/elements/connectivity"].set(connectivity)
    
    var1 = np.array([0,1],dtype=np.float32)
    var2 = np.array([1,0],dtype=np.float32)

    # create a field named var1
    mesh["fields/var1/association"] = "element"
    mesh["fields/var1/topology"] = "mesh"
    mesh["fields/var1/values"].set(var1)

    # create a field named var2
    mesh["fields/var2/association"] = "element"
    mesh["fields/var2/topology"] = "mesh"
    mesh["fields/var2/values"].set(var2)

    # make sure the mesh we created conforms to the blueprint
    verify_info = conduit.Node()
    if not conduit.blueprint.mesh.verify(mesh,verify_info):
        print("Mesh Verify failed!")
        print(verify_info.to_yaml())

def tutorial_gyre_example(time):
    """
    Helper that generates a gyre time varying example mesh.
    
    gyre ref :https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
    """
    mesh = conduit.Node()
    xy_dims = 40
    z_dims = 2
    
    conduit.blueprint.mesh.examples.braid("hexs",
                                          xy_dims,
                                          xy_dims,
                                          z_dims,
                                          mesh)

    mesh["state/time"] = time
    field = mesh["fields/gyre"]
    field["association"] = "vertex"
    field["topology"] = "mesh"
    values = np.zeros(xy_dims*xy_dims*z_dims)


    e = 0.25
    A = 0.1
    w = (2.0 * math.pi) / 10.0
    a_t = e * math.sin(w * time)
    b_t = 1.0 - 2 * e * math.sin(w * time)
    #print("e: " + str(e) + " A " + str(A) + " w " + str(w) + " a_t " + str(a_t) + " b_t " + str(b_t))
    #print(b_t)
    #print(w)
    idx = 0
    for z in range(z_dims):
        for y in range(xy_dims):
            # scale y to 0-1
            y_n = float(y)/float(xy_dims)
            y_t = math.sin(math.pi * y_n)
            for x in range(xy_dims):
                # scale x to 0-1
                x_f = float(x)/ (float(xy_dims) * .5)
                f_t = a_t * x_f * x_f + b_t * x_f
                #print(f_t)
                value = A * math.sin(math.pi * f_t) * y_t
                u = -math.pi * A * math.sin(math.pi * f_t) * math.cos(math.pi * y_n)
                df_dx = 2.0 * a_t + b_t
                #print("df_dx " + str(df_dx))
                v = math.pi * A * math.cos(math.pi * f_t) * math.sin(math.pi * y_n) * df_dx
                values[idx] = math.sqrt(u * u + v * v)
                #values[idx] = u * u + v * v
                #values[idx] = value
                #print("u " + str(u) + " v " + str(v) + " mag " + str(math.sqrt(u * u + v * v)))
                idx = idx + 1

    #print(values)
    field["values"] = values  
    return mesh

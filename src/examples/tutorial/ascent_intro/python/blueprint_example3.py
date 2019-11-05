###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


import conduit
import conduit.blueprint
import ascent

import math
import numpy as np

# The conduit blueprint library provides several 
# simple builtin examples that cover the range of
# supported coordinate sets, topologies, field etc
# 
# Here we create a mesh using the braid example
# (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid)
# and modify one of its fields to create a time-varying
# example

# Define a function that will calcualte a time varying field
def braid_time_varying(npts_x, npts_y, npts_z, interp, res):
    if npts_z < 1:
        npts_z = 1

    npts = npts_x * npts_y * npts_z
    
    res["association"] = "vertex"
    res["topology"] = "mesh"
    vals = res["values"]
    
    dx_seed_start = 0.0
    dx_seed_end   = 5.0
    dx_seed = interp * (dx_seed_end - dx_seed_start) + dx_seed_start
    
    dy_seed_start = 0.0
    dy_seed_end   = 2.0
    dy_seed = interp * (dy_seed_end - dy_seed_start) + dy_seed_start
    
    dz_seed = 3.0

    dx = (float) (dx_seed * math.pi) / float(npts_x - 1)
    dy = (float) (dy_seed * math.pi) / float(npts_y-1)
    dz = (float) (dz_seed * math.pi) / float(npts_z-1)
    
    idx = 0
    for k in range(npts_z):
        cz =  (k * dz) - (1.5 * math.pi)
        for j in range(npts_y):
            cy =  (j * dy) - (math.pi)
            for i in range(npts_x):
                cx =  (i * dx) + (2.0 * math.pi)
                cv =  math.sin( cx ) + \
                      math.sin( cy ) +  \
                      2.0 * math.cos(math.sqrt( (cx*cx)/2.0 +cy*cy) / .75) + \
                      4.0 * math.cos( cx*cy / 4.0)
                                  
                if npts_z > 1:
                    cv += math.sin( cz ) + \
                          1.5 * math.cos(math.sqrt(cx*cx + cy*cy + cz*cz) / .75)
                vals[idx] = cv
                idx+=1

# create a conduit node with an example mesh using conduit blueprint's braid function
# ref: https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#braid
mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("hexs",
                                      50,
                                      50,
                                      50,
                                      mesh)

a = ascent.Ascent()
# open ascent
a.open()

# create our actions
actions = conduit.Node()
add_act =actions.append()
add_act["action"] = "add_scenes"

# declare a scene (s1) and plot (p1)
# to render braid field 
scenes = add_act["scenes"] 
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/field"] = "braid"

print(actions.to_yaml())

# loop over a set of steps and 
# render a time varying version of the braid field

nsteps = 20
interps = np.linspace(0.0, 1.0, num=nsteps)
i = 0

for interp in interps:
    print("{}: interp = {}".format(i,interp))
    # update the braid field
    braid_time_varying(50,50,50,interp,mesh["fields/braid"])
    # update the mesh cycle
    mesh["state/cycle"] = i
    # Set the output file name (ascent will add ".png")
    scenes["s1/renders/r1/image_name"] = "out_ascent_render_braid_tv_%04d" % i
    scenes["s1/renders/r1/camera/azimuth"] = 25.0
    
    # publish mesh to ascent
    a.publish(mesh)

    # execute the actions
    a.execute(actions)
    
    i+=1

# close ascent
a.close()


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
import numpy as np


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
add_act = actions.append();
add_act["action"] = "add_pipelines"
pipelines = add_act["pipelines"]

# create a  pipeline (pl1) with a contour filter (f1)
pipelines["pl1/f1/type"] = "contour"

# extract contours where braid variable
# equals 0.2 and 0.4
contour_params = pipelines["pl1/f1/params"]
contour_params["field"] = "braid"
iso_vals = np.array([0.2, 0.4],dtype=np.float32)
contour_params["iso_values"].set(iso_vals)

# declare a scene to render several angles of
# the pipeline result (pl1) to a Cinema Image
# database

add_act2 = actions.append()
add_act2["action"] = "add_scenes"
scenes = add_act2["scenes"]

scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/pipeline"] = "pl1"
scenes["s1/plots/p1/field"] = "braid"
# select cinema path
scenes["s1/renders/r1/type"] = "cinema"
# use 5 renders in phi
scenes["s1/renders/r1/phi"] = 5
# and 5 renders in theta
scenes["s1/renders/r1/theta"] = 5
# setup to output database to:
#  cinema_databases/out_extract_cinema_contour
# you can view using a webserver to open:
#  cinema_databases/out_extract_cinema_contour/index.html
scenes["s1/renders/r1/db_name"] = "out_extract_cinema_contour"

# print our full actions tree
print(actions.to_yaml())

# execute the actions
a.execute(actions);

a.close()

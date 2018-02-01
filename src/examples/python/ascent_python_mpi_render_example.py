###############################################################################
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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

"""
 file: ascent_python_mpi_render_example.py
 
 description:
   Demonstrates using mpi-enabled ascent to render a pseudocolor plot.
 

 example usage:

 mpiexec -n 2 python ascent_python_mpi_render_example.py

"""

import conduit
import conduit.blueprint
import ascent.mpi

from mpi4py import MPI


# print details about ascent
print(ascent.mpi.about())

# open ascent
a = ascent.mpi.Ascent()
ascent_opts = conduit.Node()
ascent_opts["mpi_comm"].set(MPI.COMM_WORLD.py2f())
a.open(ascent_opts)


# create example mesh using conduit blueprint
n_mesh = conduit.Node()
conduit.blueprint.mesh.examples.braid("uniform",
                                      30,
                                      30,
                                      30,
                                      n_mesh)
# shift data for rank > 1
x_origin = MPI.COMM_WORLD.rank * 20 - 10;
n_mesh["coordsets/coords/origin/x"] = x_origin
# provide proper domain id
n_mesh["state/domain_id"] = MPI.COMM_WORLD.rank
# overwrite example field with domain id
n_mesh["fields/braid/values"][:] = MPI.COMM_WORLD.rank

# publish mesh to ascent
a.publish(n_mesh)

# declare a scene to render the dataset
scenes  = conduit.Node()
scenes["s1/plots/p1/type"] = "pseudocolor"
scenes["s1/plots/p1/params/field"] = "braid"
# Set the output file name (ascent will add ".png")
scenes["s1/image_prefix"] = "out_ascent_render_mpi_3d"

# setup actions to 
actions = conduit.Node()
add_act =actions.append()
add_act["action"] = "add_scenes"
add_act["scenes"] = scenes

actions.append()["action"] = "execute"

# execute
a.execute(actions)

# close alpine
a.close()


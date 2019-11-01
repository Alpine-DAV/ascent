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
import numpy as np

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

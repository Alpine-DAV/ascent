//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: blueprint_example1.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    //
    // Create a 3D mesh defined on a uniform grid of points
    // with a single vertex associated field named `alternating`
    //
    Node mesh;

    int numPerDim = 9;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = numPerDim;
    mesh["coordsets/coords/dims/j"] = numPerDim;
    mesh["coordsets/coords/dims/k"] = numPerDim;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = -10.0;
    mesh["coordsets/coords/origin/y"] = -10.0;
    mesh["coordsets/coords/origin/z"] = -10.0;
    double distancePerStep = 20.0/(numPerDim-1);
    mesh["coordsets/coords/spacing/dx"] = distancePerStep;
    mesh["coordsets/coords/spacing/dy"] = distancePerStep;
    mesh["coordsets/coords/spacing/dz"] = distancePerStep;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    int numVertices = numPerDim*numPerDim*numPerDim; // 3D
    float *vals = new float[numVertices];
    for (int i = 0 ; i < numVertices ; i++)
        vals[i] = ( (i%2)==0 ? 0.0 : 1.0);

    // create a vertex associated field named alternating
    mesh["fields/alternating/association"] = "vertex";
    mesh["fields/alternating/topology"] = "topo";
    mesh["fields/alternating/values"].set_external(vals, numVertices);

    // print the mesh we created
    std::cout << mesh.to_yaml() << std::endl;

    //  make sure the mesh we created conforms to the blueprint

    Node verify_info;
    if(!blueprint::mesh::verify(mesh, verify_info))
    {
        std::cout << "Mesh Verify failed!" << std::endl;
        std::cout << verify_info.to_yaml() << std::endl;
    }
    else
    {
        std::cout << "Mesh verify success!" << std::endl;
    }

    // now lets look at the mesh with Ascent
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a scene (s1) with one plot (p1) 
    // to render the dataset
    Node & scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "alternating";
    // Set the output file name (ascent will add ".png")
    scenes["s1/image_prefix"] = "out_ascent_render_uniform";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}

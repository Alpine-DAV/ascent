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
/// file: blueprint_example2.cpp
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
    // Create a 3D mesh defined on an explicit set of points,
    // composed of two tets, with two element associated fields
    //  (`var1` and `var2`)
    //

    Node mesh;

    // create an explicit coordinate set
    double X[5] = { -1.0, 0.0, 0.0, 0.0, 1.0 };
    double Y[5] = { 0.0, -1.0, 0.0, 1.0, 0.0 };
    double Z[5] = { 0.0, 0.0, 1.0, 0.0, 0.0 };
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(X, 5);
    mesh["coordsets/coords/values/y"].set_external(Y, 5);
    mesh["coordsets/coords/values/z"].set_external(Z, 5);


    // add an unstructured topology
    mesh["topologies/mesh/type"] = "unstructured";
    // reference the coordinate set by name
    mesh["topologies/mesh/coordset"] = "coords";
    // set topology shape type
    mesh["topologies/mesh/elements/shape"] = "tet";
    // add a connectivity array for the tets
    int64 connectivity[8] = { 0, 1, 3, 2, 4, 3, 1, 2 };
    mesh["topologies/mesh/elements/connectivity"].set_external(connectivity, 8);

    const int num_elements = 2;
    float var1_vals[num_elements] = { 0, 1 };
    float var2_vals[num_elements] = { 1, 0 };
    
    // create a field named var1
    mesh["fields/var1/association"] = "element";
    mesh["fields/var1/topology"] = "mesh";
    mesh["fields/var1/values"].set_external(var1_vals, 2);

    // create a field named var2
    mesh["fields/var2/association"] = "element";
    mesh["fields/var2/topology"] = "mesh";
    mesh["fields/var2/values"].set_external(var2_vals, 2);

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
    Node & add_act = actions.append();
    add_act["action"] = "add_scenes";

    // declare a scene (s1) with one plot (p1) 
    // to render the dataset
    Node & scenes = add_act["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "var1";
    // Set the output file name (ascent will add ".png")
    scenes["s1/image_name"] = "out_ascent_render_tets";

    // print our full actions tree
    std::cout <<  actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}


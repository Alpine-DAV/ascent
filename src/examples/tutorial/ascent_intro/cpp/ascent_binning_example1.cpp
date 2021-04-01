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
/// file: ascent_query_example1.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <sstream>

#include "ascent.hpp"
#include "conduit_blueprint.hpp"

#include "ascent_tutorial_cpp_utils.hpp"

using namespace ascent;
using namespace conduit;

const int EXAMPLE_MESH_SIDE_DIM = 32;

int main(int argc, char **argv)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);

    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_queries";

    // declare a queries to ask some questions
    Node &queries = add_act["queries"] ;

    // Create a 1D binning projected onto the x-axis
    queries["q1/params/expression"] = "binning('radial','max', [axis('x',num_bins=20)])";
    queries["q1/params/name"] = "1d_binning";

    // Create a 2D binning projected onto the x-y plane
    queries["q2/params/expression"] = "binning('radial','max', [axis('x',num_bins=20), axis('y',num_bins=20)])";
    queries["q2/params/name"] = "2d_binning";

    // Create a binning that emulates a line-out, that is, bin all values
    // between x = [-1,1], y = [-1,1] along the z-axis in 20 bins.
    // The result is a 1x1x20 array
    queries["q3/params/expression"] = "binning('radial','max', [axis('x',[-1,1]), axis('y', [-1,1]), axis('z', num_bins=20)])";
    queries["q3/params/name"] = "3d_binning";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();

}



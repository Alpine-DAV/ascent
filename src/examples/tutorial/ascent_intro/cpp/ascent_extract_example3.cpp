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
/// file: ascent_extract_example3.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "ascent.hpp"
#include "conduit_blueprint.hpp"

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
    //create example mesh using the conduit blueprint braid helper
    
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              mesh);

    // Use Ascent to export contours to blueprint flavored hdf5 files
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_pipelines";
    Node &pipelines = add_act["pipelines"];

    // create a  pipeline (pl1) with a contour filter (f1)
    pipelines["pl1/f1/type"] = "contour";

    // extract contours where braid variable
    // equals 0.2 and 0.4
    Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";

    double iso_vals[2] = {0.2, 0.4};
    contour_params["iso_values"].set(iso_vals,2);

    // add an extract to capture the pipeline result
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_extracts";
    Node &extracts = add_act2["extracts"];

    // add an relay extract (e1) to export the pipeline result
    // (pl1) to blueprint hdf5 files
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = "out_extract_braid_contour";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();
}




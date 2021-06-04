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
/// file: t_ascent_htg.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
TEST(ascent_htg, test_htg_save)
{
    Node n;
    ascent::about(n);

    //
    // Create a HTG.
    //
    int dims[3] = {3, 3, 3};
    double xcoord[3] = {-9, 0, 9};
    double ycoord[3] = {-9, 0, 9};
    double zcoord[3] = {-9, 0, 9};

    int ntrees = 8;
    int nlevels = 2;
    int nvertices[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    int descriptor_ntuples[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int descriptor_range[8][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1},
                                  {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    int descriptor_values[8][1] = {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}};
    int nbverticesbylevel_ntuples[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    int nbverticesbylevel_range[8][2] = {{1, 8}, {1, 8}, {1, 8}, {1, 8},
                                         {1, 8}, {1, 8}, {1, 8}, {1, 8}};
    int nbverticesbylevel_values[8][2] = {{1, 8}, {1, 8}, {1, 8}, {1, 8},
                                          {1, 8}, {1, 8}, {1, 8}, {1, 8}};
    int mask_ntuples[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int mask_range[8][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0},
                            {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    int mask_values[8][1] = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
    double var_range[8][2] = {{-5.727093109410222, 6.6233486943174285},
                              {-5.727093109410222, 6.623348694317427},
                              {2.285714285713738, 12.301382488478277},
                              {2.285714285713738, 12.301382488478277},
                              {-5.727093109410215, 6.62334869431742},
                              {-5.727093109410216, 6.623348694317419},
                              {2.285714285713738, 12.301382488478275},
                              {2.285714285713738, 12.301382488478275}};
    double var[8][9] = {{-1.4329301982925986, 6.6233486943174285, 0.15456141491812506,
                         3.1667208531123707, -1.486595032586818, 0.18342804281928857,
                         -5.727093109410222, -1.3677804043318535, -2.2857142857137376},
                        {-1.4329301982925986, 0.15456141491812492, 6.623348694317427,
                         -1.4865950325868176, 3.1667208531123725, -5.727093109410222,
                         0.18342804281928762, -2.2857142857137376, -1.3677804043318535},
                        {5.242126588494578, 6.264321477338269, 2.8281486700713954,
                         12.301382488478279, 9.27648091553249, 2.908901668841599,
                         2.285714285713738, 9.289282410464718, 7.268214373919992},
                        {5.242126588494579, 2.8281486700713954, 6.264321477338272,
                         9.27648091553249, 12.301382488478277, 2.285714285713738,
                         2.908901668841599, 7.268214373919992, 9.289282410464725},
                        {-1.4329301982925962, 0.18342804281928737, -5.727093109410215,
                         -1.3677804043318391, -2.2857142857137376, 6.62334869431742,
                         0.15456141491812272, 3.166720853112348, -1.486595032586809},
                        {-1.4329301982925966, -5.727093109410216, 0.18342804281928707,
                         -2.2857142857137376, -1.3677804043318393, 0.15456141491812367,
                         6.623348694317419, -1.486595032586809, 3.1667208531123476},
                        {5.24212658849458, 2.908901668841604, 2.285714285713738,
                         9.289282410464722, 7.268214373919974, 6.264321477338269,
                         2.828148670071402, 12.301382488478275, 9.276480915532511},
                        {5.242126588494579, 2.285714285713738, 2.9089016688416027,
                         7.268214373919974, 9.289282410464724, 2.828148670071402,
                         6.26432147733827, 9.27648091553251, 12.30138248847827}};

    Node data;
    
    data["coordsets/coords/type"] = "rectilinear";
    data["coordsets/coords/values/x"].set_external(xcoord, 3);
    data["coordsets/coords/values/y"].set_external(ycoord, 3);
    data["coordsets/coords/values/z"].set_external(zcoord, 3);
    data["topologies/mesh/type"] = "rectilinear";
    data["topologies/mesh/coordset"] = "coords";
    data["topologies/mesh/nlevels"] = nlevels;

    for (int i = 0; i < ntrees; ++i)
    {
        Node &tree = data["topologies/mesh/trees"].append();
        tree["nvertices"] = nvertices[i];
        tree["descriptor_ntuples"] = descriptor_ntuples[i];
        tree["descriptor_range"].set_external(descriptor_range[i], 2);
        tree["descriptor_values"].set_external(descriptor_values[i], 2);
        tree["nbverticesbylevel_ntuples"] = nbverticesbylevel_ntuples[i];
        tree["nbverticesbylevel_range"].set_external(nbverticesbylevel_range[i], 2);
        tree["nbverticesbylevel_values"].set_external(nbverticesbylevel_values[i], 2);
        tree["mask_ntuples"] = mask_ntuples[i];
        tree["mask_range"].set_external(mask_range[i], 2);
        tree["mask_values"].set_external(mask_values[i], 1);
        tree["var_range"].set_external(var_range[i], 2);
        tree["var_values"].set_external(var[i], 9);
    }

    data.print();

    Node verify_info;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing htg vtk save");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_htg_vtk.htg");

    // remove old file before saving
    remove_test_image(output_file);

    conduit::Node extracts;
    extracts["e1/type"]  = "htg";

    extracts["e1/params/path"] = output_file;

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}

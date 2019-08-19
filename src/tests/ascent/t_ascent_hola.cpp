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
/// file: t_ascent_hola.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_hola.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;


//-----------------------------------------------------------------------------
TEST(ascent_hola, test_hola_relay_blueprint_mesh)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }

    //
    // Create example data
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              10,
                                              10,
                                              10,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    int cycle = 101;
    data["state/cycle"] = cycle;

    // make sure the _output dir exists
    string output_path =  prepare_output_dir();


    string output_file = "tout_hola_relay_blueprint_mesh";
    //
    // Create the actions to export the dataset
    //

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extract = actions.append();
    add_extract["action"] = "add_extracts";
    add_extract["extracts/e1/type"]  = "relay";
    add_extract["extracts/e1/params/path"] = output_file;
    add_extract["extracts/e1/params/protocol"] = "blueprint/mesh/hdf5";

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["messages"] = "verbose";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // use hola to say hello to the data gain

    Node hola_data, hola_opts;
    char cyc_fmt_buff[64];
    snprintf(cyc_fmt_buff, sizeof(cyc_fmt_buff), "%06d",cycle);

    ostringstream oss;
    oss << output_file << ".cycle_" << cyc_fmt_buff << ".root";
    std::string output_root = oss.str();

    hola_opts["root_file"] = output_root;
    ascent::hola("relay/blueprint/mesh", hola_opts, hola_data);

    string output_image = conduit::utils::join_file_path(output_path,
                                            "tout_hola_bp_test_render");
    // remove old image before rendering
    remove_test_image(output_image);

    Ascent ascent2;
    ascent2.open(ascent_opts);
    //
    // Create rendering actions.
    //
    actions.reset();

    conduit::Node &add_scene = actions.append();
    add_scene["action"] = "add_scenes";
    add_scene["scenes/scene1/plots/plt1/type"]         = "pseudocolor";
    add_scene["scenes/scene1/plots/plt1/field"] = "braid";
    add_scene["scenes/scene1/image_prefix"] = output_file;


    ascent2.publish(hola_data);
    ascent2.execute(actions);
    ascent2.close();

    ASCENT_ACTIONS_DUMP(actions,output_file);

}


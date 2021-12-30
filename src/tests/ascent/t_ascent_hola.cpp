//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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

    std::string msg = "An example of using hola with a blueprint hdf5 file";
    ASCENT_ACTIONS_DUMP(actions,output_file, msg);
}


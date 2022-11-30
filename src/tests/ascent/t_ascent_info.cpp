// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to Ascent.

///
/// file: t_ascent_info.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 50;

//-----------------------------------------------------------------------------
TEST(ascent_info, info_save)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               20,
                                               20,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_render_info_test");
    // remove old images before rendering
    remove_test_image(output_file);

    // remove expected file
    conduit::utils::remove_path_if_exists("out_ascent_info_000100.yaml");

    // setup actions
    Node actions;
    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes/scene1/plots/plt1/type"]  = "pseudocolor";
    add_scenes["scenes/scene1/plots/plt1/field"] = "braid";
    add_scenes["scenes/scene1/image_prefix"] = output_file;
    actions.append()["action"] = "save_info";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);

    Node ascent_info;
    ascent.info(ascent_info);

    ascent.close();

    EXPECT_TRUE(conduit::utils::is_file("out_ascent_info_000100.yaml"));

    //compare info with info saved to file
    std::cout << ascent_info.to_yaml() << std::endl;
    conduit::Node info_load;
    info_load.load("out_ascent_info_000100.yaml");

    std::cout << info_load.to_yaml() << std::endl;
    // NOTE: some things won't be quite the same due to order of exec

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example saving info via `save_info` action.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}




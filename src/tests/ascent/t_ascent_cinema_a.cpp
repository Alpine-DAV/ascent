//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//-----------------------------------------------------------------------------
///
/// file: t_ascent_cinema_a.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 32;

//-----------------------------------------------------------------------------
TEST(ascent_cinema_a, test_cinema_a)
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
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    std::string db_name = "test_db";
    string output_path = "./cinema_databases/" + db_name;
    string output_file = conduit::utils::join_file_path(output_path, "info.json");
    // remove old file before rendering
    if(conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 0.;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"] = "pseudocolor";
    scenes["scene1/plots/plt1/pipeline"] = "pl1";
    scenes["scene1/plots/plt1/field"] = "braid";
    // setup required cinema params
    scenes["scene1/renders/r1/type"] = "cinema";
    scenes["scene1/renders/r1/phi"] = 2;
    scenes["scene1/renders/r1/theta"] = 2;
    scenes["scene1/renders/r1/db_name"] = "test_db";
    //scenes["scene1/renders/r1/annotations"] = "false";
    scenes["scene1/renders/r1/camera/zoom"] = 1.0; // no zoom

    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add scene
    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(conduit::utils::is_file(output_file));
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}



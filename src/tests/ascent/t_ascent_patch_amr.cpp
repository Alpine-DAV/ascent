//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_patch_amr.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 20;


//-----------------------------------------------------------------------------
TEST(ascent_amr, test_patct_amr_2d)
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
    // Create an example mesh.
    //
    Node data, verify_info;
    //conduit::blueprint::mesh::examples::braid("hexs",
    //                                          EXAMPLE_MESH_SIDE_DIM,
    //                                          EXAMPLE_MESH_SIDE_DIM,
    //                                          EXAMPLE_MESH_SIDE_DIM,
    //                                          data);
    conduit::blueprint::mesh::examples::misc("nestsets",10,10,0,data);
    data.print();
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 2D AMR");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_amr_2d");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p2/type"]         = "mesh";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
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



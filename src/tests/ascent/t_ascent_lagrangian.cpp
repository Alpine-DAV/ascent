//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_lagrangian.cpp
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
TEST(ascent_lagrangian, test_lagrangian_multistep)
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

    ASCENT_INFO("Testing lagrangian");


//    string output_path = prepare_output_dir();
//    string output_file = conduit::utils::join_file_path(output_path,"tout_lagrangian_3d");

    string output_path = ASCENT_T_BIN_DIR;

    ASCENT_INFO("Execute test from folder: " + output_path + "/ascent");
    output_path = conduit::utils::join_file_path(output_path,"ascent/output");
    ASCENT_INFO("Creating output folder: " + output_path);
    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    // remove old images before rendering
    string output_file1 = conduit::utils::join_file_path(output_path,"basisflows_0_5.vtk");
    remove_test_file(output_file1);
    string output_file2 = conduit::utils::join_file_path(output_path,"basisflows_0_10.vtk");
    remove_test_file(output_file2);

    ASCENT_INFO(output_file1);
    ASCENT_INFO(output_file2);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "lagrangian";
    // filter knobs
    conduit::Node &lagrangian_params = pipelines["pl1/f1/params"];
    lagrangian_params["field"] = "vel";
    lagrangian_params["step_size"] = 0.1;
    lagrangian_params["write_frequency"] = 5;
    lagrangian_params["cust_res"] = 1;
    lagrangian_params["x_res"] = 2;
    lagrangian_params["y_res"] = 2;
    lagrangian_params["z_res"] = 2;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);

    for(int cycle = 1; cycle <= 10; cycle++)
    {
      //
      // Create an example mesh.
      //
      Node data, verify_info;
      conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

      EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
      ascent.publish(data);
      ascent.execute(actions);
    }

    ascent.close();

    // check that we created the right output
    //EXPECT_TRUE(check_test_file(output_file1));
    //EXPECT_TRUE(check_test_file(output_file2));
    std::string msg = "An example of using the lagrangian flow filter.";
    ASCENT_ACTIONS_DUMP(actions,output_file1,msg);

    // clean up
    remove_test_file(output_file1);
    remove_test_file(output_file2);
    conduit::utils::remove_directory(output_path);
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



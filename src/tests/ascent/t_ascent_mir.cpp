//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_divergence.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 100;
float64 RADIUS = .25;

//-----------------------------------------------------------------------------
TEST(ascent_mir, venn_vtkm_mir_full)
{
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
    conduit::blueprint::mesh::examples::venn("full",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              RADIUS,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the MIR filter with 'full' data");

    data["state/cycle"] = 100;
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_mir_venn_full");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f1/type"] = "mir";
    conduit::Node &params = pipelines["pl1/f1/params"];
    //params["field"] = "circle_a";         // name of the vector field
    params["matset"] = "matset";         // name of the vector field
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    //params["output_name"] = "mag_vorticity";   // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
//    scenes["s1/plots/p1/matset"] = "matset";
    scenes["s1/plots/p1/field"] = "matset";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
//    scenes["s1/plots/p1/field"] = "circle_b";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    // add the extracts
//    conduit::Node &add_extracts = actions.append();
//    add_extracts["action"] = "add_extracts";
//    add_extracts["extracts"] = extracts;
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the MIR filter "
                      "and plotting the field 'cellMat'.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

}
//
////-----------------------------------------------------------------------------
TEST(ascent_mir, venn_vtkm_mir_sparse_by_element)
{
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
    conduit::blueprint::mesh::examples::venn("sparse_by_element",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              RADIUS,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the MIR filter with 'sparse by element' data");

    data["state/cycle"] = 100;
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_mir_venn_sparse_by_element");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f1/type"] = "mir";
    conduit::Node &params = pipelines["pl1/f1/params"];
    //params["field"] = "circle_a";         // name of the vector field
    params["matset"] = "matset";         // name of the vector field
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    //params["output_name"] = "mag_vorticity";   // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
//    scenes["s1/plots/p1/matset"] = "matset";
    scenes["s1/plots/p1/field"] = "matset";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
//    scenes["s1/plots/p1/field"] = "circle_b";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    // add the extracts
//    conduit::Node &add_extracts = actions.append();
//    add_extracts["action"] = "add_extracts";
//    add_extracts["extracts"] = extracts;
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the MIR filter "
                      "and plotting the field 'cellMat'.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

}

//-----------------------------------------------------------------------------
TEST(ascent_mir, venn_vtkm_mir_sparse_by_material)
{
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
    conduit::blueprint::mesh::examples::venn("sparse_by_material",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              RADIUS,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the MIR filter with 'sparse by material' data");

    data["state/cycle"] = 100;
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_mir_venn_sparse_by_material");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f1/type"] = "mir";
    conduit::Node &params = pipelines["pl1/f1/params"];
    //params["field"] = "circle_a";         // name of the vector field
    params["matset"] = "matset";         // name of the vector field
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    //params["output_name"] = "mag_vorticity";   // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
//    scenes["s1/plots/p1/matset"] = "matset";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
    scenes["s1/plots/p1/field"] = "matset";
//    scenes["s1/plots/p1/field"] = "circle_b";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    // add the extracts
//    conduit::Node &add_extracts = actions.append();
//    add_extracts["action"] = "add_extracts";
//    add_extracts["extracts"] = extracts;
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the MIR filter "
                      "and plotting the field 'cellMat'.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

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



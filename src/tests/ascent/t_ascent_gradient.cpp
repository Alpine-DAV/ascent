//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_gradient.cpp
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
TEST(ascent_gradient, test_gradient)
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the gradient of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_gradient_mag_braid");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f2/type"] = "gradient";
    conduit::Node &params2 = pipelines["pl1/f2/params"];
    params2["field"] = "braid";                  // name of the input field
    params2["output_name"] = "braid_gradient";   // name of the output field

    pipelines["pl1/f1/type"] = "vector_magnitude";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["field"] = "braid_gradient";         // name of the vector field
    params["output_name"] = "mag_grad";         // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "mag_grad";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the gradient filter "
                      "and plotting the magnitude.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

}

//-----------------------------------------------------------------------------
TEST(ascent_gradient, test_cell_gradient)
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the gradient of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_cell_gradient_mag_braid");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f2/type"] = "gradient";
    conduit::Node &params2 = pipelines["pl1/f2/params"];
    params2["field"] = "braid";                  // name of the input field
    params2["output_name"] = "braid_gradient";   // name of the output field
    params2["use_cell_gradient"] = "true";

    pipelines["pl1/f1/type"] = "vector_magnitude";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["field"] = "braid_gradient";         // name of the vector field
    params["output_name"] = "mag_grad";         // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "mag_grad";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the gradient filter "
                      "and plotting the magnitude.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

}
//-----------------------------------------------------------------------------
TEST(ascent_gradient, test_gradient_radial)
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the gradient of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_gradient_mag_radial");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f2/type"] = "gradient";
    conduit::Node &params2 = pipelines["pl1/f2/params"];
    params2["field"] = "radial";                  // name of the input field
    params2["output_name"] = "radial_gradient";   // name of the output field

    pipelines["pl1/f1/type"] = "vector_magnitude";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["field"] = "radial_gradient";         // name of the vector field
    params["output_name"] = "mag_grad";         // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "mag_grad";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the gradient filter on a element centered field"
                      "and plotting the magnitude.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

} //-----------------------------------------------------------------------------

TEST(ascent_gradient, test_cell_gradient_radial)
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing the gradient of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_cell_gradient_mag_radial");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f2/type"] = "gradient";
    conduit::Node &params2 = pipelines["pl1/f2/params"];
    params2["field"] = "radial";                  // name of the input field
    params2["output_name"] = "radial_gradient";   // name of the output field
    params2["use_cell_gradient"] = "true";

    pipelines["pl1/f1/type"] = "vector_magnitude";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["field"] = "radial_gradient";         // name of the vector field
    params["output_name"] = "mag_grad";         // name of the output field

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "mag_grad";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the gradient filter using cell gradients "
                      "on a element centered field and plotting the magnitude.";
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



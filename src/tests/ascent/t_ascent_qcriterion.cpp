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
#include <conduit_relay_io.hpp>
#include <conduit_relay_io_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;


float64 rand_float()
{
    return  static_cast <float64>(rand()) / static_cast <float64> (RAND_MAX);
}

void generate_example_mesh(index_t mesh_side_dim, conduit::Node &data)
{
    //
    // Create an example mesh.
    //
    data.reset();
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              mesh_side_dim,
                                              mesh_side_dim,
                                              mesh_side_dim,
                                              data);
    std::cout << data.to_yaml() << std::endl;
    index_t npts = data["fields/vel/values"][0].dtype().number_of_elements();
    // add a random vector field so we can have a non-zero
    // q-crit

    data["fields/rand_vec/association"] = "vertex";
    data["fields/rand_vec/topology"] = "mesh";
    data["fields/rand_vec/values/u"] = DataType::float32(npts);
    data["fields/rand_vec/values/v"] = DataType::float32(npts);
    data["fields/rand_vec/values/w"] = DataType::float32(npts);

    float32 *u_vals = data["fields/rand_vec/values/u"].value();
    float32 *v_vals = data["fields/rand_vec/values/v"].value();
    float32 *w_vals = data["fields/rand_vec/values/w"].value();

    srand(10);
    for(index_t i=0;i<npts;i++)
    {
        u_vals[i] = rand_float();
        v_vals[i] = rand_float();
        w_vals[i] = rand_float();
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
}


//-----------------------------------------------------------------------------
TEST(ascent_qcriterion, vel_qcriterion)
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
    Node data;
    generate_example_mesh(EXAMPLE_MESH_SIDE_DIM,data);
    
    
    conduit::relay::io::blueprint::save_mesh(data,"here_r","hdf5");
    

    ASCENT_INFO("Testing the qcriterion of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_qcriterion_vel");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    pipelines["pl1/f2/type"] = "qcriterion";
    conduit::Node &params2 = pipelines["pl1/f2/params"];
    params2["field"] = "rand_vec";                  // name of the input field
    params2["output_name"] = "vel_qcriterion";   // name of the output field
    params2["use_cell_gradient"] = "false";

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vel_qcriterion";
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
TEST(ascent_qcriterion, vel_qcriterion_contour)
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
    Node data;
    generate_example_mesh(EXAMPLE_MESH_SIDE_DIM,data);

    ASCENT_INFO("Testing the qcriterion of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_qcriterion_vel");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1

    // qcrit
    pipelines["pl1/f1/type"] = "qcriterion";
    pipelines["pl1/f1/params/field"] = "vel";
    pipelines["pl1/f1/params/output_name"] = "vel_qcriterion";
    pipelines["pl1/f1/params/use_cell_gradient"] = "false";

    // contour
    pipelines["pl1/f2/type"] = "contour";
    pipelines["pl1/f2/params/field"]  =  "vel_qcriterion"; // name of the input field
    pipelines["pl1/f2/params/levels"] = 5;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vel_qcriterion";
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



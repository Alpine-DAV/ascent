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

void generate_example_mesh(index_t mesh_side_dim, conduit::Node &mesh)
{
    //
    // We want input that is passable for q-crit calc 
    // Create an example mesh based on "double gyre"
    // https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
    //

    mesh.reset();
    Node verify_info;

    float64 time_val = 5;
    mesh["state/cycle"] = 100;
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = mesh_side_dim;
    mesh["coordsets/coords/dims/j"] = mesh_side_dim;
    mesh["coordsets/coords/dims/k"] = mesh_side_dim;
    mesh["topologies/topo/type"] = "uniform";
    mesh["topologies/topo/coordset"] = "coords";

    index_t num_verts = mesh_side_dim*mesh_side_dim*mesh_side_dim;

    Node &gyre_field = mesh["fields/gyre"];
    gyre_field["association"] = "vertex";
    gyre_field["topology"] = "topo";
    gyre_field["values"].set(DataType::float64(num_verts));

    Node &vel_field = mesh["fields/vel"];
    vel_field["association"] = "vertex";
    vel_field["topology"] = "topo";
    vel_field["values/u"].set(DataType::float64(num_verts));
    vel_field["values/v"].set(DataType::float64(num_verts));
    vel_field["values/w"].set(DataType::float64(num_verts));

    float64_array gyre_vals = gyre_field["values"].value();
    float64_array u_vals = vel_field["values/u"].value();
    float64_array v_vals = vel_field["values/u"].value();
    float64_array w_vals = vel_field["values/w"].value();

    float64 math_pi = 3.14159265359;
    float64 e = 0.25;
    float64 A = 0.1;
    float64 w = (2.0 * math_pi) / 10.0;
    float64 a_t = e * sin(w * time_val);
    float64 b_t = 1.0 - 2 * e * sin(w * time_val);
    index_t idx = 0;

    for(index_t z=0;z<mesh_side_dim;z++)
    {
        float64 z_n = float64(z)/float64(mesh_side_dim);

        for(index_t y=0;y<mesh_side_dim;y++)
        {
            //  scale y to 0-1
            float64 y_n = float64(y)/float64(mesh_side_dim);
            float64 y_t = sin(math_pi * y_n);
            for(index_t x=0;x<mesh_side_dim;x++)
            {
                // scale x to 0-1
                float64 x_f = float64(x)/ (float64(mesh_side_dim) * .5);
                float64 f_t = a_t * x_f * x_f + b_t * x_f;
                float64 u = -math_pi * A * sin(math_pi * f_t) * cos(math_pi * y_n);
                float64 df_dx = 2.0 * a_t + b_t;
                float64 v = math_pi * A * cos(math_pi * f_t) * sin(math_pi * y_n) * df_dx;
                gyre_vals[idx] = sqrt(u * u + v * v);
                u_vals[idx] = u;
                v_vals[idx] = v;
                // create some pattern for z vel
                w_vals[idx] = sqrt( (u -v) * ( u - v ));
                idx = idx + 1;
            }
        }
    }

    //std::cout << mesh.to_yaml() << std::endl;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh,verify_info));
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
    
    
    conduit::relay::io::blueprint::save_mesh(data,"gyre_test","hdf5");
    

    ASCENT_INFO("Testing the qcriterion of a field");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_qcriterion_vel");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "qcriterion";
    pipelines["pl1/f1/params/field"] = "vel";
    pipelines["pl1/f1/params/output_name"] = "vel_qcriterion";
    pipelines["pl1/f1/params/use_cell_gradient"] = "false";

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vel_qcriterion";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "hdf5";
    extracts["e1/params/path"] = output_file + "_extract";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    string output_file = conduit::utils::join_file_path(output_path,"tout_qcriterion_contour");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "qcriterion";
    pipelines["pl1/f1/params/field"] = "vel";
    pipelines["pl1/f1/params/output_name"] = "vel_qcriterion";
    pipelines["pl1/f1/params/use_cell_gradient"] = "false";

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vel_qcriterion";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // contour
    pipelines["pl1/f2/type"] = "contour";
    pipelines["pl1/f2/params/field"]  =  "vel_qcriterion"; // name of the input field
    pipelines["pl1/f2/params/levels"] = 5;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "hdf5";
    extracts["e1/params/path"] = output_file + "_extract";


    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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



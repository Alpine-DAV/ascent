//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_ray_surface.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;
#include <conduit_fmt/conduit_fmt.h>

index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_ray_surface, test_explicit_rays_with_rays_output)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Explicit Ray Surface intersections with a Contour with Rays Output");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_ray_surface_explicit_rays");

    //
    // Create the actions.
    //

    conduit::Node pipelines;

    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 1.5;

    pipelines["pl2/f2/type"] = "ray_surface";
    pipelines["pl2/pipeline"] = "pl1";
    
    // filter knobs
    conduit::Node &params = pipelines["pl2/f2/params"];

    std::vector<double> points = { 0.0, 0.0, 0.0, // center
                                  -2.5, 0.0, 0.0, // left
                                   2.5, 0.0, 0.0, // right
                                   0.0, -2.5, 0.0,// bottom
                                   0.0, 2.5, 0.0, // top
                                   //
                                   20.0, 20.0, 5.0, // top
                                   //
                                   0.0, 20.0, -5.0,
                                   0.0, 20.0, 5.0,
                                   0.0, 20.0, 0.0, 
                                   //
                                   20.0, -5.0, 0.0,
                                   20.0, 5.0, 0.0,
                                   20.0, 0.0, 0.0};

    std::vector<double> normals= { 0.0, 0.0, 1.0,
                                   0.0, 0.0, 1.0,
                                   0.0, 0.0, 1.0,
                                   0.0, 0.0, 1.0,
                                   0.0, 0.0, 1.0,
                                   //
                                  -0.7071, -0.7071, 0.0,
                                   //
                                   0.0, -1.0, 0.0,
                                   0.0, -1.0, 0.0,
                                   0.0, -1.0, 0.0,
                                   //
                                  -1.0, 0.0, 0.0,
                                  -1.0, 0.0, 0.0,
                                  -1.0, 0.0, 0.0};

    params["rays/points/x"].set(points.data(),points.size()/3, 0, sizeof(double)*3);
    params["rays/points/y"].set(points.data(),points.size()/3, sizeof(double), sizeof(double)*3);
    params["rays/points/z"].set(points.data(),points.size()/3, sizeof(double)*2, sizeof(double)*3);

    params["rays/normals/x"].set(normals.data(),normals.size()/3, 0, sizeof(double)*3);
    params["rays/normals/y"].set(normals.data(),normals.size()/3, sizeof(double), sizeof(double)*3);
    params["rays/normals/z"].set(normals.data(),normals.size()/3, sizeof(double)*2, sizeof(double)*3);


    params["result"] = "rays";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file + "_input";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    extracts["e2/type"]  = "relay";
    extracts["e2/pipeline"] = "pl2";

    extracts["e2/params/path"] = output_file + "_res";
    extracts["e2/params/protocol"] = "blueprint/mesh/hdf5";


    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of calculating ray surface intersections with a contour";
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



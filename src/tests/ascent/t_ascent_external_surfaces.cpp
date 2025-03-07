//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_external_surfaces.cpp
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
TEST(ascent_external_surfaces, test_external_surfaces_simple)
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
    conduit::blueprint::mesh::examples::braid("structured",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing External Surfaces Filter");


    string output_path = prepare_output_dir();
    string output_base = conduit::utils::join_file_path(output_path,"tout_external_surfaces_3d");
    string output_render = output_base + "_render";
    string output_root = output_base + ".cycle_000100.root";
    // remove old images before rendering
    remove_test_image(output_base + "_render");


    // remove old images before rendering
    remove_test_image(output_root);

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "external_surfaces";
    // pipelines["pl1/f1/params/topology"] = "mesh";
    pipelines["pl2/pipeline"] = "pl1";
    // clip so we can verify the inside elements are gone
    pipelines["pl2/f1/type"] = "clip";
    conduit::Node &clip_params = pipelines["pl2/f1/params"];
    clip_params["plane/point/x"] = 0.;
    clip_params["plane/point/y"] = 0.;
    clip_params["plane/point/z"] = 0.;
    clip_params["plane/normal/x"] = 1.;
    clip_params["plane/normal/y"] = 0.;
    clip_params["plane/normal/z"] = 0;

    // add scene to render
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl2";
    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_render;
    scenes["s1/renders/r1/camera/azimuth"] = -45.0;

    // add extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "hdf5";
    extracts["e1/params/path"] = output_base;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_render));
    std::string msg = "An example of using the external_surfaces filter.";
    ASCENT_ACTIONS_DUMP(actions,output_render,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_external_surfaces, test_external_surfaces_multi_topo)
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
    // create_example_multi_domain_multi_topo_dataset(data,0,1);
    conduit::blueprint::mesh::examples::braid("structured",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    // copy with name changes
    data["topologies/other_topo"] = data["topologies/mesh"];
    data["fields/other_braid"] = data["fields/braid"];
    data["fields/other_braid/topology"] = "other_topo";
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing External Surfaces Filter Multi Topo\n");


    string output_path = prepare_output_dir();
    string output_base = conduit::utils::join_file_path(output_path,"tout_external_surfaces_3d_multi_topo");
    string output_render = output_base + "_render";
    string output_root = output_base + ".cycle_000100.root";
    // remove old images before rendering
    remove_test_image(output_base + "_render");

    // remove old images before rendering
    remove_test_image(output_root);

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "external_surfaces";
    pipelines["pl2/pipeline"] = "pl1";
    // clip so we can verify the inside elements are gone
    pipelines["pl2/f1/type"] = "clip";
    conduit::Node &clip_params = pipelines["pl2/f1/params"];
    clip_params["plane/point/x"] = 0.;
    clip_params["plane/point/y"] = 0.;
    clip_params["plane/point/z"] = 0.;
    clip_params["plane/normal/x"] = 1.;
    clip_params["plane/normal/y"] = 0.;
    clip_params["plane/normal/z"] = 0;

    // add scene to render
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "other_braid";
    scenes["s1/plots/p1/pipeline"] = "pl2";
    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_render;
    scenes["s1/renders/r1/camera/azimuth"] = -45.0;

    // add extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "hdf5";
    extracts["e1/params/path"] = output_base;

    //
    // Run Ascent
    //
    Ascent ascent;
    conduit::Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    // expect failure
    EXPECT_THROW(ascent.execute(actions),conduit::Error);

    // fix by adding the topo to ext surf filter
    pipelines["pl1/f1/params/topology"] = "other_topo";
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_render));
    std::string msg = "An example of using the external_surfaces filter.";
    ASCENT_ACTIONS_DUMP(actions,output_render,msg);
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



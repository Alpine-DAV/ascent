//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_relay.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 10;

//-----------------------------------------------------------------------------
TEST(ascent_vtk_file_extract, basic)
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

    ASCENT_INFO("Testing vtk file extract serial single");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_vtk_file_extract_test_braid_single_domain");
    // add the extract
    extracts["e1/type"] = "vtk";
    extracts["e1/params/path"] = output_file;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    
    // check that the file exists
    EXPECT_TRUE(conduit::utils::is_file(output_file + ".visit"));
}

//-----------------------------------------------------------------------------
TEST(ascent_vtk_file_extract, bogus)
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

    ASCENT_INFO("Testing vtk file extract serial bogus path");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = "/bogus/that/can/t/exist/for/sure/100/percent/tout_here";
    // add the extract
    extracts["e1/type"] = "vtk";
    extracts["e1/params/path"] = output_file;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();
}



//-----------------------------------------------------------------------------
TEST(ascent_vtk_file_extract, basic_mulit_domain)
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
    conduit::blueprint::mesh::examples::spiral(7,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing vtk file extract serial multi domain");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_vtk_file_extract_test_spiral_7_domains");

    // add the extract
    extracts["e1/type"] = "vtk";
    extracts["e1/params/path"] = output_file;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that the file exists
    EXPECT_TRUE(conduit::utils::is_file(output_file + ".visit"));
}

//-----------------------------------------------------------------------------
TEST(ascent_vtk_file_extract, basic_multi_topo)
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
    build_multi_topo(data, 10);

    ASCENT_INFO("Testing vtk file extract serial multi domain");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_vtk_file_extract_basic_multi_topo");

    // add the extract
    extracts["e1/type"] = "vtk";
    extracts["e1/params/path"] = output_file;

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    // if there are mulit topos -- this will will fail w/o topology name
    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    // now try again with topo name
    extracts["e1/params/topology"] = "point_mesh";
    ascent.execute(actions);
    ascent.close();

    // check that the file exists
    EXPECT_TRUE(conduit::utils::is_file(output_file + ".visit"));
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



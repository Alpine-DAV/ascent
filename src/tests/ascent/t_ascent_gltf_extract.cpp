//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_gltf_extract.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <fstream>
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
bool
check_glb_magic(const std::string &path)
{
    char magic[4] = {0, 0, 0, 0};
    std::ifstream ifs(path.c_str(), std::ios::binary);
    ifs.read(magic, 4);
    return ifs.good() && std::string(magic, 4) == "glTF";
}

//-----------------------------------------------------------------------------
// remove output files from prior runs so stale files can't satisfy checks
void
remove_gltf_package_files(const std::string &package_dir)
{
    remove_test_file(conduit::utils::join_file_path(package_dir,
                                                    "manifest.json"));
    std::string domains_dir = conduit::utils::join_file_path(package_dir,
                                                             "domains");
    remove_test_file(conduit::utils::join_file_path(domains_dir,
                                                    "domain_00000000.glb"));
}

//-----------------------------------------------------------------------------
TEST(ascent_gltf_extract, contour_field)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
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

    ASCENT_INFO("Testing gltf extract of a contour with a scalar field");

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    pipelines["pl1/f1/params/field"] = "braid";
    pipelines["pl1/f1/params/iso_values"] = 0.2;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = "tout_gltf_extract_braid_contour_{cycle:06d}";
    string output_file_formatted = conduit::utils::join_file_path(output_path,
                                                "tout_gltf_extract_braid_contour_000100");

    remove_gltf_package_files(output_file_formatted);

    // add the extract
    extracts["e1/type"] = "gltf";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/field"] = "braid";
    extracts["e1/params/color_table/name"] = "Viridis";

    std::cout << actions.to_yaml() << std::endl;

    conduit::Node ascent_opts;
    ascent_opts["default_dir"] = output_path;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check the manifest
    string manifest_file = conduit::utils::join_file_path(output_file_formatted,
                                                          "manifest.json");
    EXPECT_TRUE(conduit::utils::is_file(manifest_file));

    Node manifest;
    conduit::relay::io::load(manifest_file, "json", manifest);
    EXPECT_EQ(manifest["protocol"].as_string(), "ascent-gltf");
    EXPECT_EQ(manifest["field"].as_string(), "braid");
    EXPECT_EQ(manifest["domains"].number_of_children(), 1);
    EXPECT_EQ(manifest["invalid_value_count"].to_uint64(), 0);

    // check the glb file
    string glb_file = conduit::utils::join_file_path(
                          conduit::utils::join_file_path(output_file_formatted,
                                                         "domains"),
                          "domain_00000000.glb");
    EXPECT_TRUE(conduit::utils::is_file(glb_file));
    EXPECT_TRUE(check_glb_magic(glb_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_gltf_extract, contour_no_field)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
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

    ASCENT_INFO("Testing gltf extract of a contour without a field");

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    pipelines["pl1/f1/params/field"] = "braid";
    pipelines["pl1/f1/params/iso_values"] = 0.2;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_gltf_extract_braid_contour_no_field_{cycle:06d}");
    string output_file_formatted = conduit::utils::join_file_path(output_path,
                                                "tout_gltf_extract_braid_contour_no_field_000100");

    remove_gltf_package_files(output_file_formatted);

    // add the extract
    extracts["e1/type"] = "gltf";
    extracts["e1/pipeline"] = "pl1";
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

    // check the manifest, no field metadata expected
    string manifest_file = conduit::utils::join_file_path(output_file_formatted,
                                                          "manifest.json");
    EXPECT_TRUE(conduit::utils::is_file(manifest_file));

    Node manifest;
    conduit::relay::io::load(manifest_file, "json", manifest);
    EXPECT_EQ(manifest["protocol"].as_string(), "ascent-gltf");
    EXPECT_FALSE(manifest.has_child("field"));
    EXPECT_EQ(manifest["domains"].number_of_children(), 1);

    // check the glb file
    string glb_file = conduit::utils::join_file_path(
                          conduit::utils::join_file_path(output_file_formatted,
                                                         "domains"),
                          "domain_00000000.glb");
    EXPECT_TRUE(conduit::utils::is_file(glb_file));
    EXPECT_TRUE(check_glb_magic(glb_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_gltf_extract, element_field_error)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
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

    ASCENT_INFO("Testing gltf extract rejects an element field");

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    pipelines["pl1/f1/params/field"] = "braid";
    pipelines["pl1/f1/params/iso_values"] = 0.2;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_gltf_extract_element_field_{cycle:06d}");

    // add the extract, radial is element associated
    extracts["e1/type"] = "gltf";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/field"] = "radial";

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
TEST(ascent_gltf_extract, bogus_path)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
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

    ASCENT_INFO("Testing gltf extract bogus path");

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "contour";
    pipelines["pl1/f1/params/field"] = "braid";
    pipelines["pl1/f1/params/iso_values"] = 0.2;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    string output_file = "/bogus/that/can/t/exist/for/sure/100/percent/tout_here";

    // add the extract
    extracts["e1/type"] = "gltf";
    extracts["e1/pipeline"] = "pl1";
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

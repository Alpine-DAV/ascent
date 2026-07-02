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

#include <fstream>
#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay_io_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 100;
float64 RADIUS = .25;

namespace
{

bool
copy_test_file(const std::string &src, const std::string &dest)
{
    std::ifstream input(src.c_str(), std::ios::binary);
    std::ofstream output(dest.c_str(), std::ios::binary);

    if(!input || !output)
    {
        return false;
    }

    output << input.rdbuf();
    return output.good();
}

bool
ensure_directory(const std::string &path)
{
    return conduit::utils::is_directory(path) ||
           conduit::utils::create_directory(path);
}

bool
stage_axom_klee_fixture(const std::string &fixture_name,
                        std::string &root_file)
{
    const std::string input_dir =
        conduit::utils::join_file_path(
            conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                           "axom_klee_test_data"),
            fixture_name);
    const std::string staged_dir =
        conduit::utils::join_file_path(prepare_output_dir(),
                                       "axom_klee_test_data_" + fixture_name);
    const std::string input_shaping_dir =
        conduit::utils::join_file_path(input_dir, "shaping");
    const std::string staged_shaping_dir =
        conduit::utils::join_file_path(staged_dir, "shaping");

    root_file = conduit::utils::join_file_path(staged_dir, "shaping.root");

    return ensure_directory(staged_dir) &&
           ensure_directory(staged_shaping_dir) &&
           copy_test_file(conduit::utils::join_file_path(input_dir,
                                                         "shaping.root"),
                          root_file) &&
           copy_test_file(conduit::utils::join_file_path(input_shaping_dir,
                                                         "shaping_0000000.hdf5"),
                          conduit::utils::join_file_path(staged_shaping_dir,
                                                         "shaping_0000000.hdf5"));
}

}

//-----------------------------------------------------------------------------
TEST(ascent_mir, venn_viskores_mir_full)
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
    params["matset"] = "matset";         // name of the material set  
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    params["output_name"] = "matset";   // name of the output field; default is `matset` param

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "matset";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
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
TEST(ascent_mir, venn_viskores_mir_sparse_by_element)
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
    params["matset"] = "matset";         // name of the material set  
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    params["output_name"] = "matset";   // name of the output field; default is `matset` param

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "matset";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
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
TEST(ascent_mir, venn_viskores_mir_sparse_by_material)
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
    params["matset"] = "matset";         // name of the material set  
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    params["output_name"] = "matset";   // name of the output field; default is `matset` param

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
    scenes["s1/plots/p1/field"] = "matset";
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
TEST(ascent_mir, axom_q7o5_material_boundary)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    auto mesh_domain = [](Node &mesh) -> Node *
    {
        if(mesh.has_path("fields") && mesh.has_path("topologies"))
        {
            return &mesh;
        }

        for(index_t i = 0; i < mesh.number_of_children(); ++i)
        {
            Node &candidate = mesh.child(i);
            if(candidate.has_path("fields") && candidate.has_path("topologies"))
            {
                return &candidate;
            }
        }

        return nullptr;
    };

    Node data, verify_info;
    std::string root_file;
    ASSERT_TRUE(stage_axom_klee_fixture("balls_and_jacks_q7o5", root_file));

    conduit::relay::io::blueprint::load_mesh(root_file, data);
    Node *dom = mesh_domain(data);
    ASSERT_TRUE(dom != nullptr);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(*dom, verify_info));

    ASCENT_INFO("Testing the MIR filter with Axom balls_and_jacks_q7o5 material data");

    (*dom)["state/cycle"] = 100;
    string output_path = prepare_output_dir();
    string output_file =
        conduit::utils::join_file_path(output_path,
                                       "tout_mir_axom_q7o5_material_boundary");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "mir";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["matset"] = "materials";
    params["error_scaling"] = 0.0;
    params["scaling_decay"] = 0.0;
    params["iterations"] = 0;
    params["max_error"] = 0.00001;
    params["output_name"] = "materials";

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "materials";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
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
    ascent.publish(*dom);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_file(output_file + "_000100.png"));
    std::string msg = "An example of using the MIR filter "
                      "with Axom balls_and_jacks_q7o5 material data.";
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

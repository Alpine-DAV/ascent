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

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 10;

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
  return conduit::utils::is_directory(path) || conduit::utils::create_directory(path);
}

bool
stage_axom_klee_fixture(const std::string &fixture_name, std::string &root_file)
{
  // Stage the fixture beside test output so relative HDF5 links resolve.
  const std::string input_dir  = conduit::utils::join_file_path(conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),"axom_klee_test_data"),fixture_name);
  const std::string staged_dir = conduit::utils::join_file_path(prepare_output_dir(), "axom_klee_test_data_" + fixture_name);
  const std::string input_shaping_dir  = conduit::utils::join_file_path(input_dir, "shaping");
  const std::string staged_shaping_dir = conduit::utils::join_file_path(staged_dir, "shaping");

  root_file = conduit::utils::join_file_path(staged_dir, "shaping.root");

  return ensure_directory(staged_dir) && 
         ensure_directory(staged_shaping_dir) && 
         copy_test_file(conduit::utils::join_file_path(input_dir,"shaping.root"),root_file) &&
         copy_test_file(conduit::utils::join_file_path(input_shaping_dir,"shaping_0000000.hdf5"),conduit::utils::join_file_path(staged_shaping_dir,"shaping_0000000.hdf5"));
}

}

//-----------------------------------------------------------------------------
TEST(ascent_conduit_extract, test_pass_thru)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    data["state/domain_id"] = 0;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing conduit  extract in serial");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    // add the extract
    extracts["e1/type"]  = "conduit";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    conduit::Node & info =  ascent.info();

    // copy out our extract
    conduit::Node extract_copy;
    extract_copy.set(info["extracts"][0]);

    ascent.close();
    // diff to make sure data looks as we expect
    Node diff_info;
    EXPECT_FALSE(extract_copy["data"][0].diff(data,diff_info));
}

//-----------------------------------------------------------------------------
TEST(ascent_conduit_extract, test_pipeline_result)
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

    ASCENT_INFO("Testing slice to in-memory extract");

    //
    // Create the actions.
    //
    // slice + conduit in memory extract
    conduit::Node actions;
    // add the pipeline

    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 1.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    // add the extract
    extracts["e1/type"]  = "conduit";
    extracts["e1/pipeline"] = "pl1";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    conduit::Node & info = ascent.info();

    // copy out our extract
    conduit::Node extract_copy;
    extract_copy.set(info["extracts"][0]);

    ascent.close();

    // pass back copy and render the result

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                            "tout_in_memory_extract_render_slice_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    actions.reset();

    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes  = add_scenes["scenes"];

    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/image_prefix"] = output_file;

    ascent.open();
    ascent.publish(extract_copy["data"]);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));

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

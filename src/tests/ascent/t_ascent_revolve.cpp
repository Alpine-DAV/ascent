//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_revolve.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace conduit;
using namespace ascent;

static const index_t EXAMPLE_MESH_SIDE_DIM = 32;

static bool
viskores_avalible()
{
    Node n;
    ascent::about(n);
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return false;
    }
    return true;
}

static void
setup(const std::string &tout_name, Node &data, std::string &output_file)
{
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              0,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    std::string output_path = prepare_output_dir();
    output_file = conduit::utils::join_file_path(output_path, tout_name);
    remove_test_image(output_file);
}

TEST(ascent_revolve, test_revolve)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_revolve", data, output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl1/f1/type"] = "revolve";
    conduit::Node &p = pipelines["pl1/f1/params"];
    p["topology"] = "mesh";
    p["axis/x"] = 1.0;
    p["axis/y"] = 0.0;
    p["axis/z"] = 0.0;
    p["angle"] = 360.0;
    p["num_steps"] = 24;
    p["capping"] = "false";

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example revolve filter using rotational extrusion.";
    ASCENT_ACTIONS_DUMP(actions, output_file, msg);
}

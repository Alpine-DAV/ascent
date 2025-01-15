//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_logging.cpp
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



//-----------------------------------------------------------------------------
TEST(ascent_logging, test_logging_simplest)
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
                                              5,
                                              5,
                                              5,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_logging_render1");

    // remove old images before rendering
    remove_test_image(output_file);
    conduit::utils::remove_path_if_exists("ascent_log_output.yaml");
    EXPECT_FALSE(conduit::utils::is_file("ascent_log_output.yaml"));

    conduit::Node actions;
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["logging"] = "true";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that the log file exists
    EXPECT_TRUE(conduit::utils::is_file("ascent_log_output.yaml"));

}

//-----------------------------------------------------------------------------
TEST(ascent_logging, test_logging_options)
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
                                              5,
                                              5,
                                              5,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_logging_render2");
    string log_file = conduit::utils::join_file_path(output_path,"tout_my_log.yaml");

    // remove old images before rendering
    remove_test_image(output_file);
    conduit::utils::remove_path_if_exists(log_file);
    EXPECT_FALSE(conduit::utils::is_file(log_file));

    conduit::Node actions;
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["logging/file_pattern"]  = log_file;
    ascent_opts["logging/log_threshold"] = "all";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that the log file exists
    EXPECT_TRUE(conduit::utils::is_file(log_file));

}

//-----------------------------------------------------------------------------
TEST(ascent_logging, test_logging_actions)
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
                                              5,
                                              5,
                                              5,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_logging_render3");
    string log_file = conduit::utils::join_file_path(output_path,"ascent_action_log.yaml");

    // remove old images/log files before rendering
    remove_test_image(output_file);
    conduit::utils::remove_path_if_exists(log_file);
    EXPECT_FALSE(conduit::utils::is_file(log_file));

    conduit::Node actions;
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions_begin_logs;
    conduit::Node &begin_logs= actions_begin_logs.append();
    begin_logs["action"] = "open_log";
    begin_logs["file_pattern"] = log_file;

    conduit::Node actions_flush_logs;
    conduit::Node &flush_logs= actions_flush_logs.append();
    flush_logs["action"] = "flush_log";

    conduit::Node actions_close_logs;
    conduit::Node &close_logs= actions_close_logs.append();
    close_logs["action"] = "close_log";

    //
    // Run Ascent
    //

    Ascent ascent;

    ascent.open();
    ascent.publish(data);
    ascent.execute(actions_begin_logs);
    ascent.execute(actions);
    ascent.execute(actions_flush_logs);
    ascent.execute(actions_close_logs);
    ascent.close();

    // check that the log file exists
    EXPECT_TRUE(conduit::utils::is_file(log_file));
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//-----------------------------------------------------------------------------
///
/// file: t_ascent_cinema_a.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 32;

//-----------------------------------------------------------------------------
TEST(ascent_queries, max_query)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_max_query");
    // remove old file
    if(conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }


    //
    // Create the actions.
    //
    Node actions;

    conduit::Node queries;
    queries["q1/params/expression"] = "max(field('braid'))";
    queries["q1/params/name"] = "max_braid";

    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    EXPECT_TRUE(info.has_path("expressions/max_braid/100/attrs/value"));
    info["expressions"].save(output_file, "json");
    info["expressions/max_braid"].print();

    ascent.close();

    // check that we created an image
    EXPECT_TRUE(conduit::utils::is_file(output_file));
    std::string msg = "An example of quiering the maximum value of a field.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_queries, max_query_pipeline)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled: skipping");

        return;
    }
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_max_pipeline_query");
    // remove old file
    if(conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }

    //
    // Create the actions.
    //
    Node actions;


    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node queries;
    queries["q1/params/expression"] = "max(field('braid'))";
    queries["q1/params/name"] = "max_braid_pipeline";
    queries["q1/pipeline"] = "pl1";

    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;

    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //actions.print();
    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    EXPECT_TRUE(info.has_path("expressions/max_braid_pipeline/100/attrs/value"));
    info["expressions"].print();
    info["expressions"].save(output_file, "json");

    ascent.close();

    // check that we created an image
    EXPECT_TRUE(conduit::utils::is_file(output_file));
    std::string msg = "An example of quiering the maximum value of a field from the result of a pipeline.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_queries, cycle_query)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_cycle_query");
    // remove old file
    if(conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node queries;
    queries["q1/params/expression"] = "cycle()";
    queries["q1/params/name"] = "cycle";

    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;
    //actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);

    conduit::Node info;
    ascent.info(info);
    EXPECT_TRUE(info.has_path("expressions/cycle/100/value"));
    EXPECT_TRUE(info["expressions/cycle/100/value"].to_int32() == 100);
    info["expressions"].save(output_file, "json");

    ascent.close();

    // check that we created an image
    EXPECT_TRUE(conduit::utils::is_file(output_file));
    std::string msg = "An example of quiering the current cycle.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_queries, filter_params)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled: skipping");

        return;
    }

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file =
      conduit::utils::join_file_path(output_path,"tout_filter_params_query");
    // remove old file
    if(conduit::utils::is_file(output_file))
    {
        conduit::utils::remove_file(output_file);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node queries;
    queries["q1/params/expression"] = "min(field('braid')).value";
    queries["q1/params/name"] = "min_value";
    queries["q2/params/expression"] = "max(field('braid')).value";
    queries["q2/params/name"] = "max_value";
    queries["q3/params/expression"] = "max_value - min_value";
    queries["q3/params/name"] = "length";

    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;
    //actions.print();

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "threshold";
    // filter knobs
    conduit::Node &thresh_params = pipelines["pl1/f1/params"];
    thresh_params["field"] = "braid";
    thresh_params["min_value"] = "0.75 * length + min_value";
    thresh_params["max_value"] = "max_value";

    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add the scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);


    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using queries in filter parameters.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_queries, save_session)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file =
      conduit::utils::join_file_path(output_path,"tout_save_session");

    string session_file = "ascent_session.yaml";
    // remove old file
    if(conduit::utils::is_file(output_file))
    {
      conduit::utils::remove_file(output_file);
    }
    // make sure we get rid of the session file
    if(conduit::utils::is_file(session_file))
    {
      std::cout<<"Removing session file "<<session_file<<"\n";
      conduit::utils::remove_file(session_file);
    }

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node queries;
    queries["q1/params/expression"] = "min(field('braid')).value";
    queries["q1/params/name"] = "bananas";

    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;

    conduit::Node &save_session = actions.append();
    save_session["action"] = "save_session";
    actions.print();
    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);


    ascent.close();

    EXPECT_TRUE(conduit::utils::is_file(session_file));
    std::string msg = "An example of explicitly saving a session file.";
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



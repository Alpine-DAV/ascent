//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_runtime_options.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <sstream>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, verbose_msgs)
{
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    Node &hello = actions.append();
    hello["action"]   = "hello!";
    actions.print();

    // we want the "empty" example pipeline
    Node open_opts;
    open_opts["runtime/type"] = "empty";
    open_opts["messages"] = "verbose";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, quiet_msgs)
{
    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",100,100,0,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Node actions;
    Node &hello = actions.append();
    hello["action"]   = "hello!";
    actions.print();

    // we want the "empty" example pipeline
    Node open_opts;
    open_opts["runtime/type"] = "empty";
    open_opts["messages"] = "quiet";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, forward_exceptions)
{


    // invoke error by choosing bad runtime
    Node open_opts;
    open_opts["exceptions"] = "forward";
    open_opts["runtime/type"] = "BAD";


    //
    // Run Ascent
    //
    Ascent ascent;
    EXPECT_THROW(ascent.open(open_opts),conduit::Error);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, catch_exceptions)
{


    // make sure bad runtime selection is caught
    Node open_opts;
    open_opts["exceptions"] = "catch";
    open_opts["runtime/type"] = "BAD";


    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open(open_opts);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_errors)
{
    // invoke error cases caused by not initializing ascent

    Ascent ascent;
    Node n;
    // these will error to std::out, but we want to check they dont cras
    ascent.publish(n);
    ascent.close();

    Node open_opts;
    open_opts["exceptions"] = "forward";
    ascent.open(open_opts);
    ascent.close();

    EXPECT_THROW(ascent.publish(n),conduit::Error);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_error_actions_file)
{
    // invoke error cases caused by not initializing ascent

    Ascent ascent;
    Node n;

    Node open_opts;
    open_opts["exceptions"] = "forward";
    open_opts["actions_file"] = "bananas.yaml";
    ascent.open(open_opts);
    Node actions;

    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_timings)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing custom actions file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.json");

    string timings_file = "ascent_filter_times.csv";
    timings_file = conduit::utils::join_file_path(output_path,timings_file);

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);
    remove_test_file(timings_file);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "  [\n"
                              "    {\n"
                              "      \"action\": \"add_scenes\",\n"
                              "      \"scenes\": \n"
                              "      {\n"
                              "        \"s1\": \n"
                              "        {\n"
                              "          \"plots\":\n"
                              "          {\n"
                              "            \"p1\": \n"
                              "            {\n"
                              "              \"type\": \"pseudocolor\",\n"
                              "              \"field\": \"braid\"\n"
                              "            }\n"
                              "          },\n"
                              "          \"renders\": \n"
                              "          {\n"
                              "            \"r1\": \n"
                              "            {\n"
                              "              \"image_prefix\": \"" + output_file + "\"\n"
                              "            }\n"
                              "          }\n"
                              "        }\n"
                              "      }\n"
                              "    }\n"
                              "  ]\n";

    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent_opts["timings"] = "true";
    ascent_opts["default_dir"] = output_path;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
    // check to see if we created the timings
    EXPECT_TRUE(check_test_file(timings_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_timings_tear_updown)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing custom actions file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.json");

    string timings_file = "ascent_filter_times.csv";
    timings_file = conduit::utils::join_file_path(output_path,timings_file);

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);
    remove_test_file(timings_file);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "  [\n"
                              "    {\n"
                              "      \"action\": \"add_scenes\",\n"
                              "      \"scenes\": \n"
                              "      {\n"
                              "        \"s1\": \n"
                              "        {\n"
                              "          \"plots\":\n"
                              "          {\n"
                              "            \"p1\": \n"
                              "            {\n"
                              "              \"type\": \"pseudocolor\",\n"
                              "              \"field\": \"braid\"\n"
                              "            }\n"
                              "          },\n"
                              "          \"renders\": \n"
                              "          {\n"
                              "            \"r1\": \n"
                              "            {\n"
                              "              \"image_prefix\": \"" + output_file + "\"\n"
                              "            }\n"
                              "          }\n"
                              "        }\n"
                              "      }\n"
                              "    }\n"
                              "  ]\n";

    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    // some sims bring up and tear down ascent each time.
    // this exercises that path
    const int num_iters = 2;
    for(int i = 0; i < num_iters; ++i)
    {
      //
      // Run Ascent
      //

      Ascent ascent;

      Node ascent_opts;
      //ascent_opts["ascent_info"] = "verbose";
      ascent_opts["runtime/type"] = "ascent";
      ascent_opts["actions_file"] = output_actions;
      ascent_opts["timings"] = "true";
      ascent_opts["default_dir"] = output_path;
      ascent.open(ascent_opts);
      ascent.publish(data);
      conduit::Node blank_actions;
      ascent.execute(blank_actions);
      ascent.close();
    }

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
    // check to see if we created the timings
    EXPECT_TRUE(check_test_file(timings_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_actions_file)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing custom actions file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.json");

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "  [\n"
                              "    {\n"
                              "      \"action\": \"add_scenes\",\n"
                              "      \"scenes\": \n"
                              "      {\n"
                              "        \"s1\": \n"
                              "        {\n"
                              "          \"plots\":\n"
                              "          {\n"
                              "            \"p1\": \n"
                              "            {\n"
                              "              \"type\": \"pseudocolor\",\n"
                              "              \"field\": \"braid\"\n"
                              "            }\n"
                              "          },\n"
                              "          \"renders\": \n"
                              "          {\n"
                              "            \"r1\": \n"
                              "            {\n"
                              "              \"image_prefix\": \"" + output_file + "\"\n"
                              "            }\n"
                              "          }\n"
                              "        }\n"
                              "      }\n"
                              "    }\n"
                              "  ]\n";

    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}



//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_default_dir)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing custom actions yaml file");


    string output_path = prepare_output_dir();
    std::string dir_name = "bananas";
    output_path = conduit::utils::join_file_path(output_path,dir_name);
    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    string image_name = "tout_render_actions_img_yaml";
    string actions_name = "tout_render_actions.yaml";
    // full paths
    string output_file = conduit::utils::join_file_path(output_path,image_name);
    string output_actions = conduit::utils::join_file_path(output_path,actions_name);


    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "-\n"
                              "  action: add_scenes\n"
                              "  scenes:\n"
                              "        s1:\n"
                              "          plots:\n"
                              "            p1: \n"
                              "              type: pseudocolor\n"
                              "              field: braid\n"
                              "          renders:\n"
                              "            r1:\n"
                              "              image_prefix: " + image_name + "\n";


    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent_opts["default_dir"] = output_path;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_actions_yaml_file)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing custom actions yaml file");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_actions_img_yaml");
    string output_actions = conduit::utils::join_file_path(output_path,"tout_render_actions.yaml");

    // remove old images before rendering
    remove_test_image(output_file);
    remove_test_file(output_actions);


    //
    // Create the actions.
    //
    std::string actions_file = ""
                              "-\n"
                              "  action: add_scenes\n"
                              "  scenes:\n"
                              "        s1:\n"
                              "          plots:\n"
                              "            p1: \n"
                              "              type: pseudocolor\n"
                              "              field: braid\n"
                              "          renders:\n"
                              "            r1:\n"
                              "              image_prefix: " + output_file + "\n";


    std::ofstream file(output_actions);
    file<<actions_file;
    file.close();
    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["actions_file"] = output_actions;
    ascent.open(ascent_opts);
    ascent.publish(data);
    conduit::Node blank_actions;
    ascent.execute(blank_actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    EXPECT_TRUE(check_test_file(output_actions));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering)
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

    ASCENT_INFO("Testing field filtering");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_field_filtering");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of filtering fields not present in the actions file.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering_ghosts)
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

    // add a ghost field that 
    data["fields/ascent_ghosts"].set(data["fields/radial"]);
    
    float64_array gvals = data["fields/ascent_ghosts/values"].value();
    for(int i=0; i < gvals.number_of_elements(); i++)
    {
        // ghost every other element ....
        gvals[i] = i%2;
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing field filtering with ghosts");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_field_filtering_wghosts");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering_error)
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

    ASCENT_INFO("Testing field filtering error");

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = "will never happen";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    bool error = false;
    try
    {
       ascent.execute(actions);
    }
    catch(...)
    {
      error = true;
    }
    ascent.close();

    EXPECT_TRUE(error);
}
//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering_binning)
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
    data["fields/bananas"] = data["fields/braid"];
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    data["state/cycle"] = 100;
    data["state/time"] = 1.3;
    data["state/domain_id"] = 0;

    ASCENT_INFO("Testing field filtering error");

    //
    // Create the actions.
    //

    conduit::Node queries;


    std::string bin1 = "binning('braid', 'sum', [axis('x', [0, 2.5, 5, 7.5, 10])])";
    queries["e1/params/expression"] = bin1;
    queries["e1/params/name"] = "binnning";

    std::string bin2 = "binning('braid', 'sum', [axis('bananas', [0, 2.5, 5, 7.5, 10])])";
    queries["e2/params/expression"] = bin2;
    queries["e2/params/name"] = "binnning2";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
}

//-----------------------------------------------------------------------------
TEST(ascent_runtime_options, test_field_filtering_lineout)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with dray support
    if(n["runtimes/ascent/dray/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent Devil Ray  support disabled, skipping test");
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
    data["fields/bananas"] = data["fields/braid"];
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    data["state/cycle"] = 100;
    data["state/time"] = 1.3;
    data["state/domain_id"] = 0;

    ASCENT_INFO("Testing filtering field list");

    //
    // Create the actions.
    //

    conduit::Node queries;

    std::string lineout = "lineout(10, vector(0,1,1), vector(5,5,5), fields = ['braid', 'bananas'])";
    queries["e1/params/expression"] = lineout;
    queries["e1/params/name"] = "binnning";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_queries = actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
}

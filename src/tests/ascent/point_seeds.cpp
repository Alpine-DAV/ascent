//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_particle_advection.cpp
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

void testFilter(bool isStreamline)
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


    string output_path = ASCENT_T_BIN_DIR;

    ASCENT_INFO("Execute test from folder: " + output_path + "/ascent");
    output_path = conduit::utils::join_file_path(output_path,"ascent/output");
    ASCENT_INFO("Creating output folder: " + output_path);
    if(!conduit::utils::is_directory(output_path))
    {
        conduit::utils::create_directory(output_path);
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    data["state/cycle"] = 100;
    string output_file, msg;
    if (isStreamline)
    {
      ASCENT_INFO("Testing Streamline filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_streamline");
      msg = "An example of using the streamline flow filter.";
    }
    else
    {
      ASCENT_INFO("Testing Particle Advection filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_particle_advection");
      msg = "An example of using the particle_advection flow filter.";
    }
    ASCENT_INFO("POO POO");
    ASCENT_INFO(output_file);

    // remove old stuff before rendering
    remove_test_file(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    if (isStreamline)
      pipelines["pl1/f1/type"] = "streamline";
    else
      pipelines["pl1/f1/type"] = "particle_advection";

    // filter knobs
    conduit::Node &sl_params = pipelines["pl1/f1/params"];
    sl_params["field"] = "vel";
    sl_params["num_steps"] = 100;
    sl_params["step_size"] = 0.01;
    sl_params["seeds/type"] = "point";
    double loc[3] = {-5.,-5.,0.};
    sl_params["seeds/location"].set_float64_ptr(loc,3);
    if (isStreamline)
    {
      sl_params["rendering/enable_tubes"] = "true";
      sl_params["rendering/tube_capping"] = "false";
      //sl_params["rendering/tube_capping"] = "true";
      sl_params["rendering/tube_size"] = 0.07;
      sl_params["rendering/tube_sides"] = 3;
      sl_params["rendering/tube_value"] = 0.0;
      sl_params["rendering/output_field"] = "lines";
    }

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    std::string output_image;
    if(isStreamline)
    {
      string image_path = ASCENT_T_BIN_DIR;

      image_path = conduit::utils::join_file_path(image_path,"_output");
      output_image = conduit::utils::join_file_path(image_path,
                                      "tout_render_streamlines");
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      conduit::Node &scenes = add_plots["scenes"];
      scenes["s1/plots/p1/type"]  = "pseudocolor";
      scenes["s1/plots/p1/field"] = "lines";
      scenes["s1/plots/p1/pipeline"] = "pl1";
      scenes["s1/image_prefix"] = output_image;

      // remove old image before rendering
      remove_test_image(output_image);
    }
    actions.print();

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

   // check that we created the right output
   ASCENT_ACTIONS_DUMP(actions,output_file,msg);
   if(isStreamline)
   {
     output_image = output_image + "100";
     EXPECT_TRUE(check_test_file(output_image));
   }

   // clean up
   remove_test_file(output_file);
   conduit::utils::remove_directory(output_path);
}

//-----------------------------------------------------------------------------
TEST(ascent_streamline, test_streamline)
{
  testFilter(true);
}

TEST(ascent_particle_advection, test_particle_advection)
{
  testFilter(false);
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

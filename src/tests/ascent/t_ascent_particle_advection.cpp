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

void testFilterPoint(bool isStreamline)
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
      output_file = conduit::utils::join_file_path(output_path,"tout_streamline_point");
      msg = "An example of using the streamline flow filter.";
    }
    else
    {
      ASCENT_INFO("Testing Particle Advection filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_particle_advection_point");
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
    double loc[3] = {-0.826997, -5.62082, 3.57729};
    sl_params["seeds/location"].set_float64_ptr(loc,3);
    if (isStreamline)
    {
      sl_params["rendering/enable_tubes"] = "true";
      sl_params["rendering/tube_capping"] = "false";
      //sl_params["rendering/tube_capping"] = "true";
      sl_params["rendering/tube_size"] = 0.4;
      sl_params["rendering/tube_sides"] = 4;
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
                                      "tout_render_streamlines_point");
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      conduit::Node &scenes = add_plots["scenes"];
      scenes["s1/plots/p1/type"]  = "pseudocolor";
      scenes["s1/plots/p1/field"] = "lines";
      scenes["s1/plots/p1/pipeline"] = "pl1";
      scenes["s1/renders/r1/image_prefix"]   = output_image;
      double bounds[6] = {-10.0,10.0,-10.0,10.0,-10.0,10.0};
      scenes["s1/renders/r1/dataset_bounds"].set_float64_ptr(bounds,6); 

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
     std::cerr << "output image: " << output_image << std::endl;
     EXPECT_TRUE(check_test_image(output_image));
   }

   // clean up
   //remove_test_file(output_file);
   conduit::utils::remove_directory(output_path);
}

//-----------------------------------------------------------------------------
void testFilterPointList(bool isStreamline)
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
      output_file = conduit::utils::join_file_path(output_path,"tout_streamline_point_list");
      msg = "An example of using the streamline flow filter.";
    }
    else
    {
      ASCENT_INFO("Testing Particle Advection filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_particle_advection_point_list");
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
    sl_params["seeds/type"] = "point_list";
    double loc[60] = {-0.826997, -5.62082, 3.57729,0.388328, -9.30856, 0.594004,-8.66316, 3.73545, 8.60873,3.07838, 4.02381, 5.24396,-3.43532, 5.12821, -2.69323,5.06712, -8.54628, 7.69414,-0.445364, -4.50186, -6.66986,-8.78871, 0.0904583, -3.61934,-8.18534, -8.52502, -2.31716,-0.711083, -8.99832, 5.40409, 3.76911, 2.59087, 4.50824,-3.87356, 0.265474, 6.91963,-1.69211, -0.641653, -6.43345,-9.33892, -0.0303976, 4.96585,6.84079, -5.74497, -7.39145,-1.71413, 4.19639, -5.20178,3.04117, 3.62692, -2.24549,6.91151, 9.10818, -7.03697,1.29797, -0.229709, 9.2219,2.58538, 3.02507, 6.06146};
    sl_params["seeds/location"].set_float64_ptr(loc,60);
    if (isStreamline)
    {
      sl_params["rendering/enable_tubes"] = "true";
      sl_params["rendering/tube_capping"] = "false";
      //sl_params["rendering/tube_capping"] = "true";
      sl_params["rendering/tube_size"] = 0.4;
      sl_params["rendering/tube_sides"] = 4;
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
                                      "tout_render_streamlines_point_list");
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      conduit::Node &scenes = add_plots["scenes"];
      scenes["s1/plots/p1/type"]  = "pseudocolor";
      scenes["s1/plots/p1/field"] = "lines";
      scenes["s1/plots/p1/pipeline"] = "pl1";
      scenes["s1/renders/r1/image_prefix"]   = output_image;
      double bounds[6] = {-10.0,10.0,-10.0,10.0,-10.0,10.0};
      scenes["s1/renders/r1/dataset_bounds"].set_float64_ptr(bounds,6); 

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
     EXPECT_TRUE(check_test_image(output_image));
   }

   // clean up
   remove_test_file(output_file);
   conduit::utils::remove_directory(output_path);
}

//-----------------------------------------------------------------------------
void testFilterLine(bool isStreamline)
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
      output_file = conduit::utils::join_file_path(output_path,"tout_streamline_line");
      msg = "An example of using the streamline flow filter.";
    }
    else
    {
      ASCENT_INFO("Testing Particle Advection filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_particle_advection_line");
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
    sl_params["seeds/type"] = "line";
    double start[3] = {-9,-9,-9};
    double end[3] = {9,9,9};
    sl_params["seeds/start"].set_float64_ptr(start,3);
    sl_params["seeds/end"].set_float64_ptr(end,3);
    sl_params["seeds/num_seeds"] = 10;
    sl_params["seeds/sampling_type"] = "uniform";

    if (isStreamline)
    {
      sl_params["rendering/enable_tubes"] = "true";
      sl_params["rendering/tube_capping"] = "false";
      //sl_params["rendering/tube_capping"] = "true";
      sl_params["rendering/tube_size"] = 0.1;
      sl_params["rendering/tube_sides"] = 4;
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
                                      "tout_render_streamlines_line");
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      conduit::Node &scenes = add_plots["scenes"];
      scenes["s1/plots/p1/type"]  = "pseudocolor";
      scenes["s1/plots/p1/field"] = "lines";
      scenes["s1/plots/p1/pipeline"] = "pl1";
      scenes["s1/renders/r1/image_prefix"]   = output_image;
      double bounds[6] = {-10.0,10.0,-10.0,10.0,-10.0,10.0};
      scenes["s1/renders/r1/dataset_bounds"].set_float64_ptr(bounds,6); 

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
     EXPECT_TRUE(check_test_image(output_image));
   }

   // clean up
   remove_test_file(output_file);
   conduit::utils::remove_directory(output_path);
}


//-----------------------------------------------------------------------------
void testFilterBox(bool isStreamline)
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
      output_file = conduit::utils::join_file_path(output_path,"tout_streamline_box");
      msg = "An example of using the streamline flow filter.";
    }
    else
    {
      ASCENT_INFO("Testing Particle Advection filter");
      output_file = conduit::utils::join_file_path(output_path,"tout_particle_advection_box");
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
    sl_params["seeds/type"] = "box";
    //sl_params["seeds/sampling_space"] = "boundary";
    sl_params["seeds/sampling_space"] = "interior";
    sl_params["seeds/sampling_type"] = "uniform";
    //sl_params["seeds/sampling_type"] = "random";
    //sl_params["seeds/num_seeds"] = 20;
    sl_params["seeds/num_seeds_x"] = 4;
    sl_params["seeds/num_seeds_y"] = 4;
    sl_params["seeds/num_seeds_z"] = 4;
    double extents_x[2] = {-9.0,9.0};
    double extents_y[2] = {-9.0,9.0};
    double extents_z[2] = {-9.0,9.0};
    sl_params["seeds/extents_x"].set_float64_ptr(extents_x,2);
    sl_params["seeds/extents_y"].set_float64_ptr(extents_y,2);
    sl_params["seeds/extents_z"].set_float64_ptr(extents_z,2);
    if (isStreamline)
    {
      sl_params["rendering/enable_tubes"] = "true";
      sl_params["rendering/tube_capping"] = "false";
      //sl_params["rendering/tube_capping"] = "true";
      sl_params["rendering/tube_size"] = 0.1;
      sl_params["rendering/tube_sides"] = 4;
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
                                      "tout_render_streamlines_box");
      conduit::Node &add_plots = actions.append();
      add_plots["action"] = "add_scenes";
      conduit::Node &scenes = add_plots["scenes"];
      scenes["s1/plots/p1/type"]  = "pseudocolor";
      scenes["s1/plots/p1/field"] = "lines";
      scenes["s1/plots/p1/pipeline"] = "pl1";
      scenes["s1/renders/r1/image_prefix"]   = output_image;
      double bounds[6] = {-10.0,10.0,-10.0,10.0,-10.0,10.0};
      scenes["s1/renders/r1/dataset_bounds"].set_float64_ptr(bounds,6); 

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
     EXPECT_TRUE(check_test_image(output_image));
   }

   // clean up
   remove_test_file(output_file);
   conduit::utils::remove_directory(output_path);
}

//-----------------------------------------------------------------------------
//Point Test
TEST(ascent_streamline_point, test_streamline_point)
{
  testFilterPoint(true);
}

TEST(ascent_particle_advection_point, test_particle_advection_point)
{
  testFilterPoint(false);
}

//-----------------------------------------------------------------------------
//Point List Test
TEST(ascent_streamline_point_list, test_streamline_point_list)
{
  testFilterPointList(true);
}

TEST(ascent_particle_advection_point_list, test_particle_advection_point_list)
{
  testFilterPointList(false);
}
//-----------------------------------------------------------------------------
//Line Test
TEST(ascent_streamline_line, test_streamline_line)
{
  testFilterLine(true);
}

TEST(ascent_particle_advection_line, test_particle_advection_line)
{
  testFilterLine(false);
}
//-----------------------------------------------------------------------------
//Box Test
TEST(ascent_streamline_box, test_streamline_box)
{
  testFilterBox(true);
}

TEST(ascent_particle_advection_box, test_particle_advection_box)
{
  testFilterBox(false);
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

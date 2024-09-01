//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_uniform_grid.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mpi_mesh_examples.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


int NUM_DOMAINS = 8;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_uniform_uniform_grid, test_mpi_uniform_grid)
{
    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

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
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(NUM_DOMAINS,data,comm);

    int local_domains = data.number_of_children();
    for(int i = 0; i < local_domains; ++i)
    {
      double x_min, x_max, y_min, y_max;
      Node vel_field;
      vel_field["association"] = "vertex";
      vel_field["topology"] = "topo";
      Node &dom = data.child(i);
      Node &values = dom["coordsets/coords/values"];
      Node &x = values["x"];
      Node &y = values["y"];
      int x_nvalues = x.dtype().number_of_elements(); 
      int y_nvalues = y.dtype().number_of_elements(); 
      double * x_array = x.value();
      double * y_array = y.value();
      x_max = x_array[x_nvalues-1];
      x_min = x_array[0];
      y_max = y_array[y_nvalues-1];
      y_min = y_array[0];
      int num_values = x_nvalues * y_nvalues;
      double x_step = (x_max - x_min)/num_values;
      double y_step = (y_max - y_min)/num_values;
      std::vector<double> u_array;
      std::vector<double> v_array;
      for(int i = 0; i < num_values; ++i)
      {
        u_array.push_back(x_min + x_step*i);
        v_array.push_back(y_min + y_step*i);
      }
      vel_field["values/u"].set(u_array);
      vel_field["values/v"].set(v_array);
      dom["fields/vel"] = vel_field;
    }
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing mpi uniform grid of conduit::blueprint spiral input\n");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_uniform_grid");
    string image_file = conduit::utils::join_file_path(output_path,"tout_mpi_uniform_grid10");

    // remove old images before rendering
    if(par_rank == 0)
      remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "uniform_grid";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["invalid_value"] = -100.0;      
    params["field"] = "dist"; 

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "dist";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = image_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

//    conduit::Node extracts;
//
//    extracts["e1/type"]  = "relay";
//    extracts["e1/params/path"] = output_file;
//    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//    conduit::Node &add_ext= actions.append();
//    add_ext["action"] = "add_extracts";
//    add_ext["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    if(par_rank == 0)
    {
      EXPECT_TRUE(check_test_image(output_file));
      std::string msg = "An example of using the mpi uniform grid filter.";
      ASCENT_ACTIONS_DUMP(actions,output_file,msg);
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_babelflow_comp_mpi.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_hola.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


//-----------------------------------------------------------------------------
TEST(ascent_babelflow_comp_mpi, test_babelflow_comp_radixk)
{

    // the vtkm runtime is currently our only rendering runtime
    conduit::Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    //
    // Create an example mesh.
    //
    conduit::Node data, hola_opts, verify_info;
    hola_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_002200.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
    // verify_info.print();

    ASCENT_INFO("Testing BFlow compositing (radix-k) extract");

    //
    // Make sure the output dir exists
    // 
    string output_path = "";
    if (par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path, "tout_babelflow_comp_mpi_radix_k_");

    // Remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "dray_project_colors_2d";
    // Set filter params
    conduit::Node& filt_params = pipelines["pl1/f1/params/"];
    filt_params["field"] = "density";
    filt_params["image_width"] = 1024;
    filt_params["image_height"] = 1024;
    filt_params["color_table/name"] = "cool2warm";
    filt_params["camera/azimuth"] = -30;
    filt_params["camera/elevation"] = 35;

    conduit::Node extracts;
    extracts["e1/type"] = "bflow_comp";
    extracts["e1/pipeline"] = "pl1";
    conduit::Node& comp_params = extracts["e1/params/"];
    comp_params["color_field"] = "colors";
    comp_params["depth_field"] = "depth";
    comp_params["image_prefix"] = output_file;
    comp_params["fanin"] = int64_t(2);
    comp_params["compositing"] = int64_t(2);     // 2 means radix-k compositing
    std::vector<int64_t> radices({2, 4});
    comp_params["radices"].set_int64_vector(radices);

    conduit::Node actions;

    // Add the pipelines
    conduit::Node& add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    // Add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    if (par_rank == 0)
    {
        actions.print();
    }


    //
    // Run Ascent
    //
    ascent::Ascent ascent;

    conduit::Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    // Check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.1, "2200"));
}

//-----------------------------------------------------------------------------
TEST(ascent_babelflow_comp_mpi, test_babelflow_comp_reduce)
{

    // the vtkm runtime is currently our only rendering runtime
    conduit::Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    //
    // Create an example mesh.
    //
    conduit::Node data, hola_opts, verify_info;
    hola_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    hola_opts["root_file"] = test_data_file("taylor_green.cycle_002200.root");
    ascent::hola("relay/blueprint/mesh", hola_opts, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
    // verify_info.print();

    ASCENT_INFO("Testing BFlow compositing (reduce) extract");

    //
    // Make sure the output dir exists
    // 
    string output_path = "";
    if (par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path, "tout_babelflow_comp_mpi_reduce_");

    // Remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "dray_project_colors_2d";
    // Set filter params
    conduit::Node& filt_params = pipelines["pl1/f1/params/"];
    filt_params["field"] = "density";
    filt_params["color_table/name"] = "cool2warm";
    filt_params["camera/azimuth"] = -30;
    filt_params["camera/elevation"] = 35;

    conduit::Node extracts;
    extracts["e1/type"] = "bflow_comp";
    extracts["e1/pipeline"] = "pl1";
    conduit::Node& comp_params = extracts["e1/params/"];
    comp_params["color_field"] = "colors";
    comp_params["depth_field"] = "depth";
    comp_params["image_prefix"] = output_file;
    comp_params["fanin"] = int64_t(2);
    comp_params["compositing"] = int64_t(0);     // 2 means radix-k compositing

    conduit::Node actions;

    // Add the pipelines
    conduit::Node& add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    // Add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    if (par_rank == 0)
    {
        actions.print();
    }
    

    //
    // Run Ascent
    //
    ascent::Ascent ascent;

    conduit::Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    // Check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.1, "2200"));
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

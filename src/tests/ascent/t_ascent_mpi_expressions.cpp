//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_slice.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <iostream>
#include <math.h>


#include <ascent_expression_eval.hpp>
#include <flow_workspace.hpp>

#include <mpi.h>

#include <conduit_blueprint.hpp>
#include "conduit_relay_mpi_io_blueprint.hpp"

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_expressions, mpi_binning_mesh)
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
    // Create the data.
    //
    Node data, verify_info;
    int dims = 2;
    create_3d_example_dataset(data,dims,par_rank,par_size);
    Node multi_dom;
    blueprint::mesh::to_multi_domain(data, multi_dom);

    conduit::blueprint::mesh::verify(data,verify_info);

    string output_path = "";
    if(par_rank == 0)
    {
      output_path = prepare_output_dir();
    }
    else
    {
      output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_binning_input");

    Node opts;

    conduit::relay::mpi::io::blueprint::save_mesh(data,
                                                  output_file,
                                                  "hdf5",
                                                  opts,
                                                  comm);

    std::string output_file_mesh = conduit::utils::join_file_path(output_path,"tout_mpi_binning_mesh");
    std::string output_file_bins = conduit::utils::join_file_path(output_path,"tout_mpi_binning_bins");

    // remove old images before rendering
    remove_test_image(output_file_mesh);
    remove_test_image(output_file_bins);

    conduit::Node pipelines;
    
    // use same axes for both pipelines
    conduit::Node axes;
    conduit::Node &axis0 = axes.append();
    axis0["var"] = "x";
    axis0["num_bins"] = 2;
    axis0["min_val"] = -1.0;
    axis0["max_val"] = 1.0;
    axis0["clamp"] = 1;
    
    conduit::Node &axis1 = axes.append();
    axis1["var"] = "y";
    axis1["num_bins"] = 4;
    axis1["clamp"] = 0;

    // pipeline 1 (result is on mesh)
    pipelines["pl1/f1/type"] = "binning";
    // filter knobs
    conduit::Node &pl1_params = pipelines["pl1/f1/params"];
    pl1_params["reduction_op"] = "sum";
    pl1_params["reduction_field"] = "ones_ele";
    pl1_params["output_field"] = "binning";
    pl1_params["output_type"] = "mesh";
    pl1_params["axes"].set(axes);

    // pipeline 2 (result is bins)
    pipelines["pl2/f1/type"] = "binning";
    // filter knobs
    conduit::Node &pl2_params = pipelines["pl2/f1/params"];
    pl2_params["reduction_op"] = "sum";
    pl2_params["reduction_field"] = "ones_ele";
    pl2_params["output_field"] = "binning";
    pl2_params["output_type"] = "bins";
    pl2_params["axes"].set(axes);

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "binning";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/plots/p2/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file_mesh;

    scenes["s2/plots/p1/type"] = "pseudocolor";
    scenes["s2/plots/p1/field"] = "binning";
    scenes["s2/plots/p1/pipeline"] = "pl2";
    scenes["s2/plots/p2/type"] = "mesh";
    scenes["s2/plots/p2/pipeline"] = "pl2";
    scenes["s2/image_prefix"] = output_file_bins;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    // add extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/path"] = conduit::utils::join_file_path(output_path,
                                                                "tout_mpi_binning_mesh_result_extract");

    extracts["e2/type"] = "relay";
    extracts["e2/pipeline"] = "pl2";
    extracts["e2/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e2/params/path"] = conduit::utils::join_file_path(output_path,
                                                                "tout_mpi_binning_bins_result_extract");


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    if(par_rank == 0)
    {
      EXPECT_TRUE(check_test_image(output_file_mesh, 0.1));
      EXPECT_TRUE(check_test_image(output_file_bins, 0.1));
    }
}
//
// //-----------------------------------------------------------------------------
// TEST(ascent_mpi_expressions, mpi_binning_bins)
// {
//     // the vtkm runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with vtkm support
//     if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent vtkm support disabled, skipping test");
//         return;
//     }
//     //
//     // Set Up MPI
//     //
//     int par_rank;
//     int par_size;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     MPI_Comm_rank(comm, &par_rank);
//     MPI_Comm_size(comm, &par_size);
//
//     ASCENT_INFO("Rank "
//                   << par_rank
//                   << " of "
//                   << par_size
//                   << " reporting");
//     //
//     // Create the data.
//     //
//     Node data, verify_info;
//     // this is per mpi task
//     int dims = 3;
//     create_3d_example_dataset(data,dims,par_rank,par_size);
//     Node multi_dom;
//     blueprint::mesh::to_multi_domain(data, multi_dom);
//
//     conduit::blueprint::mesh::verify(data,verify_info);
//
//     string output_path = "";
//     if(par_rank == 0)
//     {
//       output_path = prepare_output_dir();
//     }
//     else
//     {
//       output_path = output_dir();
//     }
//
//     string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_binning_bins_input");
//
//     Node opts;
//
//     conduit::relay::mpi::io::blueprint::save_mesh(data,
//                                                   output_file,
//                                                   "hdf5",
//                                                   opts,
//                                                   comm);
//
//     // for render
//     output_file = conduit::utils::join_file_path(output_path,"tout_mpi_binning_bins");
//
//     // remove old images before rendering
//     remove_test_image(output_file);
//
//
//     conduit::Node pipelines;
//     // pipeline 1
//     pipelines["pl1/f1/type"] = "binning";
//     // filter knobs
//     conduit::Node &params = pipelines["pl1/f1/params"];
//     params["reduction_op"] = "count";
//     params["var"] = "radial_ele";
//     params["output_field"] = "binning";
//     // reduced dataset of only the bins
//     params["output_type"] = "bins";
//
//     conduit::Node &axis0 = params["axes"].append();
//     axis0["var"] = "x";
//     axis0["num_bins"] = 3;
//     axis0["min_val"] = -5.0;
//     axis0["max_val"] = 5.0;
//     axis0["clamp"] = 1;
//     // axis0["var"] = "x";
//     // axis0["num_bins"] = 10;
//     // axis0["min_val"] = -10.0;
//     // axis0["max_val"] = 10.0;
//     // axis0["clamp"] = 1;
//     //
//     conduit::Node &axis1 = params["axes"].append();
//     axis1["var"] = "y";
//     axis1["num_bins"] = 6;
//     axis1["clamp"] = 0;
//     //
//     conduit::Node &axis2 = params["axes"].append();
//     axis2["var"] = "z";
//     axis2["num_bins"] = 1;
//     axis2["clamp"] = 1;
//
//     conduit::Node scenes;
//     scenes["s1/plots/p1/type"] = "pseudocolor";
//     scenes["s1/plots/p1/field"] = "radial_ele";
//     scenes["s1/image_prefix"] = output_file + "_input";
//
//     scenes["s2/plots/p1/type"] = "pseudocolor";
//     scenes["s2/plots/p1/field"] = "binning";
//     scenes["s2/plots/p1/pipeline"] = "pl1";
//     scenes["s2/image_prefix"] = output_file;
//
//     conduit::Node actions;
//     // add the pipeline
//     conduit::Node &add_pipelines= actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     add_pipelines["pipelines"] = pipelines;
//     // add the scenes
//     conduit::Node &add_scenes= actions.append();
//     add_scenes["action"] = "add_scenes";
//     add_scenes["scenes"] = scenes;
//
//     // add extract
//     conduit::Node &add_extracts = actions.append();
//     add_extracts["action"] = "add_extracts";
//     conduit::Node &extracts = add_extracts["extracts"];
//
//     extracts["e1/type"] = "relay";
//     extracts["e1/pipeline"] = "pl1";
//     extracts["e1/params/path"] = conduit::utils::join_file_path(output_path,
//                                                                 "tout_mpi_binning_bins_result_extract");
//     extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//
//     Node ascent_opts;
//     ascent_opts["runtime/type"] = "ascent";
//     ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
//     ascent.open(ascent_opts);
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     if(par_rank == 0)
//     {
//       EXPECT_TRUE(check_test_image(output_file + "_input", 0.1));
//       EXPECT_TRUE(check_test_image(output_file, 0.1));
//     }
// }

//-----------------------------------------------------------------------------
TEST(ascent_mpi_expressions, mpi_expressions)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

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
    // Create the data.
    //
    Node data, verify_info;
    int dims = 32;
    create_3d_example_dataset(data,dims,par_rank,par_size);
    Node multi_dom;
    blueprint::mesh::to_multi_domain(data, multi_dom);

    conduit::blueprint::mesh::verify(data,verify_info);

    flow::Workspace::set_default_mpi_comm(MPI_Comm_c2f(comm));

    runtime::expressions::register_builtin();
    runtime::expressions::ExpressionEval eval(&multi_dom);
    std::string expr = "magnitude(max(field('radial_vert')).position)";
    conduit::Node res = eval.evaluate(expr);

    EXPECT_EQ(res["type"].as_string(), "double");

    if(par_rank == 0)
    {
      res.print();
    }
}

int main(int argc, char* argv[])
{
    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}

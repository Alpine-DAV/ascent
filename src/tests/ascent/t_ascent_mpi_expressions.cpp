//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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
    int dims = 32;
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

    string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_binning_mesh");

    // remove old images before rendering
    remove_test_image(output_file);


    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "binning";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["reduction_op"] = "sum";
    params["var"] = "radial_vert";
    params["output_field"] = "binning";
    params["output_type"] = "mesh";

    conduit::Node &axis0 = params["axes"].append();
    axis0["var"] = "x";
    axis0["num_bins"] = 10;
    axis0["min_val"] = -10.0;
    axis0["max_val"] = 10.0;
    axis0["clamp"] = 1;

    conduit::Node &axis1 = params["axes"].append();
    axis1["var"] = "y";
    axis1["num_bins"] = 10;
    axis1["clamp"] = 0;

    conduit::Node &axis2 = params["axes"].append();
    axis2["var"] = "z";
    axis2["num_bins"] = 10;
    axis2["clamp"] = 10;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "binning";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
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
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    if(par_rank == 0)
    {
      EXPECT_TRUE(check_test_image(output_file, 0.1));
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_expressions, mpi_binning_bins)
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
    int dims = 32;
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

    string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_binning_bins");

    // remove old images before rendering
    remove_test_image(output_file);


    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "binning";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["reduction_op"] = "sum";
    params["var"] = "radial_vert";
    params["output_field"] = "binning";
    // reduced dataset of only the bins
    params["output_type"] = "bins";

    conduit::Node &axis0 = params["axes"].append();
    axis0["var"] = "x";
    axis0["num_bins"] = 10;
    axis0["min_val"] = -10.0;
    axis0["max_val"] = 10.0;
    axis0["clamp"] = 1;

    conduit::Node &axis1 = params["axes"].append();
    axis1["var"] = "y";
    axis1["num_bins"] = 10;
    axis1["clamp"] = 0;

    conduit::Node &axis2 = params["axes"].append();
    axis2["var"] = "z";
    axis2["num_bins"] = 10;
    axis2["clamp"] = 10;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "binning";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
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
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    if(par_rank == 0)
    {
      EXPECT_TRUE(check_test_image(output_file, 0.1));
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_expressions, mpi_expressoins)
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_render_2d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_render_2d, test_render_mpi_2d_default_runtime)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 2D MPI "
                      "runtime test");
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
    create_2d_example_dataset(data,par_rank,par_size);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_render_mpi_2d_default_runtime");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial_vert";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_render_2d, test_render_mpi_2d_uniform_default_runtime)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 2D MPI "
                      "runtime test");
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

    conduit::blueprint::mesh::examples::braid("uniform",
                                          10,
                                          10,
                                          1,
                                          data);
    // shift data for rank > 1
    double x_origin = par_rank * 20 - 10;

    data["state/domain_id"] = par_rank;
    data["state/cycle"] = 100;
    data["coordsets/coords/origin/x"] = x_origin;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_render_mpi_2d_uniform_default_runtime");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}


//-----------------------------------------------------------------------------
TEST(ascent_mpi_render_2d, test_render_mpi_2d_small_example)
{
    // example that demonstrates rendering bug described in
    // https://github.com/Alpine-DAV/ascent/issues/992
    // derived from example code from WillTrojak, posted in #992

    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 2D MPI "
                      "runtime test");
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

    // example mesh from #992
    conduit_float32 x[8], y[8], u[8];
    conduit_int32 conn[8];
    conduit_float32 scale = 4;

    // Simple 2 rank example with 2 elements per rank.
    if (par_rank == 0)
    {
        x[4] = -1.;
        x[5] =  0.;
        x[6] = -1.;
        x[7] =  0.;
        y[4] =  0.;
        y[5] =  0.;
        y[6] =  1.;
        y[7] =  1.;

        x[0] =  0.;
        x[1] =  1.;
        x[2] =  0.;
        x[3] =  1.;
        y[0] =  0.;
        y[1] =  0.;
        y[2] =  1.;
        y[3] =  1.;
        conn[0] = 0;
        conn[1] = 1;
        conn[2] = 3;
        conn[3] = 2;
        conn[4] = 4;
        conn[5] = 5;
        conn[6] = 7;
        conn[7] = 6;
    }
    else if (par_rank == 1)
    {
        x[0] =  0.;
        x[1] =  1.;
        x[2] =  0.;
        x[3] =  1.;
        y[0] = -1.;
        y[1] = -1.;
        y[2] =  0.;
        y[3] =  0.;

        x[4] = -1.;
        x[5] =  0.;
        x[6] = -1.;
        x[7] =  0.;
        y[4] = -1.;
        y[5] = -1.;
        y[6] =  0.;
        y[7] =  0.;
        conn[0] = 0;
        conn[1] = 1;
        conn[2] = 3;
        conn[3] = 2;
        conn[4] = 4;
        conn[5] = 5;
        conn[6] = 7;
        conn[7] = 6;
    }

    for (int i=0; i<8; i++)
    {
        u[i] = sin(x[i]) * cos(y[i]);
        x[i] = scale*x[i];
        y[i] = scale*y[i];
    }

    conduit::Node mesh;
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set(x,8);
    mesh["coordsets/coords/values/y"].set(y,8);
    mesh["topologies/mesh/type"] = "unstructured";
    mesh["topologies/mesh/coordset"] = "coords";
    mesh["topologies/mesh/elements/shape"] = "quad";
    mesh["topologies/mesh/elements/connectivity"].set(conn,8);
    mesh["state/domain_id"] = (conduit_int32)par_rank;
    mesh["fields/u/association"] = "vertex";
    mesh["fields/u/topology"] = "mesh";
    mesh["fields/u/values"].set(u,8);
    
    std::string actions_yaml = R"ST(
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "mesh"
      image_name: "tout_rmpi_2d_scene_mesh"
    s2: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "u"
      image_name: "tout_rmpi_2d_scene_u"
)ST";


    conduit::Node actions;
    actions.parse(actions_yaml,"yaml");

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    // setup outputs to write in ascent's test output dir
    string mesh_output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_mpi_2d_scene_mesh");

    // remove old images before rendering
    remove_test_image(mesh_output_file);

    actions[0]["scenes/s1/image_prefix"] = mesh_output_file;

    string u_output_file = conduit::utils::join_file_path(output_path,
                                                 "tout_render_mpi_2d_scene_u");

    // remove old images before rendering
    remove_test_image(u_output_file);

    actions[0]["scenes/s2/image_prefix"] = u_output_file;

    ascent::Ascent a;
    conduit::Node opts;
    opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
    a.open(opts);
    a.publish(mesh);
    a.execute(actions);
    a.close();

    MPI_Barrier(comm);

    EXPECT_TRUE(check_test_image(mesh_output_file));
    EXPECT_TRUE(check_test_image(u_output_file));
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



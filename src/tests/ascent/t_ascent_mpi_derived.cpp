//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_derived.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 20;

void create_test_data(Node &data)
{
  int par_rank;
  int par_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &par_rank);
  MPI_Comm_size(comm, &par_size);

  data.reset();
  if(par_rank == 0)
  {
    Node &mesh = data.append();
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);
    mesh["state/domain_id"] = 0;
  }
  else
  {
    Node &mesh = data.append();
    conduit::blueprint::mesh::examples::braid("points",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              mesh);

    // add a field that wont exist on both domains
    mesh["fields/bananas"] = mesh["fields/braid"];
    mesh["state/domain_id"] = 1;
    //std::cout<<mesh.to_summary_string()<<"\n";
  }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_derived, mpi_derived)
{
  Node n;
  ascent::about(n);
  // only run this test if ascent was built with jit support
  if(n["runtimes/ascent/jit/status"].as_string() == "disabled")
  {
      ASCENT_INFO("Ascent JIT support disabled, skipping test\n");
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
  create_test_data(data);

  conduit::blueprint::mesh::verify(data,verify_info);

  flow::Workspace::set_default_mpi_comm(MPI_Comm_c2f(comm));

  runtime::expressions::register_builtin();
  runtime::expressions::ExpressionEval eval(&data);


  std::string expr = "magnitude(max(field('bananas')).position)";
  conduit::Node res = eval.evaluate(expr);

  EXPECT_EQ(res["type"].as_string(), "double");

  if(par_rank == 0)
  {
    res.print();
  }

  // create an expression that will throw an exception (points volume)
  // and do an MPI reduction that will hang (reduce) if exceptions aren't
  // handled correctly
  expr = "sum(topo('mesh').cell.volume)";
  // normally the ascent runtime would catch this so we have to catch
  EXPECT_ANY_THROW(res = eval.evaluate(expr));

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

#include <iostream>
#include <cfloat>
#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <mpi.h>

using namespace ascent;
using namespace conduit;


int main(int argc, char **argv)
{
  using namespace std;
  int provided;
  auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(err == MPI_SUCCESS);

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  Ascent a;
  conduit::Node ascent_opt;
  ascent_opt["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);;
  ascent_opt["runtime/type"] = "ascent";
  a.open(ascent_opt);

  // create example mesh using conduit blueprint
  Node mesh;
  int xdim = 128, ydim = 128, zdim = 128;
  conduit::blueprint::mesh::examples::braid("hexs",
                                            xdim,
                                            ydim,
                                            zdim,
                                            mesh);
  // publish mesh to ascent
  a.publish(mesh);
  // publish mesh to ascent

  Node extract;
  extract["e1/type"] = "babelflow";
  extract["e1/params/task"] = "pmt";
  extract["e1/params/mpi_size"] = mpi_size;
  extract["e1/params/mpi_rank"] = mpi_rank;
  extract["e1/params/data_path"] = "fields/braid/values";
  extract["e1/params/xdim"] = xdim;
  extract["e1/params/ydim"] = ydim;
  extract["e1/params/zdim"] = zdim;
  extract["e1/params/bxdim"] = 64;
  extract["e1/params/bydim"] = 64;
  extract["e1/params/bzdim"] = 64;
  extract["e1/params/fanin"] = 2;
  extract["e1/params/threshold"] = -FLT_MAX;
  extract["e1/params/mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);

  Node action;
  Node &add_extract = action.append();
  add_extract["action"] = "add_extracts";
  add_extract["extracts"] = extract;

  action.append()["action"] = "execute";
  a.execute(action);
  a.close();
  MPI_Finalize();
  return 0;

}
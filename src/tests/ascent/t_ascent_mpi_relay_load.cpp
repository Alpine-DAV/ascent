//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_relay_load.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_fmt/conduit_fmt.h"

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_relay_load, test_load)
{

    Node n;
    ascent::about(n);
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    // Set Up MPI
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

    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 100;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string load_mesh_path = conduit::utils::join_file_path(output_path,"tout_relay_load_input");

    // save dataset
    conduit::relay::mpi::io::blueprint::save_mesh(data,load_mesh_path,"hdf5",comm);

    string output_extract_file = conduit::utils::join_file_path(output_path,"tout_relay_load_result");
    string output_render_file = conduit::utils::join_file_path(output_path,"tout_relay_load_render");
    // remove old files before rendering
    remove_test_image(output_extract_file);
    remove_test_image(output_render_file);

    // create a basic point mesh on rank zero hand to ascent
    data.reset();
    if(par_rank == 0)
    {
        data["coordsets/pt_coords/type"] = "explicit";
        data["coordsets/pt_coords/values/x"] = {0.0, 1.0, 2.0};
        data["coordsets/pt_coords/values/y"] = {0.0, 5.0, 0.0};
        data["coordsets/pt_coords/values/z"] = {-1.0, 0.0, 1.0};
        data["topologies/pt_topo/type"] = "points";
        data["topologies/pt_topo/coordset"] = "pt_coords";
        data.print();
    }

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    load_pipeline: 
      f1: 
        type: "load"
        params:
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: load_pipeline
      type: "relay"
      params:
        protocol: "hdf5"
- 
  action: "add_scenes"
  scenes: 
    s1:
      plots:
        p1:
          type: "pseudocolor"
          field: "rank_ele"
          pipeline: "load_pipeline"
      renders:
        r1:
          camera:
            azimuth: 45
)xyzxyz";

    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    // fill in dynamic parts
    actions[0]["pipelines/load_pipeline/f1/params/path"] = load_mesh_path + ".cycle_000100.root";
    actions[1]["extracts/e1/params/path"] =  output_extract_file;
    actions[2]["scenes/s1/renders/r1/image_name"] =  output_render_file;

    std::cout << actions.to_yaml() << std::endl;
    
    Ascent ascent;
    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_render_file,1e-7,""));
    std::string msg = "An mpi example of loading and plotting an addional blueprint mesh from the file system.";
    ASCENT_ACTIONS_DUMP(actions,output_render_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_mpi_relay_load, test_load_cache)
{

    Node n;
    ascent::about(n);
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    // Set Up MPI
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

    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 100;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    Ascent ascent;
    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);

    std::string acts_str = R"xyzxyz(
- 
  action: "add_extracts"
  extracts: 
    e1:
      type: "cache"
      params:
        name: "mymesh"
)xyzxyz";

    conduit::Node actions;
    actions.parse(acts_str);

    std::cout << actions.to_yaml() << std::endl;
    ascent.execute(actions);

    string output_extract_file = conduit::utils::join_file_path(output_path,"tout_relay_load_cache_result");
    string output_render_file = conduit::utils::join_file_path(output_path,"tout_relay_load_cache_render");

    // remove old files before rendering
    remove_test_image(output_extract_file);
    remove_test_image(output_render_file);

    // create a basic point mesh on rank zero hand to ascent
    data.reset();
    if(par_rank == 0)
    {
        data["coordsets/pt_coords/type"] = "explicit";
        data["coordsets/pt_coords/values/x"] = {0.0, 1.0, 2.0};
        data["coordsets/pt_coords/values/y"] = {0.0, 5.0, 0.0};
        data["coordsets/pt_coords/values/z"] = {-1.0, 0.0, 1.0};
        data["topologies/pt_topo/type"] = "points";
        data["topologies/pt_topo/coordset"] = "pt_coords";
        data.print();
    }

    acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    load_pipeline: 
      f1: 
        type: "load"
        params:
          path: "cache:mymesh"
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: load_pipeline
      type: "relay"
- 
  action: "add_scenes"
  scenes: 
    s1:
      plots:
        p1:
          type: "pseudocolor"
          field: "rank_ele"
          pipeline: "load_pipeline"
      renders:
        r1:
          camera:
            azimuth: -45
)xyzxyz";

    actions.parse(acts_str,"yaml");
    // fill in dynamic parts
    actions[1]["extracts/e1/params/path"] =  output_extract_file;
    actions[2]["scenes/s1/renders/r1/image_name"] =  output_render_file;

    std::cout << actions.to_yaml() << std::endl;

    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_render_file,1e-7,""));
    std::string msg = "An mpi example of loading and plotting an addional blueprint mesh from the file system.";
    ASCENT_ACTIONS_DUMP(actions,output_render_file,msg);
}



//
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



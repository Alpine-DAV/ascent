//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_relay.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include "conduit_fmt/conduit_fmt.h"

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_hdf5)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract");
    string output_root = output_file + ".cycle_000100.root";

    // remove old images before rendering
    remove_test_image(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_hdf5_2)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract");
    string output_root = output_file + ".cycle_000100.root";

    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));

}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_hdf5_opts)
{
    Node n;
    ascent::about(n);

    ASCENT_INFO("Testing relay extract in serial with hdf5 gzip opts");

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::basic("uniform",
                                              1001,
                                              11,
                                              11,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    // change values -- zeros compress nicely ....
    float64_array vals = data["fields/field/values"].value();
    vals.fill(0);
    index_t nele = vals.number_of_elements();

    std::cout << "Field Number of Elements: " << nele << std::endl;
    std::cout << "Field Number of Bytes:    " << nele * 8 << std::endl;

    string output_path = prepare_output_dir();

    string output_file_pre = conduit::utils::join_file_path(output_path,"tout_relay_hdf5_opts_000");
    string output_root_pre = output_file_pre + ".cycle_000100.root";

    string output_file_opts = conduit::utils::join_file_path(output_path,"tout_relay_hdf5_opts_001");
    string output_root_opts = output_file_opts + ".cycle_000100.root";

    string output_file_post = conduit::utils::join_file_path(output_path,"tout_relay_hdf5_opts_002");
    string output_root_post = output_file_post + ".cycle_000100.root";


    // remove old files before executing
    remove_test_file(output_root_pre);
    remove_test_file(output_root_opts);
    remove_test_file(output_root_post);

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    Node &extracts = add_extracts["extracts"];

    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file_pre;
    extracts["e1/params/protocol"] = "hdf5";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);

    // run again with hdf5 options that will trigger
    // compression for the known inputs in this tests
    extracts["e1/params/path"] = output_file_opts;
    extracts["e1/params/hdf5_options/chunking/enabled"]  = "true";
    // Note: https://github.com/LLNL/conduit/issues/1137
    extracts["e1/params/hdf5_options/chunking/threshold"]  = 800000-1;
    extracts["e1/params/hdf5_options/chunking/chunk_size"] = 800000;
    extracts["e1/params/hdf5_options/chunking/compression/level"] = 9;

    ascent.execute(actions);

    // back to default
    extracts["e1/params/path"] = output_file_post;
    extracts.remove("e1/params/hdf5_options");

    ascent.execute(actions);

    ascent.close();

    index_t fsize_pre  = conduit::utils::file_size(output_root_pre);
    index_t fsize_opts = conduit::utils::file_size(output_root_opts);
    index_t fsize_post = conduit::utils::file_size(output_root_post);
    // pre and post should be the same
    EXPECT_EQ(fsize_pre,fsize_post);

    // compression should help us out
    EXPECT_TRUE(fsize_pre > fsize_opts);

    std::cout << "pre file:  " << output_root_pre << std::endl;
    std::cout << "opts file: " << output_root_opts << std::endl;
    std::cout << "post file: " << output_root_post << std::endl;

    std::cout << "pre  size: " << fsize_pre  << std::endl;
    std::cout << "opts size: " << fsize_opts << std::endl;
    std::cout << "post size: " << fsize_post << std::endl;

}//

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_json)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial (json)");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_json");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/json";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_json_2)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial (json)");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_json_2");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "json";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_yaml)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial (yaml)");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_yaml");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/yaml";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_yaml_2)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial (yaml)");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_yaml_2");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "yaml";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // make sure the expected root file exists
    EXPECT_TRUE(conduit::utils::is_file(output_root));
}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_field_select_nestsets)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    blueprint::mesh::examples::julia_nestsets_complex(EXAMPLE_MESH_SIDE_DIM,
                                                      EXAMPLE_MESH_SIDE_DIM,
                                                      -2.0,  2.0, // x range
                                                      -2.0,  2.0, // y range
                                                      0.285, 0.01, // c value
                                                      1, // amr levels
                                                      data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial with nestsets");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_nestsets");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node & extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append() = "iters";

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_topo_select)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    blueprint::mesh::examples::julia_nestsets_complex(EXAMPLE_MESH_SIDE_DIM,
                                                      EXAMPLE_MESH_SIDE_DIM,
                                                      -2.0,  2.0, // x range
                                                      -2.0,  2.0, // y range
                                                      0.285, 0.01, // c value
                                                      1, // amr levels
                                                      data);

    blueprint::mesh::examples::braid("quads",10,10,0,data["domain_Z"]);
    // rename the coordset to avoid conflict
    data["domain_Z/coordsets"].rename_child("coords","braid_coords");
    data["domain_Z/topologies"][0]["coordset"] = "braid_coords";

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract topo selection");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_topo_select_case_1");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    // keep the julia data only 
    extracts["e1/params/topologies"].append() = "topo";

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    EXPECT_TRUE(conduit::utils::is_file(output_root));

    // keep the braid data only
    output_file = conduit::utils::join_file_path(output_path,"tout_relay_topo_select_case_2");
    output_root = output_file + ".cycle_000100.root";
    // remove old outputs
    remove_test_file(output_root);
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/topologies"].reset();
    extracts["e1/params/topologies"].append() = "mesh";
    ascent.execute(actions);

    EXPECT_TRUE(conduit::utils::is_file(output_root));

    // keep the braid data + iters (julia)
    output_file = conduit::utils::join_file_path(output_path,"tout_relay_topo_select_case_3");
    output_root = output_file + ".cycle_000100.root";
    // remove old outputs
    remove_test_file(output_root);
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/topologies"].reset();
    extracts["e1/params/topologies"].append() = "mesh";
    extracts["e1/params/fields"].append() = "iters";
    ascent.execute(actions);

    EXPECT_TRUE(conduit::utils::is_file(output_root));

    ascent.close();


}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_subselection_no_data)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_subset_no_data");
    string output_root = output_file + ".cycle_000100.root";
    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append() = "bananas";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    bool exception = false;
    try
    {
      ascent.execute(actions);
    }
    catch(const conduit::Error &e)
    {
      exception = true;
      std::cout<<e.what()<<"\n";
    }
    ascent.close();
    ASSERT_TRUE(exception);
}
//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_subselection)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_serial_extract_subset");
    string output_root = output_file + ".cycle_000100.root";
    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append() = "braid";
    extracts["e1/params/fields"].append() = "radial";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    std::string msg = "An example of using a relay extract to save a subset of the data.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);

}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_no_cycle)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    conduit::Node state = data["state"];
    data["state"].reset();
    data["state/time"] = state["time"];

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_no_cycle");
    string output_root = output_file + ".cycle_000100.root";
    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append() = "braid";
    extracts["e1/params/fields"].append() = "radial";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_bp_num_files)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract num_files option in serial");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        std::cout << "test nfiles = " << nfiles << std::endl;
        oss.str("");
        oss << "tout_relay_serial_extract_nfiles_" << nfiles;

        string output_base = conduit::utils::join_file_path(output_path,
                                                            oss.str());

        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        // remove existing directory
        utils::remove_directory(output_dir);
        utils::remove_directory(output_root);

        conduit::Node actions;
        // add the extracts
        conduit::Node &add_extracts = actions.append();
        add_extracts["action"] = "add_extracts";
        conduit::Node &extracts = add_extracts["extracts"];

        extracts["e1/type"]  = "relay";
        extracts["e1/params/path"] = output_base;
        extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
        extracts["e1/params/num_files"] =  nfiles;

        //
        // Run Ascent
        //

        Ascent ascent;
        ascent.open();
        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        char fmt_buff[64] = {0};
        for(int i=0;i<nfiles_to_check;i++)
        {

            std::string fprefix = "file_";
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is
                // domain_
                fprefix = "domain_";
            }
            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",i);
            oss.str("");
            oss << conduit::utils::join_file_path(output_base + ".cycle_000000",
                                                  fprefix)
                << fmt_buff << ".hdf5";
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }
    }
}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_sparse_topos)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //

    Node data;
    ostringstream oss;

    // three domains with different topos
    for(index_t d =0; d<4; d++)
    {
        std::cout << "i = " << d << std::endl;
        Node &mesh = data.append();
        oss.str("");
        oss << "my_coords_" << d ;
        std::string c_name = oss.str();

        oss.str("");
        oss << "my_topo_" << d ;
        std::string t_name = oss.str();

        oss.str("");
        oss << "my_field_" << d ;
        std::string f_name = oss.str();

        // create the coordinate set
        mesh["coordsets"][c_name]["type"] = "uniform";
        mesh["coordsets"][c_name]["dims/i"] = 3;
        mesh["coordsets"][c_name]["dims/j"] = 3;
        // add origin and spacing to the coordset (optional)
        mesh["coordsets"][c_name]["origin/x"] = -10.0;
        mesh["coordsets"][c_name]["origin/y"] = -10.0;
        mesh["coordsets"][c_name]["spacing/dx"] = 10.0;
        mesh["coordsets"][c_name]["spacing/dy"] = 10.0;

        // add the topology
        // this case is simple b/c it's implicitly derived from the coordinate set
        mesh["topologies"][t_name]["type"] = "uniform";
        // reference the coordinate set by name
        mesh["topologies"][t_name]["coordset"] = c_name;

        // add a simple element-associated field
        mesh["fields"][f_name]["association"] =  "element";
        // reference the topology this field is defined on by name
        mesh["fields"][f_name]["topology"] =  t_name;
        // set the field values, for this case we have 4 elements
        mesh["fields"][f_name]["values"].set(DataType::float64(4));

        float64 *ele_vals_ptr = mesh["fields"][f_name]["values"].value();

        for(int i=0;i<4;i++)
        {
            ele_vals_ptr[i] = float64(d);
        }
    }

    data.print();

    Node verify_info;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract in serial");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_sparse_topos");
    string output_root = output_file + ".cycle_000100.root";

    // remove old outputs
    remove_test_file(output_root);

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();


    Node n_root;
    conduit::relay::io::load(output_file + ".cycle_000000.root","hdf5",n_root);
    n_root.print();
}



//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_lor_extract)
{
    Node n;
    ascent::about(n);
    
    if(n["runtimes/ascent/mfem/status"].as_string() == "disabled")
    {
      std::cout << "mfem disabled: skipping test that requires mfem for lor" << std::endl;
      return;
    }

    //
    // load example mesh
    //
    Node data,verify_info;
    conduit::relay::io::blueprint::read_mesh(test_data_file("taylor_green.cycle_001860.root"),
                                             data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // create extract as published (no lor)
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_ho_no_lor");

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    Node &extracts = add_extracts["extracts"];

    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    // check that we created an extract file
    EXPECT_TRUE(conduit::utils::is_file(output_file + ".cycle_001860.root"));

    output_file = conduit::utils::join_file_path(output_path,"tout_relay_ho_lor_5");

    // LOR and save as create extract
    conduit::Node actions2;
    // add the extracts
    conduit::Node &add_extracts2 = actions2.append();
    add_extracts2["action"] = "add_extracts";
    Node &extracts2 = add_extracts2["extracts"];

    extracts2["e1/type"]  = "relay";

    extracts2["e1/params/path"] = output_file;
    extracts2["e1/params/refinement_level"] = 5;
    extracts2["e1/params/protocol"] = "blueprint";

    std::cout << actions2.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent2;
    ascent2.open();
    ascent2.publish(data);
    ascent2.execute(actions2);
    ascent2.close();
    // check that we created an extract file
    EXPECT_TRUE(conduit::utils::is_file(output_file + ".cycle_001860.root"));
    std::string msg = "An example of creating a relay extract "
                      "with low order refined data.";
    ASCENT_ACTIONS_DUMP(actions2,output_file,msg);
}

#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
//-----------------------------------------------------------------------------
TEST(ascent_relay, silo_spiral_multi_file)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);
    add_matset_to_spiral(data, 7);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract silo num_files option");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        oss.str("");
        oss << "tout_relay_silo_extract_nfiles_" << nfiles;

        std::string output_base = conduit::utils::join_file_path(output_path,
                                                                 oss.str());

        std::string output_dir  = output_base + ".cycle_000000";
        std::string output_root = output_base + ".cycle_000000.root";

        // remove existing directory
        utils::remove_directory(output_dir);
        utils::remove_directory(output_root);

        conduit::Node actions;
        // add the extracts
        conduit::Node &add_extracts = actions.append();
        add_extracts["action"] = "add_extracts";
        conduit::Node &extracts = add_extracts["extracts"];

        extracts["e1/type"]  = "relay";
        extracts["e1/params/path"] = output_base;
        extracts["e1/params/protocol"] = "silo";
        extracts["e1/params/num_files"] =  nfiles;

        //
        // Run Ascent
        //

        Ascent ascent;

        Node ascent_opts;
        ascent_opts["runtime"] = "ascent";
        ascent.open(ascent_opts);
        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        for(int i=0;i<nfiles_to_check;i++)
        {
            std::string fprefix = conduit_fmt::format("file_{:06d}.silo", i);
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is
                // domain_
                fprefix = conduit_fmt::format("domain_{:06d}.silo", i);
            }
            oss.str("");
            oss << conduit::utils::join_file_path(output_base + ".cycle_000000",
                                                  fprefix);
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }
    }
}
#endif

#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
//-----------------------------------------------------------------------------
TEST(ascent_relay, overlink_spiral_multi_file)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);
    add_matset_to_spiral(data, 7);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract overlink num_files option");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        oss.str("");
        oss << "tout_relay_overlink_extract_nfiles_" << nfiles;

        std::string output_base = conduit::utils::join_file_path(output_path,
                                                                 oss.str());

        std::string output_dir  = output_base;
        std::string output_root = conduit::utils::join_file_path(output_dir,
                                                                 "OvlTop.silo");;

        // remove existing directory
        utils::remove_directory(output_dir);
        utils::remove_directory(output_root);

        conduit::Node actions;
        // add the extracts
        conduit::Node &add_extracts = actions.append();
        add_extracts["action"] = "add_extracts";
        conduit::Node &extracts = add_extracts["extracts"];

        extracts["e1/type"]  = "relay";
        extracts["e1/params/path"] = output_base;
        extracts["e1/params/protocol"] = "overlink";
        extracts["e1/params/num_files"] =  nfiles;

        //
        // Run Ascent
        //

        Ascent ascent;

        Node ascent_opts;
        ascent_opts["runtime"] = "ascent";
        ascent.open(ascent_opts);
        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        for(int i=0;i<nfiles_to_check;i++)
        {
            std::string fprefix = conduit_fmt::format("domfile{:d}.silo", i);
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is
                // domain_
                fprefix = conduit_fmt::format("domain{:d}.silo", i);
            }
            oss.str("");
            oss << conduit::utils::join_file_path(output_base,
                                                  fprefix);
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }
    }
}
#endif


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



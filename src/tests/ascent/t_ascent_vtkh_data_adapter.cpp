//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_vtkh_data_adapter.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"
#include "t_vtkm_test_utils.hpp"

#include <ascent.hpp>
#include <runtimes/ascent_vtkh_data_adapter.hpp>
#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_uniform_2d_to_blueprint)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    vtkm::cont::DataSet ds = Make2DUniformDataSet0();
    conduit::Node blueprint;
    bool zero_copy = false;
    std::string topo_name = "topo";
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint,topo_name, false);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_uniform_3d_to_blueprint)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    vtkm::cont::DataSet ds = Make3DUniformDataSet0();
    conduit::Node blueprint;
    bool zero_copy = false;
    std::string topo_name = "topo";
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint,topo_name, false);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}

//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_rectilinear_3d_to_blueprint)
{
#ifdef VTKM_ENABLE_KOKKOS
    vtkh::InitializeKokkos();
#endif
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    vtkm::cont::DataSet ds = Make3DRectilinearDataSet0();
    conduit::Node blueprint;
    bool zero_copy = false;
    std::string topo_name = "topo";
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint,topo_name, false);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_rectilinear_2d_to_blueprint)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    vtkm::cont::DataSet ds = Make2DRectilinearDataSet0();
    conduit::Node blueprint;
    bool zero_copy = false;
    std::string topo_name = "topo";
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint,topo_name, false);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);
}

//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, vtkm_explicit_single_type_to_blueprint)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    vtkm::cont::DataSet ds = Make3DExplicitDataSetCowNose();
    conduit::Node blueprint;
    bool zero_copy = false;
    std::string topo_name = "topo";
    VTKHDataAdapter::VTKmToBlueprintDataSet(&ds, blueprint,topo_name, false);
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",blueprint,info);
    if(!success) info.print();
    EXPECT_TRUE(success);


    // Write out the data set for debugging
    blueprint["state/cycle"] = 1;
    blueprint["state/domain_id"] = 0;
    string output_path = "";
    output_path = prepare_output_dir();

    string output_file = conduit::utils::join_file_path(output_path,"tout_explicit_vtkm_converions");

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    Node ascent_opts;
    Ascent ascent;
    ascent.open(ascent_opts);
    ascent_opts["runtime"] = "ascent";
    ascent.publish(blueprint);
    ascent.execute(actions);
    ascent.close();

}
//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, zero_copy_test)
{
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("testing zero copy bp -> vtkm -> bp");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_contour_extract");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 0.;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
}


//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, consistent_domain_ids_check)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    Node multi_dom;
    Node &mesh1 = multi_dom.append();
    Node &mesh2 = multi_dom.append();
    conduit::blueprint::mesh::examples::braid("hexs",
                                              2,
                                              2,
                                              2,
                                              mesh1);
    conduit::blueprint::mesh::examples::braid("hexs",
                                              2,
                                              2,
                                              2,
                                              mesh2);
    mesh1.remove("state");
    mesh2["state/domain_id"] = 1;
    bool consistent_ids = false;


    //
    // Publish inconsistent ids to Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    EXPECT_THROW(ascent.publish(multi_dom),conduit::Error);
    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_data_adapter, interleaved_3d)
{
    // CYRUSH: I tried recreate an issue with interleaved coords
    // we hit in AMReX with this test case, however it does not
    // replicate it  (rendering still works with the vtk-m interleaved logic)
    // It is still a good basic interleaved test case.
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    Node mesh;
    conduit::blueprint::mesh::examples::braid("points",
                                              10,
                                              10,
                                              10,
                                              mesh);

    // change the x,y,z coords to interleaved

    Node icoords;
    conduit::blueprint::mcarray::to_interleaved(mesh["coordsets/coords/values"],icoords);
    mesh["coordsets/coords/values"].set_external(icoords);

    EXPECT_TRUE(conduit::blueprint::mcarray::is_interleaved(mesh["coordsets/coords/values"]));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                    "tout_render_3d_interleaved");

    // remove old images before rendering
    remove_test_image(output_file);


    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    // make sure we can render interleaved data
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(mesh);
    ascent.execute(actions);
    ascent.close();

    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file,0.01f));
}

//-----------------------------------------------------------------------------
TEST(ascent_multi_topo, adapter_test)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    ASCENT_INFO("Testing round trip of multi_topo");
    //
    // Create an example mesh convert it, and convert it back.
    //
    Node data;
    build_multi_topo(data, EXAMPLE_MESH_SIDE_DIM);

    VTKHCollection* collection = VTKHDataAdapter::BlueprintToVTKHCollection(data,true);

    Node out_data;
    VTKHDataAdapter::VTKHCollectionToBlueprintDataSet(collection, out_data);

    Node verify_info;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(out_data, verify_info));
    delete collection;
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



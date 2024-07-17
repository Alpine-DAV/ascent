//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_2d_mixed.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
void
gen_example_2d_mixed_mesh(Node &data)
{
    data.reset();

    Node verify_info;
    //
    // Create example mesh.
    //

    // create simple mixed 2d mesh with triangles and quads
    /*
    3     *-------*-------*       (8, 9, 10)
         / \     / \     / \
        / d \ e / f \ g / h \
       /     \ /     \ /     \
    1  *------*-------*------*    (4, 5, 6, 7)
       |   a  |   b   |  c   |
    0  *------*-------*------*
       0      1       2      3
    */
    
    data["coordsets/coords/type"] = "explicit";
    data["coordsets/coords/values/x"].set(DataType::float64(11));
    data["coordsets/coords/values/y"].set(DataType::float64(11));
    
    data["coordsets/coords/values/x"] = { 0.0, 1.0, 2.0, 3.0,
                                          0.0, 1.0, 2.0, 3.0,
                                          0.5, 1.5, 2.5 };
    data["coordsets/coords/values/y"] = { 0.0, 0.0, 0.0, 0.0,
                                          1.0, 1.0, 1.0, 1.0,
                                          3.0, 3.0, 3.0};

    data["topologies/topo/type"] = "unstructured";
    data["topologies/topo/coordset"] = "coords";
    data["topologies/topo/elements/shape"] = "mixed";
    data["topologies/topo/elements/shape_map/tri"]  = 5;
    data["topologies/topo/elements/shape_map/quad"] = 9;
    data["topologies/topo/elements/shapes"] = { 9, 9, 9,
                                                5, 5, 5, 5, 5};
    data["topologies/topo/elements/sizes"] =  { 4, 4, 4,
                                                3, 3, 3, 3, 3};
    data["topologies/topo/elements/offsets"] =  {0, 4, 8,
                                                 12, 15, 18, 21, 24};
   
    data["topologies/topo/elements/connectivity"] =  {0, 1, 5, 4,
                                                      1, 2, 6, 5,
                                                      2, 3, 7, 6,
                                                      4, 5, 8,
                                                      8, 5, 9,
                                                      5, 6, 9,
                                                      9, 6, 10,
                                                      6, 7, 10};

    data["fields/ele_id/topology"] = "topo";
    data["fields/ele_id/association"] = "element";
    data["fields/ele_id/values"] = { 0, 1, 2,
                                     3, 4, 5, 6, 7};

    // also add a points topo to help with debugging

    data["topologies/pts/type"] = "points";
    data["topologies/pts/coordset"] = "coords";
    data["fields/pts_id/topology"] = "pts";
    data["fields/pts_id/association"] = "element";
    data["fields/pts_id/values"] = { 0, 1, 2, 3,
                                     4, 5, 6, 7, 
                                     8, 9, 10};

    std::cout << data.to_yaml() << std::endl;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
    
    //std::cout << verify_info.to_yaml() << std::endl;
}


//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_2d_mixed)
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

    Node data;
    gen_example_2d_mixed_mesh(data);

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_2d_mixed");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p3/type"]  = "mesh";
    scenes["s1/plots/p3/topology"] = "topo";
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "ele_id";
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "pts_id";
    scenes["s1/plots/p2/points/radius"] = .15;
    scenes["s1/renders/r1/image_prefix"] = output_file;
    // TODO: This isn't changing the view in 2D ...
    //scenes["s1/renders/r1/camera/zoom"] = .5;
    actions.print();

    //
    // Run Ascent
    //
    conduit::Node info;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(info);
    ascent.close();

    // for debugging help
    // std::cout << info.to_yaml() << std::endl;

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_2d_mixed_bad_shape_ids_error)
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

    Node data;
    gen_example_2d_mixed_mesh(data);

    // add bogus shape map entry
    data["topologies/topo/elements/shape_map/bananas"]  = 42;

    // actions to try to draw
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "ele_id";
    actions.print();

    //
    // Run Ascent
    //
    conduit::Node info;
    Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
    Ascent ascent;
    ascent.open(ascent_opts);
    ascent.publish(data);
    // expect this to fail
    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();

    // for debugging help
    // std::cout << info.to_yaml() << std::endl;

}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_extract_and_render_2d_mixed)
{
    // exec a pipeline to make sure we got through the vtk-m conversion 
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    Node data;
    gen_example_2d_mixed_mesh(data);

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_2d_mixed_thresh_input");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    // apply threshold to make sure we do vtk-m round trip

    Node actions;
    Node &add_act = actions.append();
    add_act["action"] = "add_pipelines";
    Node &pipelines = add_act["pipelines"];

    // add a threshold (f1)
    pipelines["thresh/f1/type"] = "threshold";
    Node &thresh_params = pipelines["thresh/f1/params"];
    // keep first 6 elements
    thresh_params["field"]  = "ele_id";
    thresh_params["min_value"] = 0.0;
    thresh_params["max_value"] = 5.0;
    
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "ele_id";
    scenes["s1/plots/p1/pipeline"] = "thresh";
    scenes["s1/renders/r1/image_prefix"] = output_file;
    // TODO: This isn't changing the view in 2D ...
    //scenes["s1/renders/r1/camera/zoom"] = .5;

    string output_extract_root = conduit::utils::join_file_path(output_path,
                                                               "tout_render_2d_mixed_thresh_extract");

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "thresh";
    extracts["e1/params/path"] = output_extract_root;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    actions.print();

    //
    // Run Ascent
    //
    conduit::Node info;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(info);
    ascent.close();

    //std::cout << info.to_yaml() << std::endl;
 
    // check result
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));

    // load back the extract
    conduit::Node n_load, verify_info;


    // NOTE:
    // a bug in conduit root file creation logic leads to funky paths
    // when abs paths are used on windows
    // https://github.com/LLNL/conduit/issues/1297
    // so,  directly read the output mesh, instead of using mesh_load
    conduit::relay::io::load(output_extract_root + ".root:mesh","hdf5",n_load);

    // desired load post conduit bugfix
    //conduit::relay::io::blueprint::load_mesh(output_extract_root + ".root",n_load);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_final_render = conduit::utils::join_file_path(output_path,
                                                               "tout_render_2d_mixed_thresh_extract_render");
    actions.reset();
    conduit::Node &add_plots2 = actions.append();
    add_plots2["action"] = "add_scenes";
    conduit::Node &scenes2 = add_plots2["scenes"];
    scenes2["s1/plots/p1/type"]  = "pseudocolor";
    scenes2["s1/plots/p1/field"] = "ele_id";
    scenes2["s1/renders/r1/image_prefix"] = output_final_render;


    Ascent ascent2;
    ascent2.open();
    ascent2.publish(n_load);
    ascent2.execute(actions);
    ascent2.close();

    // check result
    EXPECT_TRUE(check_test_image(output_final_render, 0.001f, "0"));
    
}

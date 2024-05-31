//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_2d_poly.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;
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
    
    //
    // Create example mesh.
    //
    Node data, verify_info;

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
    data["topologies/elements/shape"] = "mixed";
    data["topologies/elements/shape_map/tri"]  = 5;
    data["topologies/elements/shape_map/quad"] = 9;
    data["topologies/elements/sizes"] =  {4, 4, 4,
                                          3, 3, 3, 3, 3, 3 };
    data["topologies/elements/offsets"] =  {0, 4, 8,
                                            12, 15, 18, 21, 24, 28};
   
    data["topologies/elements/connectivity"] =  {0, 1, 5, 4, 
                                                 1, 2, 6, 5, 
                                                 2, 3, 7, 6, 
                                                 4, 5, 8,
                                                 8, 5, 9,
                                                 5, 6, 9,
                                                 9, 6, 10,
                                                 6, 7, 10};
    
    
    data.print();

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

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
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "ele_id";
    scenes["s1/image_prefix"] = output_file;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_3d_mixed.cpp
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
TEST(ascent_pipeline, test_render_3d_mixed)
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

    // create simple mixed 3d mesh with hexs and pyramids
    /*
            *       *      * ( x: 0.5, 1.5, 2.5). 16, 17, 18
                             ( y:   2,   2,   2)
                             ( z: 0.5, 0.5, 0.5)

   1    *------*-------*------*  (12, 13, 14, 15) (back upper)
  z    /      /.      /.     /|
 0  1 *------*-------*------* |  (8, 9, 10, 11) (front upper)
    y |   a  |   b   |  c   | / 
    0 *------*-------*------*    (4, 5, 6, 7) (back lower)
      0      1       2      3
    */
    
    data["coordsets/coords/type"] = "explicit";
    data["coordsets/coords/values/x"] = { 0.0, 1.0, 2.0, 3.0,
                                          0.0, 1.0, 2.0, 3.0,
                                          0.0, 1.0, 2.0, 3.0,
                                          0.0, 1.0, 2.0, 3.0,
                                          0.5, 1.5, 2.5};

    data["coordsets/coords/values/y"] = { 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0,
                                          1.0, 1.0, 1.0, 1.0,
                                          1.0, 1.0, 1.0, 1.0,
                                          2.0, 2.0, 2.0};

    data["coordsets/coords/values/z"] = { 0.0, 0.0, 0.0, 0.0,
                                          1.0, 1.0, 1.0, 1.0,
                                          0.0, 0.0, 0.0, 0.0,
                                          1.0, 1.0, 1.0, 1.0,
                                          0.5, 0.5, 0.5};


    data["topologies/topo/type"] = "unstructured";
    data["topologies/topo/coordset"] = "coords";
    data["topologies/topo/elements/shape"] = "mixed";
    data["topologies/topo/elements/shape_map/hex"]  = 12;
    data["topologies/topo/elements/shape_map/pyramid"]  = 14;
    data["topologies/topo/elements/shapes"] =  { 12, 12, 12, 14, 14, 14 };
    data["topologies/topo/elements/sizes"] =   { 8, 8, 8, 5, 5, 5};
    data["topologies/topo/elements/offsets"] = { 0, 8, 16,
                                                 24, 29, 34};
   
    data["topologies/topo/elements/connectivity"] =  {  0,  1,  5,  4,  8,  9, 13, 12,
                                                        1,  2,  6,  5,  9, 10, 14, 13,
                                                        2,  3,  7,  6, 10, 11, 15, 14,
                                                        8,  9, 13, 12, 16,
                                                        9, 10, 14, 13, 17,
                                                       10, 11, 15, 14, 18
                                                    };

    data["fields/ele_id/topology"] = "topo";
    data["fields/ele_id/association"] = "element";
    data["fields/ele_id/values"] = { 0, 1, 2, 3, 4, 5};

    // also add a points topo to help with debugging

    data["topologies/pts/type"] = "points";
    data["topologies/pts/coordset"] = "coords";
    data["fields/pts_id/topology"] = "pts";
    data["fields/pts_id/association"] = "element";
    data["fields/pts_id/values"] = {  0,  1,  2,  3,
                                      4,  5,  6,  7,
                                      8,  9, 10, 11,
                                     12, 13, 14, 15,
                                     16, 17, 18};

    data.print();

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
    
    std::cout << verify_info.to_yaml() << std::endl;

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_3d_mixed");
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
    scenes["s1/renders/r1/camera/zoom"] = .5;
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


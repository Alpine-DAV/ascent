//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_tiling.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


//-----------------------------------------------------------------------------
TEST(ascent_tiling, test_tiling)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh made up of 2 tets.
    //
    Node data, verify_info;

    // create an explicit coordinate set
    double X[5] = { -1.0, 0.0, 0.0, 0.0, 1.0 };
    double Y[5] = { 0.0, -1.0, 0.0, 1.0, 0.0 };
    double Z[5] = { 0.0, 0.0, 1.0, 0.0, 0.0 };
    data["coordsets/coords/type"] = "explicit";
    data["coordsets/coords/values/x"].set_external(X, 5);
    data["coordsets/coords/values/y"].set_external(Y, 5);
    data["coordsets/coords/values/z"].set_external(Z, 5);


    // add an unstructured topology
    data["topologies/mesh/type"] = "unstructured";
    // reference the coordinate set by name
    data["topologies/mesh/coordset"] = "coords";
    // set topology shape type
    data["topologies/mesh/elements/shape"] = "tet";
    // add a connectivity array for the tets
    int64 connectivity[8] = { 0, 1, 3, 2, 4, 3, 1, 2 };
    data["topologies/mesh/elements/connectivity"].set_external(connectivity, 8);

    const int num_elements = 2;
    float var1_vals[num_elements] = { 0, 1 };
    float var2_vals[num_elements] = { 1, 0 };
    
    // create a field named var1
    data["fields/var1/association"] = "element";
    data["fields/var1/topology"] = "mesh";
    data["fields/var1/values"].set_external(var1_vals, 2);

    // create a field named var2
    data["fields/var2/association"] = "element";
    data["fields/var2/topology"] = "mesh";
    data["fields/var2/values"].set_external(var2_vals, 2);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing Tiling");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_tiling");

    //
    // Create the actions.
    //

    conduit::Node actions;
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";

    conduit::Node &scenes = add_scenes["scenes"];

    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "var1";
    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"] = "600";
    scenes["s1/renders/r1/image_height"] = "600";
    scenes["s1/renders/r1/tile_width"] = "200";
    float64 look_at_vals[3] = {0.0, 0.0, 0.0};
    Node look_at;
    look_at.set(look_at_vals, 3);
    float64 position_vals[3] = {0.0, 0.0, 3.4641};
    Node position;
    position.set(position_vals, 3);
    float64 up_vals[3] = {0.0, 1.0, 0.0};
    Node up;
    up.set(up_vals, 3);
    scenes["s1/renders/r1/camera/look_at"] = look_at;
    scenes["s1/renders/r1/camera/position"] = position;
    scenes["s1/renders/r1/camera/up"] = up;
    scenes["s1/renders/r1/camera/fov"] = "60.";
    scenes["s1/renders/r1/camera/xpan"] = "-0.125";
    scenes["s1/renders/r1/camera/ypan"] = "-0.125";
    scenes["s1/renders/r1/camera/zoom"] = "0.75";
    scenes["s1/renders/r1/camera/azimuth"] = "0.";
    scenes["s1/renders/r1/camera/elevation"] = "0.";
    scenes["s1/renders/r1/camera/near_plane"] = "0.1";
    scenes["s1/renders/r1/camera/far_plane"] = "100.1";
    scenes["s1/renders/r1/image_prefix"] = output_file;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);

    //
    // Loop over several image width, image height and tile size
    // combinations.
    //
    vector<string> image_widths  = {"600", "600", "400", "600"};
    vector<string> image_heights = {"600", "400", "600", "400"};
    vector<string> tile_widths   = {"200", "200", "200", "256"};
    for (int i = 0; i < image_widths.size(); ++i)
    {
        // remove the old image before rendering
        remove_test_image(output_file, i);

	// render the image
        scenes["s1/renders/r1/image_width"] = image_widths[i];
        scenes["s1/renders/r1/image_height"] = image_heights[i];
        scenes["s1/renders/r1/tile_width"] = tile_widths[i];
        ascent.execute(actions);

        // check the image we created
        EXPECT_TRUE(check_test_image(output_file, 0.001f, i));
        std::string msg = "Tiling an image.";
        ASCENT_ACTIONS_DUMP_CYCLE(actions,output_file, msg, i);
    }

    ascent.close();
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}



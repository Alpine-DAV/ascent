//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_transform.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
bool
viskores_avalible()
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
void
setup_input_mesh(Node &data)
{
    //
    // Create an example mesh.
    //
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
}


//-----------------------------------------------------------------------------
void
setup_output_file(const std::string &tout_name, std::string &output_file)
{
    string output_path = prepare_output_dir();
    output_file = conduit::utils::join_file_path(output_path,tout_name);

    // remove old images before rendering
    remove_test_image(output_file);
}


//-----------------------------------------------------------------------------
void
setup(const std::string &tout_name, Node &data, std::string &output_file)
{
    setup_input_mesh(data);
    setup_output_file(tout_name,output_file);
}


//-----------------------------------------------------------------------------
TEST(ascent_transform, test_translate)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_translate",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "transform";
    // filter knobs
    conduit::Node &p = pipelines["pl1/f1/params"];
    // translate by x and y
    p["translate/x"] = 23.0;
    p["translate/y"] = 15.0;
    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using translation.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_scale)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_scale",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/scale/x"]= 2.0;
    pipelines["pl1/f1/params/scale/y"]= 0.5;
    pipelines["pl1/f1/params/scale/z"]= 2.0;
    // then translate x
    pipelines["pl1/f2/type"] = "transform";
    pipelines["pl1/f2/params/translate/y"]= 50.0;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using scale.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_rotate_x)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_rotate_x",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/rotate/angle"]= 45.0;
    pipelines["pl1/f1/params/rotate/axis/x"]= 1.0;
    // then translate x
    pipelines["pl1/f2/type"] = "transform";
    pipelines["pl1/f2/params/translate/y"]= 50.0;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter rotating around the x-axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_transform, test_rotate_y)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_rotate_y",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/rotate/angle"]= 45.0;
    pipelines["pl1/f1/params/rotate/axis/y"]= 1.0;
    // then translate x
    pipelines["pl1/f2/type"] = "transform";
    pipelines["pl1/f2/params/translate/y"]= 50.0;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter rotating around the y-axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_rotate_z)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_rotate_z",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/rotate/angle"]= 45.0;
    pipelines["pl1/f1/params/rotate/axis/z"]= 1.0;
    // then translate x
    pipelines["pl1/f2/type"] = "transform";
    pipelines["pl1/f2/params/translate/y"]= 50.0;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter rotating around the z-axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_rotate_arb)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_rotate_arb",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/rotate/angle"]= 45.0;
    pipelines["pl1/f1/params/rotate/axis/x"]= .5;
    pipelines["pl1/f1/params/rotate/axis/y"]= 1.0;
    // then translate x
    pipelines["pl1/f2/type"] = "transform";
    pipelines["pl1/f2/params/translate/y"]= 50.0;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter rotating around an arbitrary axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_transform, test_matrix)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_matrix",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // filter knobs
    // scale
    pipelines["pl1/f1/type"] = "transform";
    // this matrix is equiv to
    // scale/x = 2.0
    // scale/y = 0.5
    // scale/z = 2.0
    // and
    // translate/y = 50.0
    pipelines["pl1/f1/params/matrix"] = { 2.0, 0.0, 0.0,  0.0,
                                          0.0, 0.5, 0.0, 50.0,
                                          0.0, 0.0, 2.0,  0.0,
                                          0.0, 0.0, 0.0,  1.0} ;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render translated
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // render orig (w/ different field so we can tell them apart easily)
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter rotating around an arbitrary axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_bad_params)
{
    if(!viskores_avalible())
    {
        return;
    }

    conduit::Node data;
    setup_input_mesh(data);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    {
        Ascent ascent;
        conduit::Node ascent_opts;
        ascent_opts["exceptions"] = "forward";
        ascent.open(ascent_opts);
        ascent.publish(data);
        pipelines["pl1/f1/type"] = "transform";

        // too many
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/rotate/x"]= 45.0;
        pipelines["pl1/f1/params/translate/x"]= 45.0;
        pipelines["pl1/f1/params/scale/x"]= 45.0;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        // reflect missing normal/x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/reflect/normal/xx"]= 45.0;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        // translate missing x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/translate/zz"]= 45.0;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        // scale missing x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/scale/xx"]= 45.0;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        // rot missing axis
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/rotate/angle"]= 45.0;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        // matrix bad size
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/matrix"]= { 45.0 , 0.0} ;
        EXPECT_THROW(ascent.execute(actions),conduit::Error);

        ascent.close();
    }
    // now lets see the errors after checking they are thrown
    {
        Ascent ascent;
        // no forward
        ascent.open();
        ascent.publish(data);
        pipelines.reset();
        pipelines["pl1/f1/type"] = "transform";

        // too many
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/rotate/x"]= 45.0;
        pipelines["pl1/f1/params/translate/x"]= 45.0;
        pipelines["pl1/f1/params/scale/x"]= 45.0;
        ascent.execute(actions);

        // reflect missing normal/x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/reflect/normal/xx"]= 45.0;
        ascent.execute(actions);

        // translate missing x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/translate/zz"]= 45.0;
        ascent.execute(actions);

        // scale missing x,y,z
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/scale/xx"]= 45.0;
        ascent.execute(actions);

        // rot missing axis
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/rotate/angle"]= 45.0;
        ascent.execute(actions);

        // matrix bad size
        pipelines["pl1/f1/params"].reset();
        pipelines["pl1/f1/params/matrix"]= { 45.0 , 0.0} ;
        ascent.execute(actions);

    }
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_x)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_x",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/translate/x"]= 10.0;
    pipelines["pl0/f1/params/translate/y"]= 10.0;

    // filter knobs
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/reflect/normal/x"]= 1.0;
    pipelines["pl1/pipeline"] = "pl0";

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/pipeline"] = "pl0";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect across x axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_arb)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_arb",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/translate/x"]= 10.0;
    pipelines["pl0/f1/params/translate/y"]= 10.0;

    // filter knobs
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/reflect/normal/x"]= 1.0;
    pipelines["pl1/f1/params/reflect/normal/y"]= 1.0;
    pipelines["pl1/pipeline"] = "pl0";

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/pipeline"] = "pl0";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect across arbitrary axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_y)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_y",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/translate/x"]= 10.0;
    pipelines["pl0/f1/params/translate/y"]= 10.0;

    // filter knobs
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/reflect/normal/y"]= 1.0;
    pipelines["pl1/pipeline"] = "pl0";

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/pipeline"] = "pl0";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect across the y-axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_x_max)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_x_max",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/translate/x"]= 10.0;
    pipelines["pl0/f1/params/translate/y"]= 10.0;

    // filter knobs
    pipelines["pl1/f1/type"] = "transform";
    pipelines["pl1/f1/params/reflect/normal/x"]= 1.0;
    pipelines["pl1/f1/params/reflect/point/x"]= "max";
    pipelines["pl1/pipeline"] = "pl0";


    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/pipeline"] = "pl0";

    scenes["s1/image_prefix"] = output_file;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect across the x-axis maximum.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_y_min_2d)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_y_min_2d",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/translate/x"]= 10.0;
    pipelines["pl0/f1/params/translate/y"]= 10.0;

    pipelines["pl1/f1/type"] = "slice";
    pipelines["pl1/f1/params/point/x"]= 0.0;
    pipelines["pl1/f1/params/point/y"]= 0.0;
    pipelines["pl1/f1/params/point/z"]= 0.0;

    pipelines["pl1/f1/params/normal/x"]= 0.0;
    pipelines["pl1/f1/params/normal/y"]= 0.0;
    pipelines["pl1/f1/params/normal/z"]= 1.0;
    //pipelines["pl1/pipeline"] = "pl0";
    // filter knobs
    pipelines["pl2/f1/type"] = "transform";
    pipelines["pl2/f1/params/reflect/normal/y"]= 1;
    pipelines["pl2/f1/params/reflect/point/y"]= "min";
    pipelines["pl2/pipeline"] = "pl1";


    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl2";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;


    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect a 2D slice across y axis minimum.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_reflect_over_point)
{
    if(!viskores_avalible())
    {
        return;
    }

    std::string output_file;
    conduit::Node data;
    setup("tout_transform_reflect_over_point",data,output_file);

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    // filter knobs
    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/reflect/normal/x"]= 1;
    pipelines["pl0/f1/params/reflect/point/x"]= 15;
    pipelines["pl0/f1/params/reflect/point/y"]= 0;
    pipelines["pl0/f1/params/reflect/point/z"]= 0;


    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // render transformed
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    // and orig
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "radial";
    scenes["s1/plots/p2/pipeline"] = "pl0";

    scenes["s1/image_prefix"] = output_file;


    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter to reflect across a point.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_transform, test_transform_selected_topo)
{
    if(!viskores_avalible())
    {
        return;
    }

    conduit::Node data;
    setup_input_mesh(data);

    // add a points topo to test
    data["coordsets/pts_coords/type"] = "explicit";
    data["coordsets/pts_coords/values/x"] = { -5.0, -2.5, 0.0, 2.5, 5.0};
    data["coordsets/pts_coords/values/y"] = { -2.0, -2.0, 5.0, 2.0, 2.0};
    data["coordsets/pts_coords/values/z"] = { -5.0, -5.0, 0.0, 5.0, 5.0};

    data["topologies/pts_topo/type"] = "points";
    data["topologies/pts_topo/coordset"] = "pts_coords";
    data["fields/pts_topo_vals/association"] = "vertex";
    data["fields/pts_topo_vals/topology"] = "pts_topo";
    data["fields/pts_topo_vals/values"] = { -1.0, -2.0, -3.0, -4.0, -5.0 };

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    // filter knobs
    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/topology"]= "pts_topo";
    pipelines["pl0/f1/params/translate/x"]= 42;
    pipelines["pl0/f1/params/translate/y"]= 42;
    pipelines["pl0/f1/params/translate/z"]= 42;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // before
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "pts_topo_vals";
    scenes["s1/plots/p2/points/radius"] = 1.0;

    // after
    scenes["s2/plots/p1/type"]  = "pseudocolor";
    scenes["s2/plots/p1/field"] = "braid";
    scenes["s2/plots/p1/pipeline"] = "pl0";

    scenes["s2/plots/p2/type"]  = "pseudocolor";
    scenes["s2/plots/p2/field"] = "pts_topo_vals";
    scenes["s2/plots/p2/points/radius"] = 1.0;

    scenes["s2/plots/p2/pipeline"] = "pl0";

    std::string output_file_bf, output_file_af;
    setup_output_file("tout_transform_selected_topo_before",output_file_bf);
    scenes["s1/image_prefix"] = output_file_bf;
    setup_output_file("tout_transform_selected_topo_after",output_file_af);
    scenes["s2/image_prefix"] = output_file_af;

    actions.print();

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check vs baselines
    EXPECT_TRUE(check_test_image(output_file_bf));
    EXPECT_TRUE(check_test_image(output_file_af));
    std::string msg = "An example of applying a transform to a selected topology";
    ASCENT_ACTIONS_DUMP(actions,output_file_af,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_transform, test_transform_selected_topos)
{
    if(!viskores_avalible())
    {
        return;
    }

    conduit::Node data;
    setup_input_mesh(data);

    // add a points topo to test
    data["coordsets/pts_coords/type"] = "explicit";
    data["coordsets/pts_coords/values/x"] = { -5.0, -2.5, 0.0, 2.5, 5.0};
    data["coordsets/pts_coords/values/y"] = { -2.0, -2.0, 5.0, 2.0, 2.0};
    data["coordsets/pts_coords/values/z"] = { -5.0, -5.0, 0.0, 5.0, 5.0};

    data["topologies/pts_topo/type"] = "points";
    data["topologies/pts_topo/coordset"] = "pts_coords";
    data["fields/pts_topo_vals/association"] = "vertex";
    data["fields/pts_topo_vals/topology"] = "pts_topo";
    data["fields/pts_topo_vals/values"] = { -1.0, -2.0, -3.0, -4.0, -5.0 };

    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    // filter knobs
    pipelines["pl0/f1/type"] = "transform";
    pipelines["pl0/f1/params/topologies"].append() =  "mesh";
    pipelines["pl0/f1/params/translate/x"]= 42;
    pipelines["pl0/f1/params/translate/y"]= 42;
    pipelines["pl0/f1/params/translate/z"]= 42;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // before
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p2/type"]  = "pseudocolor";
    scenes["s1/plots/p2/field"] = "pts_topo_vals";
    scenes["s1/plots/p2/points/radius"] = 1.0;

    // after
    scenes["s2/plots/p1/type"]  = "pseudocolor";
    scenes["s2/plots/p1/field"] = "braid";
    scenes["s2/plots/p1/pipeline"] = "pl0";

    scenes["s2/plots/p2/type"]  = "pseudocolor";
    scenes["s2/plots/p2/field"] = "pts_topo_vals";
    scenes["s2/plots/p2/points/radius"] = 1.0;

    scenes["s2/plots/p2/pipeline"] = "pl0";

    std::string output_file_bf, output_file_af;
    setup_output_file("tout_transform_selected_topos_before",output_file_bf);
    scenes["s1/image_prefix"] = output_file_bf;
    setup_output_file("tout_transform_selected_topos_after",output_file_af);
    scenes["s2/image_prefix"] = output_file_af;

    actions.print();

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check vs baselines
    EXPECT_TRUE(check_test_image(output_file_bf));
    EXPECT_TRUE(check_test_image(output_file_af));
    std::string msg = "An example of applying a transform to selected topologies";
    ASCENT_ACTIONS_DUMP(actions,output_file_af,msg);
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



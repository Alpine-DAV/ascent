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
vtkm_avalible()
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
void
setup(Node &data)
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
setup(const std::string &tout_name, Node &data, std::string &output_file)
{
    setup(data);
    string output_path = prepare_output_dir();
    output_file = conduit::utils::join_file_path(output_path,tout_name);

    // remove old images before rendering
    remove_test_image(output_file);
}


//-----------------------------------------------------------------------------
TEST(ascent_translate, test_translate)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_scale)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_rotate_x)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_rotate_y)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_rotate_z)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_rotate_arb)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_matrix)
{
    if(!vtkm_avalible())
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_bad_params)
{
    if(!vtkm_avalible())
    {
        return;
    }

    conduit::Node data;
    setup(data);

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
TEST(ascent_translate, test_reflect_x)
{
    if(!vtkm_avalible())
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
    pipelines["pl1/f1/params/reflect/x"]= 1.0;
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_reflect_arb)
{
    if(!vtkm_avalible())
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
    pipelines["pl1/f1/params/reflect/x"]= 1.0;
    pipelines["pl1/f1/params/reflect/y"]= 1.0;
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

    std::cout << actions.to_yaml() << std::endl;

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
TEST(ascent_translate, test_reflect_y)
{
    if(!vtkm_avalible())
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
    pipelines["pl1/f1/params/reflect/y"]= 1.0;
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

    std::cout << actions.to_yaml() << std::endl;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example transform filter using reflect across y axis.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
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




//-----------------------------------------------------------------------------
///
/// file: t_ascent_AOS_rendering.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <numeric>
#include <math.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_AOS_extract, test_pass_thru)
{
    Node a;
    ascent::about(a);

    //
    // Create an example mesh with 8 vertices at 8 corners around (0,0,0)
    //
    int N = 8;
    Node mesh, verify_info;
    struct xyzrho
    {
        float64 x;
        float64 y;
        float64 z;
        float64 rho; // a density field with values between -3 and +4
    };
    index_t stride = sizeof(xyzrho);
    xyzrho points[N];
    int i;
    i = 0;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -3.0;
    i = 1;
    points[i].x  =  1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -2.0;
    i = 2;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -1.0;
    i = 3;
    points[i].x  =  1.0; points[i].y  =  1.0; points[i].z  = -1.0; points[i].rho  =  0.0;
    i = 4;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  1.0;
    i = 5;
    points[i].x  =  1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  2.0;
    i = 6;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  3.0;
    i = 7;
    points[i].x  =  1.0; points[i].y  =  1.0; points[i].z  =  1.0; points[i].rho  =  4.0;
    mesh["state/domain_id"] = 0;

    mesh["topologies/mesh/type"] = "unstructured";
    std::vector<conduit_int32> conn(N);
    std::iota(conn.begin(), conn.end(), 0);
    mesh["topologies/mesh/elements/connectivity"].set(conn);
    mesh["topologies/mesh/elements/shape"] = "point";
    mesh["coordsets/coords/type"] = "explicit";
    mesh["topologies/mesh/coordset"] = "coords";
    mesh["coordsets/coords/values/x"].set_external(&points[0].x, N, 0, stride);
    mesh["coordsets/coords/values/y"].set_external(&points[0].y, N, 0, stride);
    mesh["coordsets/coords/values/z"].set_external(&points[0].z, N, 0, stride);
    
    mesh["fields/rho/values"].set_external(        &points[0].rho, N, 0, stride);
    mesh["fields/rho/association"] = "vertex";
    mesh["fields/rho/topology"]    = "mesh";
    mesh["fields/rho/volume_dependent"].set("false");
    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, verify_info));

    ASCENT_INFO("Testing conduit extract in serial");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    // add the extract
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = "AOS";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);

    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_AOS_rendering, test_pass_thru)
{
    Node a;
    ascent::about(a);

    //
    // Create an example mesh with 8 vertices at 8 corners around (0,0,0)
    //
    int N = 8;
    Node mesh, verify_info;
    struct xyzrho
    {
        float64 x;
        float64 y;
        float64 z;
        float64 rho; // a density field with values between -3 and +4
    };
    index_t stride = sizeof(xyzrho);
    xyzrho points[N];
    int i;
    i = 0;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -3.0;
    i = 1;
    points[i].x  =  1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -2.0;
    i = 2;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  = -1.0; points[i].rho  = -1.0;
    i = 3;
    points[i].x  =  1.0; points[i].y  =  1.0; points[i].z  = -1.0; points[i].rho  =  0.0;
    i = 4;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  1.0;
    i = 5;
    points[i].x  =  1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  2.0;
    i = 6;
    points[i].x  = -1.0; points[i].y  = -1.0; points[i].z  =  1.0; points[i].rho  =  3.0;
    i = 7;
    points[i].x  =  1.0; points[i].y  =  1.0; points[i].z  =  1.0; points[i].rho  =  4.0;
    mesh["state/domain_id"] = 0;

    mesh["topologies/mesh/type"] = "unstructured";
    std::vector<conduit_int32> conn(N);
    std::iota(conn.begin(), conn.end(), 0);
    mesh["topologies/mesh/elements/connectivity"].set(conn);
    mesh["topologies/mesh/elements/shape"] = "point";
    mesh["coordsets/coords/type"] = "explicit";
    mesh["topologies/mesh/coordset"] = "coords";
    mesh["coordsets/coords/values/x"].set_external(&points[0].x, N, 0, stride);
    mesh["coordsets/coords/values/y"].set_external(&points[0].y, N, 0, stride);
    mesh["coordsets/coords/values/z"].set_external(&points[0].z, N, 0, stride);
    
    mesh["fields/rho/values"].set_external(        &points[0].rho, N, 0, stride);
    mesh["fields/rho/association"] = "vertex";
    mesh["fields/rho/topology"]    = "mesh";
    mesh["fields/rho/volume_dependent"].set("false");
    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, verify_info));

    ASCENT_INFO("Testing rendering AOS points in serial");
    
    conduit::Node actions;
    std::cerr << "input mesh" << std::endl;
    mesh.print();

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rho";
    scenes["s1/plots/p1/color_table/name"] = "viridis";
    scenes["s1/renders/r1/color_bar_position"].set({-0.9,0.9,0.8,0.85});
    scenes["s1/renders/r1/camera/azimuth"] = 30.0;
    scenes["s1/renders/r1/camera/elevation"] = 30.0;
    scenes["s1/renders/r1/image_prefix"] = "AOS_image_%04d"; 

    conduit::Node &add_scene = actions.append();
    add_scene["action"] = "add_scenes";
    add_scene["scenes"] = scenes;
    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);

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


